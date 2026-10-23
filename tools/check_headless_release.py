"""Exercise actual EXEs with pipes, invalid display settings and isolated PATH."""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import uuid


def check_dat(path, angles, values):
    """Compare the published exchange file to the actual EXE JSON result."""
    rows = [line.split() for line in path.read_text(encoding="ascii").splitlines()]
    assert len(rows) == len(angles) == len(values), path.name
    for row, angle, value in zip(rows, angles, values):
        assert len(row) == 2, path.name
        assert math.isclose(float(row[0]), angle, rel_tol=0, abs_tol=1e-9), path.name
        assert math.isclose(float(row[1]), value, rel_tol=0, abs_tol=1e-8), path.name


def main():
    p = argparse.ArgumentParser()
    p.add_argument("release", type=Path)
    p.add_argument("report", type=Path)
    p.add_argument("--gui-smoke", action="store_true")
    args = p.parse_args()
    release = args.release.resolve()
    output = args.report.resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    work = output / ("无头 验收-" + uuid.uuid4().hex[:8])
    work.mkdir()
    (work / "temp").mkdir()
    env = os.environ.copy()
    env.update({"PATH": str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"),
                "QT_QPA_PLATFORM": "invalid-test-platform", "DISPLAY": "", "MPLBACKEND": "QtAgg",
                "TEMP": str(work / "temp"), "TMP": str(work / "temp"),
                "MPLCONFIGDIR": str(work / "mpl"), "PYTHONNOUSERSITE": "1"})
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    checks = []

    def run(*flags, gui=False, expected=0):
        exe = release / ("ArrayLab.exe" if gui else "ArrayLab-cli.exe")
        captured = subprocess.run([str(exe), *map(str, flags)], cwd=work, env=env,
                                  capture_output=True, timeout=45)
        # Decode on this thread so a bad Windows encoding cannot be hidden by a reader-thread error.
        result = subprocess.CompletedProcess(captured.args, captured.returncode,
                                              captured.stdout.decode("utf-8"), captured.stderr.decode("utf-8"))
        assert result.returncode == expected, (result.returncode, result.stdout, result.stderr)
        return result

    sample = run("--config", release / "examples/ula.json")
    sample_doc = json.loads(sample.stdout)
    assert abs(sample_doc["metrics"]["directivity_dbi"] - 7.067566609281987) < 1e-9
    assert not sample.stderr
    checks.append("console_stdout_JSON_matches_manual_example")
    images = work / "图形 数据"
    result = run("--elements", 20, "--scan-deg", 30, "--plot", "both", "--include-phase", "--output-dir", images)
    summary = json.loads(result.stdout)
    assert json.loads((images / "summary.json").read_text(encoding="utf-8")) == summary
    assert abs(summary["metrics"]["directivity_dbi"] - 13.010299956639813) < 1e-9
    assert len((images / "pattern.dat").read_text().splitlines()) == 1801
    assert len((images / "pattern.dat").read_text().splitlines()[0].split()) == 3
    assert len((images / "excitation.csv").read_text().splitlines()) == 21
    checks.append("unicode_paths_JSON_DAT_CSV_exports")
    for name in ("rectangular.png", "polar.png"):
        assert (images / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    checks.append("PNG_export_with_invalid_Qt_platform_and_forced_QtAgg_environment")
    for flags in [("--elements", "0"), ("--unknown",)]:
        error = run(*flags, expected=2)
        assert not error.stdout
        assert json.loads(error.stderr)["error"]["type"] == "input"
    checks.append("invalid_arguments_stderr_JSON_exit_2")
    checksum = hashlib.sha256((images / "summary.json").read_bytes()).digest()
    error = run("--output-dir", images, expected=3)
    assert json.loads(error.stderr)["error"]["type"] == "io"
    assert hashlib.sha256((images / "summary.json").read_bytes()).digest() == checksum
    checks.append("existing_outputs_preserved_exit_3")
    windowed_out = work / "windowed-headless"
    run("--headless", "--config", release / "examples/ula.json", "--output-dir", windowed_out, gui=True)
    doc = json.loads((windowed_out / "summary.json").read_text(encoding="utf-8"))
    assert doc["metrics"] == sample_doc["metrics"]
    checks.append("windowed_EXE_headless_dispatch_without_QApplication")
    assert "--output-dir" in run("--help").stdout
    assert sample_doc["app_version"] in run("--version").stdout
    checks.append("help_and_version")
    report = {"success": True, "checks": checks, "work_directory": str(work), "app_version": sample_doc["app_version"],
              "host": platform.platform(), "release_directory": str(release),
              "environment_scope": "Existing host; system-only PATH, no PYTHONHOME/PYTHONPATH, fresh temp and plotting cache. Not a clean OS image."}
    if args.gui_smoke:
        gui_env = env.copy()
        gui_env["QT_QPA_PLATFORM"] = "windows"
        gui_env.pop("MPLBACKEND", None)
        smoke = work / "gui-smoke"
        process = subprocess.run([str(release / "ArrayLab.exe"), "--smoke-test", str(smoke)],
                                 cwd=work, env=gui_env, timeout=120)
        gui_report = json.loads((smoke / "smoke.json").read_text(encoding="utf-8"))
        assert process.returncode == 0 and gui_report["success"]
        report["gui_checks_passed"] = len(gui_report["checks"])
        scheme_result = json.loads(run("--config", smoke / "中文 分析方案.json").stdout)
        assert scheme_result["parameters"] == gui_report["scheme_result"]["parameters"]
        assert scheme_result["metrics"] == gui_report["scheme_result"]["metrics"]
        checks.append("desktop_saved_scheme_matches_headless_EXE_parameters_and_metrics")
        for topology in ("corporate", "series", "symmetric_series"):
            directory = work / f"cli-{topology}"
            feed = json.loads(run("--feed-config", smoke / f"feed-{topology}.json", "--output-dir", directory).stdout)["feed_network"]
            assert feed == gui_report["feed_results"][topology]
            for key, pattern in feed["electrical"]["patterns"].items():
                check_dat(directory / f"feed-{key}.dat", pattern["angles_deg"], pattern["pattern_db"])
            checks.append(f"{topology}_feed_GUI_headless_EXE_and_DAT_match")
        electrical_directory = work / "cli-electrical"
        electrical = json.loads(run("--feed-config", smoke / "feed-electrical.json", "--output-dir", electrical_directory).stdout)["feed_network"]
        assert electrical == gui_report["electrical_result"]
        for key, pattern in electrical["electrical"]["patterns"].items():
            check_dat(electrical_directory / f"feed-{key}.dat", pattern["angles_deg"], pattern["pattern_db"])
        checks.append("weighted_compensated_perturbed_electrical_GUI_CLI_and_DAT_match")
        planar_directory = work / "cli-planar"
        planar = json.loads(run("--planar-config", smoke / "planar-gui.json", "--output-dir", planar_directory).stdout)["planar"]
        assert planar == gui_report["planar_result"]
        for key, pattern in planar["electrical"]["patterns"].items():
            check_dat(planar_directory / f"planar-cut-{key}.dat", pattern["cut_angles_deg"], pattern["cut_db"])
        checks.append("arbitrary_planar_GUI_headless_EXE_and_DAT_match")
        for kind in ("linear", "planar"):
            directory = work / f"cli-study-{kind}"
            scan = json.loads(run("--study-config", smoke / f"study-{kind}.json", "--output-dir", directory,
                                  "--plot", "all").stdout)["study"]
            assert scan == gui_report["scan_results"][kind]
            assert json.loads((directory / "study-report.json").read_text(encoding="utf-8")) == scan
            assert (directory / "study.png").read_bytes().startswith(b"\x89PNG")
            for row in scan["records"]:
                check_dat(directory / f"study-{row['index']:03d}.dat", scan["angles_deg"], row["pattern_db"])
            checks.append(f"{kind}_scan_GUI_headless_EXE_DAT_PNG_and_saved_configuration_match")
        def same_numbers(a, b):
            if isinstance(a, dict):
                return isinstance(b, dict) and a.keys() == b.keys() and all(same_numbers(a[k], b[k]) for k in a)
            if isinstance(a, list):
                return isinstance(b, list) and len(a) == len(b) and all(same_numbers(x, y) for x, y in zip(a, b))
            if isinstance(a, (float, int)) and not isinstance(a, bool):
                return isinstance(b, (float, int)) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
            return a == b
        sparse_out = work / "cli-sparse"
        run("--sparse", "run", "--config", smoke / "sparse-gui-config.json", "--output", sparse_out)
        sparse = json.loads((sparse_out / "summary.json").read_text(encoding="utf-8"))
        reference = gui_report["sparse_results"]["unweighted"]
        assert sparse["config"] == reference["config"]
        assert len(sparse["results"]) == len(reference["results"])
        for actual, expected in zip(sparse["results"], reference["results"]):
            for key in ("name", "indices", "xy", "weights", "metrics", "phase_quantization"):
                assert same_numbers(actual[key], expected[key]), key
        checks.append("sparse_GUI_CLI_joint_hole_spacing_weights_and_quantization_match")
        reference = gui_report["sparse_results"]["weighted"]["results"][-1]
        settings = {k: reference["weighting"][k] for k in ("method", "dr_db", "sll_db", "nbar", "sigma")}
        weight_file = work / "sparse-weighting.json"
        weight_file.write_text(json.dumps(settings), encoding="utf-8")
        weighted_out = work / "cli-sparse-weighted"
        run("--sparse", "weight", "--input", sparse_out, "--source", "annealing", "--settings", weight_file,
            "--output", weighted_out)
        weighted = json.loads((weighted_out / "summary.json").read_text(encoding="utf-8"))
        for key in ("indices", "weights", "metrics", "phase_quantization"):
            assert same_numbers(weighted["results"][-1][key], reference[key]), key
        checks.append("sparse_reweight_GUI_and_CLI_preserve_same_support_and_amplitudes")
        handoff = work / "cli-sparse-handoff"
        transfer = json.loads(run("--sparse", "planar", "--input", weighted_out, "--index", len(weighted["results"]) - 1,
                                  "--case-index", 3, "--output", handoff).stdout)
        assert transfer["ok"] and transfer["source"]["notes"][0].startswith("保持选中方案")
        converted = json.loads(run("--planar-config", handoff / "planar-config.json").stdout)["planar"]
        assert same_numbers(converted, gui_report["sparse_planar_result"])
        checks.append("sparse_selected_layout_handoff_and_planar_electrical_GUI_CLI_match")
        saved_native = smoke / "native-sparse-settings.json"
        validated = json.loads(run("--sparse", "validate", "--config", saved_native).stdout)
        assert validated["ok"] and validated["request"]["config"]["seed"] == 2**32 - 1
        checks.append("native_sparse_toolbar_saved_settings_accepted_by_CLI_with_uint32_seed")
        parallel_study=json.loads(run('--study-config',smoke/'parallel-study.json','--workers','2').stdout)['study']
        assert same_numbers(parallel_study,gui_report['parallel_study_result'])
        checks.append('parallel_desktop_scan_and_explicit_headless_worker_limit_match')
        report['multicore_performance']=gui_report['multicore_performance']
        for kind, flag, key in [("linear", "--feed-config", "feed_network"), ("planar", "--planar-config", "planar")]:
            actual = json.loads(run(flag, smoke / f"pointing-{kind}.json").stdout)[key]
            assert same_numbers(actual, gui_report["pointing_results"][kind])
            checks.append(f"visible_{kind}_pointing_controls_save_and_headless_EXE_full_report_match")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
