"""Exercise actual EXEs with pipes, invalid display settings and isolated PATH."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid


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
    env = os.environ.copy()
    env.update({"PATH": r"C:\Windows\System32", "QT_QPA_PLATFORM": "invalid-test-platform",
                "DISPLAY": "", "MPLBACKEND": "QtAgg"})
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    checks = []

    def run(*flags, gui=False, expected=0):
        exe = release / ("ArrayLab.exe" if gui else "ArrayLab-cli.exe")
        result = subprocess.run([str(exe), *map(str, flags)], cwd=work, env=env,
                                capture_output=True, text=True, encoding="utf-8", timeout=45)
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
    report = {"success": True, "checks": checks, "work_directory": str(work), "app_version": sample_doc["app_version"]}
    if args.gui_smoke:
        gui_env = os.environ.copy()
        gui_env.pop("QT_QPA_PLATFORM", None)
        gui_env.pop("MPLBACKEND", None)
        smoke = work / "gui-smoke"
        process = subprocess.run([str(release / "ArrayLab.exe"), "--smoke-test", str(smoke)],
                                 cwd=work, env=gui_env, timeout=45)
        gui_report = json.loads((smoke / "smoke.json").read_text(encoding="utf-8"))
        assert process.returncode == 0 and gui_report["success"]
        report["gui_checks_passed"] = len(gui_report["checks"])
        scheme_result = json.loads(run("--config", smoke / "中文 分析方案.json").stdout)
        assert scheme_result["parameters"] == gui_report["scheme_result"]["parameters"]
        assert scheme_result["metrics"] == gui_report["scheme_result"]["metrics"]
        checks.append("desktop_saved_scheme_matches_headless_EXE_parameters_and_metrics")
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True))


if __name__ == "__main__":
    main()
