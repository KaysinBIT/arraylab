from pathlib import Path as _RootPath
_PROJECT_ROOT = next(p for p in _RootPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from arraylab.features.linear.model import ArrayParameters, calculate
from arraylab.features.linear.dataio import read_pattern

ROOT = _PROJECT_ROOT


def run_cli(*args, guard=False):
    env = os.environ.copy()
    env.update({"QT_QPA_PLATFORM": "invalid-test-platform", "DISPLAY": "", "MPLBACKEND": "QtAgg"})
    if guard:
        script = '''
import importlib.abc,sys,runpy
class NoGUI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("PySide", "PyQt", "tkinter", "arraylab.window", "arraylab.widgets", "arraylab.feed_widgets", "arraylab.electrical_widgets", "arraylab.planar_widgets", "arraylab.study_widgets", "arraylab.sparse_widgets", "arraylab.sparse2d.gui", "arraylab.sparse2d.detail_view")):
            raise AssertionError("Headless attempted GUI import: " + fullname)
sys.meta_path.insert(0,NoGUI())
sys.argv=["arraylab", "--headless"] + sys.argv[1:]
runpy.run_module("arraylab",run_name="__main__")
'''
        command = [sys.executable, "-c", script]
    else:
        command = [sys.executable, "-m", "arraylab", "--headless"]
    return subprocess.run(command + list(map(str, args)), cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", timeout=45)


def test_default_stdout_metrics_no_gui():
    r = run_cli(guard=True)
    assert r.returncode == 0, r.stderr
    summary = json.loads(r.stdout)
    expected = calculate(ArrayParameters())
    assert summary["metrics"]["directivity_dbi"] == expected.directivity_dbi
    assert summary["metrics"]["half_power_beamwidth_deg"] == expected.beamwidth_deg
    assert summary["mode"] == "headless"
    assert r.stderr == ""


def test_config_overrides_exports_and_unicode(tmp_path):
    config = tmp_path / "参数.json"
    config.write_text(json.dumps({"elements": 5, "spacing": .4, "scan_deg": 60}), encoding="utf-8")
    out = tmp_path / "结果 with spaces"
    r = run_cli("--config", config, "--elements", 20, "--spacing", .5, "--scan-deg", 0,
                "--output-dir", out, "--include-phase", guard=True)
    assert r.returncode == 0, r.stderr
    summary = json.loads(r.stdout)
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    expected = calculate(ArrayParameters(elements=20))
    assert summary["parameters"]["elements"] == 20
    assert summary["metrics"]["directivity_dbi"] == expected.directivity_dbi
    angles, db, phase = read_pattern(out / "pattern.dat")
    np.testing.assert_allclose(db, expected.pattern_db, atol=1e-8)
    np.testing.assert_allclose(phase, expected.phase_deg)
    excitation = np.loadtxt(out / "excitation.csv", delimiter=",", skiprows=1)
    np.testing.assert_allclose(excitation[:, 1], expected.positions_mm)
    assert len(excitation) == 20
    assert not list(tmp_path.glob(".arraylab-*"))


def test_agg_png_with_gui_import_blocked(tmp_path):
    out = tmp_path / "plots"
    r = run_cli("--scan-deg", 30, "--output-dir", out, "--plot", "both", guard=True)
    assert r.returncode == 0, r.stderr
    for name in ("rectangular.png", "polar.png"):
        assert (out / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        assert (out / name).stat().st_size > 10000


@pytest.mark.parametrize("args", [
    ["--elements", "201"], ["--elements", "2.5"], ["--spacing", "nan"],
    ["--scan-deg", "inf"], ["--plot", "both"], ["--unknown-option"],
    ["--spacing-unit", "cm"], ["--step-deg", "0"],
])
def test_invalid_args_have_json_stderr_and_nonzero_code(args):
    r = run_cli(*args)
    assert r.returncode == 2
    assert r.stdout == ""
    assert json.loads(r.stderr)["error"]["type"] == "input"


@pytest.mark.parametrize("config", [
    '[]', '{"unknown":1}', '{"elements":true}', '{"elements":5.0}',
    '{"frequency_ghz":"3"}', '{"scan_deg":NaN}', '{"spacing":null}',
    '{"elements":5,"elements":8}', '{broken',
])
def test_bad_config(config, tmp_path):
    file = tmp_path / "bad.json"
    file.write_text(config, encoding="utf-8")
    out = tmp_path / "result"
    r = run_cli("--config", file, "--output-dir", out)
    assert r.returncode == 2
    assert not out.exists()


def test_existing_results_untouched(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    original = b"user results"
    (out / "summary.json").write_bytes(original)
    r = run_cli("--output-dir", out)
    assert r.returncode == 3
    assert json.loads(r.stderr)["error"]["type"] == "io"
    assert (out / "summary.json").read_bytes() == original
    assert list(out.iterdir()) == [out / "summary.json"]


def test_invalid_output_parent(tmp_path):
    file = tmp_path / "file"
    file.write_text("keep", encoding="utf-8")
    r = run_cli("--output-dir", file / "child")
    assert r.returncode == 3
    assert file.read_text() == "keep"


def test_missing_config_is_io_error(tmp_path):
    r = run_cli("--config", tmp_path / "missing.json")
    assert r.returncode == 3


@pytest.mark.parametrize("n,d", [(1, .5), (8, 1)])
def test_undefined_metrics_are_json_null_not_nan(n, d):
    r = run_cli("--elements", n, "--spacing", d)
    assert r.returncode == 0
    doc = json.loads(r.stdout)
    assert doc["metrics"]["main_lobe_angle_deg"] is None
    assert doc["metrics"]["half_power_beamwidth_deg"] is None
    assert doc["notices"]


def test_help_and_version_do_not_load_gui():
    for args in [("--help",), ("--version",)]:
        r = run_cli(*args, guard=True)
        assert r.returncode == 0
        assert r.stdout


def test_linear_and_feed_3d_export_remains_headless(tmp_path):
    out = tmp_path / "三维结果"
    r = run_cli("--elements", 6, "--scan-deg", 26, "--feed-network", "series",
                "--feed-phase-bits", 3, "--output-dir", out, "--plot", "3d", guard=True)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    for name in ("3d.png", "feed-3d.png"):
        assert (out / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    for name in ("sphere-pattern.csv", "feed-output-sphere.csv", "feed-target-sphere.csv"):
        data = np.loadtxt(out / name, delimiter=",", skiprows=1)
        assert data.shape[1] == 4 and np.isfinite(data).all()
    assert doc["files"]["feed_3d_plot"] == "feed-3d.png"
    assert doc["metrics"]["gain_dbi"] is None
