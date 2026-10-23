"""Snapshot persistence checks: complete restore, bounded parsing and atomicity."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from arraylab.core import ArrayParameters, calculate
from arraylab.feed import FeedSettings, design_feed
from arraylab.feed_diagram import diagram_svg
from arraylab import history_io
from arraylab.history_io import (
    FORMAT, SCHEMA_VERSION, MAX_BYTES, MAX_RECORDS, MAX_POINTS,
    array_result_from_snapshot, array_result_snapshot, load_history, save_history,
)


def external_record(**changes):
    record = {"label": "外部曲线 · 未推断三维指标", "angles": [-180., -30., 0., 180.],
              "db": [-42., -10., 0., -45.], "phase": None, "result": None}
    record.update(changes)
    return record


def linear_record(parameters=None):
    result = calculate(parameters or ArrayParameters(elements=5, step_deg=5, scan_deg=-21))
    return {"label": "#3 计算快照", "angles": result.angles_deg, "db": result.pattern_db,
            "phase": result.phase_deg, "result": result}


def document(records=None, kind="linear", selected=0, **changes):
    value = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "kind": kind,
             "records": [external_record()] if records is None else records, "selected": selected}
    value.update(changes)
    return value


def write_document(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


@pytest.fixture(scope="module")
def feed_report():
    return design_feed(ArrayParameters(elements=4, scan_deg=20, step_deg=5),
                       FeedSettings(topology="symmetric_series", seed=23, phase_error_deg=2, failed_ports=(1,)))


def test_linear_restore_preserves_arrays_metrics_external_curves_and_selection(tmp_path, monkeypatch):
    calculated = linear_record()
    with_phase = external_record(label="外部含相位", phase=[90., 20., 0., -90.])
    records = [calculated, external_record(), with_phase]
    path = tmp_path / "中文 计算记录.json"
    save_history(path, "linear", records, 1)

    def no_recompute(*args, **kwargs):
        raise AssertionError("loading snapshots must not recompute")

    monkeypatch.setattr("arraylab.core.calculate", no_recompute)
    loaded = load_history(path)
    assert loaded["selected"] == 1
    assert loaded["records"][1:] == [external_record(), with_phase]
    assert loaded["records"][1]["result"] is None
    restored = array_result_from_snapshot(loaded["records"][0]["result"])
    original = calculated["result"]
    assert restored.parameters == original.parameters
    assert restored.directivity_dbi == original.directivity_dbi
    assert restored.peaks_deg == original.peaks_deg
    assert restored.half_power_deg == original.half_power_deg
    assert restored.notices == original.notices
    for key in ("angles_deg", "pattern_db", "phase_deg", "positions_mm", "excitation_phase_deg"):
        np.testing.assert_array_equal(getattr(restored, key), getattr(original, key))
        assert not getattr(restored, key).flags.writeable
    again = tmp_path / "restored.json"
    save_history(again, loaded["kind"], loaded["records"], loaded["selected"])
    assert again.read_bytes() == path.read_bytes()


def test_snapshot_roundtrip_handles_single_element_and_multiple_peaks(tmp_path):
    records = [linear_record(ArrayParameters(elements=1, step_deg=5)),
               linear_record(ArrayParameters(elements=5, spacing=1.5, step_deg=5))]
    path = tmp_path / "undefined-widths.json"
    save_history(path, "linear", records, 0)
    loaded = load_history(path)
    for item, original in zip(loaded["records"], records):
        assert item["result"] == array_result_snapshot(original["result"])
        assert array_result_from_snapshot(item["result"]).half_power_deg is None


def test_loads_large_external_snapshot_above_old_configuration_limit(tmp_path):
    angles = np.linspace(-180, 180, MAX_POINTS)
    record = external_record(angles=angles, db=np.linspace(-50, 0, MAX_POINTS),
                             phase=np.linspace(-179, 179, MAX_POINTS))
    path = tmp_path / "large.json"
    save_history(path, "linear", [record], 0)
    assert 1024 * 1024 < path.stat().st_size < MAX_BYTES
    restored = load_history(path)["records"][0]
    np.testing.assert_array_equal(restored["angles"], angles)
    np.testing.assert_array_equal(restored["phase"], record["phase"])


@pytest.mark.parametrize("topology,n", [("corporate", 1), ("corporate", 5), ("series", 5), ("symmetric_series", 6)])
def test_feed_topologies_render_from_restored_reports(topology, n, tmp_path):
    report = design_feed(ArrayParameters(elements=n, step_deg=5), FeedSettings(topology=topology))
    path = tmp_path / "feed.json"
    save_history(path, "feed", [{"label": topology, "report": report}], 0)
    restored = load_history(path)["records"][0]["report"]
    assert restored == report
    assert diagram_svg(restored) == diagram_svg(report)


def test_feed_preserves_complete_science_metadata_and_failed_ports(feed_report, tmp_path, monkeypatch):
    report = deepcopy(feed_report)
    report["metadata"] = {"experiment": "测试<仅数据>", "source": "archive", "values": [1, None, False]}
    before = deepcopy(report)
    records = [{"label": "#7 带误差与失效", "report": report}, {"label": "#8 另一记录", "report": report}]
    path = tmp_path / "feed.json"
    save_history(path, "feed", records, 0)

    def no_recompute(*args, **kwargs):
        raise AssertionError("no feed synthesis or pattern calculation while loading")

    monkeypatch.setattr("arraylab.feed.design_feed", no_recompute)
    monkeypatch.setattr("arraylab.feed.synthesize_impedance", no_recompute)
    monkeypatch.setattr("arraylab.feed.target_amplitudes", no_recompute)
    monkeypatch.setattr("arraylab.complex_pattern.pattern_analysis", no_recompute)
    loaded = load_history(path)
    assert loaded["records"] == records
    assert report == before
    assert loaded["selected"] == 0
    assert loaded["records"][0]["report"]["ports"][0]["output_phase_deg"] is None


@pytest.mark.parametrize("app_version,omit_pointing", [("0.11.0", False), ("0.12.0", False), ("0.11.0", True)])
def test_feed_comparison_v1_compatibility_uses_saved_metrics(feed_report, app_version, omit_pointing, tmp_path, monkeypatch):
    report = deepcopy(feed_report)
    report["app_version"] = app_version
    expected_pointing = deepcopy(report["electrical"]["pointing"])
    if omit_pointing:
        del report["electrical"]["pointing"]
    original = deepcopy(report)
    path = tmp_path / "feed-comparison.json"
    write_document(path, {"format": "arraylab.feed_comparison", "schema_version": 1,
                          "records": [{"label": "first", "report": report}, {"label": "second", "report": report}]})
    source_bytes = path.read_bytes()

    def no_recompute(*args, **kwargs):
        raise AssertionError("legacy adaptation must use stored metrics")

    monkeypatch.setattr("arraylab.feed.design_feed", no_recompute)
    monkeypatch.setattr("arraylab.complex_pattern.pattern_analysis", no_recompute)
    loaded = load_history(path)
    assert loaded["kind"] == "feed"
    assert loaded["selected"] == 1
    restored = loaded["records"][0]["report"]
    assert restored["electrical"]["pointing"] == expected_pointing
    if omit_pointing:
        del restored["electrical"]["pointing"]
    assert restored == original
    assert report == original
    assert path.read_bytes() == source_bytes


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"schema_version": True}, {"schema_version": 1.0}, {"format": "other.app"},
    {"kind": "planar"}, {"kind": {}}, {"selected": True}, {"selected": 1}, {"selected": -1},
    {"selected": "0"}, {"records": []}, {"records": {}}, {"records": [external_record()] * (MAX_RECORDS + 1)},
    {"extra": "unknown"},
])
def test_invalid_envelope_rejected_without_partial_results(changes, tmp_path):
    path = tmp_path / "bad.json"
    write_document(path, document(**changes))
    with pytest.raises(ValueError):
        load_history(path)


@pytest.mark.parametrize("changes", [
    {"label": ""}, {"label": 5}, {"db": [1]}, {"phase": [1]}, {"angles": [0, 0, 1, 2]},
    {"angles": [-181, -1, 0, 1]}, {"angles": [[0], [1], [2], [3]]}, {"db": [0, True, 0, 0]},
    {"phase": "none"}, {"result": {}}, {"unknown": 3},
])
def test_damaged_later_linear_record_aborts_whole_load(changes, tmp_path):
    path = tmp_path / "bad-second-record.json"
    write_document(path, document(records=[external_record(), external_record(**changes)]))
    with pytest.raises(ValueError):
        load_history(path)


@pytest.mark.parametrize("field,value", [
    ("positions_mm", [1]), ("excitation_phase_deg", [0]), ("notices", [False]),
    ("half_power_deg", [1, -1]), ("peaks_deg", [False]), ("directivity_dbi", None),
    ("parameters", {"elements": 4}), ("phase_deg", [0]),
])
def test_broken_calculated_snapshot_rejected(field, value, tmp_path):
    record = linear_record()
    record = {key: val.tolist() if isinstance(val, np.ndarray) else val for key, val in record.items()}
    record["result"] = array_result_snapshot(record["result"])
    record["result"][field] = value
    path = tmp_path / "broken-result.json"
    write_document(path, document(records=[record]))
    with pytest.raises(ValueError):
        load_history(path)


@pytest.mark.parametrize("change", [
    lambda r: r.update(schema_version=3),
    lambda r: r.pop("nodes"),
    lambda r: r["nodes"][0].update(depth="0"),
    lambda r: r["nodes"][1].update(id="<unexpected-markup>"),
    lambda r: r["nodes"][1].pop("arm"),
    lambda r: r["sections"][0].pop("line_ohm"),
    lambda r: r["sections"][0].update(id="<unexpected-markup>"),
    lambda r: r["sections"][0].update(child="missing"),
    lambda r: r["sections"][0].update(outside_range=0),
    lambda r: r["ports"].pop(),
    lambda r: r["ports"][0].update(port=2),
    lambda r: r["ports"][0].pop("output_phase_deg"),
    lambda r: r["ports"][0].update(mirror_port=99),
    lambda r: r["settings"].update(phase_bits=True),
    lambda r: r["metrics"].pop("line_min_ohm"),
    lambda r: r["metrics"].update(ports=3),
    lambda r: r["electrical"]["metrics"].pop("normalization"),
    lambda r: r["electrical"]["patterns"].pop("target"),
    lambda r: r["electrical"]["patterns"]["output"]["pattern_db"].pop(),
    lambda r: r["electrical"]["patterns"]["target"]["metrics"].pop("directivity_dbi"),
    lambda r: r["electrical"]["pointing"].pop("output_cut_peak_angles_deg"),
    lambda r: r["electrical"]["pointing"].update(target_theta_deg="20"),
])
def test_damaged_feed_report_aborts_whole_load(change, feed_report, tmp_path):
    report = deepcopy(feed_report)
    change(report)
    path = tmp_path / "bad-feed.json"
    write_document(path, document(kind="feed", records=[{"label": "valid", "report": feed_report},
                                                        {"label": "invalid", "report": report}]))
    with pytest.raises(ValueError):
        load_history(path)


@pytest.mark.parametrize("raw", [
    b'{"format":"arraylab.calculation_history","schema_version":1,"schema_version":1}',
    b'{"value":NaN}', b'{"value":Infinity}', b'{"value":-Infinity}', b'{"value":1e999}',
    b'{"nested":{"same":0,"same":1}}', b'\xff\xfe', b'[' * 2000 + b']' * 2000,
], ids=["duplicate-root", "nan", "positive-infinity", "negative-infinity", "overflow",
        "duplicate-nested", "invalid-encoding", "deep-nesting"])
def test_strict_json_parser_rejects_malformed_data(raw, tmp_path):
    path = tmp_path / "malformed.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        load_history(path)


def test_load_size_limit_and_save_size_limit_are_enforced(tmp_path, monkeypatch):
    path = tmp_path / "large.json"
    save_history(path, "linear", [external_record()], 0)
    original = path.read_bytes()
    monkeypatch.setattr(history_io, "MAX_BYTES", len(original) - 1)
    with pytest.raises(ValueError):
        load_history(path)
    with pytest.raises(ValueError):
        save_history(path, "linear", [external_record()], 0)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".arraylab-history-*.tmp"))


def test_invalid_save_never_overwrites_existing_file(tmp_path):
    path = tmp_path / "history.json"
    save_history(path, "linear", [external_record()], 0)
    original = path.read_bytes()
    for record in [external_record(db=[1]), external_record(db=[float("nan")] * 4),
                   external_record(angles=np.zeros(MAX_POINTS + 1)), external_record(label=object())]:
        with pytest.raises(ValueError):
            save_history(path, "linear", [record], 0)
        assert path.read_bytes() == original
    assert not list(tmp_path.glob(".arraylab-history-*.tmp"))


@pytest.mark.parametrize("failure", ["replace", "fsync"])
def test_atomic_failure_retains_existing_file_and_cleans_temporary(tmp_path, monkeypatch, failure):
    path = tmp_path / "history.json"
    save_history(path, "linear", [external_record()], 0)
    original = path.read_bytes()

    def denied(*args):
        raise PermissionError("destination is locked")

    monkeypatch.setattr(history_io.os, failure, denied)
    with pytest.raises(PermissionError):
        save_history(path, "linear", [external_record(label="new label")], 0)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob(".arraylab-history-*.tmp"))


def test_feed_comparison_unknown_versions_and_missing_records_fail(tmp_path):
    path = tmp_path / "old-comparison.json"
    for content in ({"format": "arraylab.feed_comparison", "schema_version": 2, "records": []},
                    {"format": "arraylab.feed_comparison", "schema_version": True, "records": []},
                    {"format": "arraylab.feed_comparison", "schema_version": 1}):
        write_document(path, content)
        with pytest.raises(ValueError):
            load_history(path)


def test_history_module_has_no_gui_imports(tmp_path):
    script = """
import importlib.abc
import sys
class NoGui(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('PySide6', 'PyQt6', 'matplotlib'):
            raise AssertionError('GUI import: ' + fullname)
sys.meta_path.insert(0, NoGui())
from arraylab.history_io import save_history, load_history, array_result_snapshot, array_result_from_snapshot
from arraylab.core import ArrayParameters, calculate
result = calculate(ArrayParameters(step_deg=5))
restored = array_result_from_snapshot(array_result_snapshot(result))
assert restored.directivity_dbi == result.directivity_dbi
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
