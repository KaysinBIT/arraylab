"""Version boundaries and recorded-release compatibility for branch delays.

The regular tests are self contained. The optional acceptance-artifact tests use
unaltered reports produced by the 0.20.0/0.21.0 frozen applications; a clean
checkout without those local, ignored artifacts explicitly skips that audit.
"""
from pathlib import Path as _RootPath
_PROJECT_ROOT = next(p for p in _RootPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from arraylab.features.linear.model import ArrayParameters
from arraylab.features.feed.model import BRANCH_DELAY_FIELDS, LEGACY_SETTINGS, FeedSettings, design_feed, load_feed_configuration, save_feed_configuration, with_branch_delays
from arraylab.application.records import load_history, save_history
from arraylab.features.planar.model import PlanarParameters, design_planar, load_planar_configuration, save_planar_configuration
from arraylab.features.studies.model import StudySettings, load_study_configuration, run_study, save_study_configuration


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def configuration_case(kind, tmp_path, *, delayed=True):
    path = tmp_path / f"{kind}.json"
    settings = FeedSettings(topology="series", phase_bits=3, seed=73)
    if delayed:
        settings = with_branch_delays(settings, 4, (0, 30, 360, 90, 45, 720))
    if kind == "feed":
        parameters = ArrayParameters(elements=4, scan_deg=20, step_deg=5)
        save_feed_configuration(path, parameters, settings)
        return path, load_feed_configuration, (parameters, settings), "settings", 3
    parameters = (PlanarParameters(nx=2, ny=2, step_deg=5) if kind in ("planar", "study-planar")
                  else ArrayParameters(elements=4, scan_deg=20, step_deg=5))
    if kind == "planar":
        save_planar_configuration(path, parameters, settings)
        return path, load_planar_configuration, (parameters, settings), "settings", 2
    study = StudySettings(count=2, start_deg=0, stop_deg=20)
    save_study_configuration(path, parameters, settings, study)
    return path, load_study_configuration, (parameters, settings, study), "feed_settings", 2


KINDS = ("feed", "planar", "study-linear", "study-planar")


@pytest.mark.parametrize("kind", KINDS)
def test_versioned_branch_settings_roundtrip_and_legacy_zero_migration(kind, tmp_path):
    path, loader, expected, settings_key, version = configuration_case(kind, tmp_path)
    assert loader(path) == expected
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema_version"] == version
    assert doc[settings_key]["branch_delays_deg"] == [0, 30, 360, 90, 45, 720]

    # Previous feed v2, planar/study v1 have exactly the previous field set.
    doc["schema_version"] -= 1
    for name in BRANCH_DELAY_FIELDS:
        del doc[settings_key][name]
    write_json(path, doc)
    before = path.read_bytes()
    restored = loader(path)
    assert restored[0] == expected[0]
    assert restored[1] == replace(expected[1], branch_delays_deg=(),
                                  branch_delay_elements=0, branch_delay_topology="")
    assert restored[2:] == expected[2:]
    assert path.read_bytes() == before


def test_impedance_only_v1_preserves_no_compensation_semantics(tmp_path):
    path, loader, expected, key, _ = configuration_case("feed", tmp_path, delayed=False)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["schema_version"] = 1
    doc[key] = {name: doc[key][name] for name in LEGACY_SETTINGS}
    write_json(path, doc)
    parameters, settings = loader(path)
    assert parameters == expected[0]
    assert settings.phase_mode == "none"
    assert settings.branch_delays_deg == ()
    assert settings.branch_delay_elements == 0
    assert settings.branch_delay_topology == ""


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("damage", ("new-fields-old-version", "old-fields-new-version", "partial-fields",
                                     "unknown-setting", "bool-version", "fraction-version", "future-version",
                                     "wrong-count", "wrong-topology", "incomplete-vector"))
def test_configuration_rejects_mixed_versions_and_stale_branch_bindings(kind, damage, tmp_path):
    path, loader, _, key, _ = configuration_case(kind, tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if damage == "new-fields-old-version":
        doc["schema_version"] -= 1
    elif damage == "old-fields-new-version":
        for name in BRANCH_DELAY_FIELDS:
            del doc[key][name]
    elif damage == "partial-fields":
        del doc[key]["branch_delay_topology"]
    elif damage == "unknown-setting":
        doc[key]["unrecognized_delay"] = 15
    elif damage == "bool-version":
        doc["schema_version"] = True
    elif damage == "fraction-version":
        doc["schema_version"] = float(doc["schema_version"])
    elif damage == "future-version":
        doc["schema_version"] = 999
    elif damage == "wrong-count":
        # Keep a self-consistent vector binding, but bind it to a different array.
        doc[key]["branch_delay_elements"] = 5
        doc[key]["branch_delays_deg"] = [0] * 8
    elif damage == "wrong-topology":
        doc[key]["branch_delay_topology"] = "corporate"
    elif damage == "incomplete-vector":
        doc[key]["branch_delays_deg"].pop()
    write_json(path, doc)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        loader(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("kind", KINDS)
def test_invalid_bound_delay_save_keeps_previous_configuration(kind, tmp_path):
    path, _, expected, _, _ = configuration_case(kind, tmp_path)
    before = path.read_bytes()
    settings = with_branch_delays(expected[1], 5, (0,) * 8)
    with pytest.raises(ValueError):
        if kind == "feed":
            save_feed_configuration(path, expected[0], settings)
        elif kind == "planar":
            save_planar_configuration(path, expected[0], settings)
        else:
            save_study_configuration(path, expected[0], settings, expected[2])
    assert path.read_bytes() == before


@pytest.fixture(scope="module")
def delayed_report():
    settings = with_branch_delays(FeedSettings(topology="series", phase_bits=3),
                                  4, (10, 20, 30, 40, 50, 60))
    return design_feed(ArrayParameters(elements=4, scan_deg=20, step_deg=5), settings)


def forbid_solving(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("loading a saved snapshot must not synthesize or calculate")
    for target in ("arraylab.feed.design_feed", "arraylab.feed.synthesize_impedance",
                   "arraylab.feed.target_amplitudes", "arraylab.complex_pattern.pattern_analysis",
                   "arraylab.core.calculate", "arraylab.excitation.electrical_excitation"):
        monkeypatch.setattr(target, fail)


def test_delayed_history_roundtrip_is_lossless_and_does_not_recompute(delayed_report, tmp_path, monkeypatch):
    path = tmp_path / "支路 计算记录.json"
    records = [{"label": "延迟后", "report": deepcopy(delayed_report)}]
    records[0]["report"]["metadata"] = {"operator_note": "keep extension metadata"}
    save_history(path, "feed", records, 0)
    before = path.read_bytes()
    forbid_solving(monkeypatch)
    restored = load_history(path)
    assert restored["records"] == records
    assert restored["selected"] == 0
    assert path.read_bytes() == before
    saved = tmp_path / "再次保存.json"
    save_history(saved, "feed", restored["records"], restored["selected"])
    assert saved.read_bytes() == before


def damage_report(report, damage):
    if damage == "new-fields-old-version":
        report["schema_version"] = 2
    elif damage == "old-fields-new-version":
        for name in BRANCH_DELAY_FIELDS:
            del report["settings"][name]
    elif damage == "unknown-setting":
        report["settings"]["unsupported_delay"] = 3
    elif damage == "wrong-count":
        report["settings"]["branch_delay_elements"] = 5
        report["settings"]["branch_delays_deg"] = [0] * 8
    elif damage == "wrong-topology":
        report["settings"]["branch_delay_topology"] = "corporate"
    elif damage == "missing-section-delay":
        del report["sections"][0]["matched_delay_deg"]
    elif damage == "mismatched-section-delay":
        report["sections"][0]["matched_delay_deg"] += 1
    elif damage == "mismatched-delay-impedance":
        report["sections"][0]["matched_delay_ohm"] += 1
    elif damage == "mismatched-port-path":
        report["ports"][0]["branch_delay_path_deg"] += 1


@pytest.mark.parametrize("damage", ("new-fields-old-version", "old-fields-new-version", "unknown-setting",
                                     "wrong-count", "wrong-topology", "missing-section-delay",
                                     "mismatched-section-delay", "mismatched-delay-impedance",
                                     "mismatched-port-path"))
def test_invalid_later_delayed_snapshot_rejects_whole_load_and_preserves_saved_history(
        damage, delayed_report, tmp_path, monkeypatch):
    path = tmp_path / "valid.json"
    records = [{"label": "valid", "report": deepcopy(delayed_report)}]
    save_history(path, "feed", records, 0)
    before = path.read_bytes()
    bad_records = deepcopy(records) + [{"label": "invalid", "report": deepcopy(delayed_report)}]
    damage_report(bad_records[-1]["report"], damage)
    with pytest.raises(ValueError):
        save_history(path, "feed", bad_records, 1)
    assert path.read_bytes() == before
    broken = tmp_path / "broken.json"
    write_json(broken, {"format": "arraylab.calculation_history", "schema_version": 1,
                        "kind": "feed", "records": bad_records, "selected": 1})
    forbid_solving(monkeypatch)
    with pytest.raises(ValueError):
        load_history(broken)
    assert load_history(path)["records"] == records


def test_invalid_history_action_does_not_change_existing_workbench_state(delayed_report, tmp_path):
    pytest.importorskip("PySide6")
    from arraylab.application.desktop.history import HistoryActions

    report = deepcopy(delayed_report)
    damage_report(report, "wrong-topology")
    path = tmp_path / "invalid-history.json"
    write_json(path, {"format": "arraylab.calculation_history", "schema_version": 1,
                      "kind": "feed", "records": [{"label": "bad", "report": report}], "selected": 0})
    # Invoke the real action entry point. If validation reaches any UI operation,
    # this minimal host fails instead of silently accepting a partial restore.
    host = SimpleNamespace(_require_history_idle=lambda: None, sequence=9,
                           datasets=["existing linear curve"], last_directory="previous folder",
                           feed_panel=SimpleNamespace(history=["existing feed report"], sequence=7))
    before = deepcopy(host.__dict__)
    with pytest.raises(ValueError):
        HistoryActions.load_history_path(host, path)
    assert host.__dict__ == before


def _local_release_roots():
    repo = _PROJECT_ROOT
    found = []
    for version, pattern in (("0.20.0", "release-0200-acceptance-*"),
                             ("0.21.0", "release-0210-accepted-*")):
        for folder in sorted((repo / "tmp").glob(pattern)):
            for root in sorted(folder.glob("*/gui-smoke/history-feed.arraylab.json")):
                found.append(pytest.param((version, root.parents[1]), id=version + "-" + folder.name))
    return found or [pytest.param(None, marks=pytest.mark.skip(reason="local frozen-release acceptance artifacts are absent"))]


@pytest.fixture(params=_local_release_roots())
def recorded_release(request):
    return request.param


def assert_old_fields_equal(old, new, path="report"):
    """Strict numerical equality, allowing only new fields and descriptive text."""
    if isinstance(old, dict):
        assert isinstance(new, dict), path
        for key, value in old.items():
            if key in {"schema_version", "app_version", "notes", "execution"}:
                continue
            assert key in new, path + "." + key
            assert_old_fields_equal(value, new[key], path + "." + key)
    elif isinstance(old, list):
        assert len(old) == len(new), path
        for index, (left, right) in enumerate(zip(old, new, strict=True)):
            assert_old_fields_equal(left, right, f"{path}[{index}]")
    else:
        assert old == new, path


def test_actual_release_histories_load_without_changes_or_recalculation(recorded_release, tmp_path, monkeypatch):
    version, root = recorded_release
    forbid_solving(monkeypatch)
    for name in ("history-feed.arraylab.json", "history-legacy-feed.json"):
        path = root / "gui-smoke" / name
        original = path.read_bytes()
        document = json.loads(original)
        assert all(row["report"]["app_version"] == version for row in document["records"])
        loaded = load_history(path)
        expected = deepcopy(document["records"])
        # The pre-history comparison format deliberately omits pointing summaries.
        # Existing migration derives only that summary from saved metrics, with
        # the same result already recorded by the same released executable.
        current_envelope = json.loads((root / "gui-smoke" / "history-feed.arraylab.json").read_text(encoding="utf-8"))
        for index, record in enumerate(expected):
            if "pointing" not in record["report"]["electrical"]:
                record["report"]["electrical"]["pointing"] = current_envelope["records"][index]["report"]["electrical"]["pointing"]
        assert loaded["records"] == expected
        assert all(row["report"]["schema_version"] == 2 for row in loaded["records"])
        assert loaded["selected"] == document.get("selected", len(document["records"]) - 1)
        assert path.read_bytes() == original
        saved = tmp_path / name
        save_history(saved, "feed", loaded["records"], loaded["selected"])
        assert load_history(saved) == loaded


def test_actual_release_zero_delay_science_is_strictly_unchanged(recorded_release):
    _, root = recorded_release
    # CLI cases cover all three topologies; GUI history adds custom amplitudes,
    # quantization, seeded errors, and failed ports from the released executable.
    reports = []
    for case in ("cli-corporate", "cli-series", "cli-symmetric_series", "cli-electrical"):
        directory = root / case
        parameters, settings = load_feed_configuration(directory / "feed-config.json")
        old = json.loads((directory / "feed-network.json").read_text(encoding="utf-8"))
        assert settings.branch_delays_deg == ()
        reports.append((parameters, settings, old))
    historical = json.loads((root / "gui-smoke" / "history-feed.arraylab.json").read_text(encoding="utf-8"))
    for item in historical["records"]:
        old = item["report"]
        reports.append((ArrayParameters(**old["array_parameters"]), FeedSettings(**old["settings"]), old))
    for parameters, settings, old in reports:
        actual = design_feed(parameters, settings)
        assert_old_fields_equal(old, actual)


def test_actual_release_planar_and_study_configurations_keep_scientific_results(recorded_release):
    _, root = recorded_release
    directory = root / "cli-planar"
    parameters, settings = load_planar_configuration(directory / "planar-config.json")
    old = json.loads((directory / "planar-report.json").read_text(encoding="utf-8"))
    assert settings.branch_delays_deg == ()
    assert_old_fields_equal(old, design_planar(parameters, settings, workers=1))
    for case in ("cli-study-linear", "cli-study-planar"):
        directory = root / case
        parameters, settings, study = load_study_configuration(directory / "study-config.json")
        old = json.loads((directory / "study-report.json").read_text(encoding="utf-8"))
        assert settings.branch_delays_deg == ()
        assert_old_fields_equal(old, run_study(parameters, settings, study, workers=1))
