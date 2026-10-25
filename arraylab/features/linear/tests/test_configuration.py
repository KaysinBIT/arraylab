from dataclasses import asdict
import json

import pytest

from arraylab.features.linear.configuration import load_configuration, save_configuration, MAX_BYTES
from arraylab.features.linear.model import ArrayParameters
from test_cli import run_cli


def document(**changes):
    result = {"format": "arraylab.analysis", "schema_version": 1,
              "parameters": asdict(ArrayParameters(spacing=45.25, spacing_unit="mm", scan_deg=-27.5))}
    result.update(changes)
    return result


def test_roundtrip_and_headless_override_without_gui(tmp_path):
    path = tmp_path / "中文 分析方案.json"
    parameters = ArrayParameters(elements=5, spacing=45.25, spacing_unit="mm", scan_deg=-27.5)
    save_configuration(path, parameters)
    original = path.read_bytes()
    assert load_configuration(path) == parameters
    result = run_cli("--config", path, "--elements", 20, guard=True)
    assert result.returncode == 0, result.stderr
    expected = asdict(parameters)
    expected["elements"] = 20
    assert json.loads(result.stdout)["parameters"] == expected
    assert path.read_bytes() == original


def test_legacy_partial_bom_and_upgrade(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"elements":5,"scan_deg":60,"spacing":0.4}', encoding="utf-8-sig")
    parameters = load_configuration(path)
    assert parameters == ArrayParameters(elements=5, scan_deg=60, spacing=.4)
    original = path.read_bytes()
    upgraded = tmp_path / "new.json"
    save_configuration(upgraded, parameters)
    assert path.read_bytes() == original
    assert load_configuration(upgraded) == parameters
    assert json.loads(upgraded.read_text())["schema_version"] == 1


@pytest.mark.parametrize("doc", [
    document(schema_version=2), document(schema_version=True), document(schema_version=1.0),
    document(format="another.app"), document(extra="unknown"),
    document(parameters=[]), document(parameters={"elements":5}),
    document(parameters={**asdict(ArrayParameters()), "weight":1}),
    document(parameters={**asdict(ArrayParameters()), "spacing_unit":"cm"}),
    document(parameters={**asdict(ArrayParameters()), "elements":True}),
    document(parameters={**asdict(ArrayParameters()), "scan_deg":100}),
    document(parameters={**asdict(ArrayParameters()), "step_deg":None}),
    {"schema_version":1, "status":"ok", "parameters":asdict(ArrayParameters())},
])
def test_reject_invalid_versioned_document(doc, tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        load_configuration(path)


@pytest.mark.parametrize("raw", [
    b'{"format":"arraylab.analysis","schema_version":1,"parameters":{"elements":5,"elements":8}}',
    b'{"spacing":1e999}', b'{"spacing":Infinity}', b'\xff\xfe',
    b'[' * 2000 + b']' * 2000, b' ' * (MAX_BYTES + 1),
], ids=["duplicate", "overflow", "infinity", "encoding", "nesting", "size"])
def test_malformed_or_oversized_file_rejected(raw, tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(raw)
    with pytest.raises((ValueError, OverflowError)):
        load_configuration(path)


def test_invalid_document_cli_fails_without_output(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps(document(schema_version=2)), encoding="utf-8")
    output = tmp_path / "result"
    result = run_cli("--config", path, "--output-dir", output, guard=True)
    assert result.returncode == 2
    assert result.stdout == ""
    assert json.loads(result.stderr)["error"]["type"] == "input"
    assert not output.exists()


def test_atomic_save_failure_preserves_original(monkeypatch, tmp_path):
    path = tmp_path / "analysis.json"
    save_configuration(path, ArrayParameters())
    original = path.read_bytes()

    def denied(*args):
        raise PermissionError("destination is locked")

    monkeypatch.setattr("arraylab.configuration.os.replace", denied)
    with pytest.raises(PermissionError):
        save_configuration(path, ArrayParameters(elements=20))
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_parameters_cannot_overwrite(tmp_path):
    path = tmp_path / "analysis.json"
    path.write_bytes(b"original user file")
    with pytest.raises(ValueError):
        save_configuration(path, ArrayParameters(elements=True))
    assert path.read_bytes() == b"original user file"
    assert list(tmp_path.iterdir()) == [path]
