from dataclasses import asdict
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from arraylab.core import ArrayParameters
from arraylab.feed import (FeedSettings, synthesize_impedance, save_feed_configuration,
                           load_feed_configuration, export_feed_directory)
from arraylab.feed_diagram import diagram_svg
from test_cli import run_cli


def independent_circuit(report):
    """Reconstruct the circuit from line ABCD matrices, not synthesis ratios."""
    children = {}
    for line in report["sections"]:
        children.setdefault(line["parent"], []).append(line)
    loads = {f"P{row['port']}": row["load_ohm"] for row in report["ports"]}
    impedances, transfers = {}, {}

    def impedance(node):
        if node in loads:
            result = complex(loads[node])
        else:
            admittance = 0j
            for edge in children[node]:
                z_load = impedance(edge["child"])
                theta, z = math.radians(edge["electrical_length_deg"]), edge["line_ohm"]
                a = d = math.cos(theta)
                b, c = 1j * z * math.sin(theta), 1j * math.sin(theta) / z
                admittance += 1 / ((a * z_load + b) / (c * z_load + d))
                transfers[edge["id"]] = 1 / (a + b / z_load)
            result = 1 / admittance
        impedances[node] = result
        return result

    z_input = impedance(report["root"])
    voltages = {}

    def propagate(node, voltage):
        if node in loads:
            voltages[node] = voltage
        else:
            for edge in children[node]:
                propagate(edge["child"], voltage * transfers[edge["id"]])

    propagate(report["root"], math.sqrt(z_input.real))
    return z_input, {node: abs(v) ** 2 / loads[node] for node, v in voltages.items()}, voltages


@pytest.mark.parametrize("topology", ["corporate", "series"])
@pytest.mark.parametrize("count", [1, 2, 5, 8, 17, 200])
def test_abcd_reconstruction_input_match_equal_power_and_phase(topology, count):
    settings = FeedSettings(topology=topology, input_ohm=75, junction_ohm=33, load_ohm=100)
    report = synthesize_impedance(ArrayParameters(elements=count), settings)
    z_input, powers, voltages = independent_circuit(report)
    assert z_input == pytest.approx(75, rel=1e-11, abs=1e-9)
    assert sum(powers.values()) == pytest.approx(1, abs=1e-11)
    for port in report["ports"]:
        name = f"P{port['port']}"
        assert powers[name] == pytest.approx(1 / count, abs=1e-11)
        actual = voltages[name] / abs(voltages[name])
        expected = np.exp(1j * np.deg2rad(port["network_phase_deg"]))
        assert actual == pytest.approx(expected, abs=1e-10)
    assert len(report["sections"]) == (1 if count == 1 else 2 * (count - 1))
    assert {row["port"] for row in report["ports"]} == set(range(1, count + 1))
    assert ET.fromstring(diagram_svg(report)).tag.endswith("svg")


def test_reference_corporate_eight_and_non_power_of_two():
    report = synthesize_impedance(ArrayParameters())
    assert all(edge["line_ohm"] == pytest.approx(50 * math.sqrt(2)) for edge in report["sections"])
    assert all(edge["branch_input_ohm"] == 100 for edge in report["sections"])
    assert report["metrics"]["equal_port_phase_at_design_frequency"]
    five = synthesize_impedance(ArrayParameters(elements=5))
    first = five["sections"][:2]
    assert [row["local_power_fraction"] for row in first] == [.4, .6]
    assert [row["line_ohm"] for row in first] == pytest.approx([79.0569415042, 64.5497224368])
    assert not five["metrics"]["equal_port_phase_at_design_frequency"]


def test_reference_series_eight_and_range_advisory():
    report = synthesize_impedance(ArrayParameters(), FeedSettings(topology="series"))
    first = report["sections"][:2]
    assert [edge["local_power_fraction"] for edge in first] == [1 / 8, 7 / 8]
    assert [edge["branch_input_ohm"] for edge in first] == pytest.approx([400, 400 / 7])
    assert [edge["line_ohm"] for edge in first] == pytest.approx([141.4213562373, 53.4522483825])
    assert len(report["metrics"]["out_of_range_sections"]) == 3
    assert not report["metrics"]["equal_port_phase_at_design_frequency"]


@pytest.mark.parametrize("changes", [
    {"input_ohm":0}, {"junction_ohm":float("nan")}, {"load_ohm":float("inf")},
    {"input_ohm":True}, {"load_ohm":"50"}, {"max_line_ohm":1001},
    {"min_line_ohm":120}, {"min_line_ohm":200}, {"topology":"unknown"},
])
def test_invalid_settings(changes):
    with pytest.raises(ValueError):
        synthesize_impedance(ArrayParameters(), FeedSettings(**changes))


@pytest.mark.parametrize("topology", ["corporate", "series", "symmetric_series"])
def test_configuration_export_and_headless_roundtrip(topology, tmp_path):
    count = 6 if topology == "symmetric_series" else 5
    p = ArrayParameters(elements=count, frequency_ghz=10, spacing_unit="mm", spacing=10)
    settings = FeedSettings(topology=topology, input_ohm=75, load_ohm=100)
    config = tmp_path / "中文 馈电.json"
    save_feed_configuration(config, p, settings)
    assert load_feed_configuration(config) == (p, settings)
    original = config.read_bytes()
    expected = synthesize_impedance(p, settings)
    out = tmp_path / "结果"
    process = run_cli("--feed-config", config, "--output-dir", out, guard=True)
    assert process.returncode == 0, process.stderr
    actual = json.loads(process.stdout)
    assert actual["feed_network"] == expected
    assert json.loads((out / "feed-network.json").read_text(encoding="utf-8")) == expected
    assert load_feed_configuration(out / "feed-config.json") == (p, settings)
    assert config.read_bytes() == original
    assert len((out / "feed-sections.csv").read_text(encoding="utf-8-sig").splitlines()) == 2 * (count - 1) + 1
    assert ET.parse(out / "feed-network.svg").getroot().tag.endswith("svg")
    override = run_cli("--feed-config", config, "--elements", 8, "--feed-load-ohm", 25, guard=True)
    assert override.returncode == 0, override.stderr
    result = json.loads(override.stdout)["feed_network"]
    assert result["metrics"]["ports"] == 8 and result["settings"]["load_ohm"] == 25


@pytest.mark.parametrize("flags", [
    ["--feed-input-ohm", "75"], ["--feed-network", "series", "--feed-load-ohm", "nan"],
    ["--feed-config", "missing.json", "--config", "examples/ula.json"],
])
def test_invalid_cli_feed_options(flags):
    process = run_cli(*flags, guard=True)
    assert process.returncode == 2
    assert process.stdout == ""
    assert json.loads(process.stderr)["error"]["type"] == "input"


@pytest.mark.parametrize("change", ["version", "boolean_version", "missing", "unknown", "report"])
def test_strict_feed_configuration(change, tmp_path):
    config = tmp_path / "feed.json"
    save_feed_configuration(config, ArrayParameters(), FeedSettings())
    doc = json.loads(config.read_text(encoding="utf-8"))
    if change == "version": doc["schema_version"] = 2
    if change == "boolean_version": doc["schema_version"] = True
    if change == "missing": del doc["settings"]["input_ohm"]
    if change == "unknown": doc["array_parameters"]["unknown"] = 1
    if change == "report": doc = synthesize_impedance(ArrayParameters())
    config.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError): load_feed_configuration(config)


def test_atomic_export_preserves_existing_and_cleans_failed_staging(tmp_path, monkeypatch):
    report = synthesize_impedance(ArrayParameters())
    target = tmp_path / "result"
    export_feed_directory(target, report)
    original = (target / "feed-network.json").read_bytes()
    with pytest.raises(FileExistsError): export_feed_directory(target, report)
    assert (target / "feed-network.json").read_bytes() == original

    def fail(directory, report):
        (directory / "partial").write_text("incomplete")
        raise OSError("simulated storage failure")

    monkeypatch.setattr("arraylab.feed.write_feed_files", fail)
    with pytest.raises(OSError): export_feed_directory(tmp_path / "failed", report)
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".arraylab-feed-*"))


@pytest.mark.parametrize("count", [2, 4, 6, 8, 10, 200])
def test_symmetric_series_abcd_and_mirrored_paths(count):
    settings = FeedSettings(topology="symmetric_series", input_ohm=75, junction_ohm=33, load_ohm=100)
    report = synthesize_impedance(ArrayParameters(elements=count), settings)
    z_input, powers, voltages = independent_circuit(report)
    assert z_input == pytest.approx(75, abs=1e-9)
    assert sum(powers.values()) == pytest.approx(1, abs=1e-11)
    assert len(report["sections"]) == 2 * (count - 1)
    root_sections = [s for s in report["sections"] if s["parent"] == report["root"]]
    assert len(root_sections) == 2
    assert all(s["kind"] == "input_split" and s["local_power_fraction"] == .5 for s in root_sections)
    assert report["routing"]["left_ports"] == list(range(count // 2, 0, -1))
    assert report["routing"]["right_ports"] == list(range(count // 2 + 1, count + 1))
    nodes = {node["id"]: node for node in report["nodes"]}
    left = [s for s in report["sections"] if nodes[s["parent"]].get("arm") == "left"]
    right = [s for s in report["sections"] if nodes[s["parent"]].get("arm") == "right"]
    assert [(s["kind"], s["line_ohm"], s["local_power_fraction"]) for s in left] == [
        (s["kind"], s["line_ohm"], s["local_power_fraction"]) for s in right]
    for port in report["ports"]:
        i = port["port"]
        assert port["mirror_port"] == count + 1 - i
        assert powers[f"P{i}"] == pytest.approx(1 / count, abs=1e-11)
        assert voltages[f"P{i}"] == pytest.approx(voltages[f"P{count + 1 - i}"], abs=1e-10)
        assert voltages[f"P{i}"] / abs(voltages[f"P{i}"]) == pytest.approx(
            np.exp(1j * np.deg2rad(port["network_phase_deg"])), abs=1e-10)
    assert ET.fromstring(diagram_svg(report)).tag.endswith("svg")


def test_symmetric_series_eight_reference_and_phases():
    report = synthesize_impedance(ArrayParameters(), FeedSettings(topology="symmetric_series"))
    assert report["metrics"]["line_max_ohm"] == 100
    assert not report["metrics"]["out_of_range_sections"]
    assert [s["line_ohm"] for s in report["sections"][:4]] == pytest.approx(
        [70.7106781187, 70.7106781187, 100, 57.735026919])
    assert [p["network_phase_deg"] for p in report["ports"]] == [-360, -360, -270, -180, -180, -270, -360, -360]
    assert not report["metrics"]["equal_port_phase_at_design_frequency"]


@pytest.mark.parametrize("count", [1, 3, 5, 199])
def test_symmetric_series_rejects_odd_arrays_before_save(count, tmp_path):
    parameters = ArrayParameters(elements=count)
    settings = FeedSettings(topology="symmetric_series")
    with pytest.raises(ValueError, match="偶数"):
        synthesize_impedance(parameters, settings)
    path = tmp_path / "config.json"
    path.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="偶数"):
        save_feed_configuration(path, parameters, settings)
    assert path.read_text() == "original"


def test_symmetric_series_invalid_loaded_count_and_cli_override(tmp_path):
    path = tmp_path / "config.json"
    save_feed_configuration(path, ArrayParameters(), FeedSettings(topology="symmetric_series"))
    original = path.read_bytes()
    output = tmp_path / "output"
    process = run_cli("--feed-config", path, "--elements", 5, "--output-dir", output, guard=True)
    assert process.returncode == 2 and not process.stdout and not output.exists()
    assert path.read_bytes() == original
    doc = json.loads(original)
    doc["array_parameters"]["elements"] = 5
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="偶数"):
        load_feed_configuration(path)


def test_symmetric_series_cli_option(tmp_path):
    result = run_cli("--feed-network", "symmetric_series", "--elements", 8, guard=True)
    assert result.returncode == 0, result.stderr
    network = json.loads(result.stdout)["feed_network"]
    assert network["routing"]["left_ports"] == [4, 3, 2, 1]
    assert network["metrics"]["line_max_ohm"] == 100
