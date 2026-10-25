"""Matched branch extensions: independent circuit and persisted path verification."""
from copy import deepcopy
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from arraylab.features.linear.model import ArrayParameters
from arraylab.features.feed.model import FeedSettings, clear_branch_delays, with_branch_delays, normalized_branch_delays, synthesize_impedance, design_feed, save_feed_configuration, load_feed_configuration
from arraylab.application.records import save_history, load_history


def circuit(report):
    """Solve cascaded transmission-line ABCD matrices, recursively loading T nodes."""
    outgoing = {}
    for edge in report["sections"]:
        outgoing.setdefault(edge["parent"], []).append(edge)
    loads = {f"P{port['port']}": port["load_ohm"] for port in report["ports"]}
    transfer = {}

    def line(z, theta):
        angle = math.radians(theta)
        return np.array([[math.cos(angle), 1j * z * math.sin(angle)],
                         [1j * math.sin(angle) / z, math.cos(angle)]])

    def solve(node):
        if node in loads:
            return complex(loads[node])
        admittance = 0j
        for edge in outgoing[node]:
            load = solve(edge["child"])
            matrix = (line(edge["line_ohm"], edge["electrical_length_deg"]) @
                      line(edge["matched_delay_ohm"], edge["matched_delay_deg"]))
            a, b, c, d = matrix.ravel()
            admittance += 1 / ((a * load + b) / (c * load + d))
            transfer[edge["id"]] = 1 / (a + b / load)
        return 1 / admittance

    input_z = solve(report["root"])
    voltage = {}

    def propagate(node, value):
        if node in loads:
            voltage[node] = value
        else:
            for edge in outgoing[node]:
                propagate(edge["child"], value * transfer[edge["id"]])

    propagate(report["root"], math.sqrt(input_z.real))
    return input_z, voltage, {node: abs(v)**2 / loads[node] for node, v in voltage.items()}


@pytest.mark.parametrize("topology,n", [("corporate", 1), ("series", 1), ("corporate", 5),
                                         ("series", 5), ("symmetric_series", 6)])
@pytest.mark.parametrize("delay", [0, 45, 360, 3600, "mixed"])
def test_cascade_keeps_input_match_power_and_computes_phase(topology, n, delay):
    p = ArrayParameters(elements=n, step_deg=5)
    s = FeedSettings(topology=topology, input_ohm=75, junction_ohm=33, load_ohm=100)
    count = 1 if n == 1 else 2*n - 2
    delays = [(i * 47.5) % 361 if delay == "mixed" else delay for i in range(count)]
    report = synthesize_impedance(p, with_branch_delays(s, n, delays))
    z, voltages, powers = circuit(report)
    assert z == pytest.approx(75, abs=1e-9)
    assert sum(powers.values()) == pytest.approx(1, abs=1e-11)
    assert all(edge["electrical_length_deg"] == 90 for edge in report["sections"])
    for port in report["ports"]:
        name = f"P{port['port']}"
        assert powers[name] == pytest.approx(port["power_fraction"], abs=1e-11)
        assert voltages[name] / abs(voltages[name]) == pytest.approx(
            np.exp(1j*np.deg2rad(port["network_phase_deg"])), abs=1e-11)
        assert port["network_electrical_length_deg"] == port["transform_path_deg"] + port["branch_delay_path_deg"]
        assert port["network_phase_deg"] == -port["network_electrical_length_deg"]


@pytest.mark.parametrize("topology", ["corporate", "series", "symmetric_series"])
def test_one_edge_changes_exactly_its_descendant_phases(topology):
    p = ArrayParameters(elements=6)
    base = synthesize_impedance(p, FeedSettings(topology=topology))
    children = {}
    for edge in base["sections"]:
        children.setdefault(edge["parent"], []).append(edge["child"])

    def descendants(node):
        if node.startswith("P"):
            return {int(node[1:])}
        return set().union(*(descendants(child) for child in children[node]))

    for index, edge in enumerate(base["sections"]):
        delays = [0]*len(base["sections"])
        delays[index] = 45
        changed = synthesize_impedance(p, with_branch_delays(FeedSettings(topology=topology), p.elements, delays))
        affected = descendants(edge["child"])
        for old, new in zip(base["ports"], changed["ports"]):
            assert new["network_phase_deg"] - old["network_phase_deg"] == (-45 if old["port"] in affected else 0)
            assert new["power_fraction"] == old["power_fraction"]


@pytest.mark.parametrize("mode", ["none", "scan", "equalize"])
@pytest.mark.parametrize("topology", ["corporate", "series", "symmetric_series"])
def test_terminal_compensation_uses_extended_network(mode, topology):
    p = ArrayParameters(elements=6, scan_deg=23, step_deg=5)
    s = FeedSettings(topology=topology, phase_mode=mode, weighting="taylor", taylor_nbar=4)
    delayed = design_feed(p, with_branch_delays(s, 6, [15 + i*23 for i in range(10)]))
    z, voltages, powers = circuit(delayed)
    for port in delayed["ports"]:
        assert powers[f"P{port['port']}"] == pytest.approx(port["power_fraction"], abs=1e-11)
        v = voltages[f"P{port['port']}"]
        predicted = v / abs(v) * np.exp(-1j*np.deg2rad(port["applied_delay_deg"]))
        assert predicted == pytest.approx(np.exp(1j*np.deg2rad(port["output_phase_deg"])), abs=1e-11)
        assert port["total_electrical_length_deg"] == port["network_electrical_length_deg"] + port["applied_delay_deg"]
        if mode != "none":
            assert predicted == pytest.approx(np.exp(1j*np.deg2rad(port["target_phase_deg"])), abs=1e-11)
        else:
            assert port["applied_delay_deg"] == 0
    if mode != "none":
        original = design_feed(p, s)
        np.testing.assert_allclose(delayed["electrical"]["patterns"]["output"]["pattern_db"],
                                   original["electrical"]["patterns"]["output"]["pattern_db"], atol=1e-7)


def test_full_turn_preserved_and_phase_equality_uses_delays():
    p = ArrayParameters(elements=2)
    s = FeedSettings()
    report = synthesize_impedance(p, with_branch_delays(s, 2, [360, 0]))
    assert report["ports"][0]["network_phase_deg"] == -450
    assert report["metrics"]["equal_port_phase_at_design_frequency"]
    report = synthesize_impedance(p, with_branch_delays(s, 2, [45, 0]))
    assert not report["metrics"]["equal_port_phase_at_design_frequency"]


@pytest.mark.parametrize("change", [
    {"branch_delays_deg": "0"}, {"branch_delays_deg": (0,)},
    {"branch_delays_deg": (True, 0)}, {"branch_delays_deg": (-1, 0)},
    {"branch_delays_deg": (3601, 0)}, {"branch_delays_deg": (float("nan"), 0)},
    {"branch_delay_elements": True}, {"branch_delay_topology": "series"},
    {"branch_delays_deg": ()},
])
def test_rejects_malformed_or_partial_branch_settings(change):
    s = with_branch_delays(FeedSettings(), 2, [0, 45])
    with pytest.raises(ValueError):
        synthesize_impedance(ArrayParameters(elements=2), replace(s, **change))


def test_rejects_stale_binding_and_explicit_reset_preserves_other_inputs():
    s = with_branch_delays(FeedSettings(load_ohm=75), 2, [45, 0])
    with pytest.raises(ValueError):
        synthesize_impedance(ArrayParameters(elements=3), s)
    with pytest.raises(ValueError):
        synthesize_impedance(ArrayParameters(elements=2), replace(s, topology="series"))
    clean = clear_branch_delays(s)
    assert clean == FeedSettings(load_ohm=75)
    assert normalized_branch_delays(clean, 3) == (0,)*4


def test_configuration_and_history_roundtrip_without_recalculation(tmp_path, monkeypatch):
    p = ArrayParameters(elements=4, step_deg=5)
    s = with_branch_delays(FeedSettings(topology="symmetric_series", phase_bits=3), 4,
                          [0, 45, 360, 2.5, 90, 720])
    path = tmp_path / "config.json"
    save_feed_configuration(path, p, s)
    assert load_feed_configuration(path) == (p, s)
    report = design_feed(p, s)
    snapshot = tmp_path / "history.json"
    save_history(snapshot, "feed", [{"label": "延迟", "report": report}], 0)

    def no_calculation(*args, **kwargs):
        raise AssertionError("loading history must not solve anything")

    monkeypatch.setattr("arraylab.feed.synthesize_impedance", no_calculation)
    monkeypatch.setattr("arraylab.feed.design_feed", no_calculation)
    monkeypatch.setattr("arraylab.feed.target_amplitudes", no_calculation)
    assert load_history(snapshot)["records"][0]["report"] == report


@pytest.mark.parametrize("change", [
    lambda r: r["sections"][0].update(electrical_length_deg=91),
    lambda r: r["sections"][0].update(matched_delay_deg=44),
    lambda r: r["sections"][0].update(matched_delay_ohm=99),
    lambda r: r["sections"][0].update(total_electrical_length_deg=999),
    lambda r: r["ports"][0].update(transform_path_deg=0),
    lambda r: r["ports"][0].update(branch_delay_path_deg=0),
    lambda r: r["ports"][0].update(network_electrical_length_deg=0),
    lambda r: r["ports"][0].update(total_electrical_length_deg=0),
    lambda r: r["ports"][0].update(network_phase_deg=0),
    lambda r: r["ports"][0].update(applied_delay_deg=0),
    lambda r: r["ports"][0].update(output_phase_deg=17),
    lambda r: r["ports"][0].update(branch_delay_path_deg=True),
])
def test_history_rejects_corrupt_paths_or_new_numbers(change, tmp_path):
    p = ArrayParameters(elements=2, step_deg=5)
    report = design_feed(p, with_branch_delays(FeedSettings(), 2, [45, 12]))
    corrupt = deepcopy(report)
    change(corrupt)
    with pytest.raises(ValueError):
        save_history(tmp_path / "bad.json", "feed", [{"label": "corrupt", "report": corrupt}], 0)
