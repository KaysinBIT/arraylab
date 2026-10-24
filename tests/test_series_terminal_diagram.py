"""Check displayed connectivity against the saved electrical network."""
from copy import deepcopy
import re
import xml.etree.ElementTree as ET

import pytest

from arraylab.core import ArrayParameters
from arraylab.excitation import electrical_excitation
from arraylab.feed import FeedSettings, synthesize_impedance, with_branch_delays
from arraylab.feed_diagram import diagram_svg


NS = "{http://www.w3.org/2000/svg}"
CASES = [("series", n) for n in (1, 2, 3, 8)] + [
    ("symmetric_series", n) for n in (2, 4, 8)]


def drawing(report):
    group = ET.fromstring(diagram_svg(report)).find(f"{NS}g")
    children = list(group)
    nodes, blocks = {}, {}
    node_ids = {node["id"] for node in report["nodes"]}
    for index, item in enumerate(children):
        if item.tag != f"{NS}text":
            continue
        text = item.text or ""
        if text in node_ids:
            rect = children[index - 1]
            assert rect.tag == f"{NS}rect"
            nodes[text] = tuple(float(rect.get(key)) for key in ("x", "y", "width", "height"))
        elif re.match(r"[TDC]\d+: ", text):
            label = text.split(":", 1)[0]
            assert label not in blocks
            blocks[label] = (float(item.get("x")), float(item.get("y")) + 5,
                             text, children[index + 1].text)
    paths = {item.get("data-section"): item for item in group.findall(f"{NS}polyline")}
    assert len(paths) == len(report["sections"])
    return nodes, blocks, paths


def points(path):
    return [tuple(map(float, pair.split(","))) for pair in path.get("points").split()]


def center(rect):
    x, y, width, height = rect
    return x + width / 2, y + height / 2


@pytest.mark.parametrize("topology,count", CASES)
@pytest.mark.parametrize("snapshot", ["impedance", "electrical", "legacy"])
def test_terminal_load_continues_backbone_with_independent_t_d_c(topology, count, snapshot):
    parameters = ArrayParameters(elements=count, scan_deg=23)
    settings = FeedSettings(topology=topology, input_ohm=75, junction_ohm=33, load_ohm=100)
    if snapshot == "electrical":
        section_count = 1 if count == 1 else 2 * count - 2
        settings = with_branch_delays(settings, count, [11 + 7 * i for i in range(section_count)])
    report = synthesize_impedance(parameters, settings)
    if snapshot != "impedance":
        electrical_excitation(parameters, settings, report["ports"])
    if snapshot == "legacy":
        report["schema_version"] = 2
        for section in report["sections"]:
            for key in ("matched_delay_deg", "matched_delay_ohm", "total_electrical_length_deg"):
                del section[key]
        for port in report["ports"]:
            for key in ("transform_path_deg", "branch_delay_path_deg", "network_electrical_length_deg"):
                del port[key]
        for key in ("branch_delays_deg", "branch_delay_elements", "branch_delay_topology"):
            del report["settings"][key]
    saved = deepcopy(report)
    nodes, blocks, paths = drawing(report)
    assert report == saved
    port_map = {f"P{port['port']}": port for port in report["ports"]}
    terminals = []
    for section in report["sections"]:
        label, child = section["id"], section["child"]
        path = paths[label]
        t, d = blocks[label], blocks[f"D{label[1:]}"]
        assert t[2:] == (f"{label}: {section['line_ohm']:.5g} ohm",
                         f"{section['electrical_length_deg']:.7g} deg")
        assert d[2:] == (f"D{label[1:]}: {section.get('matched_delay_ohm', section['child_equivalent_ohm']):.5g} ohm",
                         f"{section.get('matched_delay_deg', 0):.7g} deg")
        assert path.get("stroke") == ("#b66b20" if section["outside_range"] else "#297da7")
        if child in port_map and snapshot != "impedance":
            port = port_map[child]
            assert blocks[f"C{port['port']}"][2:] == (
                f"C{port['port']}: {port['compensation_line_ohm']:.5g} ohm",
                f"{port['applied_delay_deg']:.7g} deg")
        if section["kind"] != "through" or child not in port_map:
            assert path.get("data-role") == section["kind"]
            continue
        terminals.append(section)
        assert path.get("data-role") == "terminal-through"
        parent_rect, child_rect = nodes[section["parent"]], nodes[child]
        parent_x, parent_y = center(parent_rect)
        child_x, child_y = center(child_rect)
        direction = 1 if child_x > parent_x else -1
        expected_end = (child_x - direction * child_rect[2] / 2, child_y)
        assert points(path) == [(parent_x, parent_rect[1] + parent_rect[3]),
                                (parent_x, child_y), expected_end]
        assert child_y > parent_y
        tap = next(item for item in report["sections"]
                   if item["parent"] == section["parent"] and item["kind"] == "tap")
        tap_start = points(paths[tap["id"]])[0]
        assert tap_start == (parent_x + direction * parent_rect[2] / 2, parent_y)
        assert tap_start != points(path)[0]
        stage_positions = [t[:2], d[:2]]
        if snapshot != "impedance":
            stage_positions.append(blocks[f"C{port_map[child]['port']}"][:2])
        assert all(y == child_y for x, y in stage_positions)
        ordered_x = [parent_x, *(x for x, y in stage_positions), expected_end[0]]
        assert all((b - a) * direction > 0 for a, b in zip(ordered_x, ordered_x[1:]))
    assert len(terminals) == (0 if count == 1 or (topology == "symmetric_series" and count == 2)
                              else 1 if topology == "series" else 2)
    assert sum(label.startswith("T") for label in blocks) == len(report["sections"])
    assert sum(label.startswith("D") for label in blocks) == len(report["sections"])
    assert sum(label.startswith("C") for label in blocks) == (0 if snapshot == "impedance" else count)
    if snapshot == "electrical":
        assert all(section["matched_delay_deg"] > 0 for section in report["sections"])
        assert all(port["applied_delay_deg"] > 0 for port in report["ports"])


def test_existing_tap_backbone_and_input_split_routes_are_unchanged():
    serial = synthesize_impedance(ArrayParameters(elements=3), FeedSettings(topology="series"))
    _, _, paths = drawing(serial)
    assert points(paths["T1"]) == [(153, 125), (190, 125), (190, 125), (707, 125)]
    assert points(paths["T2"]) == [(100, 147), (100, 323)]
    assert points(paths["T3"]) == [(153, 345), (190, 345), (190, 345), (707, 345)]
    symmetric = synthesize_impedance(ArrayParameters(elements=4), FeedSettings(topology="symmetric_series"))
    _, _, paths = drawing(symmetric)
    assert points(paths["T1"]) == [(1347, 125), (1310, 125), (1310, 255), (740, 255), (740, 363)]
    assert points(paths["T2"]) == [(1453, 125), (1490, 125), (1490, 255), (2060, 255), (2060, 363)]
    assert points(paths["T3"]) == [(687, 385), (650, 385), (650, 385), (133, 385)]
    assert points(paths["T5"]) == [(2113, 385), (2150, 385), (2150, 385), (2667, 385)]


@pytest.mark.parametrize("count", [1, 2, 3, 8])
def test_parallel_network_has_no_serial_terminal_route(count):
    report = synthesize_impedance(ArrayParameters(elements=count), FeedSettings(topology="corporate"))
    _, _, paths = drawing(report)
    assert all(path.get("data-role") != "terminal-through" for path in paths.values())
