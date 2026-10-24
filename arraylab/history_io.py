"""Versioned calculation snapshots, independent of Qt and of calculation routines.

``load_history`` returns JSON-compatible dictionaries. Linear records contain
label/angles/db/phase/result; ``array_result_from_snapshot`` restores an optional
ArrayResult without recalculating it. Feed records contain label/report. Report
extension metadata is retained, while all fields used to render it are checked.
The old feed_comparison v1 envelope is accepted as feed history, selecting its
last record. Missing legacy pointing summaries are derived from saved metrics.
"""
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .configuration import FIELDS, parameters_from_values
from .core import ArrayResult


FORMAT = "arraylab.calculation_history"
SCHEMA_VERSION = 1
MAX_BYTES = 64 * 1024 * 1024
MAX_RECORDS = 5
MAX_POINTS = 100_000
MAX_DEPTH = 32
MAX_STRING = 1024 * 1024
_RESULT_FIELDS = frozenset(field.name for field in fields(ArrayResult))


def _fail(where, message):
    raise ValueError(f"{where}：{message}")


def _object(value, where, required, exact=False):
    if not isinstance(value, dict):
        _fail(where, "须为JSON对象。")
    missing = set(required) - value.keys()
    if missing:
        _fail(where, "缺少字段 " + ", ".join(sorted(missing)))
    if exact and set(value) != set(required):
        _fail(where, "包含未知字段。")
    return value


def _string(value, where, limit=MAX_STRING, nonempty=False):
    if not isinstance(value, str) or len(value) > limit or (nonempty and not value.strip()):
        _fail(where, f"须为{'非空' if nonempty else ''}字符串，最长{limit}字。")
    return value


def _number(value, where, nullable=False, low=None, high=None):
    if nullable and value is None:
        return value
    if type(value) not in (int, float):
        _fail(where, "须为有限数值，不能是布尔值、字符串或null。")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (low is not None and value < low) or (high is not None and value > high):
        _fail(where, "数值非有限或超出允许范围。")
    return value


def _integer(value, where, low, high):
    if type(value) is not int or not low <= value <= high:
        _fail(where, f"须为{low}–{high}范围的整数。")
    return value


def _boolean(value, where):
    if type(value) is not bool:
        _fail(where, "须为布尔值。")


def _list(value, where, low=0, high=MAX_POINTS):
    if not isinstance(value, list) or not low <= len(value) <= high:
        _fail(where, f"须为长度{low}–{high}的列表。")
    return value


def _vector(value, where, size=None, low=0, high=MAX_POINTS, bounds=None):
    _list(value, where, low, high)
    if size is not None and len(value) != size:
        _fail(where, "数组长度不匹配。")
    for item in value:
        _number(item, where, low=None if bounds is None else bounds[0],
                high=None if bounds is None else bounds[1])
    return value


def _strings(value, where, high=10000):
    for item in _list(value, where, high=high):
        _string(item, where)


def _angles(value, where, bounds=(-180, 180)):
    _vector(value, where, low=2, bounds=bounds)
    if any(b <= a for a, b in zip(value, value[1:])):
        _fail(where, "角度须严格递增。")
    return value


def _plain(value, depth=0):
    """Copy only JSON data and explicitly supported scientific containers."""
    if depth > MAX_DEPTH:
        _fail("记录文件", "嵌套过深。")
    if isinstance(value, np.ndarray):
        if value.size > MAX_POINTS:
            _fail("记录数组", "超过允许的点数。")
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    elif isinstance(value, ArrayResult):
        value = {field.name: getattr(value, field.name) for field in fields(ArrayResult)}
        value["parameters"] = asdict(value["parameters"])
    if isinstance(value, dict):
        if len(value) > 10000 or any(not isinstance(key, str) for key in value):
            _fail("记录对象", "字段过多或字段名不是字符串。")
        return {_string(key, "字段名", 4096): _plain(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_POINTS:
            _fail("记录数组", "超过允许的点数。")
        return [_plain(item, depth + 1) for item in value]
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        return _number(value, "记录数值")
    if isinstance(value, str):
        return _string(value, "记录文本")
    _fail("记录文件", "含不支持的非JSON数据类型。")


def _parameters(value, where):
    _object(value, where, FIELDS, exact=True)
    return parameters_from_values(value)


def _validate_result(result):
    _object(result, "线阵结果", _RESULT_FIELDS, exact=True)
    parameters = _parameters(result["parameters"], "线阵参数")
    angles = _angles(result["angles_deg"], "线阵角度", (-90, 90))
    for name in ("pattern_db", "phase_deg"):
        _vector(result[name], name, len(angles))
    for name in ("positions_mm", "excitation_phase_deg"):
        _vector(result[name], name, parameters.elements)
    _number(result["directivity_dbi"], "方向性")
    _vector(result["peaks_deg"], "等高峰角度", bounds=(-90, 90))
    if result["half_power_deg"] is not None:
        edges = _vector(result["half_power_deg"], "半功率边界", 2, bounds=(-90, 90))
        if edges[0] >= edges[1]:
            _fail("半功率边界", "左右边界次序错误。")
    _strings(result["notices"], "线阵提示")


def array_result_snapshot(result):
    """Return the complete JSON-compatible snapshot of an ArrayResult/dict."""
    snapshot = _plain(result)
    _validate_result(snapshot)
    return snapshot


def array_result_from_snapshot(snapshot):
    """Recreate the immutable numeric result without invoking a solver."""
    values = array_result_snapshot(snapshot)
    values["parameters"] = parameters_from_values(values["parameters"])
    for name in ("angles_deg", "pattern_db", "phase_deg", "positions_mm", "excitation_phase_deg"):
        values[name] = np.array(values[name], dtype=float)
        values[name].setflags(write=False)
    for name in ("peaks_deg", "notices", "half_power_deg"):
        if values[name] is not None:
            values[name] = tuple(values[name])
    return ArrayResult(**values)


def _validate_linear(record):
    _object(record, "线阵记录", {"label", "angles", "db", "phase", "result"}, exact=True)
    _string(record["label"], "记录标签", 1024, nonempty=True)
    angles = _angles(record["angles"], "曲线角度")
    _vector(record["db"], "曲线幅度", len(angles))
    if record["phase"] is not None:
        _vector(record["phase"], "曲线相位", len(angles))
    if record["result"] is not None:
        result = record["result"]
        _validate_result(result)
        for record_key, result_key in (("angles", "angles_deg"), ("db", "pattern_db"), ("phase", "phase_deg")):
            if record[record_key] != result[result_key]:
                _fail("线阵记录", "显示曲线与计算快照不一致。")


def _validate_pattern(pattern, where):
    _object(pattern, where, {"angles_deg", "pattern_db", "field_phase_deg", "metrics", "notices"})
    angles = _angles(pattern["angles_deg"], where + "角度", (-90, 90))
    for name in ("pattern_db", "field_phase_deg"):
        _vector(pattern[name], where + name, len(angles))
    keys = {"directivity_dbi", "peak_angles_deg", "main_lobe_angle_deg", "half_power_beamwidth_deg",
            "half_power_edges_deg", "peak_level_db"}
    metrics = _object(pattern["metrics"], where + "指标", keys)
    for name in keys - {"peak_angles_deg", "half_power_edges_deg"}:
        _number(metrics[name], where + name, nullable=True)
    _vector(metrics["peak_angles_deg"], where + "峰值角度", bounds=(-90, 90))
    if metrics["half_power_edges_deg"] is not None:
        _vector(metrics["half_power_edges_deg"], where + "半功率边界", 2, bounds=(-90, 90))
    _strings(pattern["notices"], where + "提示")


def _validate_feed(record):
    # Imports are pure scientific modules. No widgets or Qt are needed.
    from .feed import settings_from_payload, normalized_branch_delays
    from .excitation import validate_electrical
    from .pointing import linear_pointing

    _object(record, "馈电记录", {"label", "report"}, exact=True)
    _string(record["label"], "记录标签", 1024, nonempty=True)
    report = _object(record["report"], "馈电报告", {"format", "schema_version", "app_version",
                     "array_parameters", "settings", "root", "nodes", "sections", "ports", "metrics", "notes", "electrical"})
    if report["format"] != "arraylab.feed_impedance_result" or type(report["schema_version"]) is not int or report["schema_version"] not in (2, 3):
        _fail("馈电报告", "不支持的格式或版本；需要完整的电气结果报告。")
    _string(report["app_version"], "应用版本", 128, nonempty=True)
    parameters = _parameters(report["array_parameters"], "馈电阵列参数")
    n = parameters.elements
    settings = settings_from_payload(report["settings"], report["schema_version"])
    branch_delays = normalized_branch_delays(settings, n)
    has_delays = report["schema_version"] == 3
    validate_electrical(settings, parameters)
    symmetric = settings.topology == "symmetric_series"
    if symmetric and n % 2:
        _fail("馈电设置", "对称串馈须为偶数端口。")
    nodes = _list(report["nodes"], "网络节点", 1, 2 * n)
    node_map = {}
    for node in nodes:
        _object(node, "网络节点", {"id", "kind", "depth", "first_port", "last_port", "equivalent_ohm"})
        name = _string(node["id"], "节点编号", 128, nonempty=True)
        if name in node_map:
            _fail("网络节点", "编号重复。")
        node_map[name] = node
        if node["kind"] not in ("port", "junction", "input"):
            _fail("网络节点", "未知节点类型。")
        _integer(node["depth"], "节点层数", 0, n)
        first = _integer(node["first_port"], "首端口", 1, n)
        _integer(node["last_port"], "末端口", first, n)
        _number(node["equivalent_ohm"], "节点阻抗", low=1e-300)
        # These IDs enter the existing SVG renderer; accept the report schema's
        # canonical identifiers rather than interpreting markup from a file.
        if node["kind"] == "port":
            valid_id = name == f"P{first}"
        elif node["kind"] == "input":
            valid_id = name == "IN" and n == 1
        else:
            valid_id = name in {f"J{i}" for i in range(1, n)}
        if not valid_id:
            _fail("网络节点", "节点编号与节点类型不符。")
    root = _string(report["root"], "输入节点", 128, nonempty=True)
    if root not in node_map or node_map[root]["depth"] != 0:
        _fail("输入节点", "根节点缺失或层数错误。")
    if symmetric:
        for node in nodes:
            if node["id"] != root and node.get("arm") not in ("left", "right"):
                _fail("网络节点", "缺少有效的对称支臂标识。")
    sections = _list(report["sections"], "变换线", 1, 2 * n)
    if len(sections) != len(nodes) - 1:
        _fail("馈电网络", "线段与节点数不匹配。")
    if has_delays and len(sections) != len(branch_delays):
        _fail("馈电网络", "支路数量与延迟配置不一致。")
    line_ids, children = set(), set()
    section_keys = {"id", "parent", "child", "kind", "downstream_ports", "local_power_fraction",
                    "parent_equivalent_ohm", "child_equivalent_ohm", "branch_input_ohm", "line_ohm",
                    "electrical_length_deg", "outside_range"}
    delay_section_keys = {"matched_delay_deg", "matched_delay_ohm", "total_electrical_length_deg"}
    path_lengths = {root: (0.0, 0.0)}
    for index, line in enumerate(sections, 1):
        _object(line, "变换线", section_keys)
        name = _string(line["id"], "线段编号", 128, nonempty=True)
        if name != f"T{index}":
            _fail("变换线", "编号须按T1、T2…连续排列。")
        parent = _string(line["parent"], "线段起点", 128, nonempty=True)
        child = _string(line["child"], "线段终点", 128, nonempty=True)
        if name in line_ids or parent not in node_map or child not in node_map or child == root or child in children:
            _fail("变换线", "编号或连接关系无效。")
        if node_map[child]["depth"] != node_map[parent]["depth"] + 1:
            _fail("变换线", "节点层数与连接关系不符。")
        line_ids.add(name); children.add(child)
        _string(line["kind"], "线段类型", 128, nonempty=True)
        _integer(line["downstream_ports"], "下游端口数", 1, n)
        _number(line["local_power_fraction"], "局部分配比", low=1e-300, high=1)
        for key in ("parent_equivalent_ohm", "child_equivalent_ohm", "branch_input_ohm", "line_ohm"):
            _number(line[key], key, low=1e-300)
        _number(line["electrical_length_deg"], "线段电长度")
        _boolean(line["outside_range"], "阻抗超范围标记")
        if has_delays:
            _object(line, "支路匹配延迟", delay_section_keys)
            for key in delay_section_keys:
                _number(line[key], "支路" + key, low=0)
            if parent not in path_lengths:
                _fail("变换线", "支路顺序须从输入到负载。")
            expected = {"electrical_length_deg": 90.0, "matched_delay_deg": branch_delays[index - 1],
                        "matched_delay_ohm": line["child_equivalent_ohm"],
                        "total_electrical_length_deg": 90.0 + branch_delays[index - 1]}
            if any(not math.isclose(line[key], value, rel_tol=1e-10, abs_tol=1e-8) for key, value in expected.items()):
                _fail("支路匹配延迟", "电长度、阻抗或配置与保存的支路不一致。")
            if not math.isclose(line["matched_delay_ohm"], node_map[child]["equivalent_ohm"], rel_tol=1e-10, abs_tol=1e-8):
                _fail("支路匹配延迟", "延迟线阻抗须等于下游节点等效阻抗。")
            transform, matched = path_lengths[parent]
            path_lengths[child] = (transform + 90.0, matched + line["matched_delay_deg"])
        elif delay_section_keys.intersection(line):
            _fail("馈电报告", "旧版报告不能包含新版支路延迟字段。")
    port_keys = {"port", "load_ohm", "power_fraction", "quarter_wave_sections", "network_phase_deg",
                 "target_amplitude", "target_phase_deg", "phase_offset_deg", "desired_delay_deg", "applied_delay_deg",
                 "matched_line_wavelengths", "compensation_line_ohm", "output_amplitude", "output_phase_deg",
                 "amplitude_error_db", "phase_error_deg", "failed", "output_real", "output_imag"}
    delay_port_keys = {"transform_path_deg", "branch_delay_path_deg", "network_electrical_length_deg",
                       "total_electrical_length_deg"}
    ports = _list(report["ports"], "端口", n, n)
    for index, port in enumerate(ports, 1):
        _object(port, "端口", port_keys)
        _integer(port["port"], "端口编号", index, index)
        node = node_map.get(f"P{index}")
        if node is None or node["kind"] != "port" or node["first_port"] != index or node["last_port"] != index:
            _fail("端口", "端口与网络节点不匹配。")
        _integer(port["quarter_wave_sections"], "端口段数", 0, n)
        for key in port_keys - {"port", "quarter_wave_sections", "failed"}:
            _number(port[key], "端口" + key, nullable=key == "output_phase_deg")
        _boolean(port["failed"], "失效标记")
        if has_delays:
            _object(port, "端口路径电长度", delay_port_keys)
            for key in delay_port_keys:
                _number(port[key], "端口" + key, low=0)
            if f"P{index}" not in path_lengths:
                _fail("端口路径", "未连接至输入端。")
            transform, matched = path_lengths[f"P{index}"]
            expected = {"transform_path_deg": transform, "branch_delay_path_deg": matched,
                        "network_electrical_length_deg": transform + matched,
                        "network_phase_deg": -(transform + matched),
                        "quarter_wave_sections": node["depth"],
                        "total_electrical_length_deg": transform + matched + port["applied_delay_deg"]}
            if any(not math.isclose(port[key], value, rel_tol=1e-10, abs_tol=1e-8) for key, value in expected.items()):
                _fail("端口路径", "支路累积电长度或相位与保存结果不一致。")
            desired = 0.0 if settings.phase_mode == "none" else (port["network_phase_deg"] - port["target_phase_deg"]) % 360
            if math.isclose(desired, 360.0, rel_tol=0, abs_tol=1e-9):
                desired = 0.0
            applied = desired
            if settings.phase_bits:
                step = 360.0 / 2**settings.phase_bits
                applied = (math.floor(desired / step + .5) % 2**settings.phase_bits) * step
            if (not math.isclose(port["desired_delay_deg"], desired, rel_tol=0, abs_tol=1e-8) or
                    not math.isclose(port["applied_delay_deg"], applied, rel_tol=0, abs_tol=1e-8)):
                _fail("端口路径", "末端补偿与网络相位或量化设置不一致。")
            expected_phase = (port["network_phase_deg"] - applied + port["phase_error_deg"] + 180) % 360 - 180
            if not port["failed"] and (port["output_phase_deg"] is None or
                    abs((port["output_phase_deg"] - expected_phase + 180) % 360 - 180) > 1e-8):
                _fail("端口路径", "输出相位与支路及末端补偿不一致。")
        elif delay_port_keys.intersection(port):
            _fail("馈电报告", "旧版报告不能包含新版路径电长度字段。")
        if symmetric:
            if port.get("arm") not in ("left", "right"):
                _fail("端口", "缺少有效的对称支臂标识。")
            _integer(port.get("mirror_port"), "镜像端口", n + 1 - index, n + 1 - index)
    metric_keys = {"ports", "junctions", "transformers", "input_equivalent_ohm", "line_min_ohm", "line_max_ohm",
                   "out_of_range_sections", "equal_port_phase_at_design_frequency"}
    metrics = _object(report["metrics"], "网络指标", metric_keys)
    _integer(metrics["ports"], "端口总数", n, n)
    _integer(metrics["junctions"], "节点总数", max(n - 1, 0), max(n - 1, 0))
    _integer(metrics["transformers"], "变换线总数", len(sections), len(sections))
    for key in ("input_equivalent_ohm", "line_min_ohm", "line_max_ohm"):
        _number(metrics[key], key, low=1e-300)
    _strings(metrics["out_of_range_sections"], "超范围线段", len(sections))
    if any(name not in line_ids for name in metrics["out_of_range_sections"]):
        _fail("网络指标", "引用未知线段。")
    _boolean(metrics["equal_port_phase_at_design_frequency"], "端口等相标记")
    if has_delays:
        equal = bool(np.allclose(np.exp(1j * np.deg2rad([port["network_phase_deg"] for port in ports])),
                                 np.exp(1j * np.deg2rad(ports[0]["network_phase_deg"])), rtol=0, atol=1e-10))
        if metrics["equal_port_phase_at_design_frequency"] != equal:
            _fail("网络指标", "端口等相标记与支路路径不一致。")
    _strings(report["notes"], "报告说明")
    electrical = _object(report["electrical"], "电气结果", {"metrics", "patterns"})
    electrical_keys = {"phase_mode", "target_scan_deg", "weighting", "normalization", "output_to_nominal_power_ratio",
                       "nominal_power_fraction_at_failed_ports", "active_ports", "seed", "max_active_phase_error_deg", "error_interpretation"}
    em = _object(electrical["metrics"], "电气指标", electrical_keys)
    for key in ("phase_mode", "weighting", "normalization", "error_interpretation"):
        _string(em[key], key, nonempty=True)
    for key in ("target_scan_deg", "output_to_nominal_power_ratio", "nominal_power_fraction_at_failed_ports", "max_active_phase_error_deg"):
        _number(em[key], key, nullable=key == "max_active_phase_error_deg")
    _integer(em["active_ports"], "有效端口数", 0, n)
    _integer(em["seed"], "随机种子", 0, 2**32 - 1)
    patterns = _object(electrical["patterns"], "方向图", {"target", "uncompensated", "output"}, exact=True)
    for name, pattern in patterns.items():
        _validate_pattern(pattern, name)
    if any(pattern["angles_deg"] != patterns["target"]["angles_deg"] for pattern in patterns.values()):
        _fail("方向图", "同一设计的方向图采样角度不一致。")
    if "pointing" not in electrical:
        electrical["pointing"] = linear_pointing(parameters, settings, patterns)
    pointing_keys = {"requested_theta_deg", "target_theta_deg", "phase_mode", "compensation_enabled", "output_cut_peak_deg",
                     "output_cut_peak_angles_deg", "output_cut_offset_deg", "status", "scope", "notes"}
    pointing = _object(electrical["pointing"], "指向摘要", pointing_keys)
    for key in ("requested_theta_deg", "target_theta_deg", "output_cut_peak_deg", "output_cut_offset_deg"):
        _number(pointing[key], key, nullable=key in ("output_cut_peak_deg", "output_cut_offset_deg"))
    _vector(pointing["output_cut_peak_angles_deg"], "输出峰角度", bounds=(-90, 90))
    for key in ("phase_mode", "status", "scope"):
        _string(pointing[key], key, nonempty=True)
    _boolean(pointing["compensation_enabled"], "补偿状态")
    _strings(pointing["notes"], "指向说明")


def _validate_document(document):
    _object(document, "记录文件", {"format", "schema_version", "kind", "records", "selected"}, exact=True)
    if document["format"] != FORMAT or type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        _fail("记录文件", "不支持的格式或版本。")
    if document["kind"] not in ("linear", "feed"):
        _fail("记录文件", "kind须为linear或feed。")
    records = _list(document["records"], "计算记录", 1, MAX_RECORDS)
    _integer(document["selected"], "所选记录", 0, len(records) - 1)
    validate = _validate_linear if document["kind"] == "linear" else _validate_feed
    for record in records:
        validate(record)
    return document


def _unique_keys(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            _fail("记录文件", "重复字段 " + key)
        values[key] = value
    return values


def _invalid_constant(value):
    _fail("记录文件", "非法JSON数值 " + value)


def load_history(path):
    """Load and validate every snapshot before returning any history data."""
    with Path(path).open("rb") as source:
        raw = source.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        _fail("记录文件", "不能超过64 MiB。")
    try:
        document = _plain(json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys,
                                     parse_constant=_invalid_constant))
        if isinstance(document, dict) and document.get("format") == "arraylab.feed_comparison":
            _object(document, "旧比较记录", {"format", "schema_version", "records"}, exact=True)
            if type(document["schema_version"]) is not int or document["schema_version"] != 1:
                _fail("旧比较记录", "不支持的版本。")
            records = _list(document["records"], "旧比较记录", 1, MAX_RECORDS)
            document = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "kind": "feed",
                        "records": records, "selected": len(records) - 1}
        return _validate_document(document)
    except RecursionError as exc:
        raise ValueError("记录文件嵌套过深。") from exc


def save_history(path, kind, records, selected):
    """Validate complete snapshots, then atomically replace the destination."""
    try:
        document = _plain({"format": FORMAT, "schema_version": SCHEMA_VERSION, "kind": kind,
                           "records": records, "selected": selected})
        _validate_document(document)
        raw = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except RecursionError as exc:
        raise ValueError("记录文件嵌套过深。") from exc
    if len(raw) > MAX_BYTES:
        _fail("记录文件", "不能超过64 MiB。")
    target, temporary = Path(path), None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=".arraylab-history-",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
