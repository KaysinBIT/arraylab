"""Lossless, single-frequency quarter-wave T-junction impedance synthesis.

This is an impedance network, not an isolated Wilkinson divider or an EM model.
The two topologies deliver equal power to equal, purely resistive loads.
"""
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import uuid

from . import __version__
from .configuration import FIELDS, MAX_BYTES, _invalid_constant, _unique_keys, parameters_from_values


@dataclass(frozen=True)
class FeedSettings:
    topology: str = "corporate"
    input_ohm: float = 50.0
    junction_ohm: float = 50.0
    load_ohm: float = 50.0
    min_line_ohm: float = 20.0
    max_line_ohm: float = 120.0

    def validate(self):
        if self.topology not in ("corporate", "series"):
            raise ValueError("馈电拓扑须为corporate（并联）或series（串行）。")
        for key, value in asdict(self).items():
            if key == "topology":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key}须为有限数值。")
            if not 1 <= value <= 1000:
                raise ValueError(f"{key}须在1–1000 Ω范围内。")
        if self.min_line_ohm >= self.max_line_ohm:
            raise ValueError("提示范围的下限必须小于上限。")


def synthesize_impedance(parameters, settings=FeedSettings()):
    parameters.validate()
    settings.validate()
    n = parameters.elements
    nodes, sections, ports = [], [], []

    def branch(parent, child, fraction, parent_z, child_z, count, kind):
        seen = parent_z / fraction
        z_line = math.sqrt(seen * child_z)
        sections.append({"id": f"T{len(sections) + 1}", "parent": parent, "child": child,
                         "kind": kind, "downstream_ports": count, "local_power_fraction": fraction,
                         "parent_equivalent_ohm": parent_z, "child_equivalent_ohm": child_z,
                         "branch_input_ohm": seen, "line_ohm": z_line, "electrical_length_deg": 90.0,
                         "outside_range": not settings.min_line_ohm <= z_line <= settings.max_line_ohm})

    def visit(indices, input_z, depth):
        if len(indices) == 1:
            i = indices[0]
            name = f"P{i}"
            nodes.append({"id": name, "kind": "port", "depth": depth, "first_port": i,
                          "last_port": i, "equivalent_ohm": settings.load_ohm})
            ports.append({"port": i, "load_ohm": settings.load_ohm, "power_fraction": 1.0 / n,
                          "quarter_wave_sections": depth, "network_phase_deg": -90.0 * depth})
            return name
        name = f"J{len([item for item in nodes if item['kind'] == 'junction']) + 1}"
        nodes.append({"id": name, "kind": "junction", "depth": depth,
                      "first_port": indices[0], "last_port": indices[-1], "equivalent_ohm": input_z})
        split = len(indices) // 2 if settings.topology == "corporate" else 1
        for group_index, group in enumerate((indices[:split], indices[split:])):
            child_z = settings.load_ohm if len(group) == 1 else settings.junction_ohm
            child = visit(group, child_z, depth + 1)
            branch(name, child, len(group) / len(indices), input_z, child_z, len(group),
                   ("tap" if group_index == 0 else "through") if settings.topology == "series" else "branch")
        return name

    if n == 1:
        root = "IN"
        nodes.append({"id": root, "kind": "input", "depth": 0, "first_port": 1,
                      "last_port": 1, "equivalent_ohm": settings.input_ohm})
        child = visit([1], settings.load_ohm, 1)
        branch(root, child, 1.0, settings.input_ohm, settings.load_ohm, 1, "match")
    else:
        root = visit(list(range(1, n + 1)), settings.input_ohm, 0)
    # Display sections in source-to-load order, not recursive return order.
    node_order = {node["id"]: i for i, node in enumerate(nodes)}
    sections.sort(key=lambda item: (node_order[item["parent"]], node_order[item["child"]]))
    for index, section in enumerate(sections, 1):
        section["id"] = f"T{index}"
    ports.sort(key=lambda item: item["port"])
    line_values = [item["line_ohm"] for item in sections]
    out_of_range = [item["id"] for item in sections if item["outside_range"]]
    phases_equal = len({port["quarter_wave_sections"] % 4 for port in ports}) == 1
    notes = [
        "等功率、同一纯电阻负载；50 Ω是可修改初值，不代表实测天线阻抗。",
        "各变换线在所设频率处均为90°电长度；节点理想、传输线无损，不计互耦和不连续效应。",
        "这是T型分配网络的阻抗综合，不提供Wilkinson隔离电阻，也不承诺输出隔离、带宽或实测匹配。",
        "尚未设计扫描相位、相位均衡、线宽、基板与物理长度；连接图不按物理尺寸绘制。",
        "阻抗提示范围由用户设置，只用于筛查；范围内不代表具体工艺可制造。",
    ]
    if settings.topology == "series":
        notes.append("串行定义：沿主干逐级T型抽头，抽头和续传支路各带λ/4变换线；末端接最后一个负载，无独立终端电阻。")
    if not phases_equal:
        notes.append("各端口经历的λ/4线段数不同，当前网络不是等相馈电；不能直接作为已有理想阵列方向图的实际馈电。")
    else:
        notes.append("中心频点端口相位模360°相同；扫描相位尚未加入，仍不能替代扫描阵列的完整馈电设计。")
    if out_of_range:
        notes.append(f"{len(out_of_range)}条变换线超出所设阻抗提示范围；需调整拓扑、节点阻抗或后续实现方式。")
    return {"format": "arraylab.feed_impedance_result", "schema_version": 1, "app_version": __version__,
            "array_parameters": asdict(parameters), "settings": asdict(settings), "root": root,
            "nodes": nodes, "sections": sections, "ports": ports,
            "metrics": {"ports": n, "junctions": max(n - 1, 0), "transformers": len(sections),
                        "input_equivalent_ohm": settings.input_ohm, "line_min_ohm": min(line_values),
                        "line_max_ohm": max(line_values), "out_of_range_sections": out_of_range,
                        "equal_port_phase_at_design_frequency": phases_equal}, "notes": notes}


def configuration_document(parameters, settings):
    parameters_from_values(asdict(parameters))
    settings.validate()
    return {"format": "arraylab.feed_impedance", "schema_version": 1,
            "array_parameters": asdict(parameters), "settings": asdict(settings)}


def load_feed_configuration(path):
    with Path(path).open("rb") as source:
        raw = source.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("馈电设置文件不能超过1 MiB。")
    try:
        doc = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    except RecursionError as exc:
        raise ValueError("馈电设置文件嵌套过深。") from exc
    if not isinstance(doc, dict) or set(doc) != {"format", "schema_version", "array_parameters", "settings"}:
        raise ValueError("请选择完整的馈电设置文件feed-config.json，不能使用结果报告。")
    if doc["format"] != "arraylab.feed_impedance" or type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
        raise ValueError("不支持的馈电设置格式或版本。")
    if not isinstance(doc["array_parameters"], dict) or set(doc["array_parameters"]) != FIELDS:
        raise ValueError("馈电设置须完整包含6项阵列参数。")
    if not isinstance(doc["settings"], dict) or set(doc["settings"]) != set(asdict(FeedSettings())):
        raise ValueError("馈电设置字段缺失或含未知字段。")
    p = parameters_from_values(doc["array_parameters"])
    settings = FeedSettings(**doc["settings"])
    settings.validate()
    return p, settings


def write_text_atomic(path, text):
    # Reuse the existing same-directory atomic replace contract.
    import os
    import tempfile
    target = Path(path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=target.parent,
                                         suffix=".tmp", prefix=".arraylab-feed-", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def save_feed_configuration(path, parameters, settings):
    write_text_atomic(path, json.dumps(configuration_document(parameters, settings), ensure_ascii=False,
                                      indent=2, allow_nan=False) + "\n")


def write_feed_files(directory, report):
    """Write into a caller-owned staging directory; caller publishes it atomically."""
    from .feed_diagram import diagram_svg
    directory = Path(directory)
    (directory / "feed-network.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    p = parameters_from_values(report["array_parameters"])
    settings = FeedSettings(**report["settings"])
    save_feed_configuration(directory / "feed-config.json", p, settings)
    for filename, rows in [("feed-sections.csv", report["sections"]), ("feed-ports.csv", report["ports"])]:
        with (directory / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (directory / "feed-network.svg").write_text(diagram_svg(report), encoding="utf-8")
    return {"feed_report": "feed-network.json", "feed_configuration": "feed-config.json",
            "feed_sections": "feed-sections.csv", "feed_ports": "feed-ports.csv", "feed_diagram": "feed-network.svg"}


def export_feed_directory(target, report):
    target = Path(target).absolute()
    if target.exists():
        raise FileExistsError("导出目录已存在，请选择新目录。")
    staging = target.parent / (".arraylab-feed-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        write_feed_files(staging, report)
        staging.rename(target)
    finally:
        if staging.exists():
            for item in staging.iterdir():
                item.unlink()
            staging.rmdir()
