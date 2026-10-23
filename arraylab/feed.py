"""Single-frequency T-junction synthesis and matched electrical compensation.

This is an impedance network, not an isolated Wilkinson divider or an EM model.
Power ratios follow the target amplitude taper into equal resistive loads.
"""
from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
import uuid

import numpy as np

from . import __version__
from .configuration import FIELDS, MAX_BYTES, _invalid_constant, _unique_keys, parameters_from_values
from .excitation import target_amplitudes, validate_electrical, electrical_excitation, excitation_csv

LEGACY_SETTINGS = {"topology", "input_ohm", "junction_ohm", "load_ohm", "min_line_ohm", "max_line_ohm"}


def settings_payload(settings):
    values = asdict(settings)
    for name in ("custom_amplitudes", "custom_phases_deg", "failed_ports"):
        values[name] = list(values[name])
    return values


@dataclass(frozen=True)
class FeedSettings:
    topology: str = "corporate"
    input_ohm: float = 50.0
    junction_ohm: float = 50.0
    load_ohm: float = 50.0
    min_line_ohm: float = 20.0
    max_line_ohm: float = 120.0
    phase_mode: str = "scan"
    weighting: str = "uniform"
    sidelobe_db: float = 45.0
    taylor_nbar: int = 4
    custom_amplitudes: tuple = ()
    custom_phases_deg: tuple = ()
    phase_bits: int = 0
    amplitude_error_db: float = 0.0
    phase_error_deg: float = 0.0
    failed_ports: tuple = ()
    seed: int = 0

    def __post_init__(self):
        for name in ("custom_amplitudes", "custom_phases_deg", "failed_ports"):
            if isinstance(getattr(self, name), list):
                object.__setattr__(self, name, tuple(getattr(self, name)))

    def validate(self):
        if self.topology not in ("corporate", "series", "symmetric_series"):
            raise ValueError("馈电拓扑须为corporate（并联）、series（串行）或symmetric_series（1分2后对称串馈）。")
        for key, value in asdict(self).items():
            if key not in LEGACY_SETTINGS - {"topology"}:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key}须为有限数值。")
            if not 1 <= value <= 1000:
                raise ValueError(f"{key}须在1–1000 Ω范围内。")
        if self.min_line_ohm >= self.max_line_ohm:
            raise ValueError("提示范围的下限必须小于上限。")
        validate_electrical(self)


def validate_feed_inputs(parameters, settings):
    parameters.validate()
    settings.validate()
    if settings.topology == "symmetric_series" and parameters.elements % 2:
        raise ValueError("1分2后对称串馈需要2–200个偶数阵元；奇数阵元的中心馈点结构尚未定义。")
    target_amplitudes(parameters, settings)


def synthesize_impedance(parameters, settings=FeedSettings(), amplitude_override=None):
    validate_feed_inputs(parameters, settings)
    n = parameters.elements
    amplitudes = target_amplitudes(parameters, settings) if amplitude_override is None else np.asarray(amplitude_override, dtype=float)
    if amplitudes.shape != (n,) or not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0):
        raise ValueError("目标幅度须为每端口一个正有限值。")
    powers = amplitudes**2 / np.sum(amplitudes**2)
    nodes, sections, ports = [], [], []

    def branch(parent, child, fraction, parent_z, child_z, count, kind):
        seen = parent_z / fraction
        z_line = math.sqrt(seen * child_z)
        sections.append({"id": f"T{len(sections) + 1}", "parent": parent, "child": child,
                         "kind": kind, "downstream_ports": count, "local_power_fraction": fraction,
                         "parent_equivalent_ohm": parent_z, "child_equivalent_ohm": child_z,
                         "branch_input_ohm": seen, "line_ohm": z_line, "electrical_length_deg": 90.0,
                         "outside_range": not settings.min_line_ohm <= z_line <= settings.max_line_ohm})

    def visit(indices, input_z, depth, arm=None):
        if len(indices) == 1:
            i = indices[0]
            name = f"P{i}"
            nodes.append({"id": name, "kind": "port", "depth": depth, "first_port": i,
                          "last_port": i, "equivalent_ohm": settings.load_ohm})
            ports.append({"port": i, "load_ohm": settings.load_ohm, "power_fraction": float(powers[i - 1]),
                          "quarter_wave_sections": depth, "network_phase_deg": -90.0 * depth})
            if arm is not None:
                nodes[-1]["arm"] = arm
                ports[-1].update(arm=arm, mirror_port=n + 1 - i)
            return name
        name = f"J{len([item for item in nodes if item['kind'] == 'junction']) + 1}"
        nodes.append({"id": name, "kind": "junction", "depth": depth,
                      "first_port": min(indices), "last_port": max(indices), "equivalent_ohm": input_z})
        if arm is not None:
            nodes[-1].update(arm=arm, tap_port=indices[0])
        split = len(indices) // 2 if settings.topology == "corporate" else 1
        for group_index, group in enumerate((indices[:split], indices[split:])):
            child_z = settings.load_ohm if len(group) == 1 else settings.junction_ohm
            child = visit(group, child_z, depth + 1, arm)
            fraction = (len(group) / len(indices) if settings.weighting == "uniform" and amplitude_override is None else
                        float(np.sum(powers[np.array(group) - 1]) / np.sum(powers[np.array(indices) - 1])))
            branch(name, child, fraction, input_z, child_z, len(group),
                   ("tap" if group_index == 0 else "through") if settings.topology != "corporate" else "branch")
        return name

    if n == 1:
        root = "IN"
        nodes.append({"id": root, "kind": "input", "depth": 0, "first_port": 1,
                      "last_port": 1, "equivalent_ohm": settings.input_ohm})
        child = visit([1], settings.load_ohm, 1)
        branch(root, child, 1.0, settings.input_ohm, settings.load_ohm, 1, "match")
    elif settings.topology == "symmetric_series":
        root = "J1"
        nodes.append({"id": root, "kind": "junction", "depth": 0, "first_port": 1,
                      "last_port": n, "equivalent_ohm": settings.input_ohm})
        half = n // 2
        # Physical port order remains P1..PN along the array. Both chains start
        # at the center and proceed outward, so corresponding paths are mirrored.
        for arm, indices in (("left", list(range(half, 0, -1))),
                             ("right", list(range(half + 1, n + 1)))):
            child_z = settings.load_ohm if half == 1 else settings.junction_ohm
            child = visit(indices, child_z, 1, arm)
            branch(root, child, .5, settings.input_ohm, child_z, half, "input_split")
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
        "按目标幅度平方分配功率、同一纯电阻负载；50 Ω是可修改初值，不代表实测天线阻抗。",
        "各变换线在所设频率处均为90°电长度；节点理想、传输线无损，不计互耦和不连续效应。",
        "这是T型分配网络的阻抗综合，不提供Wilkinson隔离电阻，也不承诺输出隔离、带宽或实测匹配。",
        "阻抗图仅表示功分主干；端口匹配相移与复激励见电气结果。物理尺寸暂缓，连接图不按比例绘制。",
        "阻抗提示范围由用户设置，只用于筛查；范围内不代表具体工艺可制造。",
    ]
    if settings.topology == "series":
        notes.append("串行定义：沿主干逐级T型抽头，抽头和续传支路各带λ/4变换线；末端接最后一个负载，无独立终端电阻。")
    if settings.topology == "symmetric_series":
        notes.append("输入先1:1等分，两臂各N/2端口并从中心向外串馈。端口P1…PN按阵列顺序编号，镜像配对为Pi与P(N+1−i)。仅支持偶数阵元。")
        notes.append("首级及两臂均使用理想T型λ/4变换线；左右镜像端口路径相位相同，不表示不同级的全部端口已等相。")
    if not phases_equal:
        notes.append("原始功分路径级数不同，未经补偿时端口不等相；电气结果单独记录补偿与输出激励。")
    else:
        notes.append("原始功分路径在中心频点端口等相；目标扫描相位及端口补偿见电气结果。")
    if out_of_range:
        notes.append(f"{len(out_of_range)}条变换线超出所设阻抗提示范围；需调整拓扑、节点阻抗或后续实现方式。")
    report = {"format": "arraylab.feed_impedance_result", "schema_version": 2, "app_version": __version__,
            "array_parameters": asdict(parameters), "settings": settings_payload(settings), "root": root,
            "nodes": nodes, "sections": sections, "ports": ports,
            "metrics": {"ports": n, "junctions": max(n - 1, 0), "transformers": len(sections),
                        "input_equivalent_ohm": settings.input_ohm, "line_min_ohm": min(line_values),
                        "line_max_ohm": max(line_values), "out_of_range_sections": out_of_range,
                        "equal_port_phase_at_design_frequency": phases_equal}, "notes": notes}
    if settings.topology == "symmetric_series":
        report["routing"] = {"input_split": [.5, .5], "direction": "center_to_outward",
                             "left_ports": list(range(n // 2, 0, -1)), "right_ports": list(range(n // 2 + 1, n + 1))}
        report["metrics"]["ports_per_arm"] = n // 2
    return report


def design_feed(parameters, settings=FeedSettings()):
    from .complex_pattern import pattern_analysis
    report = synthesize_impedance(parameters, settings)
    target, raw, output, metrics = electrical_excitation(parameters, settings, report["ports"])
    reference = float(np.sum(abs(target)))
    report["electrical"] = {"metrics": metrics, "patterns": {}}
    for name, weights in [("target", target), ("uncompensated", raw), ("output", output)]:
        report["electrical"]["patterns"][name] = pattern_analysis(parameters, weights, reference)
    report["notes"].extend([
        "端口补偿采用与负载同阻抗的理想无损匹配相移，输出匹配保持不变；电长度按中心频点给出，不是宽带真时延设计。",
        "目标与输出方向图使用同一目标相干和作为0 dB参考；输出激励用于全空间方向性积分，方向性不等于增益。",
        "误差为记录种子的一次随机实现，不是统计平均。失效按匹配吸收、辐射置零，不建模开路/短路或反射重分配。",
    ])
    return report


def configuration_document(parameters, settings):
    parameters_from_values(asdict(parameters))
    validate_feed_inputs(parameters, settings)
    return {"format": "arraylab.feed_impedance", "schema_version": 2,
            "array_parameters": asdict(parameters), "settings": settings_payload(settings)}


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
    if doc["format"] != "arraylab.feed_impedance" or type(doc["schema_version"]) is not int or doc["schema_version"] not in (1, 2):
        raise ValueError("不支持的馈电设置格式或版本。")
    if not isinstance(doc["array_parameters"], dict) or set(doc["array_parameters"]) != FIELDS:
        raise ValueError("馈电设置须完整包含6项阵列参数。")
    expected = LEGACY_SETTINGS if doc["schema_version"] == 1 else set(asdict(FeedSettings()))
    if not isinstance(doc["settings"], dict) or set(doc["settings"]) != expected:
        raise ValueError("馈电设置字段缺失或含未知字段。")
    p = parameters_from_values(doc["array_parameters"])
    values = dict(doc["settings"])
    if doc["schema_version"] == 1:
        values["phase_mode"] = "none"
    settings = FeedSettings(**values)
    validate_feed_inputs(p, settings)
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
    files = {"feed_report": "feed-network.json", "feed_configuration": "feed-config.json",
            "feed_sections": "feed-sections.csv", "feed_ports": "feed-ports.csv", "feed_diagram": "feed-network.svg"}
    if "electrical" in report:
        patterns = report["electrical"]["patterns"]
        data = np.column_stack([patterns["target"]["angles_deg"], *[patterns[key]["pattern_db"] for key in ("target", "uncompensated", "output")]])
        np.savetxt(directory / "feed-patterns.csv", data, delimiter=",", header="theta_deg,target_db,uncompensated_db,output_db", comments="", fmt="%.12g")
        amplitude = target_amplitudes(p, settings)
        phases = settings.custom_phases_deg if settings.weighting == "custom" else np.zeros(p.elements)
        (directory / "feed-target-excitation.csv").write_text(excitation_csv(amplitude, phases), encoding="utf-8")
        with (directory / "feed-output-excitation.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["element", "amplitude", "phase_deg", "real", "imag"])
            for port in report["ports"]:
                writer.writerow([port["port"], port["output_amplitude"], port["output_phase_deg"], port["output_real"], port["output_imag"]])
        files.update(feed_patterns="feed-patterns.csv", feed_target_excitation="feed-target-excitation.csv", feed_output_excitation="feed-output-excitation.csv")
    return files


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
