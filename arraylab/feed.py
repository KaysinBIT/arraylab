"""Single-frequency T-junction synthesis and matched electrical compensation.

This is an impedance network, not an isolated Wilkinson divider or an EM model.
Power ratios follow the target amplitude taper into equal resistive loads.
"""
from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path
import uuid

import numpy as np

from . import __version__
from .configuration import FIELDS, MAX_BYTES, _invalid_constant, _unique_keys, parameters_from_values
from .dataio import write_pattern
from .excitation import target_amplitudes, validate_electrical, electrical_excitation, excitation_csv

LEGACY_SETTINGS = {"topology", "input_ohm", "junction_ohm", "load_ohm", "min_line_ohm", "max_line_ohm"}
BRANCH_DELAY_FIELDS = {"branch_delays_deg", "branch_delay_elements", "branch_delay_topology"}


def settings_payload(settings):
    values = asdict(settings)
    for name in ("custom_amplitudes", "custom_phases_deg", "failed_ports", "branch_delays_deg"):
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
    branch_delays_deg: tuple = ()
    branch_delay_elements: int = 0
    branch_delay_topology: str = ""

    def __post_init__(self):
        for name in ("custom_amplitudes", "custom_phases_deg", "failed_ports", "branch_delays_deg"):
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
        if not isinstance(self.branch_delays_deg, (tuple, list)):
            raise ValueError("支路匹配延迟须为完整列表。")
        if type(self.branch_delay_elements) is not int or not 0 <= self.branch_delay_elements <= 200:
            raise ValueError("支路延迟绑定阵元数须为0–200的整数。")
        if not isinstance(self.branch_delay_topology, str):
            raise ValueError("支路延迟绑定拓扑须为字符串。")
        if not self.branch_delays_deg:
            if self.branch_delay_elements != 0 or self.branch_delay_topology != "":
                raise ValueError("空支路延迟不应包含阵元数或拓扑绑定。")
        else:
            n = self.branch_delay_elements
            if not 1 <= n <= 200 or self.branch_delay_topology != self.topology:
                raise ValueError("支路延迟的阵元数或拓扑绑定不匹配，请清空并重新设置。")
            if len(self.branch_delays_deg) != (1 if n == 1 else 2 * n - 2):
                raise ValueError("支路延迟须按T1…T末完整提供，每条支路一个数值。")
            for value in self.branch_delays_deg:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 3600:
                    raise ValueError("支路匹配延迟须为0–3600°的有限数值。")


def clear_branch_delays(settings):
    """Explicitly discard topology-bound branch assignments, preserving other inputs."""
    return replace(settings, branch_delays_deg=(), branch_delay_elements=0, branch_delay_topology="")


def normalized_branch_delays(settings, elements):
    """Return source-to-load T1…Tm delays, rejecting stale element/topology bindings."""
    settings.validate()
    if type(elements) is not int or not 1 <= elements <= 200:
        raise ValueError("支路延迟需要1–200个阵元。")
    if settings.branch_delays_deg:
        if settings.branch_delay_elements != elements:
            raise ValueError("阵元数已变化，旧支路延迟不能套用；请清空并重新设置。")
        return tuple(float(value) for value in settings.branch_delays_deg)
    return (0.0,) * (1 if elements == 1 else 2 * elements - 2)


def with_branch_delays(settings, elements, delays):
    """Bind a complete delay vector to the current network; empty means reset."""
    if not isinstance(delays, (tuple, list)):
        raise ValueError("支路匹配延迟须为完整列表。")
    result = (replace(settings, branch_delays_deg=tuple(delays), branch_delay_elements=elements,
                      branch_delay_topology=settings.topology) if delays else clear_branch_delays(settings))
    normalized_branch_delays(result, elements)
    return result


def settings_from_payload(values, schema_version):
    """Strict versioned migration, shared by feed, planar, study and history readers."""
    if type(schema_version) is not int:
        raise ValueError("不支持的馈电设置版本。")
    expected = set(asdict(FeedSettings()))
    if schema_version == 1:
        expected = LEGACY_SETTINGS
    elif schema_version == 2:
        expected -= BRANCH_DELAY_FIELDS
    elif schema_version != 3:
        raise ValueError("不支持的馈电设置版本。")
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("馈电设置字段缺失或含未知字段。")
    migrated = dict(values)
    if schema_version == 1:
        migrated["phase_mode"] = "none"
    result = FeedSettings(**migrated)
    result.validate()
    return result


def validate_feed_inputs(parameters, settings):
    parameters.validate()
    settings.validate()
    normalized_branch_delays(settings, parameters.elements)
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
    delays = normalized_branch_delays(settings, n)
    paths = {root: (0.0, 0.0)}
    for section, delay in zip(sections, delays, strict=True):
        section.update(matched_delay_deg=delay, matched_delay_ohm=section["child_equivalent_ohm"],
                       total_electrical_length_deg=section["electrical_length_deg"] + delay)
        transform, matched = paths[section["parent"]]
        paths[section["child"]] = (transform + section["electrical_length_deg"], matched + delay)
    ports.sort(key=lambda item: item["port"])
    for port in ports:
        transform, matched = paths[f"P{port['port']}"]
        port.update(transform_path_deg=transform, branch_delay_path_deg=matched,
                    network_electrical_length_deg=transform + matched,
                    network_phase_deg=-(transform + matched))
    line_values = [item["line_ohm"] for item in sections]
    out_of_range = [item["id"] for item in sections if item["outside_range"]]
    phases_equal = bool(np.allclose(np.exp(1j * np.deg2rad([port["network_phase_deg"] for port in ports])),
                                   np.exp(1j * np.deg2rad(ports[0]["network_phase_deg"])), rtol=0, atol=1e-10))
    notes = [
        "按目标幅度平方分配功率、同一纯电阻负载；50 Ω是可修改初值，不代表实测天线阻抗。",
        "各变换线在所设频率处均为90°电长度；节点理想、传输线无损，不计互耦和不连续效应。",
        "每条支路在90°变换线之后串接匹配延迟段，其阻抗等于下游等效阻抗；仅增加相位延迟，保持理想功率分配及输入匹配。",
        "这是T型分配网络的阻抗综合，不提供Wilkinson隔离电阻，也不承诺输出隔离、带宽或实测匹配。",
        "连接图分别标注变换段T、支路匹配延迟D和端口补偿C；复激励见电气结果。物理尺寸暂缓，连接图不按比例绘制。",
        "阻抗提示范围由用户设置，只用于筛查；范围内不代表具体工艺可制造。",
    ]
    if settings.topology == "series":
        notes.append("串行定义：沿主干逐级T型抽头，抽头和续传支路各带λ/4变换线；末端接最后一个负载，无独立终端电阻。")
    if settings.topology == "symmetric_series":
        notes.append("输入先1:1等分，两臂各N/2端口并从中心向外串馈。端口P1…PN按阵列顺序编号，镜像配对为Pi与P(N+1−i)。仅支持偶数阵元。")
        notes.append("首级及两臂均使用理想T型λ/4变换线；镜像路径仅在相应支路延迟相同时等相，不表示不同级的全部端口已等相。")
    if not phases_equal:
        notes.append("功分路径的变换线和支路延迟合计后端口不等相；电气结果单独记录末端补偿与输出激励。")
    else:
        notes.append("原始功分路径在中心频点端口等相；目标扫描相位及端口补偿见电气结果。")
    if out_of_range:
        notes.append(f"{len(out_of_range)}条变换线超出所设阻抗提示范围；需调整拓扑、节点阻抗或后续实现方式。")
    report = {"format": "arraylab.feed_impedance_result", "schema_version": 3, "app_version": __version__,
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
    from .pointing import linear_pointing
    report["electrical"]["pointing"] = linear_pointing(parameters, settings, report["electrical"]["patterns"])
    report["notes"].extend([
        "端口补偿采用与负载同阻抗的理想无损匹配相移，输出匹配保持不变；电长度按中心频点给出，不是宽带真时延设计。",
        "目标与输出方向图使用同一目标相干和作为0 dB参考；输出激励用于全空间方向性积分，方向性不等于增益。",
        "误差为记录种子的一次随机实现，不是统计平均。失效按匹配吸收、辐射置零，不建模开路/短路或反射重分配。",
    ])
    return report


def configuration_document(parameters, settings):
    parameters_from_values(asdict(parameters))
    validate_feed_inputs(parameters, settings)
    return {"format": "arraylab.feed_impedance", "schema_version": 3,
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
    if doc["format"] != "arraylab.feed_impedance" or type(doc["schema_version"]) is not int or doc["schema_version"] not in (1, 2, 3):
        raise ValueError("不支持的馈电设置格式或版本。")
    if not isinstance(doc["array_parameters"], dict) or set(doc["array_parameters"]) != FIELDS:
        raise ValueError("馈电设置须完整包含6项阵列参数。")
    p = parameters_from_values(doc["array_parameters"])
    settings = settings_from_payload(doc["settings"], doc["schema_version"])
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
        from .linear_plot import sphere_from_feed, write_sphere_csv
        for key in ("target", "uncompensated", "output"):
            filename = f"feed-{key}-sphere.csv"
            write_sphere_csv(directory / filename, sphere_from_feed(report, key))
            files[f"feed_{key}_sphere"] = filename
        patterns = report["electrical"]["patterns"]
        data = np.column_stack([patterns["target"]["angles_deg"], *[patterns[key]["pattern_db"] for key in ("target", "uncompensated", "output")]])
        np.savetxt(directory / "feed-patterns.csv", data, delimiter=",", header="theta_deg,target_db,uncompensated_db,output_db", comments="", fmt="%.12g")
        for key in ("target", "uncompensated", "output"):
            filename = f"feed-{key}.dat"
            write_pattern(directory / filename, patterns[key]["angles_deg"], patterns[key]["pattern_db"])
            files[f"feed_{key}_dat"] = filename
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
