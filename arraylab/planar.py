"""Rectangular/arbitrary planar point arrays with the same electrical feed model."""
from dataclasses import asdict, dataclass, replace
import csv
import io
import json
import math
from pathlib import Path
import uuid

import numpy as np

from . import __version__
from .core import ArrayParameters
from .configuration import MAX_BYTES, _unique_keys, _invalid_constant
from .excitation import finite_number, target_amplitudes, electrical_excitation
from .feed import FeedSettings, settings_payload, synthesize_impedance, write_text_atomic


@dataclass(frozen=True)
class PlanarParameters:
    geometry: str = "rectangular"
    frequency_ghz: float = 3.0
    nx: int = 4
    ny: int = 4
    spacing_x: float = .5
    spacing_y: float = .5
    coordinate_unit: str = "lambda"
    coordinates: tuple = ()
    scan_theta_deg: float = 30.0
    scan_phi_deg: float = 0.0
    cut_phi_deg: float = 0.0
    step_deg: float = .5

    def __post_init__(self):
        if isinstance(self.coordinates, (list, tuple)):
            object.__setattr__(self, "coordinates", tuple(tuple(row) if isinstance(row, (list, tuple)) else row for row in self.coordinates))

    @property
    def elements(self):
        return self.nx * self.ny if self.geometry == "rectangular" else len(self.coordinates)

    @property
    def wavelength_mm(self):
        return 299.792458 / self.frequency_ghz

    def array_parameters(self):
        return ArrayParameters(elements=self.elements, frequency_ghz=self.frequency_ghz, step_deg=self.step_deg)

    def positions(self):
        if self.geometry == "rectangular":
            x = (np.arange(self.nx) - (self.nx - 1) / 2) * self.spacing_x
            y = (np.arange(self.ny) - (self.ny - 1) / 2) * self.spacing_y
            xx, yy = np.meshgrid(x, y, indexing="xy")
            positions = np.column_stack([xx.ravel(), yy.ravel()])
        else:
            positions = np.array(self.coordinates, dtype=float)
        return positions if self.coordinate_unit == "lambda" else positions / self.wavelength_mm

    def validate(self):
        if self.geometry not in ("rectangular", "arbitrary") or self.coordinate_unit not in ("lambda", "mm"):
            raise ValueError("平面阵类型须为rectangular/arbitrary，坐标单位须为lambda/mm。")
        for key in ("nx", "ny"):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= 200:
                raise ValueError("Nx、Ny须为1–200的整数。")
        finite_number(self.frequency_ghz, "频率(GHz)", .001, 1000)
        finite_number(self.spacing_x, "x方向间距", 1e-9, 1e9)
        finite_number(self.spacing_y, "y方向间距", 1e-9, 1e9)
        finite_number(self.scan_theta_deg, "扫描θ(从+z轴，度)", 0, 90)
        finite_number(self.scan_phi_deg, "扫描φ(从+x轴，度)", -180, 180)
        finite_number(self.cut_phi_deg, "切面φ(度)", -180, 180)
        finite_number(self.step_deg, "切面步长(度)", .01, 5)
        if not isinstance(self.coordinates, (tuple, list)):
            raise ValueError("坐标须为二维坐标列表。")
        if self.geometry == "rectangular" and self.coordinates:
            raise ValueError("规则矩形阵不能同时带任意坐标。")
        if not 1 <= self.elements <= 200:
            raise ValueError("平面阵首版支持1–200阵元。")
        for row in self.coordinates:
            if not isinstance(row, (tuple, list)) or len(row) != 2:
                raise ValueError("每个平面阵元需要x、y两个坐标，z固定为0。")
            for value in row:
                finite_number(value, "坐标", -1e9, 1e9)
        points = self.positions()
        if np.max(abs(points)) > 1000:
            raise ValueError("坐标绝对值须不超过1000λ；可平移坐标原点。")
        distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
        if np.max(distances) > 40 + 1e-12:
            raise ValueError("本版平面阵最大两阵元距离限制为40λ，用于控制方向图搜索规模。")
        if len(points) > 1 and np.min(distances[np.triu_indices(len(points), 1)]) < 1e-6:
            raise ValueError("阵元不能重合，最小距离须至少为0.000001λ。")


def geometry_payload(parameters):
    values = asdict(parameters)
    values["coordinates"] = [list(row) for row in parameters.coordinates]
    return values


def planar_amplitudes(parameters, settings):
    parameters.validate()
    dummy = parameters.array_parameters()
    # All shared constraints, including custom count and matched failure model.
    direct = target_amplitudes(dummy, settings)
    if parameters.geometry != "rectangular" or settings.weighting not in ("chebyshev", "taylor"):
        return direct
    if settings.weighting == "taylor" and any(1 < dimension < settings.taylor_nbar for dimension in (parameters.nx, parameters.ny)):
        raise ValueError("规则平面阵Taylor nbar不能超过任一大于1的轴向阵元数。")
    clean = replace(settings, topology="corporate", failed_ports=())
    wx = target_amplitudes(replace(dummy, elements=parameters.nx), clean)
    wy = target_amplitudes(replace(dummy, elements=parameters.ny), clean)
    amplitude = np.outer(wy, wx).ravel()
    if np.min(amplitude) / np.max(amplitude) < 1e-6:
        raise ValueError("二维乘积幅度动态范围超过120 dB。")
    return amplitude / np.max(amplitude)


def validate_planar_design(parameters, settings):
    parameters.validate(); settings.validate()
    if settings.topology == "symmetric_series" and parameters.elements % 2:
        raise ValueError("对称串馈需要偶数端口。")
    planar_amplitudes(parameters, settings)


def read_coordinates_csv(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("坐标CSV不能超过1 MiB。")
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    if next(reader, None) != ["element", "x", "y"]:
        raise ValueError("坐标CSV表头须为element,x,y，单位由坐标单位选项指定。")
    rows = list(reader)
    if not 1 <= len(rows) <= 200:
        raise ValueError("坐标CSV需要1–200行。")
    points = []
    for index, row in enumerate(rows, 1):
        if len(row) != 3 or row[0].strip() != str(index):
            raise ValueError("坐标须按1起始的连续阵元编号排列，每行3列。")
        x, y = float(row[1]), float(row[2])
        finite_number(x, "x", -1e9, 1e9); finite_number(y, "y", -1e9, 1e9)
        points.append((x, y))
    return tuple(points)


def save_planar_configuration(path, parameters, settings):
    validate_planar_design(parameters, settings)
    document = {"format": "arraylab.planar_electrical", "schema_version": 1,
                "geometry": geometry_payload(parameters), "settings": settings_payload(settings)}
    write_text_atomic(path, json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def load_planar_configuration(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("平面阵设置不能超过1 MiB。")
    try:
        doc = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    except RecursionError as exc:
        raise ValueError("平面阵设置嵌套过深。") from exc
    if not isinstance(doc, dict) or set(doc) != {"format", "schema_version", "geometry", "settings"}:
        raise ValueError("请选择完整平面阵设置，不能使用结果报告。")
    if doc["format"] != "arraylab.planar_electrical" or type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
        raise ValueError("未知平面阵格式或版本。")
    if not isinstance(doc["geometry"], dict) or set(doc["geometry"]) != set(asdict(PlanarParameters())):
        raise ValueError("几何字段缺失或含未知字段。")
    if not isinstance(doc["settings"], dict) or set(doc["settings"]) != set(asdict(FeedSettings())):
        raise ValueError("电气字段缺失或含未知字段。")
    p, s = PlanarParameters(**doc["geometry"]), FeedSettings(**doc["settings"])
    validate_planar_design(p, s)
    return p, s


def design_planar(parameters, settings=FeedSettings()):
    from .planar_pattern import planar_pattern_analysis
    validate_planar_design(parameters, settings)
    points = parameters.positions()
    amplitude = planar_amplitudes(parameters, settings)
    dummy = parameters.array_parameters()
    network = synthesize_impedance(dummy, settings, amplitude_override=amplitude)
    theta = 0 if settings.phase_mode == "equalize" else parameters.scan_theta_deg
    phi = parameters.scan_phi_deg
    uv = math.sin(math.radians(theta)) * np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
    phase = -360 * (points @ uv)
    target, raw, output, metrics = electrical_excitation(dummy, settings, network["ports"], amplitude_override=amplitude, phase_override=phase)
    metrics.pop("target_scan_deg")
    metrics.update(target_scan_theta_deg=theta, target_scan_phi_deg=phi)
    for port, point in zip(network["ports"], points):
        port.update(x_lambda=float(point[0]), y_lambda=float(point[1]),
                    x_mm=float(point[0] * parameters.wavelength_mm), y_mm=float(point[1] * parameters.wavelength_mm))
    network.pop("array_parameters")
    network["notes"].append("平面阵馈电按端口编号连接，连接示意不代表实际x-y布线；对称串馈的编号镜像不保证任意坐标的空间镜像。")
    reference = float(np.sum(abs(target)))
    patterns = {name: planar_pattern_analysis(parameters, points, weights, reference, uv)
                for name, weights in [("target", target), ("uncompensated", raw), ("output", output)]}
    return {"format": "arraylab.planar_electrical_result", "schema_version": 1, "app_version": __version__,
            "geometry": geometry_payload(parameters), "settings": settings_payload(settings), "network": network,
            "electrical": {"metrics": metrics, "patterns": patterns},
            "notes": ["各向同性点源、z=0平面、无互耦；方向性为全空间积分，不是增益或全波结果。",
                      "θ从+z轴量起，φ从+x向+y；切面使用指定φ和有符号θ=-90…90°。",
                      "平面阵前后半空间对称；报告代表峰方向，不承诺唯一峰。非相干峰使用数值搜索并明确标记估计性质。",
                      "三维显示网格为3°采样，可能漏掉窄瓣；指标独立搜索，切面步长可调。物理尺寸暂缓。"]}


def write_planar_files(directory, report):
    from .feed_diagram import diagram_svg
    directory = Path(directory)
    p, s = PlanarParameters(**report["geometry"]), FeedSettings(**report["settings"])
    save_planar_configuration(directory / "planar-config.json", p, s)
    (directory / "planar-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    network = report["network"]
    for filename, rows in [("planar-ports.csv", network["ports"]), ("planar-sections.csv", network["sections"])]:
        with (directory / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    points = p.positions() if p.coordinate_unit == "lambda" else p.positions() * p.wavelength_mm
    with (directory / "planar-coordinates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream); writer.writerow(["element", "x", "y"])
        for i, point in enumerate(points, 1): writer.writerow([i, *point])
    patterns = report["electrical"]["patterns"]
    cut = np.column_stack([patterns["target"]["cut_angles_deg"], *[patterns[k]["cut_db"] for k in ("target", "uncompensated", "output")]])
    np.savetxt(directory / "planar-cut.csv", cut, delimiter=",", header="signed_theta_deg,target_db,uncompensated_db,output_db", comments="", fmt="%.12g")
    output = patterns["output"]
    theta, phi = np.meshgrid(output["sphere_theta_deg"], output["sphere_phi_deg"], indexing="ij")
    grid = np.column_stack([theta.ravel(), phi.ravel(), np.array(output["sphere_db"]).ravel()])
    np.savetxt(directory / "planar-sphere.csv", grid, delimiter=",", header="theta_deg,phi_deg,output_db", comments="", fmt="%.12g")
    diagram = dict(network); diagram["array_parameters"] = {"frequency_ghz": p.frequency_ghz}
    (directory / "planar-feed.svg").write_text(diagram_svg(diagram), encoding="utf-8")


def export_planar_directory(target, report, plot="none"):
    target = Path(target).absolute()
    if target.exists(): raise FileExistsError("输出目录已存在，请选择新目录。")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / (".arraylab-planar-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        write_planar_files(staging, report)
        if plot != "none":
            from .planar_plot import render_planar_plots
            render_planar_plots(staging, report, plot)
        staging.rename(target)
    finally:
        if staging.exists():
            for item in staging.iterdir(): item.unlink()
            staging.rmdir()
