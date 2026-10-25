"""Reproducible electrical scan sweeps; no GUI imports or error statistics."""
from dataclasses import asdict, dataclass, replace
import csv
import json
import math
from pathlib import Path
import uuid
from arraylab.shared.io import write_text_atomic, unique_keys as _unique_keys, invalid_constant as _invalid_constant

import numpy as np

from arraylab import __version__
from arraylab.features.linear import MAX_BYTES, parameters_from_values
from arraylab.features.linear import ArrayParameters
from arraylab.features.linear import write_pattern
from arraylab.features.feed import electrical_excitation, target_amplitudes, finite_number
from arraylab.features.feed import FeedSettings, settings_payload, settings_from_payload, normalized_branch_delays, synthesize_impedance
from arraylab.features.planar import PlanarParameters, geometry_payload, validate_planar_design, planar_amplitudes
from arraylab.shared.numerics.array_fields import field_uv, integrated_power
from arraylab.framework.parallel import ordered_map, requested_workers, check_cancelled, ComputationCancelled


StudyCancelled = ComputationCancelled


@dataclass(frozen=True)
class StudySettings:
    count: int = 13
    scan_axis: str = "theta"
    start_deg: float = 0.0
    stop_deg: float = 60.0

    def validate(self, parameters, settings):
        if type(self.count) is not int or not 2 <= self.count <= 200:
            raise ValueError("扫描点数须为2–200的整数。")
        if self.scan_axis not in ("theta", "phi"):
            raise ValueError("扫描轴须为theta或phi。")
        for value in (self.start_deg, self.stop_deg):
            finite_number(value, "扫描角度", -180, 180)
        if settings.phase_mode != "scan":
            raise ValueError("批量扫描需要原设计启用扫描补偿（scan），请先修改电气设置。")
        planar = isinstance(parameters, PlanarParameters)
        if not planar and self.scan_axis != "theta":
            raise ValueError("直线阵仅支持有符号theta扫描。")
        low, high = (-180, 180) if self.scan_axis == "phi" else ((0, 90) if planar else (-90, 90))
        if not low <= self.start_deg < self.stop_deg <= high:
            raise ValueError(f"扫描起止角须满足{low}≤起始角<终止角≤{high}。")
        cells = (math.ceil(180 / parameters.step_deg) + 1) * self.count
        if cells > 1_000_000:
            raise ValueError("批量切面总采样数不能超过100万；请减少次数或增大切面步长。")


def validate_study(parameters, settings, study):
    if isinstance(parameters, PlanarParameters):
        validate_planar_design(parameters, settings)
    elif isinstance(parameters, ArrayParameters):
        parameters.validate(); settings.validate()
        normalized_branch_delays(settings, parameters.elements)
        target_amplitudes(parameters, settings)
        if settings.topology == "symmetric_series" and parameters.elements % 2:
            raise ValueError("对称串馈需要偶数端口。")
    else:
        raise ValueError("批量评估需要直线阵或平面阵参数。")
    study.validate(parameters, settings)


def study_document(parameters, settings, study):
    validate_study(parameters, settings, study)
    planar = isinstance(parameters, PlanarParameters)
    return {"format": "arraylab.electrical_study", "schema_version": 2,
            "array_kind": "planar" if planar else "linear",
            "geometry": geometry_payload(parameters) if planar else asdict(parameters),
            "feed_settings": settings_payload(settings), "study_settings": asdict(study)}


def save_study_configuration(path, parameters, settings, study):
    write_text_atomic(path, json.dumps(study_document(parameters, settings, study), ensure_ascii=False,
                                      indent=2, allow_nan=False) + "\n")


def load_study_configuration(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("批量评估设置不能超过1 MiB。")
    try:
        doc = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys, parse_constant=_invalid_constant)
    except RecursionError:
        raise ValueError("批量评估设置嵌套过深。") from None
    required = {"format", "schema_version", "array_kind", "geometry", "feed_settings", "study_settings"}
    if not isinstance(doc, dict) or set(doc) != required:
        raise ValueError("请选择完整批量评估设置，不能使用结果报告。")
    if doc["format"] != "arraylab.electrical_study" or type(doc["schema_version"]) is not int or doc["schema_version"] not in (1, 2):
        raise ValueError("未知批量评估格式或版本。")
    if doc["array_kind"] not in ("linear", "planar"):
        raise ValueError("未知阵列类型。")
    expected = asdict(PlanarParameters() if doc["array_kind"] == "planar" else ArrayParameters())
    for key, fields in [("geometry", expected), ("study_settings", asdict(StudySettings()))]:
        if not isinstance(doc[key], dict) or set(doc[key]) != set(fields):
            raise ValueError(f"{key}字段缺失或含未知字段。")
    p = PlanarParameters(**doc["geometry"]) if doc["array_kind"] == "planar" else parameters_from_values(doc["geometry"])
    s = settings_from_payload(doc["feed_settings"], doc["schema_version"] + 1)
    study = StudySettings(**doc["study_settings"])
    validate_study(p, s, study)
    return p, s, study


def power_db(power):
    return 10 * np.log10(np.maximum(power, 1e-30))


def run_study(parameters, settings, study=StudySettings(), *, progress=None, cancelled=None, workers=None):
    workers = requested_workers(workers)
    config = study_document(parameters, settings, study)
    planar = isinstance(parameters, PlanarParameters)
    p = parameters
    dummy = p.array_parameters() if planar else p
    amplitudes = planar_amplitudes(p, settings) if planar else target_amplitudes(p, settings)
    coordinates = p.positions() if planar else np.column_stack([
        (np.arange(p.elements) - (p.elements - 1) / 2) * p.spacing_lambda, np.zeros(p.elements)])
    network = synthesize_impedance(dummy, settings, amplitude_override=amplitudes)
    if planar:
        network.pop("array_parameters")
        network["notes"].append("阻抗主干由端口编号确定，平面坐标见configuration.geometry；不代表实际布线。")
    angles = np.arange(-90.0, 90.0, p.step_deg)
    angles = np.append(angles[angles < 90 - 1e-10], 90.0)
    phi_cut = math.radians(p.cut_phi_deg if planar else 0)
    directions = np.column_stack([np.sin(np.deg2rad(angles)) * math.cos(phi_cut),
                                  np.sin(np.deg2rad(angles)) * math.sin(phi_cut)])
    values = np.linspace(study.start_deg, study.stop_deg, study.count)
    def calculate_point(i):
        check_cancelled(cancelled)
        key = ("scan_theta_deg" if study.scan_axis == "theta" else "scan_phi_deg") if planar else "scan_deg"
        condition = replace(p, **{key: float(values[i])})
        theta = condition.scan_theta_deg if planar else condition.scan_deg
        phi = condition.scan_phi_deg if planar else 0.0
        uv = math.sin(math.radians(theta)) * np.array([math.cos(math.radians(phi)), math.sin(math.radians(phi))])
        ports = [dict(row) for row in network["ports"]]
        target, _, output, electrical = electrical_excitation(
            condition.array_parameters() if planar else condition, settings, ports,
            amplitude_override=amplitudes, phase_override=-360 * (coordinates @ uv))
        reference = float(np.sum(abs(target)))
        field = field_uv(directions, coordinates, output)
        power = (abs(field) / reference) ** 2
        target_field = complex(field_uv(uv, coordinates, output))
        target_power = float(abs(target_field) ** 2)
        # Scale the quadratic form before integration, as the single-design path does.
        scale = float(np.max(abs(output)))
        integral = integrated_power(coordinates, output / scale) * scale**2 if scale else None
        directivity = target_power / integral if integral is not None else None
        metrics = {"target_direction_power_ratio": target_power / reference**2,
                   "target_direction_level_db": float(power_db(target_power / reference**2)),
                   "target_direction_directivity_linear": directivity,
                   "target_direction_directivity_dbi": 10 * math.log10(directivity) if directivity and directivity > 0 else None,
                   "output_to_nominal_power_ratio": electrical["output_to_nominal_power_ratio"],
                   "max_active_phase_error_deg": electrical["max_active_phase_error_deg"]}
        return {"index": i + 1, "scan_theta_deg": theta, "scan_phi_deg": phi, "seed": settings.seed,
                        "metrics": metrics, "pattern_db": power_db(power).tolist(),
                        "ports": ports}
    records, execution = ordered_map(calculate_point, range(study.count), workers=workers,
                                    work_units=study.count*p.elements*len(angles), progress=progress, cancelled=cancelled)
    execution['scope'] = 'scan_points'
    notes = ["各向同性、无互耦、中心频点理想馈电；物理尺寸暂缓。",
             "指标是指定目标方向的方向性，不是全空间最大方向性、增益、波束指向误差或旁瓣电平。",
             "方向性分母使用全空间功率积分；全失效方向性未定义。",
             "曲线共用目标相干幅度和参考；显示/文本零点下限−300 dB。",
             "扫描每点更新理想补偿电长度，使用同一组阵元误差；不是误差统计、固定馈相扫频或器件宽带能力。",
             "扫描使用包含两端点的等间隔角度；平面阵切面φ保持原设置，可能不经过目标方向。"]
    return {"format": "arraylab.electrical_study_result", "schema_version": 2, "app_version": __version__,
            "configuration": config, "runtime": {"numpy": np.__version__, "rng": type(np.random.default_rng(settings.seed).bit_generator).__name__,
                "seed_rule": "same base seed and realization at every scan point"},
            "angles_deg": angles.tolist(), "cut_phi_deg": p.cut_phi_deg if planar else 0, "execution": execution,
            "network": network, "records": records, "notes": notes}


def export_study_directory(target, report, plot=False):
    target = Path(target).absolute()
    if target.exists():
        raise FileExistsError("导出目录已存在，请选择新目录。")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / (".arraylab-study-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        for name, data in [("study-config.json", report["configuration"]), ("study-report.json", report)]:
            (staging / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        rows = [{"index": row["index"], "scan_theta_deg": row["scan_theta_deg"], "scan_phi_deg": row["scan_phi_deg"],
                 "seed": row["seed"], **row["metrics"]} for row in report["records"]]
        with (staging / "study-metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        for row in report["records"]:
            write_pattern(staging / f"study-{row['index']:03d}.dat", report["angles_deg"], row["pattern_db"])
        with (staging / "study-ports.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            columns = ["scan_index", *report["records"][0]["ports"][0]]
            writer = csv.DictWriter(stream, fieldnames=columns); writer.writeheader()
            for record in report["records"]:
                writer.writerows({"scan_index": record["index"], **port} for port in record["ports"])
        if plot:
            from arraylab.features.studies.plotting import render_study_plot
            render_study_plot(staging / "study.png", report)
        staging.rename(target)
    finally:
        if staging.exists():
            for item in staging.iterdir():
                item.unlink()
            staging.rmdir()
