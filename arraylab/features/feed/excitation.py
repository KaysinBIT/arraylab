"""Electrical targets and reproducible post-feed perturbations; no GUI imports."""
import csv
import io
import math
from pathlib import Path

import numpy as np
from scipy.signal.windows import chebwin, taylor


def finite_number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{label}须为{low}–{high}之间的有限数值。")


def validate_electrical(settings, parameters=None):
    s = settings
    if s.phase_mode not in ("none", "equalize", "scan"):
        raise ValueError("相位模式须为none、equalize或scan。")
    if s.weighting not in ("uniform", "chebyshev", "taylor", "custom"):
        raise ValueError("幅度分布须为uniform、chebyshev、taylor或custom。")
    finite_number(s.sidelobe_db, "旁瓣抑制参数(dB)", 20, 120)
    if s.weighting == "chebyshev" and s.sidelobe_db < 45:
        raise ValueError("本版Chebyshev旁瓣参数范围为45–120 dB。")
    if s.weighting == "taylor" and s.sidelobe_db > 100:
        raise ValueError("本版Taylor旁瓣参数范围为20–100 dB。")
    for key, low, high in [("taylor_nbar", 2, 20), ("phase_bits", 0, 12), ("seed", 0, 2**32 - 1)]:
        value = getattr(s, key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{key}须为{low}–{high}之间的整数。")
    if s.phase_mode == "none" and s.phase_bits:
        raise ValueError("启用移相量化前，请选择正侧向补偿或扫描补偿。")
    finite_number(s.amplitude_error_db, "幅度误差标准差(dB)", 0, 10)
    finite_number(s.phase_error_deg, "相位误差标准差(度)", 0, 180)
    for name in ("custom_amplitudes", "custom_phases_deg", "failed_ports"):
        if not isinstance(getattr(s, name), (tuple, list)):
            raise ValueError(f"{name}须为列表。")
    if len(s.custom_amplitudes) > 200 or len(s.custom_phases_deg) > 200:
        raise ValueError("自定义激励最多200行。")
    for value in s.custom_amplitudes:
        finite_number(value, "自定义相对幅度", 1e-12, 1e6)
    for value in s.custom_phases_deg:
        finite_number(value, "自定义相位偏置(度)", -36000, 36000)
    if any(type(i) is not int or not 1 <= i <= 200 for i in s.failed_ports) or len(set(s.failed_ports)) != len(s.failed_ports):
        raise ValueError("失效阵元须为不重复的1–200整数编号。")
    if s.weighting != "custom" and (s.custom_amplitudes or s.custom_phases_deg):
        raise ValueError("自定义幅相数据需要选择custom分布。")
    if parameters is not None:
        n = parameters.elements
        if s.weighting == "taylor" and n > 1 and s.taylor_nbar > n:
            raise ValueError("Taylor nbar不能大于阵元数。")
        if s.weighting == "custom" and (len(s.custom_amplitudes) != n or len(s.custom_phases_deg) != n):
            raise ValueError("自定义幅相须恰好覆盖每个阵元，不补零、不截断。")
        if any(i > n for i in s.failed_ports):
            raise ValueError("失效阵元编号超出当前阵元数。")


def target_amplitudes(parameters, settings):
    validate_electrical(settings, parameters)
    n, s = parameters.elements, settings
    if s.weighting == "custom":
        values = np.array(s.custom_amplitudes, dtype=float)
    elif n == 1 or s.weighting == "uniform":
        values = np.ones(n)
    elif s.weighting == "chebyshev":
        values = chebwin(n, at=s.sidelobe_db, sym=True)
    else:
        values = taylor(n, nbar=s.taylor_nbar, sll=s.sidelobe_db, norm=True, sym=True)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("幅度分布必须全为有限正值；零幅度失效请通过失效阵元模型设置。")
    values /= np.max(values)
    if np.min(values) < 1e-6:
        raise ValueError("目标幅度动态范围不能超过120 dB，以避免近零功分支路。")
    if s.topology == "symmetric_series" and not np.allclose(values, values[::-1], rtol=1e-10, atol=0):
        raise ValueError("1分2后对称串馈要求目标幅度镜像对称；相位偏置可用于扫描。")
    return values


def read_excitation_csv(path, count):
    """Strict interchange: exact header, sequential indices, linear positive amplitudes."""
    with Path(path).open("rb") as source:
        raw = source.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("激励CSV不能超过1 MiB。")
    reader = csv.reader(io.StringIO(raw.decode("utf-8-sig")))
    if next(reader, None) != ["element", "amplitude", "phase_deg"]:
        raise ValueError("CSV表头须为element,amplitude,phase_deg；幅度为线性值，相位为附加偏置(度)。")
    rows = list(reader)
    if len(rows) != count:
        raise ValueError(f"需要恰好{count}行激励，不接受缺失或多余行。")
    amplitudes, phases = [], []
    for index, row in enumerate(rows, 1):
        if len(row) != 3 or row[0].strip() != str(index):
            raise ValueError("阵元编号须从1开始严格连续，且每行恰好3列。")
        amplitude, phase = float(row[1]), float(row[2])
        finite_number(amplitude, "相对幅度", 1e-12, 1e6)
        finite_number(phase, "相位偏置(度)", -36000, 36000)
        amplitudes.append(amplitude); phases.append(phase)
    return tuple(amplitudes), tuple(phases)


def excitation_csv(amplitudes, phases):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["element", "amplitude", "phase_deg"])
    for i, (a, p) in enumerate(zip(amplitudes, phases, strict=True), 1):
        writer.writerow([i, format(a, ".17g"), format(p, ".17g")])
    return stream.getvalue()


def wrap_phase(values):
    return (np.asarray(values) + 180) % 360 - 180


def _relative_port_power(values, count):
    """Validate positive real amplitudes, then normalize before squaring."""
    if np.iscomplexobj(values):
        raise ValueError("目标幅度须为每端口一个正有限实数。")
    amplitudes = np.asarray(values, dtype=float)
    if amplitudes.shape != (count,) or not np.isfinite(amplitudes).all() or np.any(amplitudes <= 0):
        raise ValueError("目标幅度须为每端口一个正有限值。")
    scaled = amplitudes / np.max(amplitudes)
    power = scaled ** 2
    power /= np.sum(power)
    if np.any(power <= 0):
        raise ValueError("目标相对幅度动态范围过大，端口功率无法可靠表示。")
    return power


def electrical_excitation(parameters, settings, ports, amplitude_override=None, phase_override=None):
    """Compensation is a matched terminal phase shift (negative electrical delay)."""
    n, s = parameters.elements, settings
    amplitudes = target_amplitudes(parameters, s) if amplitude_override is None else amplitude_override
    power = _relative_port_power(amplitudes, n)
    amplitudes = np.sqrt(power)  # unit total nominal delivered power, equal resistive loads
    offsets = np.arange(n) - (n - 1) / 2
    scan = 0 if s.phase_mode == "equalize" else parameters.scan_deg
    target_phase = -360 * parameters.spacing_lambda * offsets * math.sin(math.radians(scan))
    if phase_override is not None:
        target_phase = np.asarray(phase_override, dtype=float)
    phase_offsets = np.array(s.custom_phases_deg if s.weighting == "custom" else np.zeros(n))
    target_phase = wrap_phase(target_phase + phase_offsets)
    network_phase = np.array([port["network_phase_deg"] for port in ports])
    desired_delay = (network_phase - target_phase) % 360
    desired_delay[np.isclose(desired_delay, 360, atol=1e-9, rtol=0)] = 0
    if s.phase_mode == "none":
        desired_delay[:] = 0
    delay = desired_delay.copy()
    if s.phase_bits:
        step = 360 / 2**s.phase_bits
        delay = (np.floor(delay / step + .5) % 2**s.phase_bits) * step
    rng = np.random.default_rng(s.seed)
    amplitude_error = rng.normal(0, s.amplitude_error_db, n)
    phase_error = rng.normal(0, s.phase_error_deg, n)
    output_amplitudes = amplitudes * 10**(amplitude_error / 20)
    failed = np.zeros(n, dtype=bool)
    for i in s.failed_ports:
        failed[i - 1] = True
    output_amplitudes[failed] = 0
    output_phase = wrap_phase(network_phase - delay + phase_error)
    target = amplitudes * np.exp(1j * np.deg2rad(target_phase))
    raw = amplitudes * np.exp(1j * np.deg2rad(network_phase))
    output = output_amplitudes * np.exp(1j * np.deg2rad(output_phase))
    for i, port in enumerate(ports):
        port.update(target_amplitude=float(amplitudes[i]), target_phase_deg=float(target_phase[i]),
                    phase_offset_deg=float(phase_offsets[i]), desired_delay_deg=float(desired_delay[i]),
                    applied_delay_deg=float(delay[i]), matched_line_wavelengths=float(delay[i] / 360),
                    total_electrical_length_deg=float(-network_phase[i] + delay[i]),
                    compensation_line_ohm=float(s.load_ohm), output_amplitude=float(output_amplitudes[i]),
                    output_phase_deg=None if failed[i] else float(output_phase[i]), amplitude_error_db=float(amplitude_error[i]),
                    phase_error_deg=float(phase_error[i]), failed=bool(failed[i]),
                    output_real=float(output[i].real), output_imag=float(output[i].imag))
    active = ~failed
    metrics = {"phase_mode": s.phase_mode, "target_scan_deg": scan, "weighting": s.weighting,
               "normalization": "nominal sum(abs(weight)**2)=1; common plot reference=sum(abs(target))",
               "output_to_nominal_power_ratio": float(np.sum(abs(output)**2)),
               "nominal_power_fraction_at_failed_ports": float(np.sum(power[failed])),
               "active_ports": int(np.count_nonzero(active)), "seed": s.seed,
               "max_active_phase_error_deg": float(np.max(abs(wrap_phase(output_phase[active] - target_phase[active])))) if np.any(active) else None,
               "error_interpretation": "one seeded realization; failed ports remain matched and radiate zero"}
    return target, raw, output, metrics
