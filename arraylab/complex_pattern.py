"""Complex-weight isotropic ULA pattern and full-sphere directivity.

Peak refinement works in spatial frequency, independently of display sampling.
The geometry is still the existing x-axis line array, not a full-wave model.
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar


def field_at_cycles(cycles, weights, centered=True):
    shape = np.shape(cycles)
    values = np.asarray(cycles, dtype=float).reshape(-1)
    indices = np.arange(len(weights), dtype=float)
    if centered:
        indices -= (len(weights) - 1) / 2
    output = np.empty(len(values), dtype=complex)
    for start in range(0, len(values), 512):
        part = values[start:start + 512]
        output[start:start + 512] = np.sum(weights * np.exp(2j * np.pi * part[:, None] * indices), axis=1)
    return output.reshape(shape)


def pattern_analysis(parameters, weights, reference_amplitude):
    p = parameters
    p.validate()
    if not math.isfinite(reference_amplitude) or reference_amplitude <= 0:
        raise ValueError("方向图参考幅度须为正有限数值。")
    weights = np.asarray(weights, dtype=complex)
    if weights.shape != (p.elements,) or not np.isfinite(weights).all():
        raise ValueError("复激励须为每个阵元一个有限复数。")
    angles = np.arange(-90.0, 90.0, p.step_deg)
    angles = np.append(angles[angles < 90 - 1e-10], 90.0)
    field = field_at_cycles(p.spacing_lambda * np.sin(np.deg2rad(angles)), weights)
    pattern = 20 * np.log10(np.maximum(abs(field) / reference_amplitude, 1e-15))
    result = {"angles_deg": angles.tolist(), "pattern_db": pattern.tolist(),
              "field_phase_deg": np.rad2deg(np.angle(field)).tolist(), "metrics": {
                  "directivity_dbi": None, "peak_angles_deg": [], "main_lobe_angle_deg": None,
                  "half_power_beamwidth_deg": None, "half_power_edges_deg": None, "peak_level_db": None}, "notices": []}
    scale = float(np.max(abs(weights)))
    if scale == 0:
        result["notices"].append("全部辐射激励为零；方向性、主瓣与波束宽度均不定义。")
        return result
    w = weights / scale
    d = p.spacing_lambda

    def power(t):
        return float(abs(np.sum(w * np.exp(2j * np.pi * t * np.arange(len(w)))))**2)

    # Intensity repeats every one spatial-frequency cycle. Optimize one cell.
    lo, hi = (-.5, .5) if d >= .5 else (-d, d)
    grid = np.linspace(lo, hi, max(2049, 64 * len(w) + 1))
    values = abs(field_at_cycles(grid, w, centered=False))**2
    flat = np.count_nonzero(abs(w) > 0) == 1 or np.ptp(values) <= max(float(np.max(values)), 1e-300) * 1e-12
    candidates = [(lo, power(lo)), (hi, power(hi))]
    coherent = False
    if len(w) > 1 and np.all(abs(w) > 0):
        step = np.angle(w[1] * np.conj(w[0]))
        coherent_t = -float(step) / (2 * np.pi)
        linear_phase = np.all(abs(np.angle(w[1:] * np.conj(w[:-1]) * np.exp(-1j * step))) < 1e-10)
        if linear_phase and (d >= .5 or -d - 1e-12 <= coherent_t <= d + 1e-12):
            # Positive amplitudes and linear phase attain the triangle bound.
            # This avoids numerical false multiple peaks for tiny apertures.
            candidates = [(coherent_t, float(np.sum(abs(w))**2))]
            coherent, flat = True, False
    if not flat and not coherent:
        maxima = np.flatnonzero((values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])) + 1
        for i in maxima:
            fitted = minimize_scalar(lambda t: -power(t), bounds=(grid[i - 1], grid[i + 1]),
                                     method="bounded", options={"xatol": 1e-14})
            candidates.append((float(fitted.x), power(fitted.x)))
    peak_power = max(value for _, value in candidates)
    corr = np.correlate(w, w, mode="full")[len(w) - 1:]
    denominator = math.fsum([float(corr[0].real), *(
        2 * corr[1:].real * np.sinc(2 * d * np.arange(1, len(w)))).tolist()])
    if denominator <= float(corr[0].real) * 1e-9:
        # Close-spaced cancellation can destroy precision in the sinc quadratic form.
        denominator = quad(lambda u: power(d * u), -1, 1, epsabs=1e-25, epsrel=1e-10, limit=400)[0] / 2
    if denominator <= 0 or not math.isfinite(denominator):
        raise ArithmeticError("复激励总辐射功率积分无法可靠求得。")
    metrics = result["metrics"]
    metrics["directivity_dbi"] = 10 * math.log10(peak_power / denominator)
    metrics["peak_level_db"] = 20 * math.log10(math.sqrt(peak_power) * scale / reference_amplitude)
    if flat:
        result["notices"].append("全向或数值上近似全向，没有唯一主瓣或半功率宽度。")
        return result
    peaks_t = []
    for t, value in candidates:
        if value < peak_power * (1 - 1e-8):
            continue
        shifts = range(math.ceil(-d - t - 1e-10), math.floor(d - t + 1e-10) + 1) if d >= .5 else (0,)
        for shift in shifts:
            location = min(d, max(-d, t + shift))
            if -d - 1e-10 <= t + shift <= d + 1e-10 and not any(abs(location - previous) < 1e-7 for previous in peaks_t):
                peaks_t.append(location)
    peaks_t.sort()
    peaks = [float(np.rad2deg(np.arcsin(np.clip(t / d, -1, 1)))) for t in peaks_t]
    metrics["peak_angles_deg"] = peaks
    if len(peaks) != 1:
        result["notices"].append("存在多个等高主瓣；不报告唯一主瓣角度或宽度。")
        return result
    metrics["main_lobe_angle_deg"] = peaks[0]
    center = peaks_t[0]
    edges = []
    for direction, end in [(-1, max(-d, center - 1)), (1, min(d, center + 1))]:
        side = np.linspace(center, end, max(129, 32 * len(w) + 1))
        side_power = abs(field_at_cycles(side, w, centered=False))**2
        crossings = np.flatnonzero(side_power <= peak_power / 2)
        if not len(crossings):
            break
        index = int(crossings[0])
        a, b = sorted([side[max(0, index - 1)], side[index]])
        if a == b:
            break
        root = brentq(lambda t: power(t) - peak_power / 2, a, b, xtol=1e-14)
        edges.append(float(np.rad2deg(np.arcsin(np.clip(root / d, -1, 1)))))
    if len(edges) == 2:
        metrics["half_power_edges_deg"] = edges
        metrics["half_power_beamwidth_deg"] = edges[1] - edges[0]
    else:
        result["notices"].append("所示角域未包含完整半功率边界；不报告宽度。")
    return result
