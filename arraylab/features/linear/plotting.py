"""Full-sphere linear-array views from the stored excitation, without Qt."""
import math

import numpy as np
from arraylab.shared.plotting.surface import register_surface

from arraylab.features.linear.model import ArrayParameters, normalized_field_u
from arraylab.shared.numerics.complex_pattern import field_at_cycles


def feed_weights(report, key="output"):
    """Recover the saved excitation; never resample errors or run the solver."""
    ports = report["ports"]
    if key == "output":
        weights = np.array([complex(p["output_real"], p["output_imag"]) for p in ports])
    elif key in ("target", "uncompensated"):
        phase_key = "target_phase_deg" if key == "target" else "network_phase_deg"
        weights = np.array([p["target_amplitude"] * np.exp(1j * math.radians(p[phase_key])) for p in ports])
    else:
        raise ValueError("Unknown feed pattern")
    return weights, sum(p["target_amplitude"] for p in ports)


def sphere_pattern(parameters, weights=None, reference=None, step_deg=2.0):
    """Theta is from +z; phi is from +x to +y. Values retain the cut reference."""
    parameters.validate()
    if not math.isfinite(step_deg) or not .5 <= step_deg <= 10:
        raise ValueError("球面采样步长须为 0.5–10°。")
    theta = np.linspace(0, 180, math.ceil(180 / step_deg) + 1)
    phi = np.linspace(-180, 180, math.ceil(360 / step_deg) + 1)
    tt, pp = np.meshgrid(np.deg2rad(theta), np.deg2rad(phi), indexing="ij")
    u = np.sin(tt) * np.cos(pp)
    if weights is None:
        field = normalized_field_u(u, parameters).astype(complex)
    else:
        weights = np.asarray(weights, dtype=complex)
        if weights.shape != (parameters.elements,) or not np.isfinite(weights).all():
            raise ValueError("线阵复激励须为每阵元一个有限复数。")
        if reference is None or not math.isfinite(reference) or reference <= 0:
            raise ValueError("球面方向图参考幅度须为正有限数值。")
        field = field_at_cycles(parameters.spacing_lambda * u, weights) / reference
    magnitude = np.abs(field)
    return {"theta_deg": theta, "phi_deg": phi, "magnitude": magnitude,
            "db": 20 * np.log10(np.maximum(magnitude, 1e-15)),
            "phase_deg": np.rad2deg(np.angle(field))}


def sphere_from_feed(report, key="output"):
    weights, reference = feed_weights(report, key)
    return sphere_pattern(ArrayParameters(**report["array_parameters"]), weights, reference)


def write_sphere_csv(path, sphere):
    theta, phi = np.meshgrid(sphere["theta_deg"], sphere["phi_deg"], indexing="ij")
    np.savetxt(path, np.column_stack([theta.ravel(), phi.ravel(), sphere["db"].ravel(), sphere["phase_deg"].ravel()]),
               delimiter=",", header="theta_deg,phi_deg,relative_field_db,phase_deg", comments="", fmt="%.12g")


def linear_metrics(result=None, feed_report=None, key="output", angles=None, db=None, sphere=False):
    from arraylab.shared.plotting.metrics import sampled_cut_metrics
    if feed_report is not None:
        pattern = feed_report["electrical"]["patterns"][key]
        target = feed_report["electrical"]["metrics"]["target_scan_deg"]
        metrics = sampled_cut_metrics(pattern["angles_deg"], pattern["pattern_db"], target)
        metrics.update(pattern["metrics"])
        metrics["frequency_ghz"] = feed_report["array_parameters"]["frequency_ghz"]
    elif result is not None:
        target = result.parameters.scan_deg
        metrics = sampled_cut_metrics(result.angles_deg, result.pattern_db, target)
        metrics.update(directivity_dbi=result.directivity_dbi,
                       main_lobe_angle_deg=result.peaks_deg[0] if len(result.peaks_deg) == 1 else None,
                       peak_angles_deg=list(result.peaks_deg), peak_level_db=0.0,
                       half_power_beamwidth_deg=result.beamwidth_deg,
                       half_power_edges_deg=list(result.half_power_deg) if result.half_power_deg else None,
                       frequency_ghz=result.parameters.frequency_ghz)
    else:
        metrics = sampled_cut_metrics(angles, db)
        target = None
    peak = metrics.get("main_lobe_angle_deg")
    metrics["pointing_offset_deg"] = None if peak is None or target is None else peak - target
    metrics["gain_dbi"] = None
    metrics["target_scan_deg"] = target
    if sphere:
        metrics["orthogonal_beamwidths_deg"] = [None, None]
        metrics["representative_peak_theta_deg"] = None
        metrics["representative_peak_phi_deg"] = None
        metrics["peak_is_not_claimed_unique"] = True
        if peak is not None:
            from arraylab.shared.plotting.metrics import spherical_beamwidths
            theta, phi = abs(peak), 0.0 if peak >= 0 else 180.0
            metrics["representative_peak_theta_deg"] = theta
            metrics["representative_peak_phi_deg"] = None if theta == 0 else phi
            if feed_report is not None:
                p = ArrayParameters(**feed_report["array_parameters"])
                weights, _ = feed_weights(feed_report, key)
                field = lambda xyz: field_at_cycles(p.spacing_lambda * xyz[:, 0], weights)
            else:
                p = result.parameters
                field = lambda xyz: normalized_field_u(xyz[:, 0], p)
            widths = spherical_beamwidths(field, theta, phi, step_deg=min(.25, 5 / max(1, (p.elements-1)*p.spacing_lambda)),
                                         flat_cuts=('transverse',) if theta == 0 else ())
            metrics["orthogonal_beamwidths_deg"] = widths["orthogonal_beamwidths_deg"]
    return metrics


def metric_lines(metrics, chinese=False, sphere=False, label=None):
    from arraylab.shared.plotting.metrics import metrics_text
    lines = metrics_text(metrics, chinese=chinese)
    if label:
        frequency = metrics.get("frequency_ghz")
        lines.insert(0, ("所选：" if chinese else "Selected: ") + label +
                     (f"  |  {frequency:g} GHz" if frequency is not None else ""))
    if sphere:
        lines.append("x轴旋转对称；局部正交切面宽度不代表唯一空间主瓣。" if chinese else
                     "Rotational symmetry about x; local widths do not imply a unique spatial peak.")
    return lines


def draw_sphere(figure, sphere, chinese=False, title=None):
    from matplotlib import colormaps, colors, cm
    ax = figure.add_subplot(111, projection="3d")
    tt, pp = np.meshgrid(np.deg2rad(sphere["theta_deg"]), np.deg2rad(sphere["phi_deg"]), indexing="ij")
    radius = sphere["magnitude"]
    top = max(0.0, float(np.max(sphere["db"])))
    norm = colors.Normalize(-40, top)
    x, y, z = radius*np.sin(tt)*np.cos(pp), radius*np.sin(tt)*np.sin(pp), radius*np.cos(tt)
    facecolors = colormaps["viridis"](norm(sphere["db"]))
    surface = ax.plot_surface(x, y, z,
                    facecolors=facecolors, rstride=2, cstride=2,
                    linewidth=0, antialiased=False, shade=False)
    register_surface(figure, ax, surface, x, y, z, facecolors)
    bound = max(1.0, float(np.max(radius)))
    ax.set(xlabel="x", ylabel="y", zlabel="z", xlim=(-bound,bound), ylim=(-bound,bound), zlim=(-bound,bound),
           title=title or ("线阵三维阵因子 · 半径为相对场幅度" if chinese else "Full-sphere array factor; radius = relative field"))
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=22, azim=-50)
    from matplotlib.ticker import MaxNLocator
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(3))
    figure.colorbar(cm.ScalarMappable(norm=norm, cmap="viridis"), ax=ax, shrink=.65, pad=.08,
                    label="相对场幅度 (dB)" if chinese else "Relative field (dB)")
    return ax
