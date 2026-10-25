"""Planar-report adaptation for shared scientific figure metrics."""
import math
import numpy as np
from arraylab.shared.plotting.metrics import sampled_cut_metrics, spherical_beamwidths


def planar_summary(report, pattern_key="output"):
    """Derive figure metrics from a complete planar report without mutating it."""
    from arraylab.shared.numerics.array_fields import field_uv
    from arraylab.shared.numerics.pointing import angular_separation

    pattern = report["electrical"]["patterns"][pattern_key]
    result = sampled_cut_metrics(pattern["cut_angles_deg"], pattern["cut_db"])
    result.update(pattern["metrics"])
    result["gain_dbi"] = None
    result["gain_status"] = "missing_efficiency"
    result["orthogonal_beamwidths_deg"] = [None, None]
    ports = report["network"]["ports"]
    points = np.array([[r["x_lambda"], r["y_lambda"]] for r in ports])
    if pattern_key == "output":
        weights = np.array([complex(r["output_real"], r["output_imag"]) for r in ports])
    elif pattern_key == "target":
        weights = np.array([r["target_amplitude"] * np.exp(1j * math.radians(r["target_phase_deg"])) for r in ports])
    elif pattern_key == "uncompensated":
        weights = np.array([r["target_amplitude"] * np.exp(1j * math.radians(r["network_phase_deg"])) for r in ports])
    else:
        raise ValueError("Unknown planar pattern.")
    theta = result["representative_peak_theta_deg"]
    phi = result["representative_peak_phi_deg"]
    result["pointing_offset_deg"] = None
    if theta is not None:
        diameter = float(np.max(np.linalg.norm(points[:, None] - points[None, :], axis=-1)))
        widths = spherical_beamwidths(lambda xyz: field_uv(xyz[:, :2], points, weights), theta, phi,
                                     step_deg=min(.25, 5 / max(1, diameter)))
        result["orthogonal_beamwidths_deg"] = widths["orthogonal_beamwidths_deg"]
        result["great_circle_widths"] = widths
        target_theta = report["electrical"]["metrics"]["target_scan_theta_deg"]
        target_phi = report["electrical"]["metrics"]["target_scan_phi_deg"]
        result["pointing_offset_deg"] = angular_separation(theta, phi, target_theta, target_phi)
    if not np.any(abs(weights) > 0):
        result.update(main_lobe_angle_deg=None, peak_level_db=None, peak_sidelobe_db=None,
                      half_power_beamwidth_deg=None, half_power_edges_deg=None, status="zero_field")
    result["scope"] = "cut PSL / cut HPBW; representative upper-hemisphere peak; local great-circle widths"
    return result
