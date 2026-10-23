"""Shared, explicitly scoped pattern measurements and figure captions.

Cut measurements are sample estimates, never a replacement for the exact
linear-array analysis.  A normalized cut alone does not define directivity or
gain.  Great-circle widths describe a selected lobe, not a unique space beam.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks


HALF_POWER_DB = 10 * math.log10(.5)


def sampled_cut_metrics(angles_deg, db, preferred_angle=None):
    """Measure an increasing angular cut; dB may use any common reference.

PSL excludes only the interval between the nearest sampled minima around the
selected peak. Other equal peaks are retained, including grating lobes. When
either bounding minimum is absent no PSL is inferred from a guessed mask.
"""
    x, y = np.asarray(angles_deg, dtype=float), np.asarray(db, dtype=float)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < 2 or
            not np.isfinite(x).all() or not np.isfinite(y).all() or
            not np.all(np.diff(x) > 0)):
        raise ValueError("Direction cut requires matching finite arrays and increasing angles.")
    if preferred_angle is not None and not math.isfinite(preferred_angle):
        raise ValueError("Preferred angle must be finite.")
    result = {
        "main_lobe_angle_deg": None, "peak_angles_deg": [],
        "selected_peak_angle_deg": None, "peak_level_db": float(np.max(y)),
        "half_power_beamwidth_deg": None, "half_power_edges_deg": None,
        "peak_sidelobe_db": None, "main_lobe_bounds_deg": None,
        "method": "sampled_cut_linear_db_interpolation",
        "sidelobe_scope": "current cut; exclude nearest sampled minima around selected peak",
        "status": "undefined", "notices": [],
    }
    if np.ptp(y) <= 1e-10:
        result["status"] = "flat_cut"
        result["notices"].append("Flat cut: no unique peak or half-power width.")
        return result
    indices, properties = find_peaks(y, plateau_size=True)
    candidates = [(int(i), float((x[a] + x[b]) / 2)) for i, a, b in zip(
        indices, properties["left_edges"], properties["right_edges"])]
    if y[0] >= y[1]:
        candidates.append((0, float(x[0])))
    if y[-1] >= y[-2]:
        candidates.append((len(x) - 1, float(x[-1])))
    peaks = sorted((i, a) for i, a in candidates if y[i] >= np.max(y) - 1e-6)
    result["peak_angles_deg"] = [a for _, a in peaks]
    selected = min(peaks, key=lambda pair: abs(pair[1] - preferred_angle)) if preferred_angle is not None else peaks[0]
    index, angle = selected
    result["selected_peak_angle_deg"] = angle
    unique = len(peaks) == 1
    result["status"] = "unique_sampled_peak" if unique else "multiple_sampled_peaks"
    if unique:
        result["main_lobe_angle_deg"] = angle
        level = y[index] + HALF_POWER_DB
        edges = []
        for direction in (-1, 1):
            previous = index
            for current in range(index + direction, -1 if direction == -1 else len(x), direction):
                if y[current] <= level:
                    fraction = (level - y[previous]) / (y[current] - y[previous])
                    edges.append(float(x[previous] + fraction * (x[current] - x[previous])))
                    break
                previous = current
        if len(edges) == 2:
            result["half_power_edges_deg"] = edges
            result["half_power_beamwidth_deg"] = edges[1] - edges[0]
        else:
            result["notices"].append("Cut does not contain both half-power boundaries.")
    else:
        result["notices"].append("Equal peaks: no unique main-lobe angle or width; PSL retains other peaks.")
    minima, _ = find_peaks(-y)
    left, right = minima[minima < index], minima[minima > index]
    if len(left) and len(right):
        a, b = int(left[-1]), int(right[0])
        result["main_lobe_bounds_deg"] = [float(x[a]), float(x[b])]
        outside = np.concatenate((y[:a], y[b + 1:]))
        if len(outside):
            result["peak_sidelobe_db"] = float(np.max(outside) - y[index])
    else:
        result["notices"].append("PSL unavailable: both main-lobe bounding minima are required.")
    return result


def spherical_beamwidths(field_xyz, theta_deg, phi_deg, step_deg=.25):
    """Local half-power widths on meridional and transverse great circles.

``field_xyz`` maps an (N,3) array of unit directions to N complex field values.
The selected direction is supplied by the caller's peak search. Sampling only
brackets crossings; roots use the field itself. Step controls the bracketing
resolution, so callers with large apertures should use a finer step. Near the
center geometric samples avoid skipping very narrow main lobes.
"""
    phi_deg = 0.0 if phi_deg is None else float(phi_deg)
    if (not all(math.isfinite(v) for v in (theta_deg, phi_deg, step_deg)) or
            not 0 <= theta_deg <= 180 or not .001 <= step_deg <= 5):
        raise ValueError("Invalid peak direction or great-circle sampling step.")
    t, p = np.deg2rad([theta_deg, phi_deg])
    center = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    tangents = [np.array([np.cos(t) * np.cos(p), np.cos(t) * np.sin(p), -np.sin(t)]),
                np.array([-np.sin(p), np.cos(p), 0.0])]

    def values(directions):
        answer = np.asarray(field_xyz(directions), dtype=complex)
        if answer.shape != (len(directions),) or not np.isfinite(answer).all():
            raise ValueError("Direction-field callback must return one finite complex value per direction.")
        return abs(answer) ** 2

    peak = float(values(center[None, :])[0])
    # Positive offsets, supplemented near zero to protect narrow selected lobes.
    offsets = np.unique(np.r_[np.geomspace(1e-6, step_deg, 32), np.arange(step_deg, 180, step_deg), 180.0])
    records = []
    for name, tangent in zip(("meridional", "transverse"), tangents):
        record = {"name": name, "tangent_xyz": tangent.tolist(),
                  "half_power_edges_deg": None, "half_power_beamwidth_deg": None,
                  "status": "no_half_power_boundaries"}
        if peak == 0:
            record["status"] = "zero_field_at_selected_direction"
            records.append(record)
            continue
        edges = []
        for sign in (-1, 1):
            angles = sign * offsets
            radians = np.deg2rad(angles)
            directions = np.cos(radians)[:, None] * center + np.sin(radians)[:, None] * tangent
            power = values(directions)
            crossings = np.flatnonzero(power <= peak / 2)
            if not len(crossings):
                continue
            index = int(crossings[0])
            a, b = sorted((0.0 if index == 0 else angles[index - 1], angles[index]))

            def residual(angle):
                radians = math.radians(angle)
                direction = math.cos(radians) * center + math.sin(radians) * tangent
                return float(values(direction[None, :])[0]) / peak - .5

            ra, rb = residual(a), residual(b)
            # Vectorized and scalar trigonometry can round an exact half-power
            # endpoint to opposite sides by a few ulps.
            root = a if abs(ra) <= 1e-12 else b if abs(rb) <= 1e-12 else brentq(residual, a, b, xtol=1e-10)
            edges.append(float(root))
        if len(edges) == 2:
            record.update(half_power_edges_deg=edges,
                          half_power_beamwidth_deg=edges[1] - edges[0], status="bounded_selected_lobe")
        records.append(record)
    return {"theta_deg": float(theta_deg), "phi_deg": phi_deg, "cuts": records,
            "orthogonal_beamwidths_deg": [r["half_power_beamwidth_deg"] for r in records],
            "scope": "local widths through selected direction; not evidence of a unique spatial peak",
            "notices": ["Meridional / transverse great circles; at the pole phi fixes their orientation.",
                        "Missing half-power boundaries produce no width; these are not E/H-plane labels."]}


def planar_summary(report, pattern_key="output"):
    """Derive figure metrics from a complete planar report without mutating it."""
    from .planar_pattern import field_uv
    from .pointing import angular_separation

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


def metrics_text(metrics, chinese=False):
    """Short, language-aware figure footer; unknown gain is never relabelled D."""
    missing = "未定义" if chinese else "n/a"

    def number(key, unit=""):
        value = metrics.get(key)
        if value is None or not math.isfinite(float(value)):
            return missing
        value = 0.0 if abs(float(value)) < 1e-10 else float(value)
        return f"{value:.3g}{unit}"

    degree = "°" if chinese else " deg"
    if "representative_peak_theta_deg" in metrics:
        pointing = ("代表峰 " if chinese else "Representative peak ") + "θ=" + number("representative_peak_theta_deg", degree)
        pointing += ", φ=" + number("representative_peak_phi_deg", degree)
    else:
        pointing = ("切面主瓣 " if chinese else "Cut peak ") + number("main_lobe_angle_deg", degree)
    width = ("切面 3 dB 宽度 " if chinese else "Cut HPBW ") + number("half_power_beamwidth_deg", degree)
    psl = ("切面 PSL " if chinese else "Cut PSL ") + number("peak_sidelobe_db", " dB")
    lines = ["  |  ".join([pointing, width, psl])]
    directivity = ("峰值方向性 D " if chinese else "Peak directivity D ") + number("directivity_dbi", " dBi")
    gain = (("增益 G 未计算（缺效率）" if chinese else "Gain G unavailable (efficiency missing)")
            if metrics.get("gain_dbi") is None else ("增益 G " if chinese else "Gain G ") + number("gain_dbi", " dBi"))
    lines.append(directivity + "  |  " + gain)
    if "orthogonal_beamwidths_deg" in metrics:
        widths = metrics["orthogonal_beamwidths_deg"]
        values = [missing if w is None else f"{w:.3g}{degree}" for w in widths]
        lines.append(("过代表峰正交切面宽度（子午/横向） " if chinese else "Local great-circle HPBW (meridional / transverse) ") + " / ".join(values))
    if metrics.get("pointing_offset_deg") is not None:
        lines[-1] += ("  |  指向偏差 " if chinese else "  |  Pointing offset ") + number("pointing_offset_deg", degree)
    return lines


def draw_metric_footer(figure, lines):
    """Reserve a bottom caption strip inside a Figure, included in PNG export.

Call after constructing the plot and after clearing it on each redraw.
Compatible with both constrained layout and ordinary subplots.
"""
    canvas = figure.canvas
    registry = canvas.callbacks
    if getattr(canvas, "_arraylab_footer_resize_registry", None) is not registry:
        previous_registry = getattr(canvas, "_arraylab_footer_resize_registry", None)
        previous_cid = getattr(canvas, "_arraylab_footer_resize_cid", None)
        if previous_registry is not None and previous_cid is not None:
            previous_registry.disconnect(previous_cid)
        def reflow_footer(event):
            active = event.canvas.figure
            current = getattr(active, "_arraylab_metric_footer", None)
            if current is not None and current in active.texts:
                draw_metric_footer(active, active._arraylab_footer_lines)
                event.canvas.draw_idle()

        canvas._arraylab_footer_resize_registry = registry
        canvas._arraylab_footer_resize_cid = registry.connect("resize_event", reflow_footer)
    old = getattr(figure, "_arraylab_metric_footer", None)
    if old is not None and old in figure.texts:
        old.remove()
    if isinstance(lines, str):
        lines = lines.splitlines()
    lines = list(lines)
    figure._arraylab_footer_lines = lines.copy()
    from matplotlib.font_manager import FontProperties
    import unicodedata

    font = FontProperties(size=8)
    renderer = figure.canvas.get_renderer() if hasattr(figure.canvas, "get_renderer") else None
    width_px, height_px = figure.get_size_inches() * figure.dpi
    available = max(80., width_px - 28.)

    def text_width(value):
        if renderer is not None:
            return renderer.get_text_width_height_descent(value, font, ismath=False)[0]
        units = sum(1 if unicodedata.east_asian_width(c) in ("W", "F") else .55 for c in value)
        return units * 8 * figure.dpi / 72

    wrapped = []
    for original in lines:
        for line in str(original).splitlines() or [""]:
            pending = ""
            for character in line:
                if pending and text_width(pending + character) > available:
                    boundary = pending.rfind(" ")
                    if boundary > len(pending) // 2:
                        wrapped.append(pending[:boundary].rstrip())
                        pending = pending[boundary + 1:].lstrip() + character
                    else:
                        wrapped.append(pending.rstrip())
                        pending = character.lstrip()
                else:
                    pending += character
            wrapped.append(pending.rstrip())
    line_height = 8 * figure.dpi / 72 * 1.55
    reserved = min(.65, max(.12, (len(wrapped) * line_height + 12.) / max(height_px, 1) + .025))
    engine = figure.get_layout_engine()
    if engine is not None and hasattr(engine, "set") and "rect" in engine.get():
        # ConstrainedLayout uses x/y/width/height; TightLayout uses l/b/r/t.
        engine.set(rect=(0, reserved, 1, 1 - reserved if "Constrained" in type(engine).__name__ else 1))
    else:
        figure.subplots_adjust(bottom=reserved + .08)
    artist = figure.text(.5, .025, "\n".join(wrapped), ha="center", va="bottom", fontsize=8,
                         color="#365368", linespacing=1.55, parse_math=False)
    artist.set_in_layout(False)
    figure._arraylab_metric_footer = artist
    figure._arraylab_footer_reserved = reserved
    return artist
