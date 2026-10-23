import math

import numpy as np
import pytest

from arraylab.pattern_metrics import (HALF_POWER_DB, sampled_cut_metrics,
                                      spherical_beamwidths, metrics_text, draw_metric_footer,
                                      planar_summary)


def test_sampled_width_uses_local_peak_not_absolute_minus_three():
    angles = np.linspace(-30, 30, 6001)
    db = -12 - .1 * (angles - 5) ** 2
    result = sampled_cut_metrics(angles, db)
    assert result["main_lobe_angle_deg"] == pytest.approx(5)
    expected = 2 * math.sqrt(-HALF_POWER_DB / .1)
    assert result["half_power_beamwidth_deg"] == pytest.approx(expected, abs=1e-5)
    shifted = sampled_cut_metrics(angles, db + 80)
    assert shifted["half_power_edges_deg"] == pytest.approx(result["half_power_edges_deg"])
    assert result["peak_sidelobe_db"] is None  # no first minima in a Gaussian cut


def test_ula_sidelobe_and_half_power_against_independent_known_result():
    from arraylab.core import ArrayParameters, calculate
    result = calculate(ArrayParameters(elements=8, step_deg=.01))
    sampled = sampled_cut_metrics(result.angles_deg, result.pattern_db)
    assert sampled["half_power_beamwidth_deg"] == pytest.approx(result.beamwidth_deg, abs=2e-5)
    assert sampled["peak_sidelobe_db"] == pytest.approx(-12.797, abs=.002)
    first_null = math.degrees(math.asin(1 / (8 * .5)))
    assert sampled["main_lobe_bounds_deg"] == pytest.approx([-first_null, first_null], abs=.006)


def test_equal_grating_peaks_are_not_hidden_as_a_unique_beam():
    angles = np.linspace(-180, 180, 3601)
    db = 20 * np.log10(np.maximum(abs(np.cos(np.deg2rad(angles * 3))), 1e-15))
    result = sampled_cut_metrics(angles, db, preferred_angle=0)
    assert result["main_lobe_angle_deg"] is None
    assert result["half_power_beamwidth_deg"] is None
    assert result["selected_peak_angle_deg"] == 0
    assert result["peak_sidelobe_db"] == pytest.approx(0)


def test_flat_and_truncated_patterns_do_not_invent_widths():
    assert sampled_cut_metrics([-1, 0, 1], [-3, -3, -3])["status"] == "flat_cut"
    result = sampled_cut_metrics([0, 1, 2, 3], [-5, -2, -1, 0])
    assert result["main_lobe_angle_deg"] == 3
    assert result["half_power_beamwidth_deg"] is None
    assert result["peak_sidelobe_db"] is None


@pytest.mark.parametrize("angles,values", [([0, 0], [1, 2]), ([0, 1], [1]),
                                          ([0, 1], [0, np.nan]), ([0], [0])])
def test_invalid_samples_rejected(angles, values):
    with pytest.raises(ValueError):
        sampled_cut_metrics(angles, values)


@pytest.mark.parametrize("theta,phi", [(0, 0), (35, 75), (90, -80)])
def test_great_circle_widths_invariant_for_rotated_cosine_field(theta, phi):
    t, p = np.deg2rad([theta, phi])
    center = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
    result = spherical_beamwidths(lambda xyz: xyz @ center, theta, phi)
    assert result["orthogonal_beamwidths_deg"] == pytest.approx([90, 90], abs=1e-8)
    for cut in result["cuts"]:
        assert cut["half_power_edges_deg"] == pytest.approx([-45, 45])
        assert np.dot(cut["tangent_xyz"], center) == pytest.approx(0, abs=1e-15)


def test_linear_ring_has_no_false_transverse_half_power_boundary():
    from arraylab.core import ArrayParameters, normalized_field_u, calculate
    p = ArrayParameters(elements=8)
    result = spherical_beamwidths(lambda xyz: normalized_field_u(xyz[:, 0], p), 0, 0)
    assert result["orthogonal_beamwidths_deg"][0] == pytest.approx(calculate(p).beamwidth_deg, abs=1e-8)
    assert result["orthogonal_beamwidths_deg"][1] is None


def test_zero_field_has_no_width():
    result = spherical_beamwidths(lambda xyz: np.zeros(len(xyz)), 20, 40)
    assert result["orthogonal_beamwidths_deg"] == [None, None]


def test_narrow_mainlobe_is_not_missed_by_coarse_display_step():
    from arraylab.core import ArrayParameters, normalized_field_u, calculate
    p = ArrayParameters(elements=200, spacing=10)
    result = spherical_beamwidths(lambda xyz: normalized_field_u(xyz[:, 0], p), 0, 0, step_deg=1)
    # First half-power point of a selected lobe, although many grating peaks exist.
    assert 0 < result["orthogonal_beamwidths_deg"][0] < .1


def test_planar_summary_keeps_existing_reference_and_great_circle_widths():
    from arraylab.planar import PlanarParameters, design_planar
    from arraylab.feed import FeedSettings
    report = design_planar(PlanarParameters(nx=4, ny=4, scan_theta_deg=0), FeedSettings(phase_mode="scan"), workers=1)
    summary = planar_summary(report)
    assert summary["directivity_dbi"] == report["electrical"]["patterns"]["output"]["metrics"]["directivity_dbi"]
    assert summary["orthogonal_beamwidths_deg"][0] == pytest.approx(summary["orthogonal_beamwidths_deg"][1])
    assert summary["gain_dbi"] is None
    assert summary["pointing_offset_deg"] == pytest.approx(0)
    assert "great_circle_widths" not in report["electrical"]["patterns"]["output"]["metrics"]


def test_footer_does_not_misname_directivity_as_gain_and_exports_inside_figure():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(8, 5), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [0, 1])
    lines = metrics_text({"directivity_dbi": 12, "half_power_beamwidth_deg": 15})
    assert "Peak directivity D 12 dBi" in lines[1]
    assert "Gain G unavailable" in lines[1]
    footer = draw_metric_footer(figure, lines)
    canvas.draw()
    bbox = footer.get_window_extent(canvas.get_renderer())
    assert bbox.y0 >= 0
    assert bbox.y1 < axis.get_window_extent().y0
    assert bbox.x0 >= 0 and bbox.x1 <= figure.bbox.width


def test_narrow_figure_footer_wraps_long_history_label_without_clipping():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=(4, 4), layout="constrained")
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.plot([0, 1], [0, 1])
    lines = ["Selected history: " + "abcdefghij" * 8]
    lines += metrics_text({"directivity_dbi": 10, "half_power_beamwidth_deg": 18,
                           "pointing_offset_deg": -3.5e-15})
    assert "Pointing offset 0 deg" in lines[-1]
    footer = draw_metric_footer(figure, lines)
    canvas.draw()
    bbox = footer.get_window_extent(canvas.get_renderer())
    assert bbox.x0 >= 0 and bbox.x1 <= figure.bbox.width
    assert bbox.y1 < axis.get_window_extent().y0
    assert len(footer.get_text().splitlines()) > len(lines)
    assert figure._arraylab_footer_reserved >= .2
