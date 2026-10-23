"""Check sphere/cut equivalence and saved feed excitation, not screenshot pixels."""
import numpy as np
import pytest

from arraylab.core import ArrayParameters, calculate, pattern_db_at
from arraylab.feed import FeedSettings, design_feed
from arraylab.complex_pattern import field_at_cycles
from arraylab.linear_plot import sphere_pattern, sphere_from_feed, feed_weights, write_sphere_csv, linear_metrics


@pytest.mark.parametrize("scan", [0, 30, -28])
def test_linear_sphere_matches_positive_and_negative_xz_cut(scan):
    p = ArrayParameters(scan_deg=scan)
    sphere = sphere_pattern(p)
    theta = sphere["theta_deg"]
    front = theta <= 90
    phi_zero = int(np.flatnonzero(sphere["phi_deg"] == 0)[0])
    np.testing.assert_allclose(sphere["db"][front, phi_zero],
                               np.maximum(pattern_db_at(theta[front], p), -300), atol=1e-8)
    np.testing.assert_allclose(sphere["db"][front, 0],
                               np.maximum(pattern_db_at(-theta[front], p), -300), atol=1e-8)
    np.testing.assert_allclose(sphere["db"][:, 0], sphere["db"][:, -1], atol=1e-12)


def test_single_element_sphere_and_no_unique_width():
    p = ArrayParameters(elements=1)
    sphere = sphere_pattern(p)
    np.testing.assert_array_equal(sphere["magnitude"], 1)
    metrics = linear_metrics(calculate(p))
    assert metrics["half_power_beamwidth_deg"] is None
    assert metrics["main_lobe_angle_deg"] is None
    assert metrics["directivity_dbi"] == 0
    assert metrics["gain_dbi"] is None


def test_feed_sphere_uses_recorded_complex_output_without_resampling(monkeypatch):
    report = design_feed(ArrayParameters(scan_deg=26), FeedSettings(topology="series", phase_bits=3,
                         phase_error_deg=7, amplitude_error_db=1, failed_ports=(2,), seed=91))
    weights, reference = feed_weights(report)
    assert weights[1] == 0
    monkeypatch.setattr("arraylab.feed.design_feed", lambda *a, **k: pytest.fail("must not recompute saved report"))
    sphere = sphere_from_feed(report)
    phi_zero = int(np.flatnonzero(sphere["phi_deg"] == 0)[0])
    theta = sphere["theta_deg"]
    expected = field_at_cycles(.5 * np.sin(np.deg2rad(theta)), weights) / reference
    np.testing.assert_allclose(sphere["magnitude"][:, phi_zero], abs(expected), atol=1e-14)
    target = sphere_from_feed(report, "target")
    assert not np.allclose(target["magnitude"], sphere["magnitude"])


def test_failed_feed_is_zero_radius_and_has_no_fabricated_gain():
    report = design_feed(ArrayParameters(elements=2), FeedSettings(failed_ports=(1, 2)))
    sphere = sphere_from_feed(report)
    np.testing.assert_array_equal(sphere["magnitude"], 0)
    metrics = linear_metrics(feed_report=report)
    assert metrics["directivity_dbi"] is None
    assert metrics["gain_dbi"] is None
    assert metrics["half_power_beamwidth_deg"] is None


def test_sphere_csv_contains_full_sphere_and_preserves_reference(tmp_path):
    sphere = sphere_pattern(ArrayParameters(scan_deg=30))
    path = tmp_path / "球面.csv"
    write_sphere_csv(path, sphere)
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    assert data.shape == (len(sphere["theta_deg"]) * len(sphere["phi_deg"]), 4)
    assert set(data[:, 0]).issuperset({0, 90, 180})
    assert set(data[:, 1]).issuperset({-180, 0, 180})
    assert np.isfinite(data).all()


def test_feed_raw_weights_use_network_phase():
    report = design_feed(ArrayParameters(elements=6), FeedSettings(topology="series"))
    weights, _ = feed_weights(report, "uncompensated")
    expected = np.array([p["target_amplitude"] * np.exp(1j * np.deg2rad(p["network_phase_deg"])) for p in report["ports"]])
    np.testing.assert_allclose(weights, expected)
    with pytest.raises(ValueError):
        feed_weights(report, "unknown")


def test_metric_footer_reflows_on_resize_and_moves_callback_with_canvas():
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.backend_bases import ResizeEvent
    from arraylab.pattern_metrics import draw_metric_footer
    figure = Figure(figsize=(10, 5)); canvas = FigureCanvasAgg(figure)
    figure.add_subplot(111)
    draw_metric_footer(figure, ["Long selected history label " * 7, "Directivity and beamwidth"])
    old = figure._arraylab_metric_footer
    registry = canvas.callbacks
    figure.set_size_inches(4, 5)
    ResizeEvent("resize_event", canvas)._process()
    canvas.draw()
    resized = figure._arraylab_metric_footer
    assert resized is not old
    assert resized.get_text().count("\n") > old.get_text().count("\n")
    box = resized.get_window_extent(canvas.get_renderer())
    assert box.x0 >= -1 and box.x1 <= figure.bbox.width + 1
    figure.clear()
    ResizeEvent("resize_event", canvas)._process()
    assert not figure.texts

    replacement = Figure(figsize=(4, 5)); replacement.add_subplot(111)
    replacement.set_canvas(canvas); canvas.figure = replacement
    draw_metric_footer(replacement, ["New active result"])
    assert not registry.callbacks.get("resize_event")
    assert len(canvas.callbacks.callbacks["resize_event"]) == 1
    old = replacement._arraylab_metric_footer
    ResizeEvent("resize_event", canvas)._process()
    assert replacement._arraylab_metric_footer is not old
