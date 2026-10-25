"""Plot annotations stay separate from metrics and neighbouring panels."""
from dataclasses import replace

import numpy as np
import pytest
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import ResizeEvent

from arraylab.shared.plotting.layout import responsive_subplots, reflow_plot_layout


@pytest.fixture(autouse=True)
def plot_fonts():
    with matplotlib.rc_context({'font.family': ['Microsoft YaHei', 'DejaVu Sans']}):
        yield


def test_responsive_panels_reflow_existing_artists_and_reuse_one_callback():
    figure = Figure(figsize=(12, 7), layout='constrained')
    canvas = FigureCanvasAgg(figure)
    axes = [figure.add_subplot(1, 3, i + 1) for i in range(3)]
    lines = [ax.plot([0, 1], [i, i + 1])[0] for i, ax in enumerate(axes)]
    responsive_subplots(figure, axes, 3)
    count = len(canvas.callbacks.callbacks['resize_event'])
    assert figure._arraylab_responsive_plots['shape'] == (1, 3)
    for width, expected in [(7, (2, 2)), (4.8, (3, 1)), (12, (1, 3))]:
        figure.set_size_inches(width, 7, forward=False)
        ResizeEvent('resize_event', canvas)._process()
        assert figure._arraylab_responsive_plots['shape'] == expected
        assert all(ax.lines[0] is line for ax, line in zip(axes, lines))
        for i, line in enumerate(lines):
            np.testing.assert_array_equal(line.get_ydata(), [i, i + 1])
    responsive_subplots(figure, axes, 3)
    assert len(canvas.callbacks.callbacks['resize_event']) == count
    figure.clear()
    reflow_plot_layout(figure)  # no obsolete axes are inserted after a view switch
    assert not figure.axes


@pytest.fixture(scope='module')
def planar_report():
    from arraylab.features.planar.model import PlanarParameters, design_planar
    from arraylab.features.feed.model import FeedSettings
    return design_planar(PlanarParameters(nx=2, ny=2), FeedSettings(), workers=1)


@pytest.fixture(scope='module')
def subarray_report():
    from arraylab.features.subarray.service import SubarrayConfig, optimize_subarrays
    return optimize_subarrays(replace(SubarrayConfig(), count=4, iterations=2, restarts=1, eval_grid=41), workers=1)


@pytest.mark.parametrize('kind', ['planar-cut', 'planar-map', 'subarray-layout', 'subarray-pattern', 'subarray-cuts'])
@pytest.mark.parametrize('width,height', [(4.8, 3.6), (7, 4.8)])
def test_plot_labels_and_metric_strip_do_not_overlap(kind, width, height, planar_report, subarray_report):
    from arraylab.features.planar.plotting import draw_planar_figure
    from arraylab.features.subarray.plotting import render_subarray_figure

    figure = Figure(figsize=(width, height), dpi=100)
    canvas = FigureCanvasAgg(figure)
    family, view = kind.split('-')
    if family == 'planar':
        draw_planar_figure(figure, planar_report, view, chinese=True)
    else:
        render_subarray_figure(figure, subarray_report, view, chinese=True)
    canvas.draw()
    renderer = canvas.get_renderer()
    footer = figure._arraylab_metric_footer.get_window_extent(renderer)
    for ax in figure.axes:
        for text in (ax.title, ax.xaxis.label, ax.yaxis.label):
            if not text.get_text():
                continue
            bounds = text.get_window_extent(renderer)
            assert not bounds.overlaps(footer), (kind, text.get_text())
            assert bounds.x0 >= -1 and bounds.x1 <= figure.bbox.width + 1
            assert bounds.y0 >= -1 and bounds.y1 <= figure.bbox.height + 1


def test_phase_panels_do_not_recompute_fields_when_resized(monkeypatch):
    from arraylab.features.sparse.engine.core import Config, run
    from arraylab.features.sparse.engine.plots import draw_phase_comparison
    import arraylab.features.sparse.engine.plots as plots

    bundle = run(Config(nx=4, ny=4, k=8, iterations=2, opt_grid=25, eval_grid=81, phase_bits=3), workers=1)
    figure = Figure(figsize=(4.8, 7.1), dpi=100)
    canvas = FigureCanvasAgg(figure)
    draw_phase_comparison(figure, bundle, bundle['results'][-1])
    original = [np.array(ax.lines[0].get_ydata()) for ax in figure.axes]
    monkeypatch.setattr(plots, '_plot_response', lambda *args: pytest.fail('Resize must not evaluate array fields'))
    for size in [(10, 4.5), (4.8, 7.1)]:
        figure.set_size_inches(*size, forward=False)
        reflow_plot_layout(figure)
        canvas.draw()
        boxes = [ax.get_tightbbox(canvas.get_renderer()) for ax in figure.axes]
        for index, bounds in enumerate(boxes):
            assert not any(bounds.overlaps(other) for other in boxes[index + 1:])
            np.testing.assert_array_equal(figure.axes[index].lines[0].get_ydata(), original[index])


@pytest.mark.parametrize('ratio', [1.0, 1.25, 1.5])
def test_single_plot_install_matches_current_canvas_physical_pixels(ratio, monkeypatch):
    """Synthetic DPI setup; native 125/150% switching is checked separately."""
    from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QComboBox
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from arraylab.features.sparse.desktop import SparsePanel

    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    app = QApplication.instance() or QApplication([])
    panel = QWidget()
    panel.reflow_current_figure = lambda: SparsePanel.reflow_current_figure(panel)
    panel.canvas = FigureCanvasQTAgg(Figure(dpi=100))
    panel.canvas._set_device_pixel_ratio(ratio)
    panel.canvas.resize(497, 361)
    panel.toolbar = NavigationToolbar2QT(panel.canvas, panel)
    panel.plot_layout = QVBoxLayout(panel)
    panel.plot_layout.addWidget(panel.canvas)
    panel.plot_layout.addWidget(panel.toolbar)
    panel.view = QComboBox()
    panel.view.addItem('Layout', 'layout')
    figure = Figure(dpi=140)
    figure.add_subplot(111).plot([0, 1], [0, 1])
    try:
        SparsePanel.install_figure(panel, figure)
        # Assert immediately, before a later Qt resize can mask the defect.
        np.testing.assert_allclose(figure.bbox.size,
                                   [panel.canvas.width() * ratio, panel.canvas.height() * ratio])
        assert figure.dpi == pytest.approx(100 * ratio)
        assert figure._original_dpi == 100
        assert panel.canvas.minimumHeight() == 360
    finally:
        panel.close()
        panel.deleteLater()
        app.processEvents()
