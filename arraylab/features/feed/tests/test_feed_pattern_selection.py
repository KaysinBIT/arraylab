"""A feed source selector controls the actual curves, not only the footer."""
from copy import deepcopy
import os

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
pytest.importorskip('PySide6.QtWidgets')
from PySide6.QtCore import QCoreApplication, QEvent, QTimer
from PySide6.QtWidgets import QApplication

from arraylab.features.linear.model import ArrayParameters
from arraylab.application.desktop.window import MainWindow


@pytest.fixture(scope='module')
def app():
    application = QApplication.instance() or QApplication([])
    application.setStyle('Fusion')
    return application


@pytest.fixture
def panel(app):
    window = MainWindow(auto_compute=False)
    window.activate_task('feed')
    window.apply_parameters(ArrayParameters(elements=8, spacing=.5, scan_deg=0, step_deg=.5))
    feed = window.feed_panel
    feed.topology.setCurrentIndex(feed.topology.findData('series'))
    yield window, feed
    for timer in window.findChildren(QTimer): timer.stop()
    window.close(); window.deleteLater()
    QCoreApplication.sendPostedEvents(window, QEvent.Type.DeferredDelete)
    app.processEvents()


def curve_lines(feed):
    return [line for line in feed.pattern_plot.axes.lines if not line.get_label().startswith('_')]


@pytest.mark.parametrize('angle', [0, 20])
def test_single_source_filters_real_cartesian_and_polar_artists(panel, angle, monkeypatch):
    window, feed = panel
    feed.target_theta.setText(str(angle)); feed.generate()
    saved = deepcopy(feed.report)
    monkeypatch.setattr('arraylab.feed_widgets.design_feed', lambda *a, **k: pytest.fail('View selection must not redesign'))
    for view in (0, 1):
        feed.pattern_view.setCurrentIndex(view)
        for key, color in [('output', '#6c5bc4'), ('uncompensated', '#d77a28'), ('target', '#087fa8')]:
            feed.pattern_source.setCurrentIndex(feed.pattern_source.findData(key))
            plot = feed.pattern_plot
            assert len(plot.datasets) == 1 and plot.selected == 0
            assert plot.datasets[0].pattern_key == key and plot.datasets[0].feed_report is feed.report
            lines = curve_lines(feed)
            assert len(lines) == 1 and lines[0].get_color() == color
            expected = feed.report['electrical']['patterns'][key]
            x = np.asarray(lines[0].get_xdata())
            if view: x = np.rad2deg(x)
            np.testing.assert_allclose(x, expected['angles_deg'], atol=1e-12)
            np.testing.assert_allclose(lines[0].get_ydata(),
                np.maximum(expected['pattern_db'], -60) + (60 if view else 0), atol=1e-12)
            expected_peak = expected['metrics']['main_lobe_angle_deg']
            assert plot.current_metrics['main_lobe_angle_deg'] == expected_peak
            if key != 'uncompensated':
                assert abs(x[np.argmax(lines[0].get_ydata())] - angle) < 1e-10
            else:
                assert abs(x[np.argmax(lines[0].get_ydata())] - 27.5) < 1e-10
    assert feed.report == saved


@pytest.mark.parametrize('angle', [0, -25])
def test_explicit_compensation_comparison_keeps_three_curves_and_final_metrics(panel, angle):
    window, feed = panel
    feed.target_theta.setText(str(angle)); feed.generate()
    for view in (0, 1):
        feed.pattern_view.setCurrentIndex(view)
        feed.pattern_source.setCurrentIndex(feed.pattern_source.findData('comparison'))
        plot = feed.pattern_plot
        assert [data.pattern_key for data in plot.datasets] == ['target', 'uncompensated', 'output']
        assert plot.selected == 2 and abs(plot.current_metrics['main_lobe_angle_deg'] - angle) < 1e-7
        assert [line.get_color() for line in curve_lines(feed)] == ['#087fa8', '#d77a28', '#6c5bc4']
        assert len(plot.axes.get_legend().get_texts()) == 3
        assert '图下指标对应最终输出' in feed.pattern_hint.text()


def test_comparison_switch_to_3d_uses_only_output_and_disables_comparison(panel, monkeypatch):
    window, feed = panel
    feed.generate()
    import arraylab.features.linear.plotting as plots
    original = plots.sphere_from_feed
    keys = []
    def capture(report, key='output'):
        keys.append(key); return original(report, key)
    monkeypatch.setattr(plots, 'sphere_from_feed', capture)
    comparison = feed.pattern_source.findData('comparison')
    feed.pattern_source.setCurrentIndex(comparison)
    feed.pattern_view.setCurrentIndex(2)
    assert keys == ['output']
    assert feed.pattern_source.currentData() == 'output'
    assert not feed.pattern_source.model().item(comparison).isEnabled()
    assert len(feed.pattern_plot.datasets) == 1 and feed.pattern_plot.datasets[0].pattern_key == 'output'
    assert feed.pattern_plot.three_dimensional and '三维已切换' in window.statusBar().currentMessage()
    # Even an API caller cannot send a comparison key to the surface builder.
    feed.pattern_source.setCurrentIndex(comparison)
    assert keys == ['output', 'output']
    feed.pattern_view.setCurrentIndex(1)
    assert feed.pattern_source.model().item(comparison).isEnabled()
    assert feed.pattern_source.currentData() == 'output'


def test_history_comparison_remains_record_based_and_does_not_mix_raw_fields(panel):
    window, feed = panel
    feed.generate(); first = feed.report
    feed.target_theta.setText('20'); feed.generate(); second = feed.report
    feed.pattern_source.setCurrentIndex(feed.pattern_source.findData('comparison'))
    feed.compare_history.setChecked(True)
    feed.history_selection.setCurrentIndex(0)
    plot = feed.pattern_plot
    assert len(plot.datasets) == 2 and plot.selected == 0
    assert [data.feed_report for data in plot.datasets] == [first, second]
    assert all(data.pattern_key == 'output' and data.color is None for data in plot.datasets)
    assert [line.get_color() for line in curve_lines(feed)] == ['#087fa8', '#d77a28']
    assert not feed.pattern_source.isEnabled() and '不是补偿前后对比' in feed.pattern_hint.text()
    feed.compare_history.setChecked(False)
    assert feed.pattern_source.isEnabled() and len(plot.datasets) == 3


def test_none_mode_is_not_silently_compensated_or_mislabeled(panel):
    window, feed = panel
    feed.electrical.phase_mode.setCurrentIndex(feed.electrical.phase_mode.findData('none'))
    feed.generate()
    assert feed.read_settings().phase_mode == feed.report['settings']['phase_mode'] == 'none'
    assert len(curve_lines(feed)) == 1 and abs(feed.pattern_plot.current_metrics['main_lobe_angle_deg'] - 27.49272) < 1e-5
    assert '未做目标相位补偿' in feed.pattern_hint.text()
    feed.pattern_source.setCurrentIndex(feed.pattern_source.findData('comparison'))
    assert '未做目标相位补偿' in feed.pattern_hint.text() and len(curve_lines(feed)) == 3
