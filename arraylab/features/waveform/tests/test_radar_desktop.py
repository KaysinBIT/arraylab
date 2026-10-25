"""Isolated Qt acceptance of the WaveLab 1.4 additions only."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUTPUT = ROOT/'tmp'/'wave14-ui'


def small_configuration(criterion='hpbw'):
    from arraylab.features.waveform.model import default_waveform_configuration
    c = default_waveform_configuration()
    c['params'].update(samples=64, rangeFft=128, loops=8, dopplerFft=16, targetRange=12)
    c['profileB'].update(samples=64, rangeFft=128, loops=8, dopplerFft=16)
    c['antennaDesign']['sparseOptions'].update(criterion=criterion, iterations=100, scanMaxDeg=10,
        targetResolutionDeg=8, trials=1, snapshots=32, maxApertureLambda=10)
    return c


@pytest.fixture(scope='module')
def desktop():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    settings = {'QT_QPA_PLATFORM': 'offscreen', 'ARRAYLAB_HISTORY_DIR': str(OUTPUT/'linear-history'),
                'ARRAYLAB_SPARSE_HISTORY_DIR': str(OUTPUT/'sparse-history'), 'TEMP': str(OUTPUT/'temp'),
                'TMP': str(OUTPUT/'temp'), 'MPLCONFIGDIR': str(OUTPUT/'matplotlib')}
    for key, value in settings.items():
        if key != 'QT_QPA_PLATFORM': Path(value).mkdir(parents=True, exist_ok=True)
    old = {key: os.environ.get(key) for key in settings}
    os.environ.update(settings)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFontDatabase, QFont
    from arraylab.application.desktop.window import MainWindow
    from arraylab.framework.parallel import set_workers, requested_workers
    previous_workers = requested_workers(); set_workers(1)
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion'); app.setQuitOnLastWindowClosed(False)
    assert app.platformName() == 'offscreen', 'This acceptance must not operate real user windows.'
    # The Windows offscreen plugin does not enumerate system fonts. Supply the
    # same installed family as the production stylesheet for meaningful images.
    if not QFontDatabase.families():
        font_file = Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts'/'msyh.ttc'
        if font_file.exists():
            assert QFontDatabase.addApplicationFont(str(font_file)) >= 0
            app.setFont(QFont('Microsoft YaHei', 9))
    window = MainWindow(auto_compute=False)
    window.resize(1536, 900); window.show(); window.activate_task('waveform')
    app.processEvents()
    try:
        yield app, window, window.waveform_panel
    finally:
        panel = window.waveform_panel
        if panel.thread is not None:
            panel.cancel(); wait_idle(app, panel)
        window.close(); window.deleteLater(); app.processEvents()
        set_workers(previous_workers)
        for key, value in old.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value


def wait_idle(app, panel, seconds=30):
    from PySide6.QtTest import QTest
    deadline = time.monotonic()+seconds
    while panel.thread is not None or panel._busy:
        app.processEvents(); QTest.qWait(5)
        assert time.monotonic() < deadline, 'Radar worker did not stop.'
    app.processEvents()
    assert panel._last_error is None, panel._last_error


@pytest.fixture(scope='module')
def radar_reports():
    from arraylab.features.waveform.model import calculate_waveform
    return {kind: calculate_waveform(small_configuration('dml' if kind == 'evaluate' else 'hpbw'),
                                    workers=1, radar_action=kind) for kind in ('waveform', 'sparse', 'evaluate')}


def select(control, value):
    index = control.findData(value)
    assert index >= 0
    control.setCurrentIndex(index)


def test_native_antenna_configuration_roundtrip_without_calculation(desktop):
    app, window, panel = desktop
    config = small_configuration('music')
    config['geometry'] = {'tx': [[0, 0], [2, 0], [4, 0]], 'rx': [[0, .5], [.5, .5], [1, .5], [1.5, .5]],
                          'steerAzDeg': 15, 'steerElDeg': 0, 'referenceFrequencyGHz': 77}
    config['antennaDesign'].update(section='gain', gainScope='all')
    panel.apply_configuration(config)
    from arraylab.features.waveform.model import normalize_waveform_configuration
    assert panel.config() == normalize_waveform_configuration(config)
    assert panel.thread is None and not panel._busy
    assert panel.parameter_widget.count() == 6
    assert not window.datasets and window.mimo_panel.report is None and window.sparse_panel.bundle is None


def test_high_resolution_controls_apply_algorithm_conditions(desktop):
    _, _, panel = desktop
    panel.apply_configuration(small_configuration('music'))
    controls = panel.radar_controls
    assert controls.sparse['snapshots'].isEnabled()
    assert controls.sparse['scanGridStepDeg'].isEnabled()
    assert not controls.sparse['diagonalLoading'].isEnabled()
    select(controls.sparse['smoothing'], 'spatial')
    assert controls.sparse['subarrayLength'].isEnabled()
    select(controls.sparse['criterion'], 'dml')
    assert controls.sparse['smoothing'].currentData() == 'none'
    assert not controls.sparse['smoothing'].isEnabled() and not controls.sparse['scanGridStepDeg'].isEnabled()
    assert not controls.sparse['subarrayLength'].isEnabled()
    select(controls.sparse['criterion'], 'mvdr')
    assert controls.sparse['diagonalLoading'].isEnabled()
    select(controls.sparse['criterion'], 'hpbw')
    assert not controls.sparse['snapshots'].isEnabled()


def test_real_sparse_worker_completes_with_immutable_input_snapshot(desktop):
    app, _, panel = desktop
    panel.apply_configuration(small_configuration())
    panel.start('sparse')
    assert panel.thread is not None and panel._busy and not panel.parameter_widget.isEnabled()
    assert panel.view.currentData() == 'sparse_design'
    wait_idle(app, panel)
    assert panel.report['sparse_design']['valid']
    assert panel.report['sparse_design']['action'] == 'sparse'
    assert panel.parameter_widget.isEnabled()
    old = deepcopy(panel.report['configuration'])
    panel.radar_controls.sparse['targetResolutionDeg'].setText('9')
    assert panel.report['configuration'] == old
    assert panel.config()['antennaDesign']['sparseOptions']['targetResolutionDeg'] == 9


def test_cancel_sparse_worker_retains_previous_result(desktop, radar_reports):
    app, _, panel = desktop
    from PySide6.QtCore import QTimer
    panel.accept_result(radar_reports['waveform'], restore_inputs=True)
    saved = panel.report
    config = small_configuration('dml'); config['antennaDesign']['sparseOptions']['iterations'] = 5000
    panel.apply_configuration(config)
    panel.start('sparse'); QTimer.singleShot(1, panel.cancel)
    wait_idle(app, panel)
    assert panel.report is saved and '取消' in panel.state.text()
    assert panel.parameter_widget.isEnabled()


def test_new_views_and_record_restore_use_saved_results_without_recalculation(desktop, radar_reports):
    app, _, panel = desktop
    saved = radar_reports['evaluate']
    panel.accept_result(saved, restore_inputs=True, view='sparse_design')
    record = OUTPUT/'radar-ui-record.json'
    panel.save_record_path(record)
    panel.apply_configuration(small_configuration('music'))
    with patch('arraylab.features.waveform.desktop.calculate_waveform', side_effect=AssertionError('record restoration recalculated')):
        panel.load_record_path(record)
        for view in ('antenna', 'sparse_design', 'gain_budget'):
            select(panel.view, view); app.processEvents(); panel.canvas.draw()
            assert len(panel.figure.axes) == 2
            assert panel.canvas.isVisible() and panel.check_table.isVisible()
            panel.recovery_table_toggle.setChecked(True); app.processEvents()
            assert panel.table.isVisible() and not panel.canvas.isVisible()
            assert panel.table.rowCount() > 0
            panel.recovery_table_toggle.setChecked(False)
    assert panel.report == saved
    assert panel.config() == saved['configuration']
    select(panel.view, 'sparse_design')
    assert panel.figure.axes[1].get_title().startswith('DML')
    assert panel.figure.axes[1].get_yticks().size == 0


def test_applying_synthesized_geometry_keeps_reference_frequency(desktop, radar_reports):
    _, _, panel = desktop
    saved = radar_reports['sparse']
    panel.accept_result(saved, restore_inputs=True, view='sparse_design')
    with patch('arraylab.features.waveform.antenna_controls.QMessageBox.warning') as warning:
        panel.radar_controls.apply_result()
    warning.assert_not_called()
    config = panel.config(); geometry = config['geometry']
    assert panel.radar_controls.custom.isChecked() and panel.radar_controls.fixed.isChecked()
    assert geometry['referenceFrequencyGHz'] == pytest.approx(saved['sparse_design']['referenceFrequencyGHz'])
    from arraylab.features.waveform.antenna import effective_geometry
    a = effective_geometry(geometry, 77)
    b = effective_geometry(geometry, 79)
    np.testing.assert_allclose(np.asarray(a['tx'])/77, np.asarray(b['tx'])/79, atol=1e-14)
    assert panel.view.currentData() == 'antenna'
    assert panel.report is saved  # applying geometry does not silently recalculate


def test_apply_then_f5_keeps_separate_sparse_diagnostics_without_resynthesis(desktop):
    app, _, panel = desktop
    panel.apply_configuration(small_configuration())
    panel.start('sparse'); wait_idle(app, panel)
    original = deepcopy(panel.report['sparse_design'])
    with patch('arraylab.features.waveform.antenna_controls.QMessageBox.warning') as warning:
        panel.radar_controls.apply_result()
    warning.assert_not_called()
    with patch('arraylab.features.waveform.sparse_design.design', side_effect=AssertionError('F5 must not synthesize')) as design:
        with patch('arraylab.features.waveform.sparse_design.evaluate', side_effect=AssertionError('F5 must not revalidate')) as evaluate:
            panel.start(); wait_idle(app, panel)
    design.assert_not_called(); evaluate.assert_not_called()
    retained = panel.report['sparse_design']
    assert retained['retainedFromPreviousCalculation'] is True
    assert {k: v for k, v in retained.items() if k != 'retainedFromPreviousCalculation'} == original
    select(panel.view, 'sparse_design')
    record = OUTPUT/'radar-applied-f5-record.json'
    saved = deepcopy(panel.report); panel.save_record_path(record)
    panel.apply_configuration(small_configuration('music'))
    with patch('arraylab.features.waveform.desktop.calculate_waveform', side_effect=AssertionError('load must not calculate')):
        panel.load_record_path(record)
    assert panel.report == saved
    assert panel.report['configuration']['geometry']['referenceFrequencyGHz'] == pytest.approx(original['referenceFrequencyGHz'])


def test_failed_record_display_restores_antenna_section_and_inputs(desktop, radar_reports):
    _, _, panel = desktop
    panel.accept_result(radar_reports['waveform'], restore_inputs=True)
    old = small_configuration('music'); old['antennaDesign']['section'] = 'gain'
    panel.apply_configuration(old)
    expected = panel.config()
    with patch.object(panel, '_show_view', side_effect=[RuntimeError('injected post-input rendering failure'), None]):
        with pytest.raises(RuntimeError):
            panel.accept_result(radar_reports['evaluate'], restore_inputs=True, view='sparse_design')
    assert panel.config() == expected


@pytest.mark.parametrize('invalid_rx', ['bad', 'nan, 0', '\n'.join('0, 0' for _ in range(17))])
def test_coordinate_count_update_validates_both_arrays_atomically(desktop, invalid_rx):
    _, _, panel = desktop
    panel.apply_configuration(small_configuration())
    controls = panel.radar_controls
    before = [panel.inputs[k].text() for k in ('txCount', 'rxCount')]
    controls.layout['tx'].setPlainText('0, 0\n1, 0')
    controls.layout['rx'].setPlainText(invalid_rx)
    with patch('arraylab.features.waveform.antenna_controls.QMessageBox.warning') as warning:
        controls.use_geometry_counts()
    assert warning.call_count == 1
    assert [panel.inputs[k].text() for k in ('txCount', 'rxCount')] == before


def test_custom_geometry_preserves_double_precision_on_ui_roundtrip(desktop):
    _, _, panel = desktop
    config = small_configuration()
    config['geometry'] = {'tx': [[.12345678901234567, -.23456789012345678], [2, 0], [4, 0]],
                          'rx': [[0, .5], [.5, .5], [1, .5], [1.5, .5]],
                          'steerAzDeg': .12345678901234567, 'steerElDeg': 0,
                          'referenceFrequencyGHz': 77.12345678901234}
    panel.apply_configuration(config)
    assert panel.config()['geometry'] == config['geometry']


@pytest.mark.parametrize('size', [(1280, 800), (1536, 900)])
def test_radar_layout_capture_and_visible_control_geometry(desktop, radar_reports, size):
    app, window, panel = desktop
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFormLayout
    window.resize(*size)
    panel.accept_result(radar_reports['evaluate'], restore_inputs=True, view='sparse_design')
    panel.parameter_widget.setCurrentIndex(5)
    # Capture the new result plus all four antenna parameter sections.
    for section in range(4):
        panel.radar_controls.pages.setCurrentIndex(section)
        app.processEvents(); QTest.qWait(30); panel.canvas.draw(); app.processEvents()
        renderer = panel.canvas.get_renderer()
        footer = panel.figure._arraylab_metric_footer.get_window_extent(renderer)
        assert all(not axis.get_tightbbox(renderer).overlaps(footer) for axis in panel.figure.axes)
        scroll = panel.radar_controls.pages.currentWidget()
        body = scroll.widget(); form = body.layout().itemAt(0).layout()
        assert isinstance(form, QFormLayout)
        rectangles = [form.itemAt(i).widget().geometry() for i in range(form.count()) if form.itemAt(i).widget()]
        for i, rect in enumerate(rectangles):
            assert rect.width() > 0 and rect.height() > 0
            assert all(not rect.intersects(other) for other in rectangles[i+1:]), 'Parameter form controls overlap.'
        assert scroll.horizontalScrollBar().maximum() == 0
        image = OUTPUT/f'radar-{size[0]}x{size[1]}-section{section}.png'
        assert window.grab().save(str(image))
    summary = {'requested_size': size, 'actual_size': [window.width(), window.height()],
               'canvas_size': [panel.canvas.width(), panel.canvas.height()], 'qt_platform': app.platformName(),
               'screenshots': [f'radar-{size[0]}x{size[1]}-section{s}.png' for s in range(4)]}
    (OUTPUT/f'layout-{size[0]}x{size[1]}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf8')
