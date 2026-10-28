"""Native workbench acceptance for independent Tx/Rx radar MIMO analysis."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox

from arraylab.features.mimo.model import MIMOParameters, mimo_parameters_from_dict
from arraylab.features.mimo.io import load_mimo_configuration, save_mimo_configuration
from arraylab.framework.parallel import requested_workers, set_workers
from arraylab.features.sparse.engine.core import Config, run as sparse_run
from arraylab.application.desktop.window import MainWindow


def exercise_mimo(app, window, output, report, check):
    """Use isolated windows/files; the caller's live workbench stays untouched."""
    base = Path(output)
    directory = base / ('mimo-workflow-' + uuid4().hex[:8]); directory.mkdir(parents=True)
    old_quit = app.quitOnLastWindowClosed(); app.setQuitOnLastWindowClosed(False)
    previous_workers = requested_workers(); set_workers(1)
    windows = []
    details = {'qt_platform': app.platformName(), 'directory': str(directory),
               'scope': 'Native Qt workbench, real workers and immutable saved snapshots',
               'uses_real_user_history': False, 'fault_injection': [], 'screenshots': []}
    report['mimo_ui'] = details

    def canonical(value):
        def convert(item):
            if isinstance(item, np.ndarray): return item.tolist()
            if isinstance(item, np.generic): return item.item()
            raise TypeError(type(item).__name__)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, default=convert)

    def settle(milliseconds=70):
        loop = QEventLoop(); QTimer.singleShot(milliseconds, loop.quit); loop.exec(); app.processEvents()

    def create_window():
        item = MainWindow(auto_compute=False); windows.append(item)
        item.resize(1440, 900); item.show(); item.raise_(); item.activateWindow(); settle()
        return item

    def key(item, value):
        item.raise_(); item.activateWindow(); settle(25); QTest.keyClick(item, value)

    def wait_finished(panel, name, timeout_ms=45000):
        loop = QEventLoop(); poll = QTimer(); deadline = QTimer(); deadline.setSingleShot(True)
        errors = []; expired = []
        def inspect():
            modal = app.activeModalWidget()
            if isinstance(modal, QMessageBox): errors.append(modal.text()); modal.accept()
            if panel.thread is None and not panel._busy: loop.quit()
        poll.timeout.connect(inspect); poll.start(20)
        deadline.timeout.connect(lambda: (expired.append(True), loop.quit())); deadline.start(timeout_ms)
        if panel.thread is not None or panel._busy: loop.exec()
        poll.stop(); deadline.stop(); app.processEvents()
        if errors: details.setdefault('modal_errors', []).extend(errors)
        check(not expired and not errors and panel.thread is None and not panel._busy and not panel._last_error, name)

    def f5(item, name):
        key(item, Qt.Key.Key_F5)
        started = item.mimo_panel.thread is not None
        wait_finished(item.mimo_panel, name)
        return started

    def expect_rejection(callback, name):
        try: callback()
        except (ValueError, RuntimeError): check(True, name)
        else: check(False, name)

    try:
        probe = create_window(); panel = probe.mimo_panel
        check(probe.active_task() == 'home' and panel.report is None and panel.thread is None,
              'mimo_startup_home_does_not_calculate')
        QTest.mouseClick(probe.task_cards['mimo'], Qt.MouseButton.LeftButton); settle()
        check(probe.active_task() == 'mimo' and probe.inspector_stack.currentWidget() is panel.parameter_widget,
              'mimo_home_card_routes_native_page_and_parameters')
        probe.activate_task('linear'); key(probe, Qt.Key.Key_F10); settle()
        selected=probe.tree.currentItem();ancestor=selected
        while ancestor is not None and ancestor is not probe.mimo_tree_item:ancestor=ancestor.parent()
        check(probe.active_task() == 'mimo' and ancestor is probe.mimo_tree_item
              and panel.view.currentData() == 'layout'
              and selected.data(0,Qt.ItemDataRole.UserRole)==('task_node',('mimo','result','layout')),
              'mimo_F10_routes_page_and_navigation')
        check(f5(probe, 'mimo_default_real_worker_completes'), 'mimo_F5_starts_background_worker')
        default = panel.report; default_parameters = mimo_parameters_from_dict(default['parameters'])
        check(default['counts']['tx_elements'] == 2 and default['counts']['rx_elements'] == 4
              and default['counts']['tx_rx_channels'] == default['counts']['unique_virtual_positions'] == 8,
              'mimo_default_TI_two_tx_four_rx_has_eight_distinct_virtual_positions')
        cut = default['patterns']['cut']; u = np.sin(np.deg2rad(cut['angles_deg']))
        independent = np.exp(2j * np.pi * u[:, None] * np.arange(8)[None, :] / 2).mean(axis=1)
        error = float(np.max(abs(abs(independent) ** 2 - np.power(10., np.asarray(cut['virtual_db']) / 10))))
        details['default_independent_max_power_error'] = error
        check(error < 2e-13 and abs(default['metrics']['orthogonal_beamwidths_deg'][0] - 12.80252579699) < 1e-7,
              'mimo_default_virtual_pattern_matches_independent_eight_element_ULA')
        save_mimo_configuration(base / 'mimo-linear-config.json', default_parameters)
        linear_record = directory / '一维 完整记录.json'; probe.save_history_path(linear_record)
        check(linear_record.is_file() and probe.history_save_button.isEnabled(),
              'mimo_global_record_save_writes_complete_result')
        for view in ('layout', 'cut', 'map', 'channels', 'unique'):
            panel.view.setCurrentIndex(panel.view.findData(view)); settle()
            graphical = view not in ('channels', 'unique')
            expected_rows = 8
            check(panel.report is default and probe.image_button.isEnabled() == graphical
                  and probe.copy_button.isEnabled() == graphical
                  and (bool(panel.figure.axes) and panel.canvas.isVisible() if graphical else
                       panel.table.isVisible() and panel.table.rowCount() == expected_rows),
                  'mimo_view_and_picture_actions_' + view)

        panel.apply_configuration(replace(default_parameters, theta_deg=-25)); panel.view.setCurrentIndex(panel.view.findData('cut'))
        f5(probe, 'mimo_negative_scan_real_worker_completes')
        check(abs(panel.report['metrics']['cut']['main_lobe_angle_deg'] + 25) < 1e-8,
              'mimo_signed_linear_scan_points_to_negative_angle')
        planar_parameters = MIMOParameters(dimension='planar',
            tx_positions=((0, 0), (1.5, 0), (0, 1.0)),
            rx_positions=((0, 0), (.5, 0), (0, .5), (.5, .5)),
            tx_weights=(1., .7, 1.2), rx_weights=(1., .8, 1.1, .9),
            theta_deg=20, phi_deg=30, cut_phi_deg=30, map_size=91)
        panel.apply_configuration(planar_parameters)
        f5(probe, 'mimo_planar_real_worker_completes')
        planar = panel.report
        check(planar['counts']['tx_rx_channels'] == 12 and planar['geometry']['virtual']['geometric_dimension'] == 2,
              'mimo_planar_custom_apertures_keep_pair_mapping')
        matrix = np.asarray(planar['patterns']['map']['virtual_db'], dtype=float)
        check(matrix.shape == (91, 91) and np.isnan(matrix[0, 0]) and np.isfinite(matrix[45, 45]),
              'mimo_planar_visible_disk_mask_and_saved_map_shape')
        save_mimo_configuration(base / 'mimo-planar-config.json', planar_parameters)
        report['mimo_results'] = {'linear': deepcopy(default), 'planar': deepcopy(planar)}
        saved_science = canonical(planar)
        panel.view.setCurrentIndex(panel.view.findData('map')); settle()
        panel.inputs['theta_deg'].setText('45')
        check(panel.report is planar and canonical(planar) == saved_science and '尚未计算' in panel.state.text(),
              'mimo_pending_parameters_do_not_change_saved_result')
        dirty_config = directory / '待计算 参数.json'; panel.save_path(dirty_config)
        check(load_mimo_configuration(dirty_config).theta_deg == 45 and panel.report is planar,
              'mimo_parameter_save_uses_pending_inputs_without_replacing_result')
        export = directory / '导出 已计算结果'; panel.export_path(export)
        exported = json.loads((export / 'summary.json').read_text(encoding='utf-8'))
        check(canonical(exported) == saved_science and all((export / f'mimo-{kind}.png').is_file() for kind in ('layout', 'cut', 'map')),
              'mimo_export_uses_successful_snapshot_and_all_three_figures')
        planar_record = directory / '二维 完整记录.json'; probe.save_history_path(planar_record)
        reopened = create_window(); restored = reopened.mimo_panel
        with patch('arraylab.mimo_widgets.calculate_mimo', side_effect=AssertionError('Record restore must not calculate')) as solve:
            reopened.load_history_path(planar_record); settle()
            check(solve.call_count == 0 and reopened.active_task() == 'mimo' and canonical(restored.report) == saved_science,
                  'mimo_restart_global_record_load_restores_without_solver')
        check(restored.config() == planar_parameters and restored.view.currentData() == 'map',
              'mimo_complete_record_restores_calculated_parameters_and_view')
        old_state = (restored.report, restored.figure, restored.config(), restored.view.currentData(), restored.summary.text())
        corrupt = directory / '损坏记录.json'
        content = json.loads(planar_record.read_text(encoding='utf-8')); content['report']['counts']['tx_elements'] += 1
        corrupt.write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8')
        corrupt_bytes = corrupt.read_bytes()
        expect_rejection(lambda: reopened.load_history_path(corrupt), 'mimo_corrupt_record_is_rejected')
        check((restored.report, restored.figure, restored.config(), restored.view.currentData(), restored.summary.text()) == old_state
              and corrupt.read_bytes() == corrupt_bytes, 'mimo_corrupt_record_preserves_inputs_figure_and_file')
        with patch.object(restored, '_make_figure', side_effect=RuntimeError('injected MIMO renderer failure')):
            expect_rejection(lambda: reopened.load_history_path(linear_record), 'mimo_renderer_restore_failure_is_reported')
        check((restored.report, restored.figure, restored.config(), restored.view.currentData(), restored.summary.text()) == old_state,
              'mimo_renderer_restore_failure_preserves_previous_snapshot')
        details['fault_injection'].append('record renderer failure')

        large = MIMOParameters(tx_positions=tuple((i * .1, 0) for i in range(128)),
            rx_positions=tuple((i * .1, 0) for i in range(128)), map_size=361, step_deg=.01)
        restored.apply_configuration(large); key(reopened, Qt.Key.Key_F5)
        check(restored.thread is not None and restored._busy and reopened.cancel_button.isEnabled(),
              'mimo_real_worker_exposes_cancel_action')
        QTest.mouseClick(reopened.cancel_button, Qt.MouseButton.LeftButton)
        wait_finished(restored, 'mimo_cancel_recovers_real_worker')
        check(restored.report is old_state[0] and canonical(restored.report) == saved_science
              and '取消' in restored.state.text() and restored.parameter_widget.isEnabled(),
              'mimo_cancel_preserves_previous_successful_result')

        # Handoff must use an existing successful source, never pending inputs.
        restored.apply_configuration(MIMOParameters(coordinate_unit='mm', frequency_ghz=6,
            tx_positions=((0, 0), (15, 0)), rx_positions=((0, 0), (12, 0)), theta_deg=-10))
        other_mm = restored.config().positions('rx') * restored.config().wavelength_mm
        reopened.activate_task('linear'); reopened.inputs['elements'].setText('4')
        reopened.inputs['frequency_ghz'].setText('5'); reopened.inputs['spacing'].setText('.5')
        reopened.inputs['step_deg'].setText('1'); key(reopened, Qt.Key.Key_F5); settle()
        source = reopened.selected_dataset()
        check(source is not None and source.result is not None and reopened.to_mimo_button.isEnabled(),
              'mimo_linear_handoff_requires_successful_source')
        original_mm = source.result.positions_mm.copy()
        reopened.inputs['elements'].setText('10'); reopened.inputs['frequency_ghz'].setText('60')
        reopened.to_mimo_button.menu().actions()[0].trigger(); settle()
        imported = restored.config()
        check(reopened.active_task() == 'mimo' and imported.frequency_ghz == 5 and len(imported.tx_positions) == 4
              and np.allclose(imported.positions('tx')[:, 0] * imported.wavelength_mm, original_mm),
              'mimo_linear_Tx_handoff_uses_saved_geometry_and_frequency')
        check(np.allclose(imported.positions('rx') * imported.wavelength_mm, other_mm)
              and restored.report is old_state[0] and restored.thread is None,
              'mimo_linear_handoff_preserves_other_aperture_mm_and_previous_result')
        tx_mm = imported.positions('tx') * imported.wavelength_mm
        sparse_config = Config(nx=4, ny=4, k=8, iterations=1, opt_grid=25, eval_grid=81,
                               reference_ghz=10, frequencies_ghz='8,12')
        sparse_bundle = sparse_run(sparse_config, workers=1)
        source_panel = reopened.sparse_panel
        reopened.activate_task('sparse'); source_panel.set_config(sparse_config); source_panel.receive(sparse_bundle)
        index = next(i for i, row in enumerate(sparse_bundle['results']) if row['name'] == 'Simulated annealing')
        source_panel.table.selectRow(index); source_panel.case_selector.setCurrentIndex(1); settle()
        selected = source_panel.selected_result(); sparse_science = canonical(sparse_bundle)
        expected_rx_mm = np.asarray(selected['xy']) * (299.792458 / 10)
        source_panel.inputs['reference_ghz'].setValue(50)
        reopened.to_mimo_button.menu().actions()[1].trigger(); settle()
        imported = restored.config()
        check(imported.dimension == 'planar' and imported.frequency_ghz == 12
              and np.allclose(imported.positions('rx') * imported.wavelength_mm, expected_rx_mm)
              and np.allclose(imported.weights('rx'), selected['weights'] / np.sum(selected['weights'])),
              'mimo_sparse_Rx_handoff_uses_saved_layout_weights_and_selected_frequency')
        check(np.allclose(imported.positions('tx') * imported.wavelength_mm, tx_mm)
              and canonical(sparse_bundle) == sparse_science and restored.report is old_state[0] and restored.thread is None,
              'mimo_sparse_handoff_preserves_other_aperture_mm_and_both_saved_results')
        check(imported.theta_deg == 10 and imported.phi_deg == 180,
              'mimo_linear_negative_target_survives_planar_handoff_as_same_direction')
        reopened.load_history_path(planar_record); settle()
        check(restored.config() == mimo_parameters_from_dict(planar['parameters']),
              'mimo_documentation_view_restores_matching_calculated_inputs')
        restored.view.setCurrentIndex(restored.view.findData('map')); settle()
        for width, height in ((1440, 900), (1100, 740)):
            reopened.resize(width, height); settle(160); restored.canvas.draw()
            reopened.results_scroll.ensureWidgetVisible(restored.canvas); settle()
            renderer = restored.canvas.get_renderer()
            footer = restored.figure._arraylab_metric_footer.get_window_extent(renderer)
            boxes = [axis.get_tightbbox(renderer) for axis in restored.figure.axes]
            check(all(not bounds.overlaps(footer) for bounds in boxes)
                  and footer.x0 >= -1 and footer.x1 <= restored.figure.bbox.width + 1
                  and reopened.inspector_stack.currentWidget() is restored.parameter_widget,
                  'mimo_map_metrics_and_parameter_sidebar_fit_' + str(width))
            filename = directory / f'mimo-workbench-{width}.png'
            check(reopened.grab().save(str(filename)), 'mimo_workbench_screenshot_' + str(width))
            details['screenshots'].append(str(filename))
        check(canonical(default) == canonical(report['mimo_results']['linear'])
              and canonical(planar) == canonical(report['mimo_results']['planar']),
              'mimo_handoffs_and_view_changes_preserve_CLI_reference_snapshots')
    finally:
        for item in reversed(windows):
            active = item.mimo_panel
            if active.thread is not None:
                active.cancel()
                loop = QEventLoop(); timer = QTimer(); timer.setInterval(20)
                timer.timeout.connect(lambda p=active: loop.quit() if p.thread is None else None)
                deadline = QTimer(); deadline.setSingleShot(True); deadline.timeout.connect(loop.quit)
                timer.start(); deadline.start(10000); loop.exec(); timer.stop(); deadline.stop()
            item.close()
            for timer in item.findChildren(QTimer): timer.stop()
            if active.thread is None:
                item.deleteLater(); QCoreApplication.sendPostedEvents(item, QEvent.Type.DeferredDelete)
        set_workers(previous_workers); app.setQuitOnLastWindowClosed(old_quit); app.processEvents()
        (directory / 'details.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
