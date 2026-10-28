"""Scoped native acceptance for the new waveform task and its entry points."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QMessageBox, QScrollArea

from arraylab.framework.parallel import requested_workers, set_workers
from arraylab.features.waveform.model import default_waveform_configuration
from arraylab.features.waveform.io import save_waveform_configuration, load_waveform_configuration
from arraylab.application.desktop.window import MainWindow


def exercise_waveform(app, window, output, report, check):
    """Real workers and isolated files; never calculate another workbench."""
    output = Path(output)
    directory = output / ('waveform-ui-' + uuid4().hex[:8]); directory.mkdir(parents=True)
    old_quit = app.quitOnLastWindowClosed(); app.setQuitOnLastWindowClosed(False)
    previous_workers = requested_workers(); set_workers(1)
    windows = []
    details = {'qt_platform': app.platformName(), 'scope': 'New waveform task and directly affected routing only',
               'directory': str(directory), 'uses_real_user_history': False,
               'fault_injection': [], 'screenshots': []}
    report['waveform_ui'] = details

    def canonical(value): return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)

    def settle(milliseconds=60):
        loop = QEventLoop(); QTimer.singleShot(milliseconds, loop.quit); loop.exec(); app.processEvents()

    def create_window():
        item = MainWindow(auto_compute=False); windows.append(item)
        item.resize(1440, 900); item.show(); item.raise_(); item.activateWindow(); settle()
        return item

    def key(item, value):
        item.raise_(); item.activateWindow(); settle(20); QTest.keyClick(item, value)

    def wait_idle(panel, *, allow_error=False):
        loop = QEventLoop(); poll = QTimer(); deadline = QTimer(); deadline.setSingleShot(True)
        expired, messages = [], []
        def observe():
            modal = app.activeModalWidget()
            if isinstance(modal, QMessageBox): messages.append(modal.text()); modal.accept()
            if panel.thread is None and not panel._busy: loop.quit()
        poll.timeout.connect(observe); poll.start(15)
        deadline.timeout.connect(lambda: (expired.append(True), loop.quit())); deadline.start(30000)
        if panel.thread is not None or panel._busy: loop.exec()
        poll.stop(); deadline.stop(); app.processEvents()
        if expired or messages or panel.thread is not None or panel._busy or (panel._last_error and not allow_error):
            raise AssertionError({'timeout': bool(expired), 'modal': messages, 'error': panel._last_error})

    def reject(callback):
        try: callback()
        except (ValueError, OSError, RuntimeError): return True
        return False

    try:
        probe = create_window(); panel = probe.waveform_panel
        check(probe.active_task() == 'home' and panel.report is None and panel.thread is None,
              'waveform_startup_does_not_calculate')
        QTest.mouseClick(probe.task_cards['waveform'], Qt.MouseButton.LeftButton); settle()
        check(probe.active_task() == 'waveform' and probe.tabs.currentIndex() == 10
              and probe.inspector_stack.currentWidget() is panel.parameter_widget,
              'waveform_home_card_opens_native_page_and_parameters')
        probe.activate_task('home'); key(probe, Qt.Key.Key_F11); settle()
        selected=probe.tree.currentItem();ancestor=selected
        while ancestor is not None and ancestor is not probe.waveform_tree_item:ancestor=ancestor.parent()
        check(probe.active_task() == 'waveform' and ancestor is probe.waveform_tree_item
              and panel.view.currentData() == 'overview'
              and selected.data(0,Qt.ItemDataRole.UserRole)==('task_node',('waveform','result','overview')),
              'waveform_F11_routes_task_and_navigation')
        key(probe, Qt.Key.Key_F5)
        check(panel.thread is not None and panel._busy and probe.cancel_button.isEnabled()
              and not panel.parameter_widget.isEnabled(), 'waveform_F5_starts_real_worker_and_locks_inputs')
        wait_idle(panel); single = deepcopy(panel.report)
        m = single['analysis']['profiles']['A']['result']['metrics']
        check(abs(m['rangeResolutionM'] - 299792458 / (2 * 307.2e6)) < 1e-12
              and m['txIntervalUs'] == 114 and set(single['spectra']) == {'A'},
              'waveform_default_single_profile_matches_independent_physics')
        check(not probe.datasets and probe.feed_panel.report is None and probe.planar_panel.report is None
              and probe.sparse_panel.bundle is None and probe.subarray_panel.report is None and probe.mimo_panel.report is None,
              'waveform_acceptance_does_not_compute_other_modules')
        save_waveform_configuration(output / 'waveform-single-config.json', single['configuration'])

        a_before = {key: control.text() for key, control in panel.profile_inputs['A'].items()}
        ramp = float(panel.profile_inputs['B']['rampUs'].text())
        panel.derived_inputs['B']['bandwidthMHz'].setText(str(ramp * 17))
        panel.derived_inputs['B']['bandwidthMHz'].editingFinished.emit()
        panel.derived_inputs['B']['chirpUs'].setText(str(ramp + 9))
        panel.derived_inputs['B']['chirpUs'].editingFinished.emit()
        check(float(panel.profile_inputs['B']['slopeMHzUs'].text()) == 17
              and float(panel.profile_inputs['B']['idleUs'].text()) == 9,
              'waveform_bandwidth_and_period_editing_finished_update_underlying_fields')
        check(a_before == {key: control.text() for key, control in panel.profile_inputs['A'].items()}
              and canonical(panel.report) == canonical(single),
              'waveform_editing_B_does_not_change_A_or_previous_result')
        panel.inputs['targetVelocity'].setText('1')
        check('3.6 km/h' in panel.velocity_label.text(), 'waveform_target_velocity_has_exact_kmh_conversion')

        config = default_waveform_configuration()
        for p in (config['params'], config['profileB']):
            p.update(mode='ddm', samples=64, sampleRateMsps=4, loops=32, rangeFft=1024,
                     dopplerFft=64, targetRange=60, targetVelocity=0, frameMs=50)
        config['profileB'].update(slopeMHzUs=15, rampUs=42, idleUs=12)
        config['abOptions'] = {'enabled': True, 'layout': 'alternating', 'switchGapUs': 5}
        config['activeProfile'] = 'B'
        config['rangeOptions'].update(method='dual_rate', secondSampleRateMsps=5, searchMaxM=120,
                                      knownVelocityMps=0, toleranceBins=.75)
        config['priOptions'].update(method='dual_period', period1Us=38, period2Us=43,
                                    targetRangeM=8000, timingResolutionNs=2, searchMaxM=20000)
        imported = {'app': 'WaveLab', 'schemaVersion': 2, 'name': '稳定v1.2输入格式', **deepcopy(config),
                    'geometryKind': 'ula', 'geometry': {'tx': [[0, 0], [2, 0], [4, 0]], 'rx': [[0, 0], [.5, 0], [1, 0], [1.5, 0]]}}
        import_path = output / 'waveform-import-v2.json'
        import_path.write_text(json.dumps(imported, ensure_ascii=False, indent=2), encoding='utf-8')
        panel.open_path(import_path)
        check(panel.config() == load_waveform_configuration(import_path) and panel.report is not None
              and canonical(panel.report) == canonical(single),
                  'waveform_imports_WaveLab_v2_without_recalculating_saved_results')
        parameter_path = directory / '原生 波形配置.json'; panel.save_path(parameter_path)
        check(load_waveform_configuration(parameter_path) == panel.config(),
              'waveform_native_configuration_roundtrip_preserves_both_profiles_and_experiments')
        key(probe, Qt.Key.Key_F5); wait_idle(panel); ab_result = deepcopy(panel.report)
        check(set(ab_result['spectra']) == {'A', 'B'} and panel.result_profile.currentData() == 'B',
              'waveform_real_AB_DDM_worker_completes_and_selects_saved_profile')
        analysis = ab_result['analysis']
        check(analysis['metrics']['pairIntervalUs'] == 102
              and all(analysis['profiles'][name]['acquisition']['slowTimeIntervalUs'] == 102 for name in ('A', 'B'))
              and analysis['profiles']['A']['result']['metrics']['observedBandwidthMHz'] == 192
              and analysis['profiles']['B']['result']['metrics']['observedBandwidthMHz'] == 240,
              'waveform_AB_uses_independent_bandwidth_and_actual_alternating_interval')
        recovery = ab_result['range_recovery']['A']
        check(recovery['status'] == 'resolved' and abs(recovery['resolvedRangeM'] - 60) < .05,
              'waveform_ADC_dual_rate_resolves_the_observed_demo')
        pri = ab_result['pri_recovery']
        check(pri['status'] == 'resolved' and abs(pri['resolvedRangeM'] - 8000) < .2,
              'waveform_dual_period_experiment_retains_independent_target')
        report['waveform_results'] = {'single': single, 'ab': ab_result}
        save_waveform_configuration(output / 'waveform-ab-config.json', ab_result['configuration'])

        science = canonical(panel.report)
        with patch('arraylab.waveform_widgets.calculate_waveform', side_effect=AssertionError('View must not calculate')) as solver:
            for view in ('overview', 'metrics', 'phase', 'spectra', 'rd', 'range_recovery', 'pri_recovery', 'antenna', 'gain_budget', 'comparisons', 'theory'):
                panel.view.setCurrentIndex(panel.view.findData(view)); settle()
                plotted = view in ('overview', 'phase', 'spectra', 'rd', 'range_recovery', 'pri_recovery', 'antenna', 'gain_budget')
                check(panel.view.currentData() == view and (probe.current_picture() is not None) == plotted,
                      'waveform_native_view_and_picture_actions_' + view)
            panel.view.setCurrentIndex(panel.view.findData('spectra'))
            for profile in ('A', 'B'):
                panel.result_profile.setCurrentIndex(panel.result_profile.findData(profile)); settle()
            check(solver.call_count == 0 and canonical(panel.report) == science
                  and panel.result_profile.currentData() == 'B',
                  'waveform_view_and_profile_switches_do_not_recompute_or_change_snapshot')
        panel.view.setCurrentIndex(panel.view.findData('range_recovery')); panel.recovery_table_toggle.setChecked(True); settle()
        check(panel.table.isVisible() and panel.table.rowCount() == len(ab_result['range_recovery']['B']['candidates'])
              and probe.current_picture() is None, 'waveform_candidate_table_exposes_all_candidates_without_picture_actions')
        panel.recovery_table_toggle.setChecked(False)

        for index in range(8): panel.add_comparison('对比 ' + str(index + 1))
        check(len(panel.comparisons) == 8 and reject(lambda: panel.add_comparison('第九项')),
              'waveform_comparisons_cap_at_eight_without_overwriting')
        check(panel.table.rowCount() == 18
              and [panel.table.item(row, 1).text() for row in range(18)] == ['A', 'B'] * 9,
              'waveform_comparison_expands_both_profiles_for_current_and_saved_designs')
        panel.table.selectRow(3)
        QTest.mouseClick(panel.remove_comparison_button, Qt.MouseButton.LeftButton); settle()
        check(len(panel.comparisons) == 7 and panel.table.rowCount() == 16,
              'waveform_removing_saved_B_row_removes_its_complete_comparison')
        panel.add_comparison('恢复对比名额')
        panel.view.setCurrentIndex(panel.view.findData('phase')); settle()
        record_path = directory / '完整 波形及对比.json'; probe.save_history_path(record_path)
        saved_view = panel.view.currentData(); panel.profile_inputs['A']['slopeMHzUs'].setText('11')
        dirty_path = directory / '待计算 参数.json'; panel.save_path(dirty_path)
        export_path = directory / '已算 结果'; panel.export_path(export_path)
        exported = json.loads((export_path / 'summary.json').read_text(encoding='utf-8'))
        check(load_waveform_configuration(dirty_path)['params']['slopeMHzUs'] == 11
              and exported['configuration']['params']['slopeMHzUs'] == ab_result['configuration']['params']['slopeMHzUs'],
              'waveform_pending_inputs_and_exported_snapshot_stay_separate')

        reopened = create_window(); restored = reopened.waveform_panel
        with patch('arraylab.waveform_widgets.calculate_waveform', side_effect=AssertionError('Records must not recompute')) as solver:
            reopened.load_history_path(record_path); settle()
            check(reopened.active_task() == 'waveform' and canonical(restored.report) == canonical(ab_result)
                  and len(restored.comparisons) == 8 and restored.view.currentData() == saved_view
                  and restored.result_profile.currentData() == 'B' and solver.call_count == 0,
                  'waveform_restart_global_record_load_restores_profile_view_and_comparisons_without_solver')
        check(restored.config() == ab_result['configuration'], 'waveform_record_restores_matching_calculated_inputs')
        before, inputs = restored.report, restored.config()
        corrupt = directory / '损坏记录.json'; document = json.loads(record_path.read_text(encoding='utf-8'))
        document['sha256'] = '0' * 64; corrupt.write_text(json.dumps(document, ensure_ascii=False), encoding='utf-8')
        check(reject(lambda: reopened.load_history_path(corrupt)) and restored.report is before and restored.config() == inputs,
              'waveform_corrupt_record_preserves_existing_inputs_and_result')
        with patch.object(restored, '_make_figure', side_effect=RuntimeError('injected renderer failure')):
            check(reject(lambda: reopened.load_history_path(record_path)) and restored.report is before and restored.config() == inputs,
                  'waveform_failed_render_restore_preserves_existing_state')
        details['fault_injection'].append('renderer failure')
        restored.profile_inputs['A']['samples'].setText('bad')
        with patch.object(QMessageBox, 'warning', return_value=QMessageBox.StandardButton.Ok) as warning:
            restored.start()
            check(warning.call_count == 1 and restored.thread is None and restored.report is before,
                  'waveform_invalid_input_does_not_start_worker_or_replace_result')
        restored.apply_configuration(inputs)
        restored.start(); active = restored.thread; restored.cancel(); wait_idle(restored)
        check(active is not None and restored.report is before and '取消' in restored.state.text(),
              'waveform_real_worker_cancel_preserves_previous_result')

        reopened.load_history_path(record_path); settle()

        # WaveLab 1.4 additions: native controls, a bounded real worker, and a
        # complete saved result. This does not run any other ArrayLab task.
        radar_config = default_waveform_configuration()
        for profile in (radar_config['params'], radar_config['profileB']):
            profile.update(samples=64, rangeFft=128, loops=8, dopplerFft=16, targetRange=12)
        radar_config['antennaDesign']['sparseOptions'].update(criterion='dml', iterations=100,
            scanMaxDeg=10, targetResolutionDeg=8, trials=1, snapshots=32, maxApertureLambda=10)
        restored.apply_configuration(radar_config)
        controls = restored.radar_controls
        check(restored.parameter_widget.count() == 6 and controls.pages.count() == 4
              and not controls.sparse['smoothing'].isEnabled() and not controls.sparse['scanGridStepDeg'].isEnabled(),
              'radar_native_antenna_pages_and_DML_conditional_controls')
        controls.sparse['criterion'].setCurrentIndex(controls.sparse['criterion'].findData('music'))
        check(controls.sparse['smoothing'].isEnabled() and controls.sparse['scanGridStepDeg'].isEnabled()
              and not controls.sparse['diagonalLoading'].isEnabled(), 'radar_MUSIC_controls_are_method_specific')
        controls.sparse['criterion'].setCurrentIndex(controls.sparse['criterion'].findData('dml'))
        restored.start('evaluate'); wait_idle(restored)
        radar_result = deepcopy(restored.report)
        check(radar_result['source']['version'] == '1.4' and radar_result['sparse_design']['valid']
              and radar_result['sparse_design']['action'] == 'evaluate'
              and radar_result['sparse_design']['superresolution']['method'] == 'dml',
              'radar_real_worker_evaluates_current_geometry_with_DML')
        for view in ('antenna', 'sparse_design', 'gain_budget'):
            restored.view.setCurrentIndex(restored.view.findData(view)); settle(); restored.canvas.draw()
            check(len(restored.figure.axes) == 2 and reopened.current_picture() is restored.canvas,
                  'radar_saved_numeric_result_native_plot_' + view)
            restored.recovery_table_toggle.setChecked(True); settle()
            check(restored.table.isVisible() and restored.table.rowCount() > 0 and reopened.current_picture() is None,
                  'radar_complete_result_table_' + view)
            restored.recovery_table_toggle.setChecked(False)
        restored.view.setCurrentIndex(restored.view.findData('sparse_design'))
        radar_record = directory / '完整 雷达1.4记录.json'; restored.save_record_path(radar_record)
        with patch('arraylab.waveform_widgets.calculate_waveform', side_effect=AssertionError('Radar record must not recompute')) as solver:
            restored.load_record_path(radar_record); settle()
            check(canonical(restored.report) == canonical(radar_result) and solver.call_count == 0
                  and restored.config() == radar_result['configuration'], 'radar_record_restores_inputs_and_DML_diagnostics_without_solver')
        controls.apply_result()
        applied = restored.config()['geometry']
        check(applied is not None and abs(applied['referenceFrequencyGHz']-radar_result['sparse_design']['referenceFrequencyGHz']) < 1e-9
              and controls.custom.isChecked() and controls.fixed.isChecked()
              and restored.report['sparse_design']['action'] == 'evaluate',
              'radar_apply_coordinates_pins_reference_frequency_without_recalculation')
        with patch('arraylab.features.waveform.sparse_design.design', side_effect=AssertionError('F5 must not synthesize')) as designer:
            with patch('arraylab.features.waveform.sparse_design.evaluate', side_effect=AssertionError('F5 must not revalidate')) as evaluator:
                restored.start(); wait_idle(restored)
        retained = restored.report['sparse_design']
        check(retained.get('retainedFromPreviousCalculation') is True and designer.call_count == evaluator.call_count == 0
              and retained['superresolution'] == radar_result['sparse_design']['superresolution'],
              'radar_F5_after_apply_retains_original_diagnostics_without_revalidation')
        applied_record = directory / '应用后 F5完整记录.json'; restored.save_record_path(applied_record)
        applied_snapshot = canonical(restored.report)
        with patch('arraylab.waveform_widgets.calculate_waveform', side_effect=AssertionError('Applied record must not recompute')):
            restored.load_record_path(applied_record)
        check(canonical(restored.report) == applied_snapshot
              and restored.report['sparse_design']['retainedFromPreviousCalculation'],
              'radar_applied_F5_record_retains_complete_independent_diagnostics')
        # Restore the established A/B result for the existing responsive checks.
        reopened.load_history_path(record_path); settle()

        for width, height in ((1440, 900), (1100, 740), (1100, 640)):
            reopened.resize(width, height); restored.view.setCurrentIndex(restored.view.findData('spectra')); settle(160)
            reopened.results_scroll.ensureWidgetVisible(restored.canvas); settle(); restored.canvas.draw()
            renderer = restored.canvas.get_renderer(); footer = restored.figure._arraylab_metric_footer.get_window_extent(renderer)
            bounds = [axis.get_tightbbox(renderer) for axis in restored.figure.axes]
            scrolls = []
            for index in range(restored.parameter_widget.count()):
                restored.parameter_widget.setCurrentIndex(index); settle()
                scroll = restored.parameter_widget.widget(index)
                if isinstance(scroll, QScrollArea):
                    scroll.verticalScrollBar().setValue(scroll.verticalScrollBar().maximum()); settle()
                    scrolls.append(scroll.horizontalScrollBar().maximum() == 0)
            check(all(not box.overlaps(footer) for box in bounds) and len(scrolls) == 6 and all(scrolls),
                  f'waveform_plot_metrics_and_all_parameter_pages_fit_{width}x{height}')
            restored.parameter_widget.setCurrentIndex(0)
            restored.parameter_widget.widget(0).verticalScrollBar().setValue(0); settle()
            filename = directory / f'waveform-workbench-{width}x{height}.png'
            if not reopened.grab().save(str(filename)): raise AssertionError('Cannot save waveform screenshot')
            details['screenshots'].append(str(filename))
        close_probe = create_window(); close_probe.activate_task('waveform'); closing = close_probe.waveform_panel
        closing.apply_configuration(inputs); closing.start(); active = closing.thread; close_probe.close()
        requested = getattr(close_probe, '_waveform_close_pending', False)
        wait_idle(closing); settle()
        check(active is not None and requested and closing.thread is None and not close_probe.isVisible(),
              'waveform_close_cancels_and_reaps_real_worker_before_closing_window')
        check(canonical(single) == canonical(report['waveform_results']['single'])
              and canonical(ab_result) == canonical(report['waveform_results']['ab']),
              'waveform_reference_reports_remain_immutable_for_frozen_CLI_crosscheck')
    finally:
        for item in reversed(windows):
            panel = item.waveform_panel
            if panel.thread is not None:
                panel.cancel()
                try: wait_idle(panel, allow_error=True)
                except AssertionError: pass
            item.close()
            for timer in item.findChildren(QTimer): timer.stop()
            if panel.thread is None:
                item.deleteLater(); QCoreApplication.sendPostedEvents(item, QEvent.Type.DeferredDelete)
        set_workers(previous_workers); app.setQuitOnLastWindowClosed(old_quit); app.processEvents()
        (directory / 'details.json').write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
