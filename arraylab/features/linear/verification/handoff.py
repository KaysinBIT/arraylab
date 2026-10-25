"""Exercise the selected-array toolbar handoff through native Qt events."""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QCoreApplication, QEvent, QTimer, Qt
from PySide6.QtTest import QTest

from arraylab.features.feed.model import FeedSettings
from arraylab.application.desktop.window import MainWindow
from arraylab.features.linear.widgets import PatternSet


def exercise_linear_feed_handoff(app, window, output, report, check):
    old_quit = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    probe = MainWindow(auto_compute=False)
    details = {'qt_platform': app.platformName(), 'cases': []}
    original = (id(window.datasets), id(window.feed_panel.report), window.active_task())

    def settle():
        QTest.qWait(45)
        app.processEvents()

    def f5():
        probe.raise_(); probe.activateWindow(); settle()
        QTest.keyClick(probe, Qt.Key.Key_F5); settle()

    try:
        probe.show(); probe.activate_task('linear'); settle()
        check(probe.to_feed_button.isVisible() and not probe.to_feed_button.isEnabled(),
              'linear_handoff_disabled_without_calculated_array')
        check(probe.history_tree_item.text(0).startswith('历史计算记录'),
              'linear_history_navigation_uses_history_name')
        probe.inputs['elements'].setText('8')
        probe.inputs['spacing'].setText('0.5')
        probe.inputs['step_deg'].setText('1')
        probe.phase_mode.setCurrentIndex(1)
        probe.inputs['scan_deg'].setText('20')
        f5()
        source = probe.selected_dataset()
        parameters = source.result.parameters
        source_values = asdict(parameters)
        check(probe.to_feed_button.isEnabled() and probe.to_feed_action.isEnabled(),
              'linear_handoff_toolbar_and_menu_enabled_for_snapshot')

        feed = probe.feed_panel
        probe.activate_task('feed')
        feed.topology.setCurrentIndex(feed.topology.findData('series'))
        feed.electrical.apply(FeedSettings(topology='series', phase_mode='equalize',
            weighting='custom', custom_amplitudes=(1, .8, .6, .5, .5, .6, .8, 1),
            custom_phases_deg=(10,) * 8, phase_bits=2, amplitude_error_db=.1,
            phase_error_deg=3, failed_ports=(1,)))
        feed.generate(); settle()
        feed.compare_history.setChecked(True)
        feed.pattern_source.setCurrentIndex(feed.pattern_source.findData('uncompensated'))
        previous_report = feed.report
        previous_history = list(feed.history)
        probe.activate_task('linear')
        probe.inputs['elements'].setText('12')
        probe.inputs['scan_deg'].setText('45')
        probe.inputs['frequency_ghz'].setText('9')
        QTest.mouseClick(probe.to_feed_button, Qt.MouseButton.LeftButton); settle()
        check(probe.active_task() == 'feed' and asdict(probe.read_parameters()) == source_values,
              'linear_handoff_click_imports_selected_20deg_not_dirty_45deg_inputs')
        settings = feed.read_settings()
        check(settings.phase_mode == 'scan' and settings.weighting == 'uniform'
              and settings.phase_bits == 0 and not settings.custom_phases_deg
              and not settings.failed_ports and settings.amplitude_error_db == 0
              and settings.phase_error_deg == 0 and not settings.branch_delays_deg,
              'linear_handoff_removes_old_feed_phase_and_error_settings')
        check(feed.report is previous_report and feed.history == previous_history
              and 'F5' in feed.state.text() and feed.linear_import_source.source_label == source.label,
              'linear_handoff_leaves_previous_results_until_explicit_F5')

        for topology in ('corporate', 'series', 'symmetric_series'):
            feed.topology.setCurrentIndex(feed.topology.findData(topology)); f5()
            actual = feed.report
            power = np.power(10., np.array(actual['electrical']['patterns']['output']['pattern_db']) / 10.)
            reference = np.power(10., source.db / 10.)
            error = float(np.max(np.abs(power - reference)))
            pointing = actual['electrical']['pointing']
            check(actual['array_parameters'] == source_values and error < 2e-12
                  and abs(pointing['target_theta_deg'] - 20) < 1e-12
                  and all(abs(angle - 20) < 1e-7 for angle in pointing['output_cut_peak_angles_deg']),
                  'linear_handoff_F5_matches_original_field_and_pointing_' + topology)
            details['cases'].append({'topology': topology, 'target_deg': 20,
                'peaks_deg': pointing['output_cut_peak_angles_deg'], 'max_power_error': error})
        check(feed.pattern_source.currentData() == 'output' and not feed.compare_history.isChecked(),
              'linear_handoff_defaults_to_final_electrical_output')
        check(asdict(source.result.parameters) == source_values and probe.datasets[0] is source,
              'linear_handoff_preserves_original_calculation_snapshot')

        probe.activate_task('linear')
        probe.add_dataset(PatternSet('外部曲线', np.array([-90., 0., 90.]), np.array([-20., 0., -20.])))
        check(not probe.to_feed_button.isEnabled() and not probe.to_feed_action.isEnabled(),
              'linear_handoff_external_curve_without_geometry_is_disabled')
        probe.selection.setCurrentIndex(0)
        probe.to_feed_action.trigger(); settle()
        check(probe.active_task() == 'feed' and asdict(probe.read_parameters()) == source_values,
              'linear_handoff_menu_uses_same_snapshot_path')
        path = Path(output) / 'linear-feed-handoff.png'
        check(probe.grab().save(str(path)), 'linear_handoff_screenshot_saved')
    finally:
        for timer in probe.findChildren(QTimer): timer.stop()
        probe.close(); probe.deleteLater()
        QCoreApplication.sendPostedEvents(probe, QEvent.Type.DeferredDelete)
        app.setQuitOnLastWindowClosed(old_quit)
        app.processEvents()
    check((id(window.datasets), id(window.feed_panel.report), window.active_task()) == original,
          'linear_handoff_probe_preserves_main_window')
    report['linear_feed_handoff'] = details
    (Path(output) / 'linear-feed-handoff.json').write_text(
        json.dumps(details, ensure_ascii=False, indent=2), encoding='utf-8')
