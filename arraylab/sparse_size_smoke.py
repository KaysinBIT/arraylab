"""Real Qt acceptance for exact custom sparse dimensions and element counts."""
from dataclasses import asdict, replace
import json
import time

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from .sparse2d.core import Config
from .sparse2d.headless import source_index
from .sparse_responsiveness_smoke import _observe_job, _wait_until


def _warning_on_start(panel):
    """Dismiss a real validation dialog and record its visible message."""
    messages = []
    ticks = []
    timer = QTimer(); timer.setInterval(20)

    def tick():
        ticks.append(time.monotonic())
        modal = QApplication.activeModalWidget()
        if isinstance(modal, QMessageBox):
            messages.append(modal.text()); modal.accept()

    timer.timeout.connect(tick); timer.start()
    started = time.monotonic()
    try:
        QTest.mouseClick(panel.host.compute_button, Qt.MouseButton.LeftButton)
    finally:
        timer.stop()
    return {'messages': messages, 'elapsed_seconds': time.monotonic()-started,
            'heartbeat_count': len(ticks), 'last_error': panel._last_error,
            'worker_started': panel.thread is not None, 'busy': panel._busy}


def exercise_sparse_sizes(app, window, output, report, check):
    """Exercise actual source/frozen process computation; restore caller state."""
    panel = window.sparse_panel
    initial = {'task': window.active_task(), 'config': panel.config_from_inputs(),
               'bundle': panel.bundle, 'row': panel.table.currentRow(),
               'case': panel.case_selector.currentIndex(), 'phase': panel.phase_view.currentIndex(),
               'view': panel.view.currentIndex(), 'tab': panel.tabs.currentIndex(),
               'workers': window.worker_choice.currentIndex(), 'error': panel._last_error,
               'state': panel.state.text()}
    observations = {'qt_platform': app.platformName(), 'warning_checks': []}
    report['sparse_custom_sizes'] = observations
    try:
        check(panel.thread is None and not panel._busy, 'sparse_custom_sizes_starts_idle')
        window.activate_task('sparse')
        window.worker_choice.setCurrentIndex(window.worker_choice.findData(0))
        panel.tabs.setCurrentIndex(0)
        panel.view.setCurrentIndex(panel.view.findData('layout'))
        original_bundle = panel.bundle

        # These checks exercise text entry, not merely assigning a Config object.
        base = Config(iterations=40, opt_grid=25, eval_grid=81)
        for nx, ny, k in ((21, 23, 101), (24, 25, 100)):
            panel.set_config(base)
            for key, value in (('nx', nx), ('ny', ny), ('k', k)):
                field = panel.inputs[key]
                field.selectAll(); QTest.keyClicks(field, str(value))
            config = panel.config_from_inputs()
            check((config.nx, config.ny, config.k) == (nx, ny, k),
                  f'sparse_custom_text_entry_{nx}x{ny}_K{k}_exact')
            path = output/f'sparse-custom-{nx}x{ny}-K{k}.json'
            panel.save_settings_path(path)
            saved = json.loads(path.read_text(encoding='utf-8'))['config']
            panel.set_config(Config()); panel.load_settings_path(path)
            check(panel.config_from_inputs() == config and
                  (saved['nx'], saved['ny'], saved['k']) == (nx, ny, k),
                  f'sparse_custom_settings_roundtrip_{nx}x{ny}_K{k}_exact')
            check(panel.bundle is original_bundle and panel.thread is None,
                  f'sparse_custom_settings_{nx}x{ny}_preserve_previous_result')

        # The numerical operation uses the real supervised CLI process. Its
        # explicitly reduced iteration/sampling settings keep this UI regression
        # small; full default 20x20 performance has a separate acceptance case.
        odd_config = replace(base, nx=21, ny=23, k=101)
        panel.set_config(odd_config); app.processEvents()
        success = _observe_job(panel, lambda: QTest.mouseClick(window.compute_button, Qt.MouseButton.LeftButton))
        success['config'] = asdict(odd_config)
        observations['odd_21x23_K101_process'] = success
        check(success['finished'] and not success['last_error'] and not success['modal_or_callback_errors'],
              'sparse_custom_odd_21x23_K101_real_process_completes')
        check(panel.bundle is not None and panel.bundle['config']['nx'] == 21
              and panel.bundle['config']['ny'] == 23 and panel.bundle['config']['k'] == 101,
              'sparse_custom_odd_result_preserves_requested_dimensions')
        for source in ('random', 'greedy', 'annealing', 'l1-topk'):
            row = panel.bundle['results'][source_index(panel.bundle, source)]
            points = np.asarray(row['xy'])
            point_set = {tuple(point) for point in points}
            check(len(points) == 101 and row['metrics']['elements'] == 101
                  and np.count_nonzero(np.all(points == 0, axis=1)) == 1
                  and all(tuple(-point) in point_set for point in points),
                  'sparse_custom_odd_' + source + '_has_exact_K_and_central_symmetry')
        check(success['heartbeat_count'] >= 2 and success['heartbeat_max_gap_seconds'] < 1.5,
              'sparse_custom_odd_process_keeps_Qt_responsive')
        check(window.grab().save(str(output/'sparse-custom-21x23-K101.png')),
              'sparse_custom_odd_screenshot_saved')
        previous = panel.bundle

        panel.set_config(replace(base, nx=1, ny=9, k=1))
        singleton = _observe_job(panel, lambda: QTest.mouseClick(window.compute_button, Qt.MouseButton.LeftButton))
        observations['singleton_1x9_K1_process'] = singleton
        check(singleton['finished'] and not singleton['last_error'] and '未定义（单阵元）' in panel.summary.text(),
              'sparse_custom_singleton_displays_without_distance_format_failure')
        check(panel.bundle is not None and panel.selected_result()['metrics']['elements'] == 1,
              'sparse_custom_singleton_keeps_exact_one_element')
        previous = panel.bundle

        # Qt spin boxes used to truncate large values. Keep all three integers
        # beyond signed 32-bit range and save/load them without starting a job.
        huge = 10**12 + 1
        huge_config = replace(base, nx=huge, ny=huge, k=huge)
        changed = time.monotonic()
        panel.set_config(huge_config); app.processEvents()
        observations['huge_input_update_seconds'] = time.monotonic()-changed
        check(observations['huge_input_update_seconds'] < 1.5
              and panel.config_from_inputs() == huge_config,
              'sparse_custom_trillion_size_edit_is_exact_and_responsive')
        huge_path = output/'sparse-custom-large-integers.json'
        panel.save_settings_path(huge_path)
        panel.set_config(Config()); panel.load_settings_path(huge_path)
        check(panel.config_from_inputs() == huge_config
              and all(panel.inputs[key].text() == str(huge) for key in ('nx', 'ny', 'k')),
              'sparse_custom_large_integers_settings_roundtrip_without_clamping')
        warning = _warning_on_start(panel)
        observations['huge_start_resource_warning'] = warning
        check(bool(warning['messages']) and '预算' in ' '.join(warning['messages'])
              and not warning['worker_started'] and not warning['busy'],
              'sparse_custom_oversized_job_reports_resource_budget_before_worker')
        check(warning['elapsed_seconds'] < 2. and warning['heartbeat_count'] >= 1
              and panel.bundle is previous and panel.run_button.isEnabled(),
              'sparse_custom_oversized_rejection_is_responsive_and_preserves_result')

        cases = [
            ('odd_K_even_dimension', replace(base, nx=24, ny=25, k=101), None),
            ('odd_K_center_forbidden', replace(odd_config, hole_shape='circle', hole_x=.2), None),
            ('empty_dimension', odd_config, ('nx', '')),
            ('negative_dimension', odd_config, ('ny', '-3')),
            ('zero_K', odd_config, ('k', '0')),
            ('negative_K', odd_config, ('k', '-1')),
            ('fractional_K', odd_config, ('k', '1.5')),
            ('empty_K', odd_config, ('k', '')),
        ]
        for name, config, text in cases:
            panel.set_config(config)
            if text is not None: panel.inputs[text[0]].setText(text[1])
            warning = _warning_on_start(panel); warning['case'] = name
            observations['warning_checks'].append(warning)
            check(bool(warning['messages']) and bool(warning['last_error'])
                  and not warning['worker_started'] and not warning['busy']
                  and panel.bundle is previous and panel.run_button.isEnabled(),
                  'sparse_custom_invalid_' + name + '_warns_without_losing_result')
    finally:
        if panel.thread is not None:
            panel.cancel(); _wait_until(lambda: panel.thread is None and not panel._busy, 10000)
        if panel.thread is None and not panel._busy:
            panel.set_config(initial['config'])
            panel.view.blockSignals(True); panel.view.setCurrentIndex(initial['view']); panel.view.blockSignals(False)
            if initial['bundle'] is not None:
                panel.receive(initial['bundle'], draw=False, selected_index=initial['row'], case_index=initial['case'])
                panel.phase_view.blockSignals(True); panel.phase_view.setCurrentIndex(initial['phase']); panel.phase_view.blockSignals(False)
                panel.redraw()
            else:
                panel.bundle = None; panel.table.setRowCount(0); panel.scheme_selector.clear(); panel.case_selector.clear()
                panel.port_table.setRowCount(0); panel.case_table.setRowCount(0)
                panel.summary.setText('尚无稀疏方案。'); panel.redraw()
            panel.tabs.setCurrentIndex(initial['tab']); panel._last_error = initial['error']
            panel.set_state(initial['state']); panel.set_busy(False)
        window.worker_choice.setCurrentIndex(initial['workers'])
        window.activate_task(initial['task'])
