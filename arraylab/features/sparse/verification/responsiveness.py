"""Real Qt responsiveness and process-lifecycle acceptance for 20-by-20 synthesis.

The renderer failure is deliberately injected and is recorded separately from
the actual successful, cancelled, and window-close subprocess jobs.
"""
import copy
from dataclasses import asdict
import time

from PySide6.QtCore import QEventLoop, QThread, QTimer, Qt, Signal
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from arraylab.features.sparse.engine.core import Config


class _CompletedWorker(QThread):
    """Only exercises GUI delivery failure; it is not numerical acceptance."""
    progress = Signal(str)

    def __init__(self, result, parent):
        super().__init__(parent)
        self.result = result
        self.error = None
        self.cancelled = False

    def run(self):
        pass


def _wait_until(predicate, timeout_ms):
    """Use a real Qt event loop so Python/native worker threads can progress."""
    if predicate(): return True
    loop = QEventLoop()
    poll = QTimer(); poll.setInterval(20)
    deadline = QTimer(); deadline.setSingleShot(True)
    poll.timeout.connect(lambda: loop.quit() if predicate() else None)
    deadline.timeout.connect(loop.quit)
    poll.start(); deadline.start(timeout_ms)
    try: loop.exec()
    finally: poll.stop(); deadline.stop()
    return predicate()


def _observe_job(panel, launch, *, on_stage=None, closed_window=None, timeout_ms=180000):
    started = time.monotonic()
    times = [started]
    events = []
    errors = []
    last_state = []
    heartbeat = QTimer(); heartbeat.setInterval(20)

    def tick():
        times.append(time.monotonic())
        if panel._busy:
            text = panel.state.text()
            if not last_state or last_state[-1] != text:
                last_state.append(text)
        modal = QApplication.activeModalWidget()
        if isinstance(modal, QMessageBox):
            errors.append(modal.text()); modal.accept()

    def progress(text):
        events.append({'seconds': time.monotonic()-started, 'stage': text})
        if on_stage is not None:
            try: on_stage(text)
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
                panel.cancel()

    def done():
        finished = panel.thread is None and not panel._busy
        return finished and (closed_window is None or not closed_window.isVisible())

    heartbeat.timeout.connect(tick)
    heartbeat.start()
    try:
        launch()
        worker = panel.thread
        if worker is not None:
            worker.progress.connect(progress, Qt.ConnectionType.QueuedConnection)
        finished = _wait_until(done, timeout_ms)
        ended = time.monotonic()
    finally:
        heartbeat.stop()
        times.append(time.monotonic())
    gaps = [right-left for left, right in zip(times, times[1:])]
    record = {'elapsed_seconds': ended-started, 'finished': finished,
              'heartbeat_interval_ms': 20, 'heartbeat_count': max(0, len(times)-2),
              'heartbeat_max_gap_seconds': max(gaps, default=0.),
              'heartbeat_gaps_seconds': gaps, 'stages': events,
              'state_text_changes': len(last_state), 'modal_or_callback_errors': errors,
              'last_error': panel._last_error, 'state': panel.state.text()}
    if not finished and panel.thread is not None:
        panel.cancel()
        record['timeout_cleanup_completed'] = _wait_until(done, 10000)
    return record


def exercise_sparse_responsiveness(app, window, output, report, check):
    """Run once from the main source/frozen Windows Qt smoke entrypoint."""
    from arraylab.application.desktop.window import MainWindow

    panel = window.sparse_panel
    initial = {'task': window.active_task(), 'config': panel.config_from_inputs(),
               'bundle': panel.bundle, 'row': panel.table.currentRow(),
               'case': panel.case_selector.currentIndex(), 'phase': panel.phase_view.currentIndex(),
               'view': panel.view.currentIndex(), 'tab': panel.tabs.currentIndex(),
               'workers': window.worker_choice.currentIndex(), 'error': panel._last_error,
               'state': panel.state.text()}
    observations = {'qt_platform': app.platformName(), 'success_config': asdict(Config(nx=20, ny=20)),
                    'cancel_limit_seconds': 3., 'heartbeat_limit_seconds': 1.5}
    report['sparse_responsiveness'] = observations
    auxiliary = None
    original_make_figure = None
    try:
        check(app.platformName() == 'windows', 'sparse_responsiveness_uses_real_Windows_Qt')
        check(panel.thread is None and not panel._busy, 'sparse_responsiveness_starts_idle')
        window.activate_task('sparse')
        window.worker_choice.setCurrentIndex(window.worker_choice.findData(0))
        panel.tabs.setCurrentIndex(0)
        panel.view.setCurrentIndex(panel.view.findData('layout'))
        panel.set_config(Config(nx=20, ny=20))
        app.processEvents()
        success = _observe_job(panel, lambda: QTest.mouseClick(window.compute_button, Qt.MouseButton.LeftButton))
        observations['default_20x20_success'] = success
        check(success['finished'] and not success['last_error'] and not success['modal_or_callback_errors'],
              'sparse_20x20_default_auto_completes')
        check(panel.bundle is not None and panel.bundle['config']['nx'] == 20 and panel.bundle['config']['ny'] == 20,
              'sparse_20x20_default_success_keeps_requested_geometry')
        check(panel.bundle['config']['iterations'] == 1400 and panel.bundle['config']['opt_grid'] == 51
              and panel.bundle['config']['eval_grid'] == 181 and panel.bundle['config']['k'] == 36,
              'sparse_20x20_acceptance_uses_full_default_workload')
        check(success['heartbeat_count'] >= 5 and success['heartbeat_max_gap_seconds'] < 1.5,
              'sparse_20x20_default_keeps_Qt_heartbeat_responsive')
        check(success['state_text_changes'] >= 2, 'sparse_20x20_stage_and_elapsed_display_updates')
        check(panel.run_button.isEnabled() and not panel._busy and panel.thread is None,
              'sparse_20x20_success_restores_controls')
        observations['default_20x20_success']['lp'] = panel.bundle.get('lp', {})
        check(window.grab().save(str(output/'sparse-20x20-responsive.png')),
              'sparse_20x20_responsiveness_screenshot_saved')

        previous = panel.bundle
        cancellation = {'stage_seen': False, 'requested_at': None}
        cancel_timer = QTimer(); cancel_timer.setSingleShot(True)
        def cancel_in_solver():
            cancellation['requested_at'] = time.monotonic()
            cancellation['worker_present_at_request'] = panel.thread is not None
            QTest.mouseClick(panel.cancel_button, Qt.MouseButton.LeftButton)
        cancel_timer.timeout.connect(cancel_in_solver)
        def cancel_stage(text):
            if '首次最小最大 LP：highs，' in text and not cancellation['stage_seen']:
                cancellation['stage_seen'] = True
                cancel_timer.start(100)
        panel.set_config(Config(nx=20, ny=20, lp_solver='highs'))
        app.processEvents()
        cancelled = _observe_job(panel, lambda: QTest.mouseClick(window.compute_button, Qt.MouseButton.LeftButton),
                                 on_stage=cancel_stage, timeout_ms=60000)
        cancel_timer.stop()
        latency = time.monotonic()-cancellation['requested_at'] if cancellation['requested_at'] is not None else None
        cancelled.update(stage_seen=cancellation['stage_seen'], cancel_latency_seconds=latency,
                         worker_present_at_request=cancellation.get('worker_present_at_request', False))
        observations['explicit_highs_cancel'] = cancelled
        check(cancelled['stage_seen'] and cancelled['worker_present_at_request'],
              'sparse_20x20_cancel_is_requested_during_first_highs_minimax')
        check(cancelled['finished'] and latency is not None and latency < 3.,
              'sparse_20x20_highs_cancellation_finishes_within_three_seconds')
        check(panel.bundle is previous and '取消' in panel.state.text() and not panel._last_error,
              'sparse_20x20_highs_cancel_retains_previous_bundle_identity')
        check(cancelled['heartbeat_count'] >= 2 and cancelled['heartbeat_max_gap_seconds'] < 1.5,
              'sparse_20x20_highs_cancel_keeps_Qt_heartbeat_responsive')

        # A second real MainWindow checks close semantics independently of the
        # smoke runner's window, which must stay alive to record the outcome.
        auxiliary = MainWindow(auto_compute=False)
        auxiliary.resize(1100, 740); auxiliary.show(); auxiliary.activate_task('sparse')
        close_panel = auxiliary.sparse_panel
        close_panel.set_config(Config(nx=20, ny=20, lp_solver='highs', iterations=1))
        app.processEvents()
        closing = {'requested_at': None}
        def close_stage(text):
            if closing['requested_at'] is None:
                closing['requested_at'] = time.monotonic()
                closing['worker_present_at_request'] = close_panel.thread is not None
                auxiliary.close()
                closing['close_pending_observed'] = bool(getattr(auxiliary, '_sparse_close_pending', False))
        closed = _observe_job(close_panel,
                              lambda: QTest.mouseClick(auxiliary.compute_button, Qt.MouseButton.LeftButton),
                              on_stage=close_stage, closed_window=auxiliary, timeout_ms=30000)
        closed.update(close_requested=closing['requested_at'] is not None,
                      close_pending_observed=closing.get('close_pending_observed', False),
                      close_latency_seconds=time.monotonic()-closing['requested_at'] if closing['requested_at'] else None)
        observations['close_running_window'] = closed
        check(closed['close_requested'] and closed['close_pending_observed'] and closed['finished'],
              'sparse_window_close_requests_cancel_and_finishes_closing')
        check(not auxiliary.isVisible() and close_panel.thread is None and not close_panel._busy
              and not getattr(auxiliary, '_sparse_close_pending', False),
              'sparse_window_close_leaves_no_busy_worker')
        auxiliary.deleteLater(); auxiliary = None

        # Separate fault injection, explicitly not reported as a real solver run.
        panel.table.selectRow(0); app.processEvents()
        selection = (panel.table.currentRow(), panel.case_selector.currentIndex(), panel.phase_view.currentIndex())
        original_make_figure = panel.make_figure
        failed = {'calls': 0}
        def fail_first_figure(*args, **kwargs):
            failed['calls'] += 1
            if failed['calls'] == 1: raise RuntimeError('injected sparse figure failure')
            return original_make_figure(*args, **kwargs)
        panel.make_figure = fail_first_figure
        delivered = copy.deepcopy(previous)
        failure = _observe_job(panel,
            lambda: panel.begin_worker(_CompletedWorker(delivered, panel), 'synthesis'), timeout_ms=10000)
        failure['evidence'] = 'injected one-shot renderer failure after a fake successful delivery; no numerical solve'
        observations['renderer_failure_recovery'] = failure
        check(failure['finished'] and 'injected sparse figure failure' in (panel._last_error or ''),
              'sparse_renderer_failure_is_reported_and_worker_is_released')
        check(panel.bundle is previous and selection ==
              (panel.table.currentRow(), panel.case_selector.currentIndex(), panel.phase_view.currentIndex()),
              'sparse_renderer_failure_preserves_previous_bundle_and_selection')
        check(not panel._busy and panel.run_button.isEnabled() and panel.thread is None,
              'sparse_renderer_failure_restores_editable_controls')
        panel.make_figure = original_make_figure; original_make_figure = None
    finally:
        if original_make_figure is not None: panel.make_figure = original_make_figure
        if panel.thread is not None:
            panel.cancel(); _wait_until(lambda: panel.thread is None and not panel._busy, 10000)
        if auxiliary is not None:
            other = auxiliary.sparse_panel
            if other.thread is not None:
                other.cancel(); _wait_until(lambda: other.thread is None and not other._busy, 10000)
            auxiliary.close(); auxiliary.deleteLater()
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
