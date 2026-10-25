"""Isolated native Qt acceptance for lazy sparse-history restoration."""
import copy
import json
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch
from uuid import uuid4

from PySide6.QtCore import QCoreApplication, QEvent, QEventLoop, QThread, QTimer, Signal
from PySide6.QtWidgets import QApplication, QMessageBox

from arraylab.framework.parallel import requested_workers, set_workers
from arraylab.features.sparse.engine.core import Config
from arraylab.features.sparse.engine.headless import jsonable
from arraylab.features.sparse.history_store import SparseHistoryError, SparseHistoryStore
from arraylab.application.desktop.window import MainWindow


class _Delivery(QThread):
    """Explicit fault-injection worker; never described as a solver execution."""
    progress = Signal(str)

    def __init__(self, result=None, error=None, cancelled=False, parent=None):
        super().__init__(parent)
        self.result, self.error, self.cancelled = result, error, cancelled

    def run(self):
        self.progress.emit('隔离交付检查')


def exercise_sparse_history(app, output, report, check):
    output = Path(output) / ('sparse-history-autosave-' + uuid4().hex[:8])
    output.mkdir(parents=True)
    path = output / '稀疏 历史.sqlite3'
    previous_quit = app.quitOnLastWindowClosed(); app.setQuitOnLastWindowClosed(False)
    previous_workers = requested_workers(); set_workers(1)
    windows = []
    details = {'qt_platform': app.platformName(), 'directory': str(output),
               'uses_real_user_archive': False, 'fault_injection': []}
    report['sparse_autosave'] = details

    def settle(milliseconds=50):
        loop = QEventLoop(); QTimer.singleShot(milliseconds, loop.quit); loop.exec()

    def open_window(db=path):
        window = MainWindow(auto_compute=False, sparse_history_path=db)
        windows.append(window); window.show(); settle()
        return window

    def finish(panel, timeout=40):
        start = time.monotonic(); loop = QEventLoop(); timer = QTimer()
        timer.setInterval(20)
        timer.timeout.connect(lambda: loop.quit() if (panel.thread is None and not panel._busy)
                              or time.monotonic() - start > timeout else None)
        timer.start(); loop.exec(); timer.stop()
        check(panel.thread is None and not panel._busy, 'sparse_history_worker_finishes_and_releases_busy')
        if panel._last_error: raise AssertionError(panel._last_error)

    def canonical(bundle):
        return json.dumps(jsonable(bundle), ensure_ascii=False, sort_keys=True, allow_nan=False)

    def table_values(table):
        return [[table.item(row, column).text() if table.item(row, column) else None
                 for column in range(table.columnCount())] for row in range(table.rowCount())]

    def no_solve(*args, **kwargs):
        raise AssertionError('Restoring sparse history must not start synthesis or weighting')

    def discard_on_close(window, answer):
        timer = QTimer(window); timer.setInterval(10)
        def respond():
            dialog = QApplication.activeModalWidget()
            if isinstance(dialog, QMessageBox) and dialog.windowTitle() == '稀疏历史尚未自动保存':
                timer.stop(); dialog.button(answer).click()
        timer.timeout.connect(respond); timer.start()
        try: window.close(); settle()
        finally: timer.stop(); timer.deleteLater()

    try:
        first = open_window(); panel = first.sparse_panel
        check(panel.bundle is None and panel.history_entries == [], 'sparse_history_initial_index_does_not_calculate')
        first.activate_task('sparse')
        config = Config(nx=4, ny=4, k=8, iterations=2, opt_grid=25, eval_grid=81,
                        phase_bits=3, scan_angles='0,0;20,0', frequencies_ghz='9,10')
        panel.set_config(config); panel.start(); finish(panel)
        check(len(panel.history_entries) == 1 and panel.current_history_id is not None,
              'sparse_history_successful_real_synthesis_appends_one_snapshot')
        first_id = panel.current_history_id; original = canonical(panel.bundle)
        panel.table.selectRow(0); panel.case_selector.setCurrentIndex(1)
        panel.view.setCurrentIndex(panel.view.findData('cuts')); panel.tabs.setCurrentIndex(1)
        panel.pattern_view.setCurrentIndex(panel.pattern_view.findData('array_factor'))
        panel.phase_view.setCurrentIndex(panel.phase_view.findData('ideal')); settle(240)
        saved_state = panel.history_view_state()
        entry = SparseHistoryStore(path).load_record(first_id)
        check(canonical(entry['bundle']) == original and entry['view_state'] == saved_state,
              'sparse_history_view_changes_preserve_complete_snapshot_and_save_selection')
        first.close(); settle()

        store_load = SparseHistoryStore.load_record; loads = []
        def count_load(store, record_id):
            loads.append(record_id); return store_load(store, record_id)
        with patch.object(SparseHistoryStore, 'load_record', count_load), \
                patch('arraylab.sparse_widgets.run_sparse_process', no_solve):
            second = open_window(); restored = second.sparse_panel
            check(restored.bundle is None and len(restored.history_entries) == 1 and not loads,
                  'sparse_history_restart_reads_metadata_without_loading_payload')
            second.activate_task('sparse'); settle()
            check(loads == [first_id] and restored.current_history_id == first_id
                  and canonical(restored.bundle) == original and restored.history_view_state() == saved_state,
                  'sparse_history_task_entry_restores_saved_bundle_and_view_without_solver')
            check(restored.config_from_inputs() == config and restored.thread is None,
                  'sparse_history_restores_parameters_and_embedded_result_state')
        restored.view.setCurrentIndex(restored.view.findData('pattern'))
        before_weight_state = restored.history_view_state()
        restored.weight_method.setCurrentIndex(restored.weight_method.findData('uniform'))
        restored.apply_weighting(); finish(restored)
        second_id = restored.current_history_id
        check(second_id != first_id and len(restored.history_entries) == 2
              and len(restored.bundle['results']) == len(entry['bundle']['results']) + 1,
              'sparse_history_real_weighting_creates_distinct_history_with_added_scheme')
        check(canonical(SparseHistoryStore(path).load_record(first_id)['bundle']) == original,
              'sparse_history_weighting_does_not_mutate_earlier_saved_snapshot')
        check(SparseHistoryStore(path).load_record(first_id)['view_state'] == before_weight_state,
              'sparse_history_start_job_flushes_latest_view_before_debounce_deadline')

        restored.view.setCurrentIndex(restored.view.findData('coarray'))
        quick_state = restored.history_view_state()
        check(restored.restore_history(first_id)
              and SparseHistoryStore(path).load_record(second_id)['view_state'] == quick_state,
              'sparse_history_switch_flushes_previous_view_before_debounce_deadline')
        check(restored.restore_history(second_id), 'sparse_history_returns_to_weighted_snapshot_after_fast_switch')

        # Failed/cancelled completion is explicit fault injection, not another solve.
        for outcome in ('error', 'cancel'):
            before = restored.bundle; rows_before = len(restored.history_entries)
            worker = _Delivery(error='injected solver failure' if outcome == 'error' else None,
                               cancelled=outcome == 'cancel', parent=restored)
            restored.begin_worker(worker, 'synthesis')
            loop = QEventLoop(); timer = QTimer(); timer.setInterval(15)
            timer.timeout.connect(lambda: loop.quit() if restored.thread is None else None)
            timer.start(); loop.exec(); timer.stop()
            check(restored.bundle is before and len(restored.history_entries) == rows_before,
                  'sparse_history_' + outcome + '_does_not_add_or_replace_snapshot')
            details['fault_injection'].append(outcome)

        check(restored.restore_history(second_id), 'sparse_history_restore_weighted_record_succeeds')
        old_bundle, old_figure = restored.bundle, restored.figure
        old_tables = (restored.summary.text(), table_values(restored.port_table), table_values(restored.case_table))
        old_input = restored.config_from_inputs(); old_state = restored.history_view_state()
        make_figure = restored.make_figure
        with patch.object(restored, 'make_figure', side_effect=RuntimeError('injected history render failure')):
            check(restored.restore_history(first_id) is False, 'sparse_history_renderer_failure_is_reported')
        check(restored.bundle is old_bundle and restored.figure is old_figure
              and restored.current_history_id == second_id and restored.config_from_inputs() == old_input
              and restored.history_view_state() == old_state
              and old_tables == (restored.summary.text(), table_values(restored.port_table), table_values(restored.case_table)),
              'sparse_history_renderer_failure_preserves_parameters_bundle_tables_and_figure')
        details['fault_injection'].append('history renderer failure')

        restored.preview_element(); settle(120)
        count_before = len(restored.history_entries)
        check(restored.preview_active and len(restored.history_entries) == count_before,
              'sparse_history_pending_element_preview_is_not_an_archived_result')
        before_preview = restored.figure
        with patch.object(restored, 'make_figure', side_effect=RuntimeError('injected preview restore failure')):
            check(not restored.restore_history(first_id), 'sparse_history_failed_restore_from_preview_is_reported')
        check(restored.preview_active and restored.figure is before_preview
              and old_tables == (restored.summary.text(), table_values(restored.port_table), table_values(restored.case_table)),
              'sparse_history_failed_restore_preserves_pending_preview_and_saved_tables')
        restored.redraw()

        append = restored._history_store.append
        def disk_failure(*args, **kwargs): raise SparseHistoryError('injected disk full')
        restored._history_store.append = disk_failure
        delivery = copy.deepcopy(restored.bundle)
        restored.begin_worker(_Delivery(result=delivery, parent=restored), 'synthesis'); finish(restored)
        check(restored.bundle is delivery and restored.current_history_id is None
              and len(restored._history_pending) == 1 and len(restored.history_entries) == count_before,
              'sparse_history_disk_failure_keeps_new_result_pending_and_old_disk_entries')
        discard_on_close(second, QMessageBox.StandardButton.No)
        check(second.isVisible(), 'sparse_history_unsaved_close_can_be_cancelled_for_backup')
        restored._history_store.append = append
        check(restored.retry_history_save() and restored.current_history_id is not None
              and len(restored.history_entries) == count_before + 1,
              'sparse_history_retry_commits_pending_snapshot_once')
        recovered_id = restored.current_history_id
        check(restored.retry_history_save() and len(restored.history_entries) == count_before + 1,
              'sparse_history_repeated_retry_does_not_duplicate_committed_result')
        details['fault_injection'].append('disk full after fake successful delivery')
        second.close(); settle()
        third = open_window(); third.activate_task('sparse'); settle()
        check(third.sparse_panel.current_history_id == recovered_id
              and len(third.sparse_panel.history_entries) == count_before + 1,
              'sparse_history_recovered_pending_result_survives_restart')
        details.update(records=len(third.sparse_panel.history_entries), selected_id=recovered_id,
                       science_snapshot_unchanged=True, lazy_restore=True)
    finally:
        for window in windows:
            panel = window.sparse_panel
            if panel.thread is not None:
                panel.cancel(); start = time.monotonic()
                while panel.thread is not None and time.monotonic() - start < 8: settle(30)
            # Cleanup belongs only to this isolated acceptance data; failure
            # details and on-disk archives remain in the test directory.
            panel._history_pending.clear()
            for timer in window.findChildren(QTimer): timer.stop()
            window.close(); window.deleteLater()
            QCoreApplication.sendPostedEvents(window, QEvent.Type.DeferredDelete)
        set_workers(previous_workers); app.setQuitOnLastWindowClosed(previous_quit)
        app.processEvents()


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('tmp/sparse-history-smoke')
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication([]); app.setStyle('Fusion')
    report = {'checks': [], 'qt_platform': app.platformName()}
    def check(condition, name):
        if not condition: raise AssertionError(name)
        report['checks'].append(name)
    code = 0
    try: exercise_sparse_history(app, output, report, check); report['success'] = True
    except Exception:
        report['success'] = False; report['error'] = traceback.format_exc(); code = 1
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'success': report['success'], 'checks': len(report['checks']), 'error': report.get('error')}, ensure_ascii=False))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
