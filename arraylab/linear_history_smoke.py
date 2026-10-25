"""Isolated native Qt acceptance for automatic linear calculation history."""
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch
from uuid import uuid4

from PySide6.QtCore import QCoreApplication, QEvent, QTimer
from PySide6.QtWidgets import QApplication, QMessageBox

from .core import ArrayParameters
from .history_io import load_history, save_history
from .linear_history_store import LinearHistoryStore, HistoryStoreError
from .window import MainWindow


def exercise_linear_history(app, output, report, check):
    output = Path(output) / ("linear-history-autosave-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    path = output / "自动 历史.sqlite3"
    previous_quit = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    windows = []
    details = {"qt_platform": app.platformName(), "directory": str(output), "uses_real_user_archive": False}
    report["linear_autosave"] = details

    def open_window(db=path):
        window = MainWindow(auto_compute=False, history_path=db)
        windows.append(window)
        window.show(); app.processEvents()
        return window

    def close_window(window):
        window.close(); app.processEvents()
        check(not window.isVisible(), "autosave_window_closes_after_committed_results")

    def no_compute(*args, **kwargs):
        raise AssertionError("History restoration must not run a solver")

    def respond_to_close(window, answer):
        timer = QTimer(window)
        timer.setInterval(10)
        def respond():
            dialog = QApplication.activeModalWidget()
            if isinstance(dialog, QMessageBox) and dialog.windowTitle() == "计算历史尚未自动保存":
                timer.stop(); dialog.button(answer).click()
        timer.timeout.connect(respond); timer.start()
        try:
            window.close(); app.processEvents()
        finally:
            timer.stop(); timer.deleteLater()

    try:
        first = open_window()
        first.activate_task("linear")
        for index in range(8):
            first.keep_history.setChecked(False)
            first.apply_parameters(ArrayParameters(elements=4, scan_deg=-20+5*index, step_deg=5))
            first.compute()
        check(len(first.datasets) == 8 and first.history_tree_item.childCount() == 8,
              "autosave_every_successful_F5_keeps_more_than_five_records")
        check(len(first.plot.datasets) == 1, "autosave_comparison_off_only_hides_other_curves")
        first.keep_history.setChecked(True)
        first.selection.setCurrentIndex(1); first.redraw(); app.processEvents()
        check(len(first.datasets) == 8 and len(first.plot.datasets) == 5
              and any(item is first.datasets[1] for item in first.plot.datasets),
              "autosave_comparison_is_bounded_and_includes_selected_old_record")
        stored, sequence, selected_id = LinearHistoryStore(path).load()
        check(len(stored) == 8 and sequence == 8 and selected_id == first.datasets[1]._archive_id,
              "autosave_commits_each_record_and_selected_identity")
        manual = output / "八条记录.arraylab.json"
        first.save_history_path(manual)
        original_file = manual.read_bytes()
        check(len(load_history(manual)["records"]) == 8, "autosave_manual_export_supports_more_than_five")
        close_window(first)

        with patch("arraylab.window.calculate", no_compute), patch("arraylab.core.calculate", no_compute):
            second = open_window()
            check(second.active_task() == "home" and len(second.datasets) == 8,
                  "autosave_restart_restores_all_results_on_home_without_calculation")
            check(second.sequence == 8 and second.selection.currentIndex() == 1
                  and second.read_parameters().scan_deg == -15,
                  "autosave_restart_restores_sequence_selection_and_saved_parameters")
            second.activate_task("linear")
            restored = output / "重启后记录.arraylab.json"
            second.save_history_path(restored)
            check(load_history(restored) == load_history(manual), "autosave_restart_preserves_all_scientific_snapshots")
            second.load_history_path(manual)
            check(len(second.datasets) == 16 and second.selection.currentIndex() == 9
                  and manual.read_bytes() == original_file,
                  "autosave_manual_import_appends_and_preserves_original_file")
        close_window(second)
        third = open_window()
        check(len(third.datasets) == 16 and third.selection.currentIndex() == 9,
              "autosave_imported_records_survive_restart")
        third.activate_task("linear")
        while third.datasets:
            third.selection.setCurrentIndex(len(third.datasets)-1)
            third.remove_selected()
        check(LinearHistoryStore(path).load() == ([], 8, -1), "autosave_explicit_delete_all_retains_sequence")
        close_window(third)
        fourth = open_window(); fourth.activate_task("linear")
        fourth.apply_parameters(ArrayParameters(elements=5, scan_deg=23, step_deg=5))
        fourth.compute()
        check(fourth.sequence == 9 and fourth.datasets[0].label.startswith("#9 "),
              "autosave_new_calculation_after_empty_restart_does_not_reuse_numbers")
        # A second live window must not flush a stale list over the other one's
        # newly appended results.
        peer = open_window(); peer.activate_task("linear")
        fourth.compute(); peer.compute()
        rows, sequence, _ = LinearHistoryStore(path).load()
        check(len(rows) == 3 and sequence == 11
              and [row[1]["label"].split()[0] for row in rows] == ["#9", "#10", "#11"],
              "autosave_two_live_windows_append_without_overwriting_each_other")
        close_window(peer)

        count_before = len(LinearHistoryStore(path).load()[0])
        original_append = fourth._history_store.append
        def write_failure(*args, **kwargs):
            raise HistoryStoreError("simulated disk full")
        fourth._history_store.append = write_failure
        fourth.compute()
        check(len(fourth.datasets) == 3 and len(LinearHistoryStore(path).load()[0]) == count_before
              and "自动保存失败" in fourth.state_label.text(),
              "autosave_disk_failure_keeps_new_result_in_memory_and_old_disk_records")
        respond_to_close(fourth, QMessageBox.StandardButton.No)
        check(fourth.isVisible(), "autosave_failed_close_requires_explicit_discard_or_cancel")
        fourth._history_store.append = original_append
        close_window(fourth)
        check(len(LinearHistoryStore(path).load()[0]) == count_before + 1,
              "autosave_close_retries_pending_result_after_storage_recovers")

        retry_path = output / "重试序号.sqlite3"
        retry = open_window(retry_path); retry.activate_task("linear")
        retry_append = retry._history_store.append
        retry._history_store.append = write_failure
        for _ in range(2):
            retry.apply_parameters(ArrayParameters(elements=4, step_deg=5)); retry.compute()
        check([data.label.split()[0] for data in retry.datasets] == ["#1", "#2"],
              "autosave_failed_results_retain_individual_generation_numbers")
        retry.selection.setCurrentIndex(0)
        retry._history_store.append = retry_append
        close_window(retry)
        retry_rows, retry_sequence, retry_selected = LinearHistoryStore(retry_path).load()
        check([row[1]["label"].split()[0] for row in retry_rows] == ["#1", "#2"]
              and retry_sequence == 2 and retry_selected == retry_rows[0][0],
              "autosave_close_retry_preserves_each_sequence_and_current_selection")

        normal_directory = output / "normal-launch"
        normal_path = normal_directory / "linear-history.sqlite3"
        normal_store = LinearHistoryStore(normal_path)
        normal_store.append(retry_rows[0][1], 1)
        with patch.dict(os.environ, {"ARRAYLAB_HISTORY_DIR": str(normal_directory)}), \
                patch("arraylab.window.calculate", no_compute):
            normal = MainWindow(); windows.append(normal); normal.show(); app.processEvents()
            check(normal.active_task() == "home" and len(normal.datasets) == 1
                  and normal._history_path == normal_path,
                  "autosave_normal_launch_uses_redirected_archive_without_calculation")
            probe = MainWindow(auto_compute=False); windows.append(probe)
            check(not probe.datasets and probe._history_store is None,
                  "autosave_test_windows_ignore_real_or_redirected_archive_by_default")
            close_window(normal)

        corrupt_path = output / "损坏但保留.sqlite3"
        original_corrupt = b"not an ArrayLab database; never overwrite"
        corrupt_path.write_bytes(original_corrupt)
        corrupt = open_window(corrupt_path); corrupt.activate_task("linear")
        check(corrupt._history_warning and corrupt_path.read_bytes() == original_corrupt,
              "autosave_corrupt_archive_is_reported_and_preserved")
        corrupt.apply_parameters(ArrayParameters(elements=4, step_deg=5)); corrupt.compute()
        check(len(corrupt.datasets) == 1 and corrupt_path.read_bytes() == original_corrupt,
              "autosave_corrupt_archive_does_not_prevent_new_in_memory_calculation")
        respond_to_close(corrupt, QMessageBox.StandardButton.Yes)
        check(not corrupt.isVisible() and corrupt_path.read_bytes() == original_corrupt,
              "autosave_explicit_discard_preserves_original_corrupt_archive")
        details.update(records_before_explicit_delete=16, independent_live_windows=True,
                       restart_preserves_results=True, corruption_preserves_original=True)
    finally:
        for window in windows:
            # Failure cleanup only affects these isolated fixture windows.
            window._history_requested = False
            for timer in window.findChildren(QTimer): timer.stop()
            window.close(); window.deleteLater()
            QCoreApplication.sendPostedEvents(window, QEvent.Type.DeferredDelete)
        app.setQuitOnLastWindowClosed(previous_quit)
        app.processEvents()


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tmp/linear-history-smoke")
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication([]); app.setStyle("Fusion")
    report = {"checks": [], "qt_platform": app.platformName()}
    def check(condition, name):
        if not condition: raise AssertionError(name)
        report["checks"].append(name)
    code = 0
    try:
        exercise_linear_history(app, output, report, check)
        report["success"] = True
    except Exception:
        report["success"] = False; report["error"] = traceback.format_exc(); code = 1
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"success": report["success"], "checks":len(report["checks"]), "error":report.get("error")},ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
