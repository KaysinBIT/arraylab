"""Real Qt file-dialog checks for complete, calculation-free history restores."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
import time
import traceback
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QTimer, Qt, QPoint
from PySide6.QtGui import QCursor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit

from .core import ArrayParameters
from .dataio import write_pattern
from .feed import FeedSettings, settings_payload, with_branch_delays
from .history_io import array_result_snapshot, load_history


def exercise_history(app, window, output, report, check):
    """Use actual generated snapshots, visible dialogs, and fresh MainWindows."""
    from .window import MainWindow

    def file_dialog(owner, path, saving):
        errors, seen = [], []

        def accept_file():
            modal = QApplication.activeModalWidget()
            try:
                check(isinstance(modal, QFileDialog), f"{path.stem}_{'save' if saving else 'open'}_is_qt_dialog")
                check(modal.testOption(QFileDialog.Option.DontUseNativeDialog),
                      f"{path.stem}_{'save' if saving else 'open'}_non_native_dialog")
                if saving:
                    check(not modal.testOption(QFileDialog.Option.DontConfirmOverwrite),
                          f"{path.stem}_save_keeps_overwrite_confirmation")
                seen.append(True)
                modal.findChild(QLineEdit, "fileNameEdit").setText(str(path.resolve()))
                modal.accept()
            except Exception:
                errors.append(traceback.format_exc())
                if modal is not None:
                    modal.reject()

        owner.show(); owner.raise_(); owner.activateWindow()
        owner.ribbon.setCurrentIndex(0)
        app.processEvents()
        QTimer.singleShot(150, accept_file)
        button = owner.history_save_button if saving else owner.history_open_button
        check(button.isVisible() and button.isEnabled(),
              f"{path.stem}_{'save' if saving else 'open'}_toolbar_action_available")
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        app.processEvents()
        if errors:
            raise AssertionError(errors[0])
        check(bool(seen), f"{path.stem}_{'save' if saving else 'open'}_dialog_exercised")

    @contextmanager
    def no_calculation():
        with ExitStack() as stack:
            engines = [stack.enter_context(patch(name, side_effect=AssertionError("history restore invoked calculation")))
                       for name in ("arraylab.window.calculate", "arraylab.core.calculate",
                                    "arraylab.feed_widgets.design_feed", "arraylab.feed.design_feed")]
            yield
            check(all(engine.call_count == 0 for engine in engines), "history_restore_never_invokes_calculation_engines")

    def state(owner):
        panel = owner.feed_panel
        return (id(owner.datasets), tuple(id(data) for data in owner.datasets), owner.selection.currentIndex(),
                owner.sequence, tuple((key, field.text()) for key, field in owner.inputs.items()),
                owner.unit.currentIndex(), owner.phase_mode.currentIndex(), owner.keep_history.isChecked(),
                id(panel.history), id(panel.report), panel.history_selection.currentIndex(), panel.sequence,
                panel.topology.currentIndex(), tuple((key, field.text()) for key, field in panel.fields.items()),
                settings_payload(panel.read_settings()), owner.active_task(), owner.last_directory,
                owner.state_label.text(), panel.state.text())

    def reject_without_changes(owner, path, name):
        before = state(owner)
        try:
            owner.load_history_path(path)
        except (ValueError, OSError):
            pass
        else:
            raise AssertionError(f"{name}_was_accepted")
        check(state(owner) == before, f"{name}_preserves_all_existing_state")

    def check_linear(owner, expected, name):
        check(owner.active_task() == "linear" and len(owner.datasets) == 3,
              f"{name}_restores_three_records_and_task")
        check(owner.selection.currentIndex() == expected["selected"]
              and [data.label for data in owner.datasets] == [record["label"] for record in expected["records"]],
              f"{name}_restores_labels_and_selected_record")
        for data, record in zip(owner.datasets, expected["records"]):
            check(np.array_equal(data.angles, record["angles"]) and np.array_equal(data.db, record["db"])
                  and ((data.phase is None and record["phase"] is None)
                       or (data.phase is not None and np.array_equal(data.phase, record["phase"]))),
                  f"{name}_restores_curve_and_phase_{data.label}")
            check((data.result is None and record["result"] is None)
                  or (data.result is not None and array_result_snapshot(data.result) == record["result"]),
                  f"{name}_restores_complete_result_{data.label}")
        check(owner.keep_history.isChecked(), f"{name}_continues_history_after_reopen")
        check("未重新计算" in owner.state_label.text(), f"{name}_identifies_saved_snapshot")

    window.activate_task("linear")
    window.keep_history.setChecked(False)
    for index, parameters in enumerate((
            ArrayParameters(elements=6, scan_deg=-12, step_deg=1),
            ArrayParameters(elements=8, frequency_ghz=5.8, spacing=22, spacing_unit="mm", scan_deg=17, step_deg=1),
            ArrayParameters(elements=10, scan_deg=31, step_deg=1))):
        window.apply_parameters(parameters)
        window.compute()
        window.keep_history.setChecked(True)
    window.datasets[1].label = "中间设计 · 17°"
    window.selection.setItemText(1, window.datasets[1].label)
    window.selection.setCurrentIndex(1)
    linear_parameters = window.selected_dataset().result.parameters
    window.inputs["scan_deg"].setText("42")
    linear_path = output / "history-linear.arraylab.json"
    file_dialog(window, linear_path, True)
    linear_expected = load_history(linear_path)
    check(linear_expected["records"][1]["result"]["parameters"]["scan_deg"] == 17,
          "history_save_uses_result_parameters_instead_of_uncomputed_inputs")

    with no_calculation():
        reopened = MainWindow(auto_compute=False)
        try:
            reopened.show()
            file_dialog(reopened, linear_path, False)
            check_linear(reopened, linear_expected, "linear_history")
            check(reopened.read_parameters() == linear_parameters and reopened.unit.currentData() == "mm",
                  "linear_history_restores_selected_parameters_and_unit")
            check(reopened.metrics["directivity"].text() == f"{reopened.selected_dataset().result.directivity_dbi:.4f} dBi"
                  and reopened.phase_export.isEnabled() and reopened.excitation_table.model().rowCount() == 8,
                  "linear_history_restores_metrics_and_excitation_table")
            previous_busy = reopened.sparse_panel._busy
            reopened.sparse_panel._busy = True
            try:
                reject_without_changes(reopened, linear_path, "busy_sparse_history_restore")
            finally:
                reopened.sparse_panel._busy = previous_busy
            invalid = deepcopy(linear_expected)
            invalid["records"][2]["angles"].pop()
            invalid_path = output / "history-invalid-linear.json"
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
            reject_without_changes(reopened, invalid_path, "invalid_unselected_linear_record")
            check(reopened.grab().save(str(output / "history-linear-restored.png")), "linear_history_restored_screenshot")
        finally:
            reopened.close(); reopened.deleteLater()

    window.selection.setCurrentIndex(2)
    window.remove_selected()
    external_path = output / "history-external.dat"
    write_pattern(external_path, [-90, -45, 0, 45, 90], [-22, -13, -3, -15, -24], [0, 40, 80, 120, 160])
    window.import_path(external_path)
    mixed_path = output / "history-external.arraylab.json"
    file_dialog(window, mixed_path, True)
    mixed_expected = load_history(mixed_path)
    with no_calculation():
        reopened = MainWindow(auto_compute=False)
        try:
            external_inputs = ArrayParameters(elements=12, scan_deg=-47, step_deg=1)
            reopened.apply_parameters(external_inputs)
            reopened.activate_task("planar")
            file_dialog(reopened, mixed_path, False)
            check_linear(reopened, mixed_expected, "mixed_history")
            check(reopened.selected_dataset().result is None and reopened.read_parameters() == external_inputs,
                  "external_history_keeps_current_inputs_without_inventing_model")
            check(reopened.metrics["directivity"].text() == "—" and reopened.phase_export.isEnabled(),
                  "external_history_restores_phase_but_no_model_metrics")
            check(reopened.grab().save(str(output / "history-external-restored.png")), "external_history_restored_screenshot")
        finally:
            reopened.close(); reopened.deleteLater()

    window.activate_task("feed")
    feed = window.feed_panel
    feed.history = []
    feed.sequence = 0
    for index, topology in enumerate(("corporate", "series", "symmetric_series")):
        window.apply_parameters(ArrayParameters(elements=6, scan_deg=(-15, 26, 38)[index], step_deg=1))
        settings = FeedSettings(topology=topology, phase_mode="scan", weighting="custom",
                                custom_amplitudes=(.6, .8, 1, 1, .8, .6),
                                custom_phases_deg=(0, 4, 8, 12, 16, 20),
                                phase_bits=3, phase_error_deg=1.5, amplitude_error_db=.2,
                                failed_ports=(2,) if index == 1 else (), seed=100 + index)
        feed.topology.setCurrentIndex(feed.topology.findData(topology))
        feed.electrical.apply(settings)
        feed.generate()
    feed.history[1]["label"] = "选中馈电 · 有量化和失效端口"
    feed.history_selection.setItemText(1, feed.history[1]["label"])
    feed.history_selection.setCurrentIndex(1)
    expected_feed_report = deepcopy(feed.report)
    window.inputs["scan_deg"].setText("-42")
    feed.electrical.phase_mode.setCurrentIndex(feed.electrical.phase_mode.findData("none"))
    feed_path = output / "history-feed.arraylab.json"
    file_dialog(window, feed_path, True)
    feed_expected = load_history(feed_path)
    legacy = {"format": "arraylab.feed_comparison", "schema_version": 1, "records": deepcopy(feed_expected["records"])}
    for record in legacy["records"]:
        record["report"]["electrical"].pop("pointing")
    legacy_path = output / "history-legacy-feed.json"
    legacy_path.write_text(json.dumps(legacy, ensure_ascii=False), encoding="utf-8")

    with no_calculation():
        reopened = MainWindow(auto_compute=False)
        try:
            file_dialog(reopened, feed_path, False)
            restored = reopened.feed_panel
            check(reopened.active_task() == "feed" and len(restored.history) == 3,
                  "feed_history_restores_three_records_and_task")
            check(restored.history == feed_expected["records"] and restored.history_selection.currentIndex() == 1,
                  "feed_history_restores_complete_reports_labels_and_selection")
            check(restored.report == expected_feed_report and restored.report is restored.history[1]["report"],
                  "feed_history_displays_selected_snapshot")
            check(reopened.read_parameters() == ArrayParameters(**expected_feed_report["array_parameters"])
                  and settings_payload(restored.read_settings()) == expected_feed_report["settings"],
                  "feed_history_restores_selected_geometry_and_all_electrical_settings")
            check(restored.sequence == 3 and restored.ports.rowCount() == 6
                  and "26" in restored.pattern_metrics.text() and "未重新计算" in restored.state.text(),
                  "feed_history_restores_sequence_tables_pointing_and_snapshot_status")
            check(reopened.grab().save(str(output / "history-feed-restored.png")), "feed_history_restored_screenshot")
            invalid = deepcopy(feed_expected)
            invalid["records"][2]["report"]["electrical"]["patterns"]["output"]["angles_deg"].pop()
            invalid_path = output / "history-invalid-feed.json"
            invalid_path.write_text(json.dumps(invalid), encoding="utf-8")
            reject_without_changes(reopened, invalid_path, "invalid_unselected_feed_record")
            reject_without_changes(reopened, output / "history-does-not-exist.json", "missing_history_file")
            reopened.load_history_path(legacy_path)
            check(reopened.active_task() == "feed" and len(restored.history) == 3
                  and restored.history_selection.currentIndex() == 2
                  and all("pointing" in item["report"]["electrical"] for item in restored.history),
                  "legacy_feed_comparison_restores_and_upgrades_saved_pointing")
            check(settings_payload(restored.read_settings()) == legacy["records"][2]["report"]["settings"],
                  "legacy_feed_history_restores_selected_settings")
            # The normal result-export action produces feed-network.json rather
            # than a history envelope. Open that real export through the same
            # visible file dialog, retaining every saved electrical result.
            exported_report = deepcopy(restored.report)
            exported_directory = output / "history-exported-feed"
            restored.export_path(exported_directory)
            exported_path = exported_directory / "feed-network.json"
            exported_bytes = exported_path.read_bytes()
            file_dialog(reopened, exported_path, False)
            check(len(restored.history) == 1 and restored.history_selection.currentIndex() == 0
                  and restored.report == exported_report and restored.history[0]["report"] is restored.report,
                  "feed_result_export_reopens_as_one_complete_saved_design")
            check(restored.ports.rowCount() == len(exported_report["ports"])
                  and restored.sections.rowCount() == len(exported_report["sections"])
                  and settings_payload(restored.read_settings()) == exported_report["settings"]
                  and "未重新计算" in restored.state.text(),
                  "feed_result_export_restores_tables_settings_and_snapshot_status")
            imported_history = output / "history-from-feed-export.arraylab.json"
            file_dialog(reopened, imported_history, True)
            check(load_history(imported_history)["records"][0]["report"] == exported_report
                  and exported_path.read_bytes() == exported_bytes,
                  "feed_result_export_can_be_saved_as_history_without_altering_original")
            bad_export = deepcopy(exported_report)
            bad_export["sections"][0].pop("line_ohm")
            bad_export_path = output / "history-invalid-feed-export.json"
            bad_export_path.write_text(json.dumps(bad_export), encoding="utf-8")
            reject_without_changes(reopened, bad_export_path, "invalid_exported_feed_report")
            reject_without_changes(reopened, exported_directory / "feed-config.json", "settings_are_not_saved_results")
        finally:
            reopened.close(); reopened.deleteLater()

    saved = feed_path.read_bytes()
    window.activate_task("planar")
    try:
        window.save_history_path(feed_path)
    except ValueError:
        pass
    else:
        raise AssertionError("unsupported_task_saved_history")
    check(feed_path.read_bytes() == saved, "unsupported_task_cannot_overwrite_saved_history")
    window.activate_task("feed")
    window.show(); window.raise_(); window.activateWindow()
    app.processEvents()
    report["history_results"] = {
        "linear": {"records": 3, "selected": linear_expected["selected"]},
        "with_external_curve": {"records": 3, "selected": mixed_expected["selected"]},
        "feed": {"records": 3, "selected": feed_expected["selected"]},
        "legacy_feed": {"records": 3, "selected": 2},
    }
    exercise_feed_record_selection(app, output, report, check)


def exercise_feed_record_selection(app, output, report, check):
    """Qt event-driven checks; these are not a human OS-mouse acceptance test."""
    from .window import MainWindow

    probe = MainWindow(auto_compute=False)
    probe.show(); probe.activate_task("feed"); app.processEvents()
    panel = probe.feed_panel
    combo = panel.history_selection
    old_cursor = QCursor.pos()
    selection_times = []

    def select_with_events(index, opener="text", mouse=True):
        if opener == "space":
            combo.setFocus(); QTest.keyClick(combo, Qt.Key.Key_Space)
        elif opener == "alt-down":
            combo.setFocus(); QTest.keyClick(combo, Qt.Key.Key_Down, Qt.KeyboardModifier.AltModifier)
        else:
            x = combo.width() - 8 if opener == "arrow" else 8
            QTest.mouseClick(combo, Qt.MouseButton.LeftButton, pos=QPoint(x, combo.height() // 2))
        app.processEvents()
        view = combo.view()
        check(view.isVisible(), f"feed_record_popup_opens_{combo.count()}_{opener}")
        if mouse:
            # Do not let the synthetic item click fall inside Qt's popup
            # double-click suppression interval after opening the combo.
            QTest.qWait(app.doubleClickInterval() + 20)
            target = view.model().index(index, 0)
            view.scrollTo(target); app.processEvents()
            item = view.visualRect(target).intersected(view.viewport().rect())
            check(not item.isEmpty(), "feed_record_popup_target_visible")
            # Align the native cursor with Qt's event position: Qt's popup mouse
            # release handling also checks that the cursor remains in the item.
            QCursor.setPos(view.viewport().mapToGlobal(item.center())); QTest.qWait(80)
            QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, pos=item.center())
            QTest.qWait(40)
            started = time.perf_counter()
            QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, pos=item.center())
        else:
            QTest.keyClick(view, Qt.Key.Key_Home)
            for _ in range(index):
                QTest.keyClick(view, Qt.Key.Key_Down)
            started = time.perf_counter()
            QTest.keyClick(view, Qt.Key.Key_Return)
        app.processEvents()
        selection_times.append({"records": combo.count(), "selected": index,
                                "restore_and_draw_ms": round((time.perf_counter() - started) * 1000, 2)})
        check(not view.isVisible() and combo.currentIndex() == index, "feed_record_popup_commits_selection")

    def restored(index):
        saved = panel.history[index]["report"]
        check(panel.report is saved and probe.read_parameters() == ArrayParameters(**saved["array_parameters"])
              and settings_payload(panel.read_settings()) == settings_payload(FeedSettings(**saved["settings"])),
              "feed_record_selection_restores_geometry_electrical_and_branch_delays")
        check("未重新计算" in panel.state.text(), "feed_record_selection_identifies_saved_result")

    try:
        check(combo.count() == 0 and not combo.isEnabled() and "暂无设计记录" in combo.placeholderText(),
              "empty_feed_record_selector_explains_missing_records")
        for index, topology in enumerate(("corporate", "series", "symmetric_series", "series", "corporate")):
            n = 4 + index * 2
            parameters = ArrayParameters(elements=n, scan_deg=(-15, 26, 38, -8, 5)[index], step_deg=5)
            settings = with_branch_delays(FeedSettings(topology=topology, load_ohm=40 + index * 5,
                                                       input_ohm=50 + index, phase_bits=index % 3, seed=80 + index),
                                         n, tuple(10 * (j + 1) for j in range(2 * n - 2)))
            probe.apply_parameters(parameters)
            panel.topology.setCurrentIndex(panel.topology.findData(topology))
            for key, edit in panel.fields.items():
                edit.setText(str(getattr(settings, key)))
            panel.electrical.apply(settings); panel.apply_branch_delays(settings); panel.generate(); app.processEvents()
            if index == 0:
                check(combo.count() == 1 and combo.isEnabled(), "generated_record_enables_selector")
                probe.inputs["elements"].setText("12")
                panel.target_theta.setText("43")
                panel.fields["load_ohm"].setText("75")
                with patch("arraylab.feed_widgets.design_feed", side_effect=AssertionError("record selection recalculated")):
                    select_with_events(0, "text")
                    restored(0)
        records = deepcopy(panel.history)
        labels = [combo.itemText(i) for i in range(combo.count())]
        with ExitStack() as stack:
            engines = [stack.enter_context(patch(name, side_effect=AssertionError("record selection recalculated")))
                       for name in ("arraylab.window.calculate", "arraylab.feed_widgets.design_feed", "arraylab.feed.design_feed")]
            with patch.object(panel, "display_report", wraps=panel.display_report) as display:
                select_with_events(0, "arrow")
                restored(0)
                check(display.call_count == 1, "changed_record_draws_once_for_changed_and_activated_signals")
            probe.inputs["elements"].setText("not a number")
            panel.target_theta.setText("42")
            panel.electrical.fields["phase_bits"].setText("bad")
            with patch.object(panel, "display_report", wraps=panel.display_report) as display:
                select_with_events(0, "space", mouse=False)
                restored(0)
                check(display.call_count == 1, "reselecting_current_record_restores_unrun_invalid_inputs_once")
            for size, index, opener in (((1100, 740), 4, "text"), ((1440, 900), 1, "alt-down")):
                probe.resize(*size); app.processEvents()
                select_with_events(index, opener, mouse=opener == "text")
                restored(index)
            check(panel.history == records and [combo.itemText(i) for i in range(combo.count())] == labels,
                  "record_selection_preserves_all_five_reports_and_labels")
            previous = panel.report
            previous_parameters, previous_settings = probe.read_parameters(), panel.read_settings()
            valid_record = panel.history[0]["report"]
            panel.history[0]["report"] = deepcopy(valid_record)
            panel.history[0]["report"]["settings"]["load_ohm"] = "invalid saved value"
            try:
                with patch("arraylab.feed_widgets.QMessageBox.warning") as warning:
                    combo.setCurrentIndex(0)
                    panel.activate_history(0)
                check(warning.call_count == 1 and combo.currentIndex() == 1 and panel.report is previous
                      and probe.read_parameters() == previous_parameters and panel.read_settings() == previous_settings,
                      "invalid_record_keeps_previous_selection_parameters_and_result")
            finally:
                panel.history[0]["report"] = valid_record
            check(all(engine.call_count == 0 for engine in engines), "record_selection_never_runs_calculation_engines")
            saved = output / "history-selected-feed-record.arraylab.json"
            probe.save_history_path(saved)
            probe.load_history_path(saved)
            check(combo.isEnabled() and combo.currentIndex() == 1 and load_history(saved)["selected"] == 1,
                  "selected_feed_record_survives_save_and_open")
            restored(1)
        check(probe.grab().save(str(output / "history-record-selector-restored.png")), "feed_record_selector_screenshot")
        report["feed_record_selection"] = {"empty_state": True, "record_counts": [0, 1, 5],
                                           "parameter_restore": True, "same_item_restore": True,
                                           "selection_times": selection_times,
                                           "event_driver": "Qt events with cursor alignment, not manual OS mouse acceptance"}
    finally:
        QCursor.setPos(old_cursor)
        probe.close(); probe.deleteLater(); app.processEvents()
