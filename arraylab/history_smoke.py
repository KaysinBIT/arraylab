"""Real Qt file-dialog checks for complete, calculation-free history restores."""
from contextlib import ExitStack, contextmanager
from copy import deepcopy
import json
import traceback
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit

from .core import ArrayParameters
from .dataio import write_pattern
from .feed import FeedSettings, settings_payload
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
