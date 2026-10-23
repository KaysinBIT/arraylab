"""Reproducible integration check, also runnable in the frozen executable."""
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog, QLineEdit

from .dataio import read_pattern
from .configuration import load_configuration


def schedule_smoke_test(app, window, output: Path):
    output.mkdir(parents=True, exist_ok=True)

    def run():
        report = {"packaged": bool(getattr(sys, "frozen", False)), "qt_platform": app.platformName(), "checks": []}
        code = 1

        def check(condition, name):
            if not condition:
                raise AssertionError(name)
            report["checks"].append(name)

        def click(widget):
            QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
            app.processEvents()

        try:
            click(window.compute_button)
            check(abs(window.selected_dataset().result.directivity_dbi - 9.03089987) < 1e-7, "default_8_element_directivity")
            sequence = window.sequence
            QTest.keyClick(window, Qt.Key.Key_F5)
            app.processEvents()
            check(window.sequence == sequence + 1, "F5_shortcut_works_in_workbench")
            click(window.model_button)
            check(window.tabs.currentIndex() == 3 and window.geometry.isVisible(), "ribbon_opens_model_view")
            window.tree.setCurrentItem(window.data_tree_item)
            app.processEvents()
            check(window.tabs.currentIndex() == 1, "navigation_tree_opens_data_table")
            click(window.result_button)
            check(window.tabs.currentIndex() == 0, "ribbon_returns_to_pattern")
            window.inspector_dock.hide()
            window.nav_dock.hide()
            window.messages_dock.hide()
            window.ribbon.setCurrentIndex(3)
            click(window.restore_layout_button)
            check(all(d.isVisible() and not d.isFloating() for d in
                      (window.inspector_dock, window.nav_dock, window.messages_dock)), "closed_docks_restore_without_losing_results")
            window.ribbon.setCurrentIndex(0)
            check("布局已恢复" in window.message_log.toPlainText(), "message_window_records_operations")
            window.phase_mode.setCurrentIndex(1)
            window.inputs["scan_deg"].setText("30")
            check("尚未重新计算" in window.state_label.text(), "edited_inputs_mark_results_stale")
            click(window.compute_button)
            check(abs(window.selected_dataset().result.peaks_deg[0] - 30) < 1e-9, "scan_change_recomputes_peak")
            window.phase_export.setChecked(True)
            file = output / "中文 文件验证.dat"
            window.export_path(file)
            theta, db, phase = read_pattern(file)
            check(phase is not None and np.allclose(db, window.selected_dataset().db), "unicode_path_phase_export_roundtrip")
            dialog_path = output.resolve() / "dialog-result.dat"
            dialog_path.unlink(missing_ok=True)

            def accept_file_dialog():
                modal = QApplication.activeModalWidget()
                if isinstance(modal, QFileDialog):
                    # selectFile() does not replace an already populated name
                    # in a visible Qt dialog. Exercise its actual filename field.
                    modal.findChild(QLineEdit, "fileNameEdit").setText(str(dialog_path))
                    modal.accept()

            QTimer.singleShot(200, accept_file_dialog)
            click(window.save_button)
            check(dialog_path.exists(), "save_dialog_creates_file")
            QTimer.singleShot(200, accept_file_dialog)
            window.open_dialog()
            check(window.selected_dataset().result is None, "open_dialog_reads_saved_file")
            window.phase_mode.setCurrentIndex(1)
            window.inputs["scan_deg"].setText("30")
            click(window.compute_button)
            selected_params = window.selected_dataset().result.parameters
            window.unit.setCurrentIndex(1)
            check(abs(window.read_parameters().spacing_lambda - selected_params.spacing_lambda) < 1e-10, "unit_switch_preserves_physical_spacing")
            # Save pending inputs, not the selected curve's older parameter snapshot.
            window.inputs["scan_deg"].setText("-27.5")
            window.inputs["spacing"].setText("45.25")
            scheme_parameters = window.read_parameters()
            selected_before = window.selected_dataset()
            sequence_before = window.sequence
            dialog_path = output.resolve() / "中文 分析方案.json"
            dialog_path.unlink(missing_ok=True)
            QTimer.singleShot(200, accept_file_dialog)
            click(window.save_scheme_button)
            check(load_configuration(dialog_path) == scheme_parameters, "scheme_save_dialog_uses_pending_inputs")
            window.reset_parameters()
            QTimer.singleShot(200, accept_file_dialog)
            click(window.load_scheme_button)
            check(window.read_parameters() == scheme_parameters and window.unit.currentData() == "mm",
                  "scheme_open_dialog_restores_parameters_and_units")
            check(window.sequence == sequence_before and window.selected_dataset() is selected_before
                  and "尚未重新计算" in window.state_label.text(), "scheme_load_preserves_old_result_and_marks_pending")
            from .window import MainWindow
            reopened = MainWindow(auto_compute=False)
            try:
                reopened.load_configuration_path(dialog_path)
                check(reopened.read_parameters() == scheme_parameters and not reopened.datasets,
                      "scheme_load_into_new_window_restores_analysis_inputs")
            finally:
                reopened.close()
                reopened.deleteLater()
            bad_scheme = output / "invalid-scheme.json"
            bad_scheme.write_text('{"format":"arraylab.analysis","schema_version":99,"parameters":{}}', encoding="utf-8")
            try:
                window.load_configuration_path(bad_scheme)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid_scheme_was_accepted")
            check(window.read_parameters() == scheme_parameters and window.selected_dataset() is selected_before,
                  "invalid_scheme_preserves_inputs_and_selected_result")
            saved_bytes = dialog_path.read_bytes()
            window.inputs["elements"].setText("0")
            try:
                window.save_configuration_path(dialog_path)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid_inputs_were_saved")
            check(dialog_path.read_bytes() == saved_bytes, "invalid_input_save_preserves_existing_scheme")
            window.load_configuration_path(dialog_path)
            click(window.compute_button)
            check(window.selected_dataset().result.parameters == scheme_parameters,
                  "scheme_compute_uses_restored_parameters")
            from .cli import summary
            report["scheme_result"] = summary(window.selected_dataset().result)
            window.plot_type.setCurrentIndex(1)
            window.plot.canvas.draw()
            app.processEvents()
            check(window.plot.axes.name == "polar", "polar_plot_switch")
            check(window.grab().save(str(output / "polar.png")), "polar_screenshot_saved")
            click(window.example_button)
            check(abs(window.selected_dataset().result.directivity_dbi - 7.06756660928) < 1e-8, "manual_page48_example")
            window.keep_history.setChecked(True)
            for scan in (0, 10, 20, 30, 40, 50):
                window.inputs["scan_deg"].setText(str(scan))
                click(window.compute_button)
            check(len(window.datasets) == 5, "history_bounded_at_five")
            window.selection.setCurrentIndex(0)
            check(window.selected_dataset().result.parameters.scan_deg == 10, "history_selection_drives_metrics")
            window.tree.setCurrentItem(window.history_tree_item.child(1))
            app.processEvents()
            check(window.selected_dataset().result.parameters.scan_deg == 20, "history_tree_selects_matching_dataset")
            window.selection.setCurrentIndex(0)
            window.inputs["scan_deg"].setText("25")
            window.export_path(output / "selected_not_edited.dat")
            check(np.allclose(read_pattern(output / "selected_not_edited.dat")[1], window.datasets[0].db), "export_uses_selected_snapshot_not_uncomputed_inputs")
            window.keep_history.setChecked(False)
            window.reset_parameters()
            window.inputs["spacing"].setText("1")
            click(window.compute_button)
            check(window.selected_dataset().result.beamwidth_deg is None and window.notice.isVisible(), "grating_lobes_suppress_width")
            window.inputs["elements"].setText("201")
            dismissed = []

            def close_warning():
                modal = QApplication.activeModalWidget()
                if isinstance(modal, QMessageBox):
                    dismissed.append(modal.text())
                    modal.accept()

            QTimer.singleShot(150, close_warning)
            click(window.compute_button)
            check(bool(dismissed) and window.focusWidget() is window.inputs["elements"], "invalid_input_dialog_and_focus")
            window.import_path(file)
            check(window.metrics["directivity"].text() == "—", "import_does_not_invent_directivity")
            click(window.remove_button)
            check(window.selected_dataset() is None and not window.save_button.isEnabled(), "last_result_removal_disables_save")
            window.plot_type.setCurrentIndex(0)
            click(window.example_button)
            window.plot.canvas.draw()
            app.processEvents()
            check(window.grab().save(str(output / "main.png")), "main_screenshot_saved")
            report["example_directivity_dbi"] = window.selected_dataset().result.directivity_dbi
            window.reset_parameters()
            retained = window.selected_dataset()
            window.ribbon.setCurrentIndex(1)
            click(window.feed_button)
            feed = window.feed_panel
            check(window.tabs.currentIndex() == 4 and feed.report["metrics"]["ports"] == 8,
                  "feed_ribbon_opens_impedance_panel_with_current_inputs")
            check(abs(feed.report["sections"][0]["line_ohm"] - 70.71067811865476) < 1e-10,
                  "corporate_8_port_impedance_reference")
            check(window.selected_dataset() is retained, "feed_design_does_not_replace_array_results")
            report["feed_results"] = {}
            for topology in ("corporate", "series"):
                feed.topology.setCurrentIndex(feed.topology.findData(topology))
                click(feed.design_button)
                dialog_path = output.resolve() / f"feed-{topology}.json"
                dialog_path.unlink(missing_ok=True)
                QTimer.singleShot(200, accept_file_dialog)
                click(feed.save_button)
                check(dialog_path.exists(), f"feed_{topology}_settings_save_dialog")
                report["feed_results"][topology] = feed.report
            check(len(feed.report["metrics"]["out_of_range_sections"]) == 3,
                  "series_high_impedance_sections_flagged")
            old_feed = feed.report
            feed.fields["load_ohm"].setText("75")
            feed.export_path(output / "feed-export")
            exported = json.loads((output / "feed-export/feed-network.json").read_text(encoding="utf-8"))
            check(exported == old_feed and "尚未重新生成" in feed.state.text(),
                  "feed_export_uses_displayed_snapshot_not_pending_inputs")
            QTimer.singleShot(200, accept_file_dialog)
            click(feed.open_button)
            check(feed.read_settings().load_ohm == 50 and "请重新生成" in feed.state.text(),
                  "feed_settings_open_dialog_restores_parameters_pending_regeneration")
            bad_scheme.write_text('{"format":"arraylab.feed_impedance","schema_version":99}', encoding="utf-8")
            before = feed.read_settings()
            try:
                feed.load_path(bad_scheme)
            except ValueError:
                pass
            else:
                raise AssertionError("invalid_feed_settings_accepted")
            check(feed.read_settings() == before and feed.report is old_feed,
                  "invalid_feed_settings_preserve_inputs_and_result")
            feed.fields["input_ohm"].setText("0")
            QTimer.singleShot(150, close_warning)
            click(feed.design_button)
            check(feed.report is old_feed and len(dismissed) == 2, "invalid_feed_input_dialog_preserves_result")
            feed.fields["input_ohm"].setText("50")
            click(feed.design_button)
            app.processEvents()
            check(window.grab().save(str(output / "feed-series.png")), "feed_series_screenshot_saved")
            report["success"] = True
            code = 0
        except Exception:
            report["success"] = False
            report["error"] = traceback.format_exc()
        finally:
            (output / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            app.exit(code)

    QTimer.singleShot(300, run)
