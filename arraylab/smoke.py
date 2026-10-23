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
            report["success"] = True
            code = 0
        except Exception:
            report["success"] = False
            report["error"] = traceback.format_exc()
        finally:
            (output / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            app.exit(code)

    QTimer.singleShot(300, run)
