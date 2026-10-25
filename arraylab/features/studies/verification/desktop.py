"""Actual Qt interaction checks for scan studies; called by the packaged smoke suite."""
import json
import traceback
from dataclasses import asdict

from PySide6.QtCore import QTimer, Qt, QElapsedTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QLineEdit

from arraylab.features.studies.desktop import ScanStudyDialog


def exercise_scan_dialogs(app, window, output, report, check):
    report["scan_results"] = {}
    for kind, panel in [("linear", window.feed_panel), ("planar", window.planar_panel)]:
        errors = []

        def exercise():
            dialog = QApplication.activeModalWidget()
            try:
                check(isinstance(dialog, ScanStudyDialog), f"{kind}_scan_button_opens_dialog")
                source = panel.report
                expected = source["array_parameters"] if kind == "linear" else source["geometry"]
                check(json.loads(json.dumps(asdict(dialog.parameters))) == expected,
                      f"{kind}_scan_uses_selected_generated_design")
                dialog.count.setValue(3); dialog.start_angle.setText("0"); dialog.stop_angle.setText("40")
                if kind == "planar":
                    dialog.axis.setCurrentIndex(dialog.axis.findData("phi"))

                def wait_finished():
                    timer = QElapsedTimer(); timer.start()
                    while dialog.worker is not None and timer.elapsed() < 15000:
                        QTest.qWait(20)
                    check(dialog.worker is None, f"{kind}_scan_worker_finished")

                QTest.mouseClick(dialog.run_button, Qt.MouseButton.LeftButton); wait_finished()
                check(dialog.report is not None and len(dialog.report["records"]) == 3, f"{kind}_scan_completes")
                original = dialog.report
                report["scan_results"][kind] = original
                config = output.resolve() / f"study-{kind}.json"

                def accept_file():
                    modal = QApplication.activeModalWidget()
                    if isinstance(modal, QFileDialog):
                        modal.findChild(QLineEdit, "fileNameEdit").setText(str(config)); modal.accept()

                QTimer.singleShot(150, accept_file)
                QTest.mouseClick(dialog.save_button, Qt.MouseButton.LeftButton)
                check(config.exists(), f"{kind}_scan_save_settings_dialog")
                dialog.stop_angle.setText("55")
                dialog.export_path(output / f"study-{kind}-export")
                check(json.loads((output / f"study-{kind}-export/study-report.json").read_text(encoding="utf-8")) == original,
                      f"{kind}_scan_export_preserves_successful_snapshot")
                QTimer.singleShot(150, accept_file)
                QTest.mouseClick(dialog.open_button, Qt.MouseButton.LeftButton)
                check(dialog.read_study().stop_deg == 40 and dialog.report is original, f"{kind}_scan_load_preserves_result")
                for key in ("response", "map", "pattern"):
                    dialog.view.setCurrentIndex(dialog.view.findData(key)); dialog.selection.setCurrentIndex(1)
                    app.processEvents(); dialog.canvas.draw()
                    check(bool(dialog.figure.axes), f"{kind}_scan_{key}_renders")
                check(dialog.ports.rowCount() == dialog.parameters.elements, f"{kind}_scan_per_port_phase_table")
                dialog.view.setCurrentIndex(dialog.view.findData("map")); dialog.canvas.draw()
                check(dialog.grab().save(str(output / f"study-{kind}.png")), f"{kind}_scan_screenshot")
                bad = json.loads(config.read_text(encoding="utf-8")); bad["study_settings"]["count"] = 0
                bad_path = output / f"study-{kind}-invalid.json"; bad_path.write_text(json.dumps(bad), encoding="utf-8")
                previous = dialog.read_study()
                try:
                    dialog.load_path(bad_path)
                except ValueError:
                    pass
                else:
                    raise AssertionError("invalid_scan_config_loaded")
                check(dialog.read_study() == previous and dialog.report is original, f"{kind}_scan_invalid_load_preserves_state")
                dialog.count.setValue(100); dialog.start(); dialog.cancel(); wait_finished()
                check(dialog.report is original and "取消" in dialog.status.text(), f"{kind}_scan_cancel_keeps_result")
                dialog.start(); dialog.reject(); wait_finished()
                check(not dialog.isVisible() and dialog.report is original, f"{kind}_scan_close_waits_for_worker")
            except Exception:
                errors.append(traceback.format_exc())
            finally:
                if isinstance(dialog, ScanStudyDialog):
                    dialog.reject()

        window.tabs.setCurrentIndex(4 if kind == "linear" else 5)
        if kind == "linear":
            window.inputs["scan_deg"].setText("17")
        else:
            panel.geometry_fields["scan_theta_deg"].setText("10")
        app.processEvents()
        window.ribbon.setCurrentIndex(0)
        QTimer.singleShot(150, exercise)
        QTest.mouseClick(window.scan_button, Qt.MouseButton.LeftButton)
        if errors:
            raise AssertionError(errors[0])
        cached=panel.study_dialog
        def reopen():
            modal=QApplication.activeModalWidget()
            try:
                check(modal is cached,f"{kind}_scan_reopens_same_design_window")
                check(modal.report is report['scan_results'][kind],f"{kind}_scan_reopen_keeps_completed_result")
                check(not modal._close_pending,f"{kind}_scan_reopen_clears_previous_close_request")
            except Exception:errors.append(traceback.format_exc())
            finally:
                if isinstance(modal,ScanStudyDialog):modal.reject()
        QTimer.singleShot(150,reopen)
        QTest.mouseClick(window.scan_button,Qt.MouseButton.LeftButton)
        if errors:raise AssertionError(errors[0])
