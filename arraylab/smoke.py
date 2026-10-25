"""Reproducible integration check, also runnable in the frozen executable."""
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import QTimer, Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QFileDialog, QLineEdit, QDialog, QTableWidget, QInputDialog

from .dataio import read_pattern
from .configuration import load_configuration
from .smoke_suites import new_report, SmokeGroup, run_group


def schedule_smoke_test(app, window, output: Path):
    output.mkdir(parents=True, exist_ok=True)

    def run():
        report = new_report(app.platformName(), "all")
        started = time.perf_counter()
        core_group = SmokeGroup(report, "desktop-core")
        code = 1

        def check(condition, name):
            if not condition:
                raise AssertionError(name)
            report["checks"].append(name)

        def click(widget):
            QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
            app.processEvents()

        try:
            from .icons import application_icon
            icon = application_icon()
            check(not icon.isNull(), "application_icon_loads")
            check({16, 24, 32, 48, 64, 128, 256}.issubset({size.width() for size in icon.availableSizes()}),
                  "application_icon_contains_windows_sizes")
            check(not app.windowIcon().isNull() and app.windowIcon().cacheKey() == icon.cacheKey(),
                  "application_default_icon_uses_array_and_beam_artwork")
            check(window.windowIcon().cacheKey() == icon.cacheKey(), "main_window_uses_application_icon")
            check(window.active_task()=='home' and not window.datasets and window.feed_panel.report is None and window.planar_panel.report is None,
                  'startup_home_does_not_calculate')
            click(window.task_cards['linear'])
            click(window.compute_button)
            check(abs(window.selected_dataset().result.directivity_dbi - 9.03089987) < 1e-7, "default_8_element_directivity")
            sequence = window.sequence
            QTest.keyClick(window, Qt.Key.Key_F5)
            app.processEvents()
            check(window.sequence == sequence + 1, "F5_shortcut_works_in_workbench")
            window.ribbon.setCurrentIndex(1)
            click(window.model_button)
            check(window.tabs.currentIndex() == 3 and window.geometry.isVisible(), "ribbon_opens_model_view")
            window.tree.setCurrentItem(window.data_tree_item)
            app.processEvents()
            check(window.tabs.currentIndex() == 1, "navigation_tree_opens_data_table")
            window.ribbon.setCurrentIndex(1)
            click(window.result_button)
            check(window.tabs.currentIndex() == 0, "ribbon_returns_to_pattern")
            window.inspector_dock.hide()
            window.nav_dock.hide()
            window.messages_dock.hide()
            window.ribbon.setCurrentIndex(1)
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
            window.ribbon.setCurrentIndex(1)
            click(window.example_button)
            check(abs(window.selected_dataset().result.directivity_dbi - 7.06756660928) < 1e-8, "manual_page48_example")
            window.keep_history.setChecked(True)
            first_new_record = len(window.datasets)
            for scan in (0, 10, 20, 30, 40, 50):
                window.inputs["scan_deg"].setText(str(scan))
                click(window.compute_button)
            check(len(window.datasets) == first_new_record + 6, "history_retains_every_successful_calculation")
            check(len(window.plot.datasets) <= 5, "history_plot_overlay_is_bounded_without_discarding_records")
            window.selection.setCurrentIndex(first_new_record)
            check(window.selected_dataset().result.parameters.scan_deg == 0, "history_selection_drives_metrics")
            window.tree.setCurrentItem(window.history_tree_item.child(first_new_record + 2))
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
            before_removal = len(window.datasets)
            click(window.remove_button)
            check(len(window.datasets) == before_removal - 1,
                  "remove_result_keeps_other_historical_calculations")
            while window.datasets:
                click(window.remove_button)
            check(window.selected_dataset() is None and not window.save_button.isEnabled(), "last_result_removal_disables_save")
            window.plot_type.setCurrentIndex(0)
            window.ribbon.setCurrentIndex(1)
            click(window.example_button)
            window.plot.canvas.draw()
            app.processEvents()
            check(window.grab().save(str(output / "main.png")), "main_screenshot_saved")
            report["example_directivity_dbi"] = window.selected_dataset().result.directivity_dbi
            window.reset_parameters()
            retained = window.selected_dataset()
            window.activate_task('home')
            click(window.feed_button)
            check(window.feed_panel.report is None,'feed_task_entry_does_not_auto_generate')
            click(window.compute_button)
            feed = window.feed_panel
            check(window.tabs.currentIndex() == 4 and feed.report["metrics"]["ports"] == 8,
                  "feed_ribbon_opens_impedance_panel_with_current_inputs")
            check(abs(feed.report["sections"][0]["line_ohm"] - 70.71067811865476) < 1e-10,
                  "corporate_8_port_impedance_reference")
            check(window.selected_dataset() is retained, "feed_design_does_not_replace_array_results")
            report["feed_results"] = {}
            for topology in ("corporate", "series"):
                feed.topology.setCurrentIndex(feed.topology.findData(topology))
                click(window.compute_button)
                dialog_path = output.resolve() / f"feed-{topology}.json"
                dialog_path.unlink(missing_ok=True)
                QTimer.singleShot(200, accept_file_dialog)
                click(window.save_scheme_button)
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
            click(window.load_scheme_button)
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
            click(window.compute_button)
            check(feed.report is old_feed and len(dismissed) == 2, "invalid_feed_input_dialog_preserves_result")
            feed.fields["input_ohm"].setText("50")
            click(window.compute_button)
            app.processEvents()
            check(window.grab().save(str(output / "feed-series.png")), "feed_series_screenshot_saved")
            feed.topology.setCurrentIndex(feed.topology.findData("symmetric_series"))
            click(window.compute_button)
            check(feed.report["metrics"]["line_max_ohm"] == 100 and feed.ports.horizontalHeaderItem(feed.ports.columnCount() - 1).text() == "镜像端口",
                  "symmetric_series_impedances_and_mirror_columns")
            check(feed.report["routing"]["left_ports"] == [4, 3, 2, 1]
                  and feed.report["routing"]["right_ports"] == [5, 6, 7, 8],
                  "symmetric_series_arms_route_center_outward")
            dialog_path = output.resolve() / "feed-symmetric_series.json"
            dialog_path.unlink(missing_ok=True)
            QTimer.singleShot(200, accept_file_dialog)
            click(window.save_scheme_button)
            check(dialog_path.exists(), "symmetric_series_settings_save_dialog")
            original_bytes = dialog_path.read_bytes()
            old_symmetric = feed.report
            window.inputs["elements"].setText("5")
            QTimer.singleShot(150, close_warning)
            click(window.compute_button)
            check(feed.report is old_symmetric and len(dismissed) == 3 and "偶数" in dismissed[-1],
                  "symmetric_series_odd_count_dialog_preserves_result")
            try:
                feed.save_path(dialog_path)
            except ValueError:
                pass
            else:
                raise AssertionError("odd_symmetric_settings_saved")
            check(dialog_path.read_bytes() == original_bytes, "odd_symmetric_save_preserves_existing_settings")
            odd_document = json.loads(original_bytes)
            odd_document["array_parameters"]["elements"] = 3
            bad_scheme.write_text(json.dumps(odd_document), encoding="utf-8")
            try:
                feed.load_path(bad_scheme)
            except ValueError:
                pass
            else:
                raise AssertionError("odd_symmetric_settings_loaded")
            check(window.read_parameters().elements == 5 and feed.report is old_symmetric,
                  "odd_symmetric_load_preserves_inputs_and_result")
            QTimer.singleShot(200, accept_file_dialog)
            click(window.load_scheme_button)
            check(window.read_parameters().elements == 8 and feed.read_settings().topology == "symmetric_series",
                  "symmetric_series_settings_open_dialog")
            click(window.compute_button)
            report["feed_results"]["symmetric_series"] = feed.report
            feed.export_path(output / "feed-symmetric-export")
            check(json.loads((output / "feed-symmetric-export/feed-network.json").read_text(encoding="utf-8")) == feed.report,
                  "symmetric_series_result_export")
            app.processEvents()
            check(window.grab().save(str(output / "feed-symmetric.png")), "symmetric_series_screenshot_saved")
            electrical = feed.electrical
            window.phase_mode.setCurrentIndex(1); window.inputs["scan_deg"].setText("30")
            feed.settings_tabs.setCurrentIndex(1)
            electrical.phase_mode.setCurrentIndex(electrical.phase_mode.findData("scan"))
            electrical.weighting.setCurrentIndex(electrical.weighting.findData("taylor"))
            electrical.fields["sidelobe_db"].setText("30")
            click(window.compute_button)
            metrics = feed.report["electrical"]["patterns"]["output"]["metrics"]
            check(abs(metrics["main_lobe_angle_deg"] - 30) < 1e-7 and feed.report["electrical"]["metrics"]["max_active_phase_error_deg"] < 1e-9,
                  "taylor_weighted_scan_compensation_reaches_target")
            feed.result_tabs.setCurrentIndex(3)
            check(len(feed.pattern_plot.datasets) == 1 and feed.pattern_plot.datasets[0].pattern_key == "output",
                  "electrical_final_source_displays_only_final_output")
            feed.pattern_source.setCurrentIndex(feed.pattern_source.findData("comparison"))
            check(len(feed.pattern_plot.datasets) == 3, "electrical_target_raw_output_patterns_displayed")
            feed.pattern_source.setCurrentIndex(feed.pattern_source.findData("output"))
            feed.result_tabs.setCurrentIndex(2)
            previewed = []
            def accept_preview():
                modal = QApplication.activeModalWidget()
                if isinstance(modal, QDialog):
                    table = modal.findChild(QTableWidget)
                    if table:
                        previewed.append(table.rowCount())
                    modal.accept()
            QTimer.singleShot(200, accept_preview)
            click(electrical.preview_button)
            check(previewed == [8], "electrical_input_preview_dialog")
            from .excitation import excitation_csv
            dialog_path = output.resolve() / "custom-excitation.csv"
            dialog_path.write_text(excitation_csv([1,2,3,4,4,3,2,1], [5,-5,0,0,0,0,-5,5]), encoding="utf-8")
            QTimer.singleShot(200, accept_file_dialog)
            click(electrical.import_button)
            check(electrical.weighting.currentData() == "custom" and len(electrical.custom_amplitudes) == 8,
                  "custom_excitation_import_dialog")
            previous_values = electrical.custom_amplitudes
            invalid_csv = output / "invalid-excitation.csv"; invalid_csv.write_text("element,amplitude,phase_deg\n1,1,0\n")
            try:
                electrical.import_path(invalid_csv)
            except ValueError:
                pass
            else:
                raise AssertionError("incomplete_excitation_imported")
            check(electrical.custom_amplitudes == previous_values, "bad_excitation_preserves_previous_inputs")
            click(window.compute_button)
            electrical.fields["phase_bits"].setText("3")
            electrical.fields["amplitude_error_db"].setText("0.5")
            electrical.fields["phase_error_deg"].setText("2")
            electrical.fields["seed"].setText("23"); electrical.failed.setText("2")
            click(window.compute_button)
            first_realization = feed.report
            click(window.compute_button)
            check(feed.report == first_realization and feed.report["ports"][1]["output_amplitude"] == 0
                  and feed.report["ports"][1]["output_phase_deg"] is None,
                  "seeded_error_quantization_and_matched_failure_reproducible")
            check(len(feed.history) == 5, "electrical_history_bounded_at_five")
            feed.history_selection.setCurrentIndex(0)
            selected_report = feed.report
            feed.compare_history.setChecked(True)
            check(len(feed.pattern_plot.datasets) == 5 and feed.report is feed.history[0]["report"],
                  "electrical_history_selection_drives_plot_and_metrics")
            def name_history():
                modal = QApplication.activeModalWidget()
                if isinstance(modal, QInputDialog):
                    modal.setTextValue("电气验证记录"); modal.accept()
            QTimer.singleShot(150, name_history)
            feed.rename_history()
            check(feed.history[0]["label"] == "电气验证记录", "electrical_history_label_edit")
            window.inputs["scan_deg"].setText("15")
            feed.export_path(output / "electrical-selected-export")
            check(json.loads((output / "electrical-selected-export/feed-network.json").read_text(encoding="utf-8")) == selected_report,
                  "electrical_export_follows_selected_history")
            for key in ("target", "uncompensated", "output"):
                theta, db, phase = read_pattern(output / f"electrical-selected-export/feed-{key}.dat")
                pattern = selected_report["electrical"]["patterns"][key]
                check(phase is None and np.allclose(theta, pattern["angles_deg"], rtol=0, atol=1e-9)
                      and np.allclose(db, pattern["pattern_db"], rtol=0, atol=1e-8),
                      f"electrical_{key}_DAT_matches_selected_history_JSON")
            feed.export_history_path(output / "electrical-comparison.json")
            comparison = json.loads((output / "electrical-comparison.json").read_text(encoding="utf-8"))
            check(len(comparison["records"]) == 5 and comparison["records"][0]["report"] == selected_report,
                  "electrical_comparison_export_contains_full_snapshots")
            window.inputs["scan_deg"].setText("30")
            feed.compare_history.setChecked(False)
            click(window.compute_button)
            dialog_path = output.resolve() / "feed-electrical.json"
            dialog_path.unlink(missing_ok=True)
            QTimer.singleShot(200, accept_file_dialog)
            click(window.save_scheme_button)
            report["electrical_result"] = feed.report
            check(dialog_path.exists(), "electrical_settings_save_dialog")
            feed.result_tabs.setCurrentIndex(3)
            check(feed.settings_tabs.isVisible() and feed.parameters_in_sidebar, "electrical_parameters_stay_in_dedicated_sidebar")
            app.processEvents()
            check(window.grab().save(str(output / "electrical.png")), "electrical_workbench_screenshot")
            line_parameters = window.read_parameters()
            window.activate_task('home')
            click(window.planar_button)
            check(window.planar_panel.report is None,'planar_task_entry_does_not_auto_generate')
            click(window.compute_button)
            planar = window.planar_panel
            check(window.tabs.currentIndex() == 5 and len(planar.report["network"]["ports"]) == 16,
                  "planar_ribbon_opens_rectangular_design")
            check(window.read_parameters() == line_parameters, "planar_parameters_independent_of_line_array")
            planar.geometry_fields["scan_theta_deg"].setText("33")
            planar.geometry_fields["scan_phi_deg"].setText("47")
            planar.geometry_fields["cut_phi_deg"].setText("47")
            click(window.compute_button)
            pm = planar.report["electrical"]["patterns"]["output"]["metrics"]
            check(abs(pm["representative_peak_theta_deg"] - 33) < 1e-8 and abs(pm["representative_peak_phi_deg"] - 47) < 1e-8,
                  "rectangular_scan_theta_phi_compensation")
            for name in ("cut", "map", "3d"):
                planar.view.setCurrentIndex(planar.view.findData(name)); app.processEvents()
                planar.canvas.draw()
                check(bool(planar.figure.axes) and (name != "3d" or planar.figure.axes[0].name == "3d"),
                      f"planar_{name}_view_renders")
                if name == "3d":
                    axes = planar.figure.axes[0]; before_view = (axes.azim, axes.elev)
                    box = axes.get_window_extent()
                    point = QPoint(int((box.x0 + box.x1) / 2), int(planar.canvas.height() - (box.y0 + box.y1) / 2))
                    QTest.mousePress(planar.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
                    QTest.mouseMove(planar.canvas, point + QPoint(45,20), 30)
                    QTest.mouseRelease(planar.canvas, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point + QPoint(45,20))
                    app.processEvents()
                    check(before_view != (axes.azim, axes.elev), "planar_3d_mouse_rotation")
            planar.settings_tabs.setCurrentIndex(0)
            dialog_path = output.resolve() / "arbitrary-coordinates.csv"
            dialog_path.write_text("element,x,y\n1,-1,-0.3\n2,-0.3,0.6\n3,0.2,-0.5\n4,0.8,0.7\n5,0.5,0.1\n6,-0.7,-0.8\n", encoding="utf-8")
            QTimer.singleShot(200, accept_file_dialog)
            click(planar.import_coordinates_button)
            check(planar.geometry.currentData() == "arbitrary" and len(planar.coordinates) == 6,
                  "arbitrary_planar_coordinate_import_dialog")
            click(window.compute_button)
            arbitrary_report = planar.report
            dialog_path = output.resolve() / "planar-gui.json"
            dialog_path.unlink(missing_ok=True)
            QTimer.singleShot(200, accept_file_dialog)
            click(window.save_scheme_button)
            check(dialog_path.exists(), "planar_settings_save_dialog")
            planar.geometry_fields["scan_theta_deg"].setText("10")
            planar.export_path(output / "planar-selected-export")
            check(json.loads((output / "planar-selected-export/planar-report.json").read_text(encoding="utf-8")) == arbitrary_report,
                  "planar_export_uses_generated_snapshot")
            for key in ("target", "uncompensated", "output"):
                theta, db, phase = read_pattern(output / f"planar-selected-export/planar-cut-{key}.dat")
                pattern = arbitrary_report["electrical"]["patterns"][key]
                check(phase is None and np.allclose(theta, pattern["cut_angles_deg"], rtol=0, atol=1e-9)
                      and np.allclose(db, pattern["cut_db"], rtol=0, atol=1e-8),
                      f"planar_{key}_DAT_matches_generated_snapshot_JSON")
            QTimer.singleShot(200, accept_file_dialog)
            click(window.load_scheme_button)
            check(planar.read_parameters().scan_theta_deg == 33 and len(planar.coordinates) == 6,
                  "planar_settings_load_restores_geometry")
            bad_planar = json.loads(dialog_path.read_text(encoding="utf-8"))
            bad_planar["geometry"]["coordinates"][1] = bad_planar["geometry"]["coordinates"][0]
            bad_scheme.write_text(json.dumps(bad_planar), encoding="utf-8")
            previous_geometry = planar.read_parameters()
            try:
                planar.load_path(bad_scheme)
            except ValueError:
                pass
            else:
                raise AssertionError("overlapping_planar_elements_loaded")
            check(planar.read_parameters() == previous_geometry and planar.report is arbitrary_report,
                  "invalid_planar_geometry_preserves_inputs_and_result")
            planar.geometry_fields["frequency_ghz"].setText("0")
            QTimer.singleShot(150, close_warning)
            click(window.compute_button)
            check(planar.report is arbitrary_report and len(dismissed) == 4, "planar_invalid_input_dialog")
            planar.geometry_fields["frequency_ghz"].setText("3")
            report["planar_result"] = arbitrary_report
            planar.view.setCurrentIndex(planar.view.findData("3d")); app.processEvents(); planar.canvas.draw()
            check(window.grab().save(str(output / "planar.png")), "planar_workbench_screenshot")
            for filename, values in [
                ("electrical-selected-export/feed-output.dat", selected_report["electrical"]["patterns"]["output"]["pattern_db"]),
                ("planar-selected-export/planar-cut-output.dat", arbitrary_report["electrical"]["patterns"]["output"]["cut_db"]),
            ]:
                window.import_path(output / filename)
                dataset = window.selected_dataset()
                check(dataset.result is None and np.allclose(dataset.db, values, rtol=0, atol=1e-8),
                      f"{Path(filename).name}_desktop_import_without_inferred_directivity")
            core_group.finish(True)
            core_group = None
            from .waveform_smoke import exercise_waveform
            run_group(report, 'waveform', lambda: exercise_waveform(app, window, output, report, check))
            from .mimo_smoke import exercise_mimo
            run_group(report, 'mimo', lambda: exercise_mimo(app, window, output, report, check))
            from .study_smoke import exercise_scan_dialogs
            run_group(report, 'scan-dialogs', lambda: exercise_scan_dialogs(app, window, output, report, check))
            from .sparse_smoke import exercise_sparse
            run_group(report, 'sparse-workflow', lambda: exercise_sparse(app, window, output, report, check))
            from .sparse_overview_smoke import exercise as exercise_sparse_overview
            run_group(report, "sparse-overview", lambda: exercise_sparse_overview(app, window, output, check, click, report))
            from .sparse_navigation_smoke import exercise_sparse_navigation
            run_group(report, 'sparse-navigation', lambda: exercise_sparse_navigation(app, output, report, check))
            from .sparse_history_smoke import exercise_sparse_history
            run_group(report, 'sparse-history', lambda: exercise_sparse_history(app, output, report, check))
            from .pointing_smoke import exercise_pointing
            run_group(report, 'pointing', lambda: exercise_pointing(app, window, output, report, check))
            from .multicore_smoke import exercise_multicore
            run_group(report, 'multicore', lambda: exercise_multicore(app, window, output, report, check))
            from .history_smoke import exercise_history
            run_group(report, 'feed-history', lambda: exercise_history(app, window, output, report, check))
            from .workflow_smoke import exercise_workflow
            run_group(report, 'workflow', lambda: exercise_workflow(app, window, output, report, check))
            from .linear_workflow_smoke import exercise_linear_feed_handoff
            run_group(report, 'linear-feed-handoff', lambda: exercise_linear_feed_handoff(app, window, output, report, check))
            from .linear_history_smoke import exercise_linear_history
            run_group(report, 'linear-history', lambda: exercise_linear_history(app, output, report, check))
            from .layout_smoke import exercise_layout
            run_group(report, 'layout', lambda: exercise_layout(app, output, check, report, window))
            from .next_stage_smoke import exercise_next_stage
            run_group(report, 'next-stage', lambda: exercise_next_stage(app, window, output, report, check))
            from .branch_lengths_smoke import exercise_branch_lengths
            run_group(report, 'branch-lengths', lambda: exercise_branch_lengths(app, window, output, report, check))
            from .sparse_responsiveness_smoke import exercise_sparse_responsiveness
            run_group(report, 'sparse-responsiveness', lambda: exercise_sparse_responsiveness(app, window, output, report, check))
            from .sparse_size_smoke import exercise_sparse_sizes
            run_group(report, 'sparse-sizes', lambda: exercise_sparse_sizes(app, window, output, report, check))
            from .interaction_smoke import exercise_3d_interaction
            run_group(report, '3d-interaction', lambda: exercise_3d_interaction(app, window, output, report, check))
            report["success"] = True
            code = 0
        except Exception:
            report["success"] = False
            report["error"] = traceback.format_exc()
        finally:
            if core_group is not None:
                core_group.finish(False)
            report["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            (output / "smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            app.exit(code)

    QTimer.singleShot(300, run)
