"""Real Qt interactions for editable branch delays and immutable result snapshots."""
from copy import deepcopy
import json
import traceback

import numpy as np
from PySide6.QtCore import QTimer, Qt, QByteArray
from PySide6.QtGui import QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLineEdit, QMessageBox

from arraylab.features.feed.branch_widgets import BranchDelayDialog
from arraylab.features.linear.model import ArrayParameters
from arraylab.features.feed.model import FeedSettings, design_feed, synthesize_impedance, with_branch_delays
from arraylab.features.feed.diagram import diagram_svg


def exercise_branch_lengths(app, window, output, report, check):
    from arraylab.application.desktop.window import MainWindow
    panel = window.feed_panel
    previous_size = window.size()
    window.resize(1100, 740)
    window.activate_task("feed")
    window.ribbon.setCurrentIndex(0)
    window.apply_parameters(ArrayParameters(elements=4, spacing=.45, scan_deg=20, step_deg=1))
    settings = FeedSettings(topology="series", phase_mode="none")
    panel.topology.setCurrentIndex(panel.topology.findData(settings.topology))
    panel.electrical.apply(settings)
    panel.apply_branch_delays(settings)
    panel.history.clear()
    panel.settings_tabs.setCurrentIndex(0)
    window.show()
    app.processEvents()
    panel.generate()
    baseline = panel.report
    check(panel.branch_delay_button.isVisible(), "branch_editor_visible_in_feed_sidebar")
    check(panel.branch_delay_button.visibleRegion().contains(panel.branch_delay_button.rect()),
          "branch_editor_button_fully_visible_at_1100x740")
    check(window.grab().save(str(output / "branch-length-compact.png")), "branch_compact_sidebar_screenshot")

    def click(widget):
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
        app.processEvents()

    def edit_value(dialog, row, value):
        table = dialog.table
        table.setCurrentCell(row, dialog.DELAY_COLUMN)
        table.editItem(table.item(row, dialog.DELAY_COLUMN))
        editor = QApplication.focusWidget()
        check(isinstance(editor, QLineEdit), "branch_value_uses_real_qt_editor")
        editor.selectAll()
        QTest.keyClicks(editor, str(value))
        QTest.keyClick(editor, Qt.Key.Key_Tab)

    def editor_action(action):
        errors, seen = [], []
        panel.settings_tabs.setCurrentIndex(0)
        app.processEvents()
        def interact():
            modal = QApplication.activeModalWidget()
            try:
                check(isinstance(modal, BranchDelayDialog), "branch_edit_button_opens_real_dialog")
                seen.append(True)
                action(modal)
            except Exception:
                errors.append(traceback.format_exc())
                if modal is not None:
                    modal.reject()
        QTimer.singleShot(150, interact)
        click(panel.branch_delay_button)
        if errors:
            raise AssertionError(errors[0])
        check(bool(seen), "branch_editor_interaction_completed")

    def first_edit(dialog):
        check(dialog.table.rowCount() == 6, "branch_editor_has_all_series_sections")
        check(not bool(dialog.table.item(0, 4).flags() & Qt.ItemFlag.ItemIsEditable),
              "branch_transform_length_is_read_only")
        edit_value(dialog, 0, 3601)
        warnings = []
        def dismiss_warning():
            modal = QApplication.activeModalWidget()
            if isinstance(modal, QMessageBox):
                warnings.append(modal.text())
                modal.accept()
        QTimer.singleShot(100, dismiss_warning)
        click(dialog.apply_button)
        report["branch_invalid_input_dialog"] = {
            "entered": dialog.table.item(0, dialog.DELAY_COLUMN).text(),
            "warnings": list(warnings), "editor_visible": dialog.isVisible(),
            "accepted_delays": dialog.result_delays,
        }
        check(bool(warnings) and dialog.isVisible() and dialog.result_delays is None,
              "branch_invalid_delay_warns_without_accepting_dialog")
        edit_value(dialog, 0, 405)
        check(dialog.table.item(0, 6).text() == "495", "branch_dialog_unwrapped_total_preserves_full_turn")
        check(dialog.grab().save(str(output / "branch-length-editor.png")), "branch_editor_screenshot")
        click(dialog.apply_button)
    editor_action(first_edit)
    check(panel.report is baseline and panel.read_settings().branch_delays_deg[0] == 405,
          "branch_edit_changes_pending_inputs_only")
    check("尚未重新生成" in panel.state.text(), "branch_edit_shows_pending_state")
    click(window.compute_button)
    changed = panel.report
    check(changed is not baseline and changed["sections"][0]["matched_delay_deg"] == 405,
          "branch_f5_generates_new_snapshot")
    check(changed["ports"][0]["branch_delay_path_deg"] == 405,
          "branch_path_accumulation_is_displayed_without_modulo")
    check(not np.allclose(changed["electrical"]["patterns"]["output"]["pattern_db"],
                          baseline["electrical"]["patterns"]["output"]["pattern_db"]),
          "branch_delay_changes_uncompensated_pattern")
    headers = [panel.sections.horizontalHeaderItem(i).text() for i in range(panel.sections.columnCount())]
    check(all(text in headers for text in ("变换段 (°)", "附加延迟 (°)", "匹配延迟 Zc (Ω)", "支路合计 (°)")),
          "branch_table_separates_transform_and_matched_delay")
    port_headers = [panel.ports.horizontalHeaderItem(i).text() for i in range(panel.ports.columnCount())]
    check(all(text in port_headers for text in ("网络路径合计 (°)", "端口补偿 (°)", "完整路径 (°)")),
          "branch_port_table_separates_network_and_terminal_compensation")
    panel.result_tabs.setCurrentIndex(1)
    panel.diagram.fit()
    app.processEvents()
    check(panel.diagram.renderer.isValid(), "branch_connection_svg_valid_in_qt")
    check(window.grab().save(str(output / "branch-length-network.png")), "branch_network_screenshot")

    previous_settings = panel.read_settings()
    def cancel_edit(dialog):
        edit_value(dialog, 0, 90)
        dialog.reject()
    editor_action(cancel_edit)
    check(panel.read_settings() == previous_settings and panel.report is changed,
          "branch_cancel_preserves_pending_settings_and_results")
    config = output / "branch-length-settings.json"
    panel.save_path(config)
    def clear_edit(dialog):
        click(dialog.clear_button)
        click(dialog.apply_button)
    editor_action(clear_edit)
    check(panel.read_settings().branch_delays_deg == (), "branch_clear_removes_old_binding")
    panel.load_path(config)
    check(panel.read_settings() == previous_settings and panel.report is changed,
          "branch_configuration_restores_pending_delays_without_calculation")

    history = output / "branch-length-history.json"
    window.save_history_path(history)
    probe = MainWindow(auto_compute=False)
    try:
        probe.show()
        probe.load_history_path(history)
        check(probe.feed_panel.read_settings() == previous_settings,
              "branch_history_restores_bound_delays")
        check(probe.feed_panel.report == changed, "branch_history_preserves_complete_snapshot")
    finally:
        probe.close()
        probe.deleteLater()
    panel.history_selection.setCurrentIndex(0)
    check(panel.report is baseline and panel.sections.item(0, headers.index("附加延迟 (°)")).text() == "0",
          "branch_selected_history_controls_section_table")
    panel.history_selection.setCurrentIndex(1)
    editor_action(clear_edit)
    export = output / "branch-length-selected-export"
    panel.export_path(export)
    check(json.loads((export / "feed-network.json").read_text(encoding="utf-8")) == changed,
          "branch_export_uses_selected_snapshot_not_pending_inputs")
    svg = (export / "feed-network.svg").read_text(encoding="utf-8")
    check("D1: " in svg and "405 deg" in svg and "C1: " in svg,
          "branch_export_svg_has_independent_delay_and_compensation")
    check((export / "feed-sections.csv").read_text(encoding="utf-8-sig").splitlines()[0].find("matched_delay_deg") >= 0,
          "branch_csv_exports_delay_field")

    panel.apply_branch_delays(previous_settings)
    panel.topology.setCurrentIndex(panel.topology.findData("corporate"))
    check("需重新设置" in panel.branch_delay_button.text(), "branch_topology_change_marks_stale_binding")
    try:
        panel.read_settings()
    except ValueError:
        rejected = True
    else:
        rejected = False
    check(rejected, "branch_topology_change_rejects_stale_T_numbers")
    def stale_cancel(dialog):
        check(dialog.reset_notice.isVisible() and "不匹配" in dialog.reset_notice.text(),
              "branch_stale_reset_is_explicit")
        check(all(dialog.table.item(i, 5).text() == "0" for i in range(dialog.table.rowCount())),
              "branch_stale_editor_does_not_remap_delays")
        dialog.reject()
    editor_action(stale_cancel)
    check(panel.branch_delays_deg[0] == 405, "branch_stale_cancel_keeps_old_settings")
    editor_action(clear_edit)
    check(panel.read_settings().branch_delays_deg == (), "branch_stale_reset_acceptance_clears_binding")

    window.apply_parameters(ArrayParameters(elements=200, step_deg=1))
    def many_rows(dialog):
        check(dialog.table.rowCount() == 398, "branch_editor_supports_200_element_network")
        dialog.table.scrollToBottom()
        app.processEvents()
        check(dialog.table.verticalScrollBar().maximum() > 0, "branch_large_network_scrolls")
        dialog.reject()
    editor_action(many_rows)
    window.apply_parameters(ArrayParameters(elements=4, step_deg=1))

    # Stored legacy reports acquire zero-delay display defaults without mutation.
    legacy = deepcopy(baseline)
    for line in legacy["sections"]:
        for name in ("matched_delay_deg", "matched_delay_ohm", "total_electrical_length_deg"):
            line.pop(name, None)
    for port in legacy["ports"]:
        for name in ("transform_path_deg", "branch_delay_path_deg", "network_electrical_length_deg", "total_electrical_length_deg"):
            port.pop(name, None)
    original = deepcopy(legacy)
    panel.display_report(legacy)
    check(legacy == original and panel.sections.item(0, headers.index("附加延迟 (°)")).text() == "0",
          "branch_legacy_display_is_derived_without_mutating_snapshot")

    for topology in ("corporate", "series", "symmetric_series"):
        parameters = ArrayParameters(elements=8, step_deg=1)
        topology_settings = with_branch_delays(FeedSettings(topology=topology), 8, tuple(range(14)))
        network = design_feed(parameters, topology_settings)
        raw_svg = diagram_svg(network)
        renderer = QSvgRenderer(QByteArray(raw_svg.encode("utf-8")))
        check(renderer.isValid(), f"branch_{topology}_svg_renders")
        size = renderer.defaultSize()
        image = QImage(size, QImage.Format.Format_ARGB32)
        image.fill(Qt.GlobalColor.white)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        check(image.save(str(output / f"branch-{topology}.png")), f"branch_{topology}_diagram_image")
    panel.display_report(changed)
    window.resize(previous_size)
    report["branch_length_result"] = changed
