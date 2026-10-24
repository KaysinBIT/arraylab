"""Task workflow regression checks using real Qt widgets and an isolated window."""
import copy
import json
from pathlib import Path
import sys
import traceback

from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QScrollArea

from .parallel import requested_workers, set_workers
from .window import MainWindow


def exercise_workflow(app, window, output, report, check):
    """Check task routing without changing the caller's calculated snapshots.

    Sparse busy checks exercise only the UI state machine. If the caller already
    has a sparse result, a deep copy is reused for screenshots without rerunning it.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    original_workers = requested_workers()
    original_quit = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    retained = None if window is None else (
        window.datasets, window.feed_panel.report, window.planar_panel.report,
        window.sparse_panel.bundle, window.sequence, window.selection.currentIndex(),
    )
    probe = None
    watcher = None
    warnings = []
    details = {"qt_platform": app.platformName(), "screenshots": [],
               "sparse_busy_check": "UI state machine; no synthesis started",
               "horizontal_scroll_checks": []}

    def settle(milliseconds=35):
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()
        app.processEvents()

    def click(widget):
        check(widget.isVisible() and widget.isEnabled(),
              "workflow_clickable_" + (widget.objectName() or widget.text()).replace("\n", "_"))
        QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
        settle()

    def task(kind):
        probe.activate_task(kind)
        settle()

    def f5():
        probe.raise_()
        probe.activateWindow()
        settle()
        QTest.keyClick(probe, Qt.Key.Key_F5)
        settle()
        check(not warnings, "workflow_F5_without_modal_error")

    def graphics():
        probe.ribbon.setCurrentIndex(1)
        settle()

    def picture_actions():
        actions = []
        for menu_action in probe.workbench_menu.actions():
            menu = menu_action.menu()
            if menu is not None:
                actions.extend(a for a in menu.actions()
                               if a.text() in ("保存当前图形…", "复制当前图形"))
        return actions

    def check_no_horizontal_scroll(kind, tab_index):
        settle(50)
        active = probe.inspector_stack.currentWidget()
        scrolls = ([active] if isinstance(active, QScrollArea) else []) + active.findChildren(QScrollArea)
        visible = [scroll for scroll in scrolls if scroll.isVisible()]
        check(bool(visible), f"workflow_{kind}_parameters_have_scroll_container_{tab_index}")
        ranges = [scroll.horizontalScrollBar().maximum() for scroll in visible]
        check(all(value == 0 for value in ranges),
              f"workflow_{kind}_parameters_fit_1100_tab_{tab_index}")
        details["horizontal_scroll_checks"].append({"task": kind, "tab": tab_index,
                                                    "maximums": ranges})

    try:
        set_workers(1)
        probe = MainWindow()
        probe.show()
        probe.raise_()
        probe.activateWindow()
        watcher = QTimer(probe)

        def dismiss_unexpected_message():
            modal = QApplication.activeModalWidget()
            if isinstance(modal, QMessageBox) and probe.isAncestorOf(modal):
                warnings.append(modal.text())
                modal.accept()

        watcher.timeout.connect(dismiss_unexpected_message)
        watcher.start(50)
        settle(150)
        check(probe.active_task() == "home" and probe.sequence == 0 and not probe.datasets
              and probe.feed_panel.report is None and probe.planar_panel.report is None
              and probe.sparse_panel.bundle is None and probe.sparse_panel.thread is None,
              "workflow_default_home_has_no_automatic_calculation")
        check(not probe.inspector_dock.isVisible() and not probe.result_bar.isVisible(),
              "workflow_home_hides_task_parameters_and_results")

        for kind in ("linear", "feed", "planar", "sparse"):
            task("home")
            click(probe.task_cards[kind])
            check(probe.active_task() == kind and probe.sequence == 0
                  and probe.feed_panel.report is None and probe.planar_panel.report is None
                  and probe.sparse_panel.bundle is None and probe.sparse_panel.thread is None,
                  f"workflow_{kind}_card_opens_without_computing")
            check(not probe.image_button.isEnabled() and not probe.copy_button.isEnabled(),
                  f"workflow_{kind}_empty_picture_toolbar_disabled")
            actions = picture_actions()
            check(len(actions) == 2 and all(not action.isEnabled() for action in actions),
                  f"workflow_{kind}_empty_picture_menu_disabled")
            check(probe.current_picture() is None, f"workflow_{kind}_empty_picture_guard")

        task("linear")
        check(probe.inputs["frequency_ghz"].isVisible() and probe.metrics["peak"].isVisible()
              and not probe.feed_parameter_host.isVisible() and probe.result_bar.isVisible(),
              "workflow_linear_sidebar_contains_only_linear_parameters_and_metrics")
        f5()
        check(probe.sequence == 1 and probe.selected_dataset() is not None
              and probe.feed_panel.report is None and probe.planar_panel.report is None
              and probe.sparse_panel.thread is None,
              "workflow_linear_F5_runs_only_linear_analysis")
        linear_result = probe.selected_dataset()
        probe.inputs["elements"].setText("10")
        settle()
        check(probe.state_label.isVisible() and "尚未" in probe.state_label.text()
              and probe.selected_dataset() is linear_result
              and linear_result.result.parameters.elements == 8,
              "workflow_pending_linear_inputs_have_visible_snapshot_warning")
        task("sparse")
        sparse = probe.sparse_panel
        sparse.inputs["nx"].setValue(sparse.inputs["nx"].value() + 1)
        settle()
        sparse_state = sparse.state.text()
        task("linear")
        check(probe.state_label.isVisible() and probe.state_label.text() != sparse_state
              and ("尚未" in probe.state_label.text() or "不同" in probe.state_label.text()),
              "workflow_sparse_status_cannot_replace_linear_snapshot_warning")

        task("feed")
        feed = probe.feed_panel
        check(probe.inspector_stack.currentIndex() == 0 and probe.feed_parameter_host.isVisible()
              and probe.inputs["frequency_ghz"].isVisible() and not probe.metrics["peak"].isVisible()
              and not probe.inputs["scan_deg"].isVisible() and not probe.phase_mode.isVisible()
              and probe.inspector_stack.isAncestorOf(feed.settings_tabs),
              "workflow_feed_sidebar_uses_shared_geometry_without_linear_result_metrics")
        f5()
        check(feed.report is not None and feed.sequence == 1 and probe.sequence == 1
              and probe.selected_dataset() is linear_result and probe.planar_panel.report is None,
              "workflow_feed_F5_runs_only_feed_design")
        feed.target_theta.setText("20")
        f5()
        feed.history_selection.setCurrentIndex(0)
        click(feed.compare_history)
        graphics()
        click(probe.polar_button)
        check(feed.history_selection.currentIndex() == 0 and feed.pattern_plot.selected == 0
              and feed.pattern_plot.polar,
              "workflow_feed_polar_preserves_selected_comparison_record")
        feed.history_selection.setCurrentIndex(1)
        settle()
        check(feed.pattern_plot.selected == 1 and feed.pattern_plot.polar,
              "workflow_feed_history_switch_preserves_polar_mode")
        click(probe.reset_view_button)
        check(feed.pattern_plot.selected == 1 and feed.pattern_plot.polar,
              "workflow_feed_pattern_reset_preserves_polar_and_selection")

        feed_report = feed.report
        task("planar")
        planar = probe.planar_panel
        check(probe.inspector_stack.currentIndex() == 2
              and probe.inspector_stack.isAncestorOf(planar.settings_tabs)
              and planar.geometry_fields["nx"].isVisible()
              and not probe.inputs["frequency_ghz"].isVisible() and not probe.result_bar.isVisible(),
              "workflow_planar_sidebar_is_independent_of_linear_geometry")
        shared_parameters = probe.read_parameters()
        f5()
        check(planar.report is not None and probe.sequence == 1 and feed.report is feed_report
              and probe.read_parameters() == shared_parameters and sparse.thread is None,
              "workflow_planar_F5_runs_only_planar_design")

        task("linear")
        probe.tree.setCurrentItem(probe.model_tree_item)
        settle()
        probe.geometry.zoom = 2.0
        probe.geometry.update()
        graphics()
        click(probe.reset_view_button)
        check(probe.tabs.currentIndex() == 3 and probe.geometry.zoom == 1.0,
              "workflow_reset_current_linear_model_restores_zoom")
        task("feed")
        feed.result_tabs.setCurrentIndex(1)
        settle()
        feed.diagram.scale(2.0, 2.0)
        graphics()
        click(probe.reset_view_button)
        check(abs(feed.diagram.transform().m11() - 1.0) < 1e-12,
              "workflow_reset_current_feed_diagram_restores_transform")

        task("home")
        graphics()
        click(probe.restore_layout_button)
        check(not probe.inspector_dock.isVisible() and not probe.messages_dock.isVisible(),
              "workflow_home_layout_restore_hides_unrelated_parameters")
        task("sparse")
        check(probe.inspector_stack.currentWidget() is sparse.parameter_widget
              and not probe.inputs["frequency_ghz"].isVisible(),
              "workflow_sparse_sidebar_contains_only_sparse_parameters")
        check(sparse.thread is None and not sparse._busy, "workflow_sparse_busy_probe_starts_idle")
        try:
            sparse.set_busy(True)
            settle()
            locked = (probe.compute_button, probe.run_action, probe.save_button, probe.save_action,
                      probe.load_scheme_button, probe.load_parameters_action,
                      probe.save_scheme_button, probe.save_parameters_action,
                      probe.history_open_button, probe.history_open_action)
            check(all(not item.isEnabled() for item in locked) and probe.cancel_button.isEnabled(),
                  "workflow_sparse_busy_locks_toolbar_and_menu_consistently")
            QTest.keyClick(probe, Qt.Key.Key_F5)
            settle()
            check(sparse.thread is None and sparse.bundle is None,
                  "workflow_sparse_busy_F5_does_not_start_a_job")
            click(probe.home_button)
            check(probe.active_task() == "home" and not probe.resume_button.isEnabled()
                  and not probe.history_open_button.isEnabled() and not probe.history_open_action.isEnabled(),
                  "workflow_sparse_busy_home_locks_all_record_restore_entries")
        finally:
            sparse.set_busy(False)
        check(probe.resume_button.isEnabled() and probe.history_open_action.isEnabled(),
              "workflow_record_restore_entries_reenable_after_sparse_idle")

        probe.resize(1100, 740)
        for kind in ("linear", "feed", "planar", "sparse"):
            task(kind)
            tabs = {"feed": feed.settings_tabs, "planar": planar.settings_tabs,
                    "sparse": sparse.settings_tabs}.get(kind)
            if tabs is None:
                check_no_horizontal_scroll(kind, 0)
            else:
                for index in range(tabs.count()):
                    tabs.setCurrentIndex(index)
                    check_no_horizontal_scroll(kind, index)
                tabs.setCurrentIndex(0)
        check(probe.width() == 1100, "workflow_compact_window_really_uses_1100_pixels")

        if window is not None and window.sparse_panel.bundle is not None:
            from .sparse2d.core import Config
            sparse.set_config(Config(**window.sparse_panel.bundle["config"]))
            sparse.receive(copy.deepcopy(window.sparse_panel.bundle))
            sparse.set_busy(False)
            details["sparse_screenshot"] = "deep copy of previously tested snapshot; no recomputation"
        else:
            details["sparse_screenshot"] = "uncomputed task; no synthesis claimed"
        for kind in ("home", "linear", "feed", "planar", "sparse"):
            task(kind)
            if kind == "feed":
                feed.result_tabs.setCurrentIndex(3)
            elif kind == "planar":
                planar.view.setCurrentIndex(planar.view.findData("cut"))
            for width, height in ((1440, 900), (1100, 740)):
                probe.resize(width, height)
                settle(80)
                path = output / f"workflow-{kind}-{width}.png"
                check(probe.grab().save(str(path)), f"workflow_{kind}_{width}_screenshot_saved")
                details["screenshots"].append(path.name)
        check(not warnings, "workflow_no_unexpected_modal_warnings")
        report["workflow"] = details
    finally:
        # Closing a probe does not synchronously delete its child QTimer.
        # Keep this guard from consuming later exercises' validation dialogs.
        if watcher is not None:
            watcher.stop()
            watcher.timeout.disconnect()
        if probe is not None:
            if probe.sparse_panel.thread is None:
                probe.sparse_panel.set_busy(False)
            probe.close()
            probe.deleteLater()
        set_workers(original_workers)
        app.setQuitOnLastWindowClosed(original_quit)
        if window is not None:
            window.raise_()
            window.activateWindow()
        app.processEvents()
    if retained is not None:
        check(window.datasets is retained[0] and window.feed_panel.report is retained[1]
              and window.planar_panel.report is retained[2] and window.sparse_panel.bundle is retained[3]
              and window.sequence == retained[4] and window.selection.currentIndex() == retained[5],
              "workflow_isolated_window_preserves_main_smoke_snapshots")


def main():
    """Standalone native Qt driver; never forces an offscreen platform."""
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tmp/workflow-smoke")
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setStyle("Fusion")
    report = {"packaged": bool(getattr(sys, "frozen", False)),
              "qt_platform": app.platformName(), "checks": []}

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        report["checks"].append(name)

    code = 0
    try:
        exercise_workflow(app, None, output, report, check)
        report["ok"] = True
    except Exception:
        report["ok"] = False
        report["error"] = traceback.format_exc()
        code = 1
    (output / "workflow-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "checks": len(report["checks"]),
                      "qt_platform": app.platformName(), "error": report.get("error")}, ensure_ascii=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
