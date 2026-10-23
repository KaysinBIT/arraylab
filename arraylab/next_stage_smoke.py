"""Real Qt acceptance for sphere views, metric captions and subarray placement."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import traceback

from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from .core import ArrayParameters
from .parallel import requested_workers, set_workers
from .subarray import SubarrayConfig, expanded_coordinates, layout_valid
from .window import MainWindow


def exercise_next_stage(app, window, output, report, check):
    """Use an isolated visible window; do not rerun or mutate the caller's state."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    prior_workers = requested_workers()
    prior_quit = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    probe = None
    warnings = []
    details = {"qt_platform": app.platformName(), "screenshots": [], "footer_bounds": [],
               "subarray_execution": "real QThread optimization and cooperative cancellation"}
    retained = None if window is None else (window.datasets, window.feed_panel.report,
                                           window.planar_panel.report, window.sparse_panel.bundle,
                                           window.subarray_panel.report)

    def settle(milliseconds=45):
        loop = QEventLoop()
        QTimer.singleShot(milliseconds, loop.quit)
        loop.exec()
        app.processEvents()

    def wait_idle(panel, timeout_ms=100000):
        if panel.thread is None:
            return
        loop = QEventLoop()
        tick = QTimer(); tick.setInterval(25)
        limit = QTimer(); limit.setSingleShot(True)
        tick.timeout.connect(lambda: loop.quit() if panel.thread is None else None)
        limit.timeout.connect(loop.quit)
        tick.start(); limit.start(timeout_ms)
        loop.exec()
        tick.stop(); limit.stop()
        if panel.thread is not None:
            raise TimeoutError("Subarray worker did not finish within the bounded Qt event loop")
        app.processEvents()

    def task(name):
        probe.activate_task(name); settle()

    def footer_snapshot(canvas, label):
        canvas.draw()
        figure = canvas.figure
        artist = getattr(figure, "_arraylab_metric_footer", None)
        check(artist is not None and artist in figure.texts, "next_"+label+"_footer_exists")
        renderer = canvas.get_renderer()
        bounds = artist.get_window_extent(renderer)
        total = figure.bbox
        path = output / ("next-" + label + ".png")
        check(canvas.grab().save(str(path)), "next_"+label+"_PNG_saved")
        details["screenshots"].append(path.name)
        details["footer_bounds"].append({"view": label, "footer": list(bounds.bounds),
                                         "canvas": list(total.bounds),
                                         "axes_bottom": figure.axes[0].get_window_extent(renderer).y0})
        check(bounds.x0 >= total.x0-2 and bounds.x1 <= total.x1+2 and
              bounds.y0 >= total.y0-2 and bounds.y1 <= total.y1+2,
              "next_"+label+"_footer_inside_canvas")
        check(bounds.y1 <= figure.axes[0].get_window_extent(renderer).y0 + 1,
              "next_"+label+"_footer_below_axes")
        text = artist.get_text()
        check(("3 dB" in text or "HPBW" in text) and ("方向性" in text or "directivity" in text),
              "next_"+label+"_core_metrics_present")
        check("缺效率" in text or "efficiency missing" in text,
              "next_"+label+"_gain_not_fabricated")

    try:
        set_workers(2)
        probe = MainWindow(auto_compute=False)
        probe.resize(1440, 900); probe.show(); probe.raise_(); probe.activateWindow()
        watcher = QTimer(probe)

        def dismiss_unexpected_modal():
            modal = QApplication.activeModalWidget()
            if isinstance(modal, QMessageBox):
                warnings.append(modal.text()); modal.accept()

        watcher.timeout.connect(dismiss_unexpected_modal); watcher.start(50)
        settle(120)
        check(set(probe.task_cards) == {"linear", "feed", "planar", "sparse", "subarray"},
              "next_home_contains_five_task_cards")
        check(probe.active_task() == "home" and probe.subarray_panel.report is None
              and probe.subarray_panel.thread is None, "next_home_has_no_subarray_auto_run")
        QTest.keyClick(probe, Qt.Key.Key_F9); settle()
        check(probe.active_task() == "subarray" and probe.inspector_stack.currentIndex() == 3,
              "next_F9_opens_native_subarray_parameters")

        task("linear")
        probe.keep_history.setChecked(True)
        for angle in (24, -16):
            probe.apply_parameters(ArrayParameters(scan_deg=angle)); probe.compute_linear()
        probe.selection.setCurrentIndex(0); probe.plot_type.setCurrentIndex(2); settle()
        check(probe.plot.axes.name == "3d" and probe.plot.current_metrics["target_scan_deg"] == 24,
              "next_linear_3d_uses_selected_history")
        history_path = output / "next-linear-history.json"
        probe.save_history_path(history_path)
        probe.selection.setCurrentIndex(1); probe.load_history_path(history_path); settle()
        check(probe.selection.currentIndex() == 0 and probe.plot.three_dimensional
              and probe.plot.current_metrics["target_scan_deg"] == 24,
              "next_linear_history_restores_selected_3d_snapshot")

        task("feed")
        feed = probe.feed_panel
        feed.generate(); feed.result_tabs.setCurrentIndex(3)
        feed.pattern_view.setCurrentIndex(2); settle()
        check(feed.pattern_plot.axes.name == "3d"
              and feed.pattern_plot.datasets[feed.pattern_plot.selected].feed_report is feed.report,
              "next_feed_3d_uses_saved_actual_excitation")
        feed.pattern_source.setCurrentIndex(0); settle()
        check(feed.pattern_plot.datasets[feed.pattern_plot.selected].pattern_key == "target",
              "next_feed_3d_target_source_selectable")
        feed.pattern_source.setCurrentIndex(2)
        feed_history_path = output / "next-feed-history.json"
        probe.save_history_path(feed_history_path); probe.load_history_path(feed_history_path); settle()
        check(feed.pattern_plot.three_dimensional and feed.pattern_plot.current_metrics["directivity_dbi"] ==
              feed.report["electrical"]["patterns"]["output"]["metrics"]["directivity_dbi"],
              "next_feed_history_restores_matching_3d_metrics")

        task("planar")
        planar = probe.planar_panel
        planar.geometry_fields["nx"].setText("3"); planar.geometry_fields["ny"].setText("3")
        planar.geometry_fields["scan_theta_deg"].setText("20")
        planar.geometry_fields["scan_phi_deg"].setText("30")
        planar.geometry_fields["cut_phi_deg"].setText("30")
        planar.generate(); planar.view.setCurrentIndex(planar.view.findData("cut")); settle()
        check(planar.report is not None and hasattr(planar.figure, "_arraylab_metric_footer"),
              "next_planar_output_has_metric_summary")

        task("subarray")
        panel = probe.subarray_panel
        config = SubarrayConfig(count=4, nx=2, ny=2, aperture_x=4., aperture_y=4.,
                               iterations=20, restarts=2, opt_grid=21, eval_grid=41,
                               theta=20., phi=30.)
        panel.apply_config(config)
        path = output / "next-subarray-parameters.json"
        panel.save_path(path); panel.inputs["count"].setText("5"); panel.open_path(path)
        check(panel.config() == config, "next_subarray_parameter_roundtrip")
        start_time = time.monotonic()
        panel.start()
        check(panel.thread is not None and panel._busy and not probe.compute_button.isEnabled()
              and probe.cancel_button.isEnabled(), "next_subarray_actual_async_start_and_busy_controls")
        wait_idle(panel)
        check(panel.report is not None and not panel._busy and not warnings,
              "next_subarray_actual_worker_completed")
        check(layout_valid(config, panel.report["best"]["centers"]) and
              len(expanded_coordinates(config, panel.report["best"]["centers"])) == 16,
              "next_subarray_result_contains_four_complete_templates")
        details["subarray_seconds"] = time.monotonic()-start_time
        saved_report = panel.report
        saved_json = json.dumps(saved_report, sort_keys=True, allow_nan=False)
        panel.apply_config(replace(config, iterations=20000))
        panel.start()
        check(panel.thread is not None and panel._busy, "next_subarray_cancel_starts_real_worker")
        probe.cancel_current()
        wait_idle(panel)
        check(panel.report is saved_report and json.dumps(panel.report, sort_keys=True, allow_nan=False) == saved_json
              and "取消" in panel.state.text(), "next_subarray_cancel_preserves_previous_result")
        panel.apply_config(config)

        # Redraw after each actual resize so renderer-based caption wrapping uses
        # the same canvas dimensions as the user's visible window.
        for width, height in ((1440, 900), (1100, 740)):
            probe.resize(width, height); settle(80)
            check(probe.width() == width and probe.height() == height,
                  f"next_window_really_uses_{width}x{height}")
            task("linear")
            for index, name in enumerate(("rectangular", "polar", "3d")):
                probe.plot_type.setCurrentIndex(index); probe.redraw(); settle()
                footer_snapshot(probe.plot.canvas, f"linear-{name}-{width}")
            task("feed"); feed.result_tabs.setCurrentIndex(3)
            for index, name in enumerate(("rectangular", "polar", "3d")):
                feed.pattern_view.setCurrentIndex(index); feed.redraw_patterns(); settle()
                footer_snapshot(feed.pattern_plot.canvas, f"feed-{name}-{width}")
            task("planar")
            for name in ("cut", "map", "3d"):
                planar.view.setCurrentIndex(planar.view.findData(name)); planar.redraw(); settle()
                footer_snapshot(planar.canvas, f"planar-{name}-{width}")
            task("subarray")
            for name in ("layout", "pattern", "cuts", "3d"):
                panel.view.setCurrentIndex(panel.view.findData(name)); panel.redraw(); settle()
                footer_snapshot(panel.canvas, f"subarray-{name}-{width}")
            desktop_path = output / f"next-subarray-workbench-{width}.png"
            check(probe.grab().save(str(desktop_path)), f"next_subarray_workbench_{width}_saved")
            details["screenshots"].append(desktop_path.name)

        # Resizing alone must reflow existing summaries; switching views would
        # hide a broken resize handler by constructing a fresh caption.
        for kind in ("linear", "feed", "planar", "subarray"):
            task(kind); probe.resize(1440, 900); settle(80)
            if kind == "linear":
                probe.selected_dataset().label = "线阵缩放验收" * 12
                probe.plot_type.setCurrentIndex(2); probe.redraw(); canvas = probe.plot.canvas
            elif kind == "feed":
                feed.result_tabs.setCurrentIndex(3); feed.pattern_view.setCurrentIndex(2)
                feed.redraw_patterns(); canvas = feed.pattern_plot.canvas
            elif kind == "planar":
                planar.view.setCurrentIndex(planar.view.findData("3d")); planar.redraw(); canvas = planar.canvas
            else:
                panel.view.setCurrentIndex(panel.view.findData("3d")); panel.redraw(); canvas = panel.canvas
            settle()
            previous_artist = canvas.figure._arraylab_metric_footer
            probe.resize(1100, 740); settle(100)
            check(canvas.figure._arraylab_metric_footer is not previous_artist,
                  f"next_{kind}_resize_alone_reflows_existing_footer")
            footer_snapshot(canvas, f"{kind}-resize-only-1100")

        task("linear")
        probe.plot.draw_sets([], -1, view="3d")
        probe.resize(1440, 900); settle(80)
        check(getattr(probe.plot.figure, "_arraylab_metric_footer", None) not in probe.plot.figure.texts,
              "next_resize_does_not_resurrect_cleared_footer")

        check(not warnings, "next_no_unexpected_modal_warnings")
        report["next_stage"] = details
    finally:
        if probe is not None:
            if probe.subarray_panel.thread is not None:
                probe.subarray_panel.cancel()
                wait_idle(probe.subarray_panel)
            probe.close(); probe.deleteLater()
        set_workers(prior_workers)
        app.setQuitOnLastWindowClosed(prior_quit)
        if window is not None:
            window.raise_(); window.activateWindow()
        app.processEvents()
    if retained is not None:
        check(window.datasets is retained[0] and window.feed_panel.report is retained[1]
              and window.planar_panel.report is retained[2] and window.sparse_panel.bundle is retained[3]
              and window.subarray_panel.report is retained[4], "next_isolated_probe_preserves_callers_results")


def main():
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tmp/next-stage-smoke")
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication([]); app.setStyle("Fusion")
    report = {"packaged": bool(getattr(sys, "frozen", False)), "qt_platform": app.platformName(), "checks": []}

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        report["checks"].append(name)

    try:
        exercise_next_stage(app, None, output, report, check)
        report["ok"] = True
    except Exception:
        report["ok"] = False; report["error"] = traceback.format_exc()
    (output / "next-stage-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": report["ok"], "checks": len(report["checks"]), "error": report.get("error")}, ensure_ascii=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
