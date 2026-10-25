"""Layout regression checks without calculations or caller-window mutations.

Use the Windows Qt platform for Windows acceptance: offscreen font metrics do
not reproduce the original ribbon overlap. Other platforms are recorded as
geometry checks only, never as native Windows evidence.
"""
import json
from pathlib import Path
import sys
import traceback

from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QAbstractSpinBox, QCheckBox, QComboBox, QFrame, QLabel,
    QLineEdit, QPushButton, QScrollArea, QToolButton, QWidget,
)

from .window import MainWindow


_CONTROLS = (QLabel, QLineEdit, QComboBox, QPushButton, QCheckBox, QAbstractSpinBox)
_TASKS = ("home", "linear", "feed", "planar", "sparse", "subarray", "mimo", "waveform")


def _name(widget):
    text = (widget.text() if hasattr(widget, "text") else
            widget.currentText() if isinstance(widget, QComboBox) else "")
    return f"{type(widget).__name__}:{widget.objectName()}:{text[:70]}"


def _contents_issues(container):
    """Check application controls, not intentional viewport clipping.

    A scroll viewport clips children by design, and tabs hide inactive pages.
    Only visible controls sharing the same direct parent are compared; nested
    widgets (for example a spinbox editor) are not sibling overlaps.
    """
    controls = [item for item in container.findChildren(QWidget)
                if isinstance(item, _CONTROLS) and item.isVisible()]
    issues = []
    for index, item in enumerate(controls):
        parent = item.parentWidget()
        if not parent.rect().contains(item.geometry()):
            issues.append({"kind": "outside_parent", "control": _name(item)})
        if (isinstance(item, QLabel) and item.wordWrap() and item.text()
                and item.heightForWidth(item.width()) > item.height() + 2):
            issues.append({"kind": "wrapped_text_height", "control": _name(item)})
        for other in controls[index + 1:]:
            if parent is not other.parentWidget():
                continue
            overlap = item.geometry().intersected(other.geometry())
            if overlap.width() > 1 and overlap.height() > 1:
                issues.append({"kind": "sibling_overlap", "control": _name(item),
                               "other": _name(other)})
    return issues


def _caller_state(window):
    if window is None:
        return None
    # Identity checks cover immutable result snapshots; UI values cover task
    # context and selection. No result copies or calculations are needed.
    return (
        id(window.datasets), id(window.feed_panel.history), id(window.feed_panel.report),
        id(window.planar_panel.report), id(window.sparse_panel.bundle),
        id(window.subarray_panel.report), window.sequence,
        window.active_task(), window.tabs.currentIndex(), window.ribbon.currentIndex(),
        window.selection.currentIndex(), window.feed_panel.history_selection.currentIndex(),
        tuple((key, value.text()) for key, value in window.inputs.items()),
        window.size().width(), window.size().height(),
    )


def exercise_layout(app, output, check, report, window=None):
    """Check every task layout in an isolated, hidden native Qt window."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    caller = _caller_state(window)
    previous_quit = app.quitOnLastWindowClosed()
    app.setQuitOnLastWindowClosed(False)
    probe = None
    details = {
        "qt_platform": app.platformName(),
        "native_windows": app.platformName() == "windows",
        "scope": "Uncomputed widget geometry; no business plots or calculations",
        "sizes": [],
    }
    report["layout"] = details

    def settle():
        # Layout requests are queued after visibility and tab changes. This
        # drains them without starting a long nested event loop or a worker.
        for _ in range(5):
            app.processEvents()

    try:
        probe = MainWindow(auto_compute=False)
        probe.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        probe.show()
        settle()
        details["device_pixel_ratio"] = probe.devicePixelRatioF()
        details["logical_dpi"] = probe.logicalDpiX()
        initial_task = probe.active_task()
        parameter_tabs = {
            "feed": probe.feed_panel.settings_tabs,
            "planar": probe.planar_panel.settings_tabs,
            "sparse": probe.sparse_panel.settings_tabs,
            "subarray": probe.subarray_panel.parameter_widget,
            "mimo": probe.mimo_panel.parameter_widget,
            "waveform": probe.waveform_panel.parameter_widget,
        }
        tab_indices = {key: tabs.currentIndex() for key, tabs in parameter_tabs.items()}
        canvases = {
            "linear": probe.plot.canvas, "feed": probe.feed_panel.pattern_plot.canvas,
            "planar": probe.planar_panel.canvas, "sparse": probe.sparse_panel.canvas,
            "subarray": probe.subarray_panel.canvas,
            "mimo": probe.mimo_panel.canvas,
            "waveform": probe.waveform_panel.canvas,
        }

        for width, height in ((1440, 900), (1100, 740), (1100, 640)):
            probe.resize(width, height)
            settle()
            row = {"requested": [width, height], "actual": [probe.width(), probe.height()],
                   "ribbon": [], "parameters": [], "canvases": []}
            details["sizes"].append(row)
            check(row["actual"] == row["requested"], f"layout_actual_window_{width}x{height}")
            for kind in _TASKS:
                probe.activate_task(kind)
                settle()
                ribbon_issues = []
                visible_groups = 0
                for page in range(probe.ribbon.count()):
                    probe.ribbon.setCurrentIndex(page)
                    settle()
                    for group in probe.ribbon.findChildren(QFrame, "ribbonGroup"):
                        if not group.isVisible():
                            continue
                        visible_groups += 1
                        caption = group.findChild(QLabel, "ribbonCaption")
                        for button in group.findChildren(QToolButton):
                            if button.isVisible() and button.geometry().intersects(caption.geometry()):
                                ribbon_issues.append({"button": _name(button), "caption": caption.text()})
                row["ribbon"].append({"task": kind, "groups": visible_groups, "issues": ribbon_issues})
                check(visible_groups > 0 and not ribbon_issues,
                      f"layout_ribbon_no_caption_overlap_{kind}_{width}x{height}")
                probe.ribbon.setCurrentIndex(0)
                settle()

                if kind == "home":
                    home_issues = _contents_issues(probe.start_page.widget())
                    row["home_issues"] = home_issues
                    check(len(probe.task_cards) == len(_TASKS)-1 and not home_issues,
                          f"layout_home_cards_and_text_fit_{width}x{height}")
                    continue

                tabs = parameter_tabs.get(kind)
                for index in range(tabs.count() if tabs is not None else 1):
                    if tabs is not None:
                        tabs.setCurrentIndex(index)
                    settle()
                    active = probe.inspector_stack.currentWidget()
                    scrolls = ([active] if isinstance(active, QScrollArea) else [])
                    scrolls += active.findChildren(QScrollArea)
                    scrolls = [scroll for scroll in scrolls if scroll.isVisible()]
                    parameter = {"task": kind, "tab": index, "scrolls": [], "issues": []}
                    if not scrolls:
                        parameter["issues"].append({"kind": "missing_scroll_container"})
                    for scroll in scrolls:
                        bar = scroll.verticalScrollBar()
                        bar.setValue(bar.maximum())
                        settle()
                        issues = _contents_issues(scroll.widget())
                        leaves = [item for item in scroll.widget().findChildren(QWidget)
                                  if isinstance(item, _CONTROLS) and item.isVisible()]
                        bottom = max((item.mapTo(scroll.viewport(), QPoint(0, item.height())).y()
                                      for item in leaves), default=0)
                        state = {"vertical_max": bar.maximum(), "vertical_value": bar.value(),
                                 "horizontal_max": scroll.horizontalScrollBar().maximum(),
                                 "last_control_bottom": bottom, "viewport_height": scroll.viewport().height()}
                        parameter["scrolls"].append(state)
                        parameter["issues"].extend(issues)
                        if (state["horizontal_max"] != 0 or bar.value() != bar.maximum()
                                or bottom > scroll.viewport().height() + 2):
                            parameter["issues"].append({"kind": "bottom_unreachable_or_horizontal_scroll"})
                    row["parameters"].append(parameter)

                if kind == "feed":
                    probe.feed_panel.result_tabs.setCurrentIndex(3)
                settle()
                canvas = canvases[kind]
                result_scroll = probe.results_scroll
                result_scroll.verticalScrollBar().setValue(result_scroll.verticalScrollBar().maximum())
                settle()
                row["canvases"].append({
                    "task": kind, "visible": canvas.isVisible(),
                    "size": [canvas.width(), canvas.height()],
                    "result_height": result_scroll.widget().height(),
                    "viewport_height": result_scroll.viewport().height(),
                    "vertical_max": result_scroll.verticalScrollBar().maximum(),
                    "vertical_value": result_scroll.verticalScrollBar().value(),
                })
            check(all(not item["issues"] for item in row["parameters"]),
                  f"layout_all_parameter_tabs_bottom_reachable_{width}x{height}")
            check(len(row["canvases"]) == len(canvases) and all(
                item["visible"] and item["size"][0] >= 480 and item["size"][1] >= 360
                for item in row["canvases"]), f"layout_all_canvases_keep_readable_minimum_{width}x{height}")
            if height == 640:
                check(all(item["vertical_max"] > 0
                          and item["vertical_value"] == item["vertical_max"]
                          and item["result_height"] > item["viewport_height"]
                          for item in row["canvases"]), "layout_short_window_scrolls_results_instead_of_squeezing_plots")

        # An inactive sparse page can have a much taller minimum than the
        # current task. It must not enlarge the linear or planar result page.
        # This changes widget constraints only; no sparse synthesis is run.
        probe.resize(1100, 740)
        isolation = {"baseline": {}, "hidden_tall_sparse": {}, "restored": {}}
        details["inactive_page_height_isolation"] = isolation

        def result_geometry(kind):
            probe.activate_task(kind)
            settle()
            canvas = canvases[kind]
            scroll = probe.results_scroll
            return {"canvas_height": canvas.height(), "result_height": scroll.widget().height(),
                    "viewport_height": scroll.viewport().height(),
                    "vertical_max": scroll.verticalScrollBar().maximum()}

        for kind in ("linear", "planar"):
            isolation["baseline"][kind] = result_geometry(kind)
        sparse_canvas = canvases["sparse"]
        original_minimum = sparse_canvas.minimumSize()
        try:
            sparse_canvas.setMinimumHeight(1800)
            for kind in ("linear", "planar"):
                isolation["hidden_tall_sparse"][kind] = result_geometry(kind)
            check(all(isolation["hidden_tall_sparse"][kind] == isolation["baseline"][kind]
                      for kind in ("linear", "planar")),
                  "layout_inactive_sparse_height_does_not_enlarge_linear_or_planar")
            sparse_geometry = result_geometry("sparse")
            scroll = probe.results_scroll
            bar = scroll.verticalScrollBar()
            bar.setValue(bar.maximum())
            settle()
            sparse_geometry["vertical_value"] = bar.value()
            sparse_geometry["canvas_bottom_at_scroll_end"] = sparse_canvas.mapTo(
                scroll.viewport(), QPoint(0, sparse_canvas.height())).y()
            isolation["active_tall_sparse"] = sparse_geometry
            check(sparse_geometry["canvas_height"] >= 1800 and bar.maximum() > 0
                  and bar.value() == bar.maximum()
                  and sparse_geometry["canvas_bottom_at_scroll_end"] <= scroll.viewport().height() + 2,
                  "layout_active_tall_sparse_keeps_full_canvas_and_scrolls_to_bottom")
        finally:
            sparse_canvas.setMinimumSize(original_minimum)
            settle()
        for kind in ("linear", "planar"):
            isolation["restored"][kind] = result_geometry(kind)
        check(sparse_canvas.minimumSize() == original_minimum
              and isolation["restored"] == isolation["baseline"],
              "layout_restoring_sparse_minimum_preserves_other_task_sizes")

        for kind, tabs in parameter_tabs.items():
            tabs.setCurrentIndex(tab_indices[kind])
        probe.activate_task(initial_task)
        settle()
        check(probe.active_task() == initial_task and all(
            tabs.currentIndex() == tab_indices[kind] for kind, tabs in parameter_tabs.items()),
            "layout_probe_task_and_parameter_tabs_restored")
        check(not probe.datasets and not probe.feed_panel.history and probe.feed_panel.report is None
              and probe.planar_panel.report is None and probe.sparse_panel.bundle is None
              and probe.subarray_panel.report is None and probe.sparse_panel.thread is None
              and probe.subarray_panel.thread is None, "layout_probe_never_starts_business_calculations")
    finally:
        if probe is not None:
            # Stop resize/interaction/progress timers before deferred deletion so
            # this probe cannot redraw or consume later smoke exercises' events.
            for timer in probe.findChildren(QTimer):
                timer.stop()
            probe.close()
            probe.deleteLater()
            QCoreApplication.sendPostedEvents(probe, QEvent.Type.DeferredDelete)
        app.setQuitOnLastWindowClosed(previous_quit)
        app.processEvents()
    check(_caller_state(window) == caller, "layout_preserves_caller_task_inputs_and_result_snapshots")
    (output / "layout-details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    """Standalone driver; platform and DPI are reported, never overridden."""
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tmp/layout-smoke")
    output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setStyle("Fusion")
    report = {"packaged": bool(getattr(sys, "frozen", False)), "qt_platform": app.platformName(), "checks": []}

    def check(condition, name):
        if not condition:
            raise AssertionError(name)
        report["checks"].append(name)

    code = 0
    try:
        exercise_layout(app, output, check, report)
        report["success"] = True
    except Exception:
        report["success"] = False
        report["error"] = traceback.format_exc()
        code = 1
    (output / "layout-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success": report["success"], "checks": len(report["checks"]),
                      "qt_platform": app.platformName(), "error": report.get("error")}, ensure_ascii=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
