"""Named desktop acceptance scopes, with explicit coverage and group timings.

The small sparse-view scope is for iteration on the linked sparse plots. It
does not replace the default complete desktop/packaged acceptance suite.
"""
import json
from pathlib import Path
import sys
import time
import traceback


SUITES = ('all', 'sparse-view', 'mimo', 'waveform')
SCOPES = {
    'all': 'Complete desktop workflow, numerical handoffs, history, layout and responsiveness acceptance.',
    'sparse-view': 'Focused sparse overview, navigation and automatic-history acceptance; excludes other workbenches and large-array performance.',
    'mimo': 'Focused radar MIMO calculation, array handoff, records and view acceptance; excludes other complete workbench workflows.',
    'waveform': 'Only new waveform/A-B/distance-recovery and directly affected task/record entrypoints; no unrelated calculations.',
}


def new_report(qt_platform, suite):
    if suite not in SUITES:
        raise ValueError('Unknown desktop acceptance suite: ' + str(suite))
    return {'packaged': bool(getattr(sys, 'frozen', False)), 'qt_platform': qt_platform,
            'suite': suite, 'scope': SCOPES[suite], 'full_desktop_suite': suite == 'all',
            'checks': [], 'groups': []}


class SmokeGroup:
    """Keep partial results/timing even when a group raises an assertion."""
    def __init__(self, report, name):
        self.report = report
        self.started = time.perf_counter()
        self.initial_checks = len(report['checks'])
        self.entry = {'name': name, 'success': False, 'checks_passed': 0}
        report['groups'].append(self.entry)

    def finish(self, success):
        self.entry.update(success=bool(success),
                          checks_passed=len(self.report['checks']) - self.initial_checks,
                          elapsed_seconds=round(time.perf_counter() - self.started, 6))


def run_group(report, name, callback):
    group = SmokeGroup(report, name)
    try:
        result = callback()
    except BaseException:
        group.finish(False)
        raise
    group.finish(True)
    return result


def schedule_sparse_view_test(app, window, output):
    schedule_feature_test(app, window, output, 'sparse-view')


def schedule_mimo_test(app, window, output):
    schedule_feature_test(app, window, output, 'mimo')


def schedule_waveform_test(app, window, output):
    schedule_feature_test(app, window, output, 'waveform')


def schedule_feature_test(app, window, output, suite):
    """Reuse real Qt exercises with isolated history, without full-suite setup."""
    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtTest import QTest

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    def run():
        report = new_report(app.platformName(), suite)
        started = time.perf_counter()
        code = 1

        def check(condition, name):
            if not condition:
                raise AssertionError(name)
            report['checks'].append(name)

        def click(widget):
            QTest.mouseClick(widget, Qt.MouseButton.LeftButton)
            app.processEvents()

        try:
            if suite == 'waveform':
                from .waveform_smoke import exercise_waveform
                run_group(report, 'waveform', lambda: exercise_waveform(app, window, output, report, check))
            elif suite == 'mimo':
                from .mimo_smoke import exercise_mimo
                run_group(report, 'mimo', lambda: exercise_mimo(app, window, output, report, check))
            else:
                from .sparse_overview_smoke import exercise
                run_group(report, 'sparse-overview',
                          lambda: exercise(app, window, output, check, click, report))
                from .sparse_navigation_smoke import exercise_sparse_navigation
                run_group(report, 'sparse-navigation',
                          lambda: exercise_sparse_navigation(app, output, report, check))
                from .sparse_history_smoke import exercise_sparse_history
                run_group(report, 'sparse-history',
                          lambda: exercise_sparse_history(app, output, report, check))
            report['success'] = True
            code = 0
        except Exception:
            report['success'] = False
            report['error'] = traceback.format_exc()
        finally:
            report['elapsed_seconds'] = round(time.perf_counter() - started, 6)
            (output / 'smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            app.exit(code)

    QTimer.singleShot(300, run)
