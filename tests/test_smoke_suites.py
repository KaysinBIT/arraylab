"""Acceptance-runner routing/claims only; these mocks are not GUI evidence."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from arraylab.smoke_suites import new_report, run_group, schedule_sparse_view_test


ROOT = Path(__file__).resolve().parents[1]


def test_failed_group_preserves_partial_checks_and_failure_status():
    report = new_report('mock', 'all')
    report['checks'].append('earlier_group')
    def partial_failure():
        report['checks'].append('passed_before_failure')
        raise AssertionError('expected failure')
    with pytest.raises(AssertionError, match='expected failure'):
        run_group(report, 'fault', partial_failure)
    group = report['groups'][0]
    assert report['checks'] == ['earlier_group', 'passed_before_failure']
    assert group['success'] is False and group['checks_passed'] == 1
    assert group['elapsed_seconds'] >= 0


@pytest.mark.parametrize('failing_group', [None, 'navigation'])
def test_focused_dispatch_records_real_scope_and_stops_on_failure(tmp_path, monkeypatch, failing_group):
    from PySide6.QtCore import QTimer
    invoked, exits = [], []
    def exercise(name, check):
        invoked.append(name)
        check(True, name + '_checkpoint')
        if name == failing_group:
            raise AssertionError('injected orchestration failure')
    monkeypatch.setitem(sys.modules, 'arraylab.sparse_overview_smoke', SimpleNamespace(
        exercise=lambda app, window, output, check, click, report: exercise('overview', check)))
    monkeypatch.setitem(sys.modules, 'arraylab.sparse_navigation_smoke', SimpleNamespace(
        exercise_sparse_navigation=lambda app, output, report, check: exercise('navigation', check)))
    monkeypatch.setitem(sys.modules, 'arraylab.sparse_history_smoke', SimpleNamespace(
        exercise_sparse_history=lambda app, output, report, check: exercise('history', check)))
    monkeypatch.setattr(QTimer, 'singleShot', lambda delay, callback: callback())
    app = SimpleNamespace(platformName=lambda: 'mock-dispatch', exit=exits.append)
    schedule_sparse_view_test(app, object(), tmp_path)
    report = json.loads((tmp_path / 'smoke.json').read_text(encoding='utf-8'))
    assert report['suite'] == 'sparse-view' and report['full_desktop_suite'] is False
    assert 'excludes' in report['scope'] and report['qt_platform'] == 'mock-dispatch'
    assert report['elapsed_seconds'] >= 0
    assert len(report['checks']) == sum(group['checks_passed'] for group in report['groups'])
    if failing_group:
        assert exits == [1] and invoked == ['overview', 'navigation']
        assert report['success'] is False and report['groups'][-1]['success'] is False
        assert 'injected orchestration failure' in report['error']
    else:
        assert exits == [0] and invoked == ['overview', 'navigation', 'history']
        assert report['success'] is True and all(group['success'] for group in report['groups'])


@pytest.mark.parametrize('command,required', [
    (['-m', 'arraylab', '--smoke-suite', 'sparse-view'], '--smoke-test'),
    (['-m', 'arraylab', '--smoke-test', 'unused', '--smoke-suite', 'unknown'], 'invalid choice'),
    (['tools/check_headless_release.py', 'unused', 'unused.json', '--gui-suite', 'sparse-view'], '--gui-smoke'),
    (['tools/check_headless_release.py', 'unused', 'unused.json', '--gui-suite', 'unknown'], 'invalid choice'),
])
def test_invalid_suite_arguments_fail_before_starting_a_window_or_executable(command, required):
    result = subprocess.run([sys.executable, *command], cwd=ROOT, capture_output=True,
                            text=True, encoding='utf-8', timeout=15)
    assert result.returncode == 2
    assert required in result.stderr and not result.stdout
