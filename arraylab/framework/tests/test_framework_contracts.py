"""Only startup/routing/output contracts changed by structural extraction.

Fake Qt checks below verify orchestration order, not visual GUI acceptance.
The headless isolation check runs in a separate process with Qt blocked.
"""
from pathlib import Path as _RootPath
_PROJECT_ROOT = next(p for p in _RootPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
import builtins
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from arraylab.framework.bootstrap import run
from arraylab.framework.cli_support import Parser, UsageError, emit
from arraylab.framework.registry import CliRoute, dispatch


ROOT = _PROJECT_ROOT


def test_headless_dispatch_precedes_qt_and_desktop_argument_parsing():
    source = r'''
import importlib.abc, json, sys
class NoQt(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('PySide6', 'matplotlib')):
            raise AssertionError('Headless bootstrap imported GUI dependency: ' + fullname)
sys.meta_path.insert(0, NoQt())
from arraylab.framework.bootstrap import run
seen=[]
def headless(argv):
    seen.append(argv)
    return 17
def forbidden(*args, **kwargs):
    raise AssertionError('Desktop factory was called')
arguments=['--headless','--waveform-config','输入.json','--business-only-option']
assert run(arguments,headless_main=headless,window_factory=forbidden,
           icon_factory=forbidden,version='test',smoke_schedulers={}) == 17
assert seen == [arguments]
assert not any(k.startswith(('PySide6','matplotlib')) for k in sys.modules)
print(json.dumps({'success':True,'arguments_preserved':True}))
'''
    env = os.environ.copy()
    result = subprocess.run([sys.executable, '-c', source], cwd=ROOT, env=env,
                            text=True, capture_output=True, encoding='utf-8', timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {'success': True, 'arguments_preserved': True}


@pytest.mark.parametrize('arguments', [
    ['--smoke-suite', 'waveform'],
    ['--smoke-test', 'unused', '--smoke-suite', 'unknown'],
])
def test_invalid_desktop_arguments_exit_before_qt(arguments, monkeypatch):
    imports = []
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.startswith('PySide6'):
            imports.append(name)
            raise AssertionError('Invalid arguments imported Qt')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(SystemExit) as failure:
        run(arguments, headless_main=lambda _: 0, window_factory=lambda **_: None,
            icon_factory=lambda: None, version='test',
            smoke_schedulers={'all': lambda *a: None, 'waveform': lambda *a: None})
    assert failure.value.code == 2
    assert not imports


def test_route_order_is_preserved_and_unselected_factories_stay_lazy():
    events = []
    def load(name):
        events.append('load:' + name)
        return lambda argv: events.append((name, argv)) or 3
    routes = [
        CliRoute('waveform', lambda argv: '--waveform-config' in argv,
                 lambda: load('waveform'), lambda argv: [v for v in argv if v != '--headless']),
        CliRoute('mimo', lambda argv: '--mimo-config' in argv, lambda: load('mimo')),
    ]
    argv = ['--headless', '--mimo-config', 'm.json', '--waveform-config', 'w.json']
    assert dispatch(routes, argv, lambda: load('default')) == 3
    assert events == ['load:waveform', ('waveform', argv[1:])]
    assert argv[0] == '--headless'


def test_fallback_factory_receives_unchanged_arguments():
    seen = []
    routes = [CliRoute('unselected', lambda args: False,
                       lambda: pytest.fail('unselected handler was imported'))]
    assert dispatch(routes, ['--elements', '8'], lambda: lambda args: seen.append(args) or 0) == 0
    assert seen == [['--elements', '8']]


def test_mutating_predicate_cannot_change_later_route_arguments():
    original = ['--config', 'example.json']
    def first(arguments):
        arguments.clear()
        return False
    seen = []
    assert dispatch([CliRoute('probe', first, lambda: None)], original,
                    lambda: lambda args: seen.append(args) or 0) == 0
    assert original == ['--config', 'example.json'] and seen == [original]


def test_parser_raises_existing_usage_error_and_json_stays_ascii_safe():
    parser = Parser(allow_abbrev=False)
    parser.add_argument('--value', type=int)
    with pytest.raises(UsageError):
        parser.parse_args(['--value', 'not-an-int'])
    class Sink(io.StringIO):
        flushes = 0
        def flush(self): self.flushes += 1
    sink = Sink()
    emit({'消息': '完成', 'value': 1.25}, sink)
    output = sink.getvalue()
    assert output.isascii() and output.endswith('\n')
    assert json.loads(output) == {'消息': '完成', 'value': 1.25}
    assert sink.flushes == 1
    emit({'not_serializable': object()}, None)
    with pytest.raises(ValueError): emit({'value': float('nan')}, sink)


@pytest.mark.parametrize('smoke', [False, True])
def test_desktop_bootstrap_preserves_order_settings_and_factories(monkeypatch, tmp_path, smoke):
    import arraylab.framework.bootstrap as bootstrap
    events = []
    icon = object()
    class App:
        @staticmethod
        def setAttribute(attribute, enabled): events.append(('attribute', attribute, enabled))
        def __init__(self, arguments): events.append(('app', arguments)); self.arguments = arguments
        def setApplicationName(self, value): events.append(('name', value))
        def setWindowIcon(self, value): assert value is icon; events.append('icon-installed')
        def setApplicationVersion(self, value): events.append(('version', value))
        def setStyle(self, value): events.append(('style', value))
        def primaryScreen(self):
            return SimpleNamespace(availableGeometry=lambda: SimpleNamespace(width=lambda: 1920, height=lambda: 1080))
        def exec(self): events.append('exec'); return 23
    class Window:
        def resize(self, width, height): events.append(('resize', width, height))
        def show(self): events.append('show')
    window = Window()
    def window_factory(**kwargs): events.append(('window', kwargs)); return window
    def icon_factory(): events.append('icon-created'); return icon
    def scheduler(app, current, path):
        assert isinstance(app, App) and current is window and path == tmp_path
        events.append('schedule')
    monkeypatch.setitem(sys.modules, 'PySide6.QtWidgets', SimpleNamespace(QApplication=App))
    monkeypatch.setitem(sys.modules, 'PySide6.QtCore', SimpleNamespace(Qt=SimpleNamespace(
        ApplicationAttribute=SimpleNamespace(AA_DontUseNativeDialogs='non-native',AA_ShareOpenGLContexts='shared-gl'))))
    monkeypatch.setattr(bootstrap, 'set_windows_application_id', lambda: events.append('windows-id'))
    monkeypatch.setenv('MPLCONFIGDIR', 'keep-this-cache')
    args = ['--smoke-test', str(tmp_path), '--smoke-suite', 'waveform'] if smoke else []
    assert run(args, headless_main=lambda _: pytest.fail('wrong dispatch'), window_factory=window_factory,
               icon_factory=icon_factory, version='unchanged', smoke_schedulers={'all': scheduler, 'waveform': scheduler}) == 23
    assert events[0:2] == ['windows-id', ('attribute', 'non-native', True)]
    assert events.index(('attribute','shared-gl',True)) < next(i for i,e in enumerate(events) if isinstance(e,tuple) and e[0]=='app')
    assert ('name', 'ArrayLab') in events and ('style', 'Fusion') in events and ('version', 'unchanged') in events
    assert events.index('icon-installed') < events.index(('window', {'auto_compute': not smoke}))
    assert ('resize', 1440, 900) in events
    assert ('schedule' in events) is smoke and events[-1] == 'exec'
    assert os.environ['MPLCONFIGDIR'] == 'keep-this-cache'
