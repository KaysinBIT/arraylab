"""Only namespace, layer, resource and import contracts affected by relocation."""
import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
from tools.check_architecture import inspect_project


def project(tmp_path, sources):
    for name, source in sources.items():
        path = tmp_path/'arraylab'/name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(source, encoding='utf-8')
    return tmp_path


def test_framework_and_shared_reject_business_dependencies_even_inside_functions(tmp_path):
    root = project(tmp_path, {
        '__init__.py': '',
        'framework/start.py': "def start():\n    from arraylab.features.linear import calculate\n",
        'shared/common.py': "def operation():\n    import arraylab.application.composition\n",
        'features/linear/__init__.py': '', 'application/composition.py': ''})
    report = inspect_project(root)
    assert {r['rule'] for r in report['violations']} == {'framework_depends_on_higher_layer', 'shared_depends_on_business'}
    assert report['success'] is False


def test_public_feature_facade_is_allowed_but_internal_private_and_legacy_paths_are_not(tmp_path):
    root = project(tmp_path, {
        '__init__.py': '', 'core.py': '', 'features/linear/__init__.py': '',
        'features/linear/model.py': '',
        'features/feed/model.py': "from arraylab.features.linear import ArrayParameters\nfrom arraylab.features.linear.model import calculate\nfrom arraylab.features.linear import _helper\nfrom arraylab.core import calculate\n"})
    report = inspect_project(root)
    assert {r['rule'] for r in report['violations']} == {'cross_feature_internal_import', 'cross_feature_private_symbol', 'legacy_import'}
    assert all(r['line'] != 1 for r in report['violations'])


def test_delayed_facade_imports_are_part_of_the_cycle_graph(tmp_path):
    root = project(tmp_path, {
        '__init__.py': '',
        'features/one/__init__.py': "def __getattr__(name):\n    from .model import Value\n    return Value\n",
        'features/one/model.py': 'from arraylab.features.two import Value\n',
        'features/two/__init__.py': "def __getattr__(name):\n    from .model import Value\n    return Value\n",
        'features/two/model.py': 'from arraylab.features.one import Value\n'})
    report = inspect_project(root)
    assert report['cycles'] == [[
        'arraylab.features.one', 'arraylab.features.one.model',
        'arraylab.features.two', 'arraylab.features.two.model']]


def test_module_aliases_cannot_reach_cross_feature_internals_or_legacy_paths(tmp_path):
    root = project(tmp_path, {'__init__.py': '', 'core.py': '',
        'features/linear/__init__.py': '', 'features/linear/model.py': '',
        'features/feed/model.py': "import arraylab.features.linear as linear\nimport arraylab\na=linear.model.calculate\nb=linear._private\nc=arraylab.core.calculate\n"})
    report = inspect_project(root)
    assert {r['rule'] for r in report['violations']} == {
        'cross_feature_internal_import', 'cross_feature_private_symbol', 'legacy_import'}


def test_tests_and_verification_do_not_create_business_layer_violations(tmp_path):
    root = project(tmp_path, {'__init__.py': '',
        'framework/tests/test_start.py': 'from arraylab.features.linear.model import Value\n',
        'features/linear/verification/desktop.py': 'from arraylab.application.desktop.window import MainWindow\n',
        'features/linear/__init__.py': '', 'features/linear/model.py': ''})
    report = inspect_project(root)
    assert report['success'] and report['checked_modules'] == 3


def test_literal_importlib_calls_are_checked_and_dynamic_registry_targets_are_not_guessed(tmp_path):
    root = project(tmp_path, {'__init__.py': '', 'features/linear/__init__.py': '',
        'framework/registry.py': "from importlib import import_module as load\ndef create(target):\n    load('arraylab.features.linear')\n    return load(target)\n"})
    report = inspect_project(root)
    assert report['violations'][0]['rule'] == 'framework_depends_on_higher_layer'
    assert report['unresolved_dynamic_imports'][0]['line'] == 4


def test_current_runtime_dependency_direction_and_cycles():
    report = inspect_project(ROOT)
    assert report['success'], json.dumps({'violations': report['violations'], 'cycles': report['cycles']}, ensure_ascii=False, indent=2)


def compatibility_aliases():
    """Read static wrappers; no test dependency on a temporary migration map."""
    for path in sorted((ROOT/'arraylab').glob('*.py')) + sorted((ROOT/'arraylab/sparse2d').glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        implementation = next((alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                               for alias in node.names if alias.asname == '_implementation'), None)
        redirected = any(isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
                         and node.value.id == '_implementation' and any(
                             isinstance(target, ast.Subscript) and isinstance(target.value, ast.Attribute)
                             and target.value.attr == 'modules' for target in node.targets) for node in ast.walk(tree))
        if implementation and redirected:
            old = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
            yield old, implementation


@pytest.mark.parametrize('old,new', list(compatibility_aliases()), ids=lambda name: name)
def test_old_module_imports_share_the_canonical_module_and_patch_state(old, new, monkeypatch):
    try:
        legacy = importlib.import_module(old); current = importlib.import_module(new)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith(('PySide6', 'matplotlib')):
            pytest.skip('Optional desktop/plotting dependency is not installed')
        raise
    assert legacy is current
    token = object(); monkeypatch.setattr(legacy, '_architecture_probe', token, raising=False)
    assert current._architecture_probe is token
    other = object(); monkeypatch.setattr(current, '_architecture_probe', other)
    assert legacy._architecture_probe is other


def test_split_metric_compatibility_forwards_mutation_to_the_real_owner(monkeypatch):
    old = importlib.import_module('arraylab.pattern_metrics')
    common = importlib.import_module('arraylab.shared.plotting.metrics')
    planar = importlib.import_module('arraylab.features.planar.metrics')
    assert old.planar_summary is planar.planar_summary
    token = lambda *args, **kwargs: None
    monkeypatch.setattr(old, '_wrap_footer_lines', token)
    assert common._wrap_footer_lines is token
    second = lambda *args, **kwargs: ()
    monkeypatch.setattr(common, '_wrap_footer_lines', second)
    assert old._wrap_footer_lines is second
    monkeypatch.setattr(old, 'planar_summary', token)
    assert planar.planar_summary is token


def test_relocated_resources_and_sparse_source_command_keep_their_contract():
    from arraylab.framework.resources import package_root, source_root
    from arraylab.sparse2d.runtime_paths import resource_path
    from arraylab.sparse_process import _cli_command
    assert source_root() == ROOT and package_root() == ROOT/'arraylab'
    assert resource_path('MANUAL.md').is_file()
    assert resource_path('examples/element_grid_multifrequency.csv').is_file()
    command, cwd = _cli_command()
    assert command[0] == sys.executable and command[-3:] == ['arraylab', '--headless', '--sparse']
    assert cwd == ROOT


@pytest.mark.parametrize('module,arguments', [
    ('arraylab', ['--headless', '--help']), ('arraylab.cli', ['--help']),
    ('arraylab.mimo_cli', ['--version']), ('arraylab.waveform_cli', ['--version']),
    ('arraylab.sparse2d.headless', ['--help']), ('arraylab.planar_cli', ['--help']),
    ('arraylab.study_cli', ['--help']), ('arraylab.subarray_cli', ['--help'])])
def test_original_cli_module_entrypoints_still_run_without_gui_imports(module, arguments, tmp_path):
    bootstrap = """import importlib.abc,runpy,sys
class NoQt(importlib.abc.MetaPathFinder):
    def find_spec(self,name,path=None,target=None):
        if name.startswith(('PySide','PyQt','tkinter')):raise AssertionError('Qt import attempted: '+name)
sys.meta_path.insert(0,NoQt())
module=sys.argv[1];sys.argv=[module]+sys.argv[2:]
runpy.run_module(module,run_name='__main__')
"""
    env = {**os.environ, 'PYTHONUTF8': '1', 'QT_QPA_PLATFORM': 'invalid-platform',
           'MPLCONFIGDIR': str(tmp_path/'mpl'), 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    result = subprocess.run([sys.executable, '-c', bootstrap, module, *arguments], cwd=ROOT,
                            env=env, capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
