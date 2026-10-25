from pathlib import Path as _RootPath
_PROJECT_ROOT = next(p for p in _RootPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from arraylab.features.mimo.model import MIMOParameters, calculate_mimo
from arraylab.features.mimo.io import save_mimo_configuration


ROOT = _PROJECT_ROOT


@pytest.fixture
def config(tmp_path):
    path = tmp_path/'MIMO 参数.json'
    save_mimo_configuration(path, MIMOParameters(map_size=81, step_deg=2))
    return path


def invoke(args, tmp_path, *, block_qt=False):
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', MPLCONFIGDIR=str(tmp_path/'mpl'), MPLBACKEND='QtAgg',
               QT_QPA_PLATFORM='invalid-test-platform', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if block_qt:
        bootstrap = '''import importlib.abc,sys,runpy
class BlockQt(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(('PySide','PyQt','tkinter')):raise RuntimeError('GUI import blocked: '+fullname)
sys.meta_path.insert(0,BlockQt())
sys.argv=['arraylab.mimo_cli']+sys.argv[1:]
runpy.run_module('arraylab.mimo_cli',run_name='__main__')
'''
        command = [sys.executable, '-c', bootstrap, *map(str,args)]
    else: command = [sys.executable, '-m', 'arraylab.mimo_cli', *map(str,args)]
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', timeout=45)


def test_headless_configuration_matches_core_and_preserves_channels(config, tmp_path):
    result = invoke(['--mimo-config',config,'--workers','1'], tmp_path, block_qt=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)['mimo']
    expected = calculate_mimo(MIMOParameters(map_size=81,step_deg=2),workers=1)
    assert actual == json.loads(json.dumps(expected))
    assert actual['counts']['tx_rx_channels'] == 8


def test_png_export_forces_agg_without_qt(config, tmp_path):
    out = tmp_path/'中文 PNG'
    result = invoke(['--config',config,'--output-dir',out,'--plot','all'], tmp_path, block_qt=True)
    assert result.returncode == 0, result.stderr
    assert len(list(out.glob('*.png'))) == 3
    assert all(path.stat().st_size > 1000 for path in out.glob('*.png'))
    document = json.loads(result.stdout)
    assert json.loads((out/'summary.json').read_text(encoding='utf-8')) == document['mimo']


@pytest.mark.parametrize('arguments', [('--unknown',),('--plot','all'),('--workers','9'),
                                     ('--workers','not-int'),('--plot','3d')])
def test_invalid_flags_return_json_input_error(arguments, tmp_path):
    result = invoke(arguments, tmp_path)
    assert result.returncode == 2 and not result.stdout
    error = json.loads(result.stderr)
    assert error['error']['type'] == 'input' and error['exit_code'] == 2


def test_mutually_exclusive_configuration_arguments(config, tmp_path):
    result = invoke(['--config',config,'--mimo-config',config],tmp_path)
    assert result.returncode == 2
    assert json.loads(result.stderr)['error']['type'] == 'input'


def test_existing_output_returns_three_and_keeps_bytes(config,tmp_path):
    out=tmp_path/'existing';out.mkdir();sentinel=out/'keep.txt';sentinel.write_bytes(b'keep')
    result=invoke(['--config',config,'--output-dir',out],tmp_path)
    assert result.returncode==3 and not result.stdout
    assert json.loads(result.stderr)['error']['type']=='output_exists'
    assert sentinel.read_bytes()==b'keep' and list(out.iterdir())==[sentinel]


def test_help_and_version_need_no_qt(tmp_path):
    for option in ('--help','--version'):
        result=invoke([option],tmp_path,block_qt=True)
        assert result.returncode==0 and not result.stderr
        assert 'ArrayLab' in result.stdout
