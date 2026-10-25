"""Real CLI child lifetime, cancellation and numerical parity for the desktop supervisor."""
from pathlib import Path as _RootPath
_PROJECT_ROOT = next(p for p in _RootPath(__file__).resolve().parents if (p / "pyproject.toml").is_file())
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

import arraylab.features.sparse.process as sparse_process
from arraylab.framework.parallel import ComputationCancelled
from arraylab.features.sparse.engine.core import Config, run
from arraylab.features.sparse.engine.elements import write_example_csv
from arraylab.features.sparse.engine.headless import jsonable, source_index
from arraylab.features.sparse.engine.weighting import Weighting, add_weighting


@pytest.fixture(scope='module')
def small_config():
    return Config(nx=4, ny=4, k=8, iterations=5, opt_grid=25, eval_grid=81, theta=20, phi=35)


@pytest.fixture(scope='module')
def direct_bundle(small_config):
    return run(small_config, workers=1)


def _compare_rows(actual, expected):
    assert len(actual['results']) == len(expected['results'])
    assert actual['config'] == expected['config']
    for left, right in zip(actual['results'], expected['results']):
        assert left['name'] == right['name']
        np.testing.assert_array_equal(left['indices'], right['indices'])
        np.testing.assert_allclose(left['weights'], right['weights'], rtol=0, atol=1e-12)
        assert left['metrics'] == right['metrics']


def test_real_cli_result_and_progress_equal_direct(small_config, direct_bundle):
    progress = []
    result = sparse_process.run_sparse_process(small_config, workers=1, progress=progress.append)
    _compare_rows(result, direct_bundle)
    assert len(progress) > 2
    assert progress[-1] == '读取并校验计算结果'
    assert result['headless']['operation'] == 'run'


def test_real_cli_weight_keeps_source_and_matches_direct(direct_bundle):
    index = source_index(direct_bundle, 'annealing')
    settings = Weighting(method='gaussian', sigma=.5)
    before = jsonable(direct_bundle)
    expected = add_weighting(direct_bundle, index, settings, workers=1)
    actual = sparse_process.run_sparse_process(weighting=(direct_bundle, index, settings), workers=1)
    _compare_rows(actual, expected)
    assert jsonable(direct_bundle) == before
    assert actual['headless']['operation'] == 'weight'


def test_relative_import_path_uses_caller_cwd_and_retains_snapshot(monkeypatch, tmp_path):
    folder = tmp_path / '中文 相对路径'
    folder.mkdir()
    write_example_csv(folder / 'element.csv')
    monkeypatch.chdir(folder)
    config = Config(nx=4, ny=4, k=8, iterations=1, opt_grid=25, eval_grid=81,
                    element_model='imported', element_file='element.csv')
    result = sparse_process.run_sparse_process(config, workers=1)
    assert result['config']['element_file'] == 'element.csv'
    assert result['element_pattern']['model'] == 'imported'
    (folder / 'element.csv').unlink()
    weighted = sparse_process.run_sparse_process(weighting=(result, 0, Weighting(method='uniform')), workers=1)
    assert weighted['config']['element_file'] == 'element.csv'
    assert weighted['element_pattern'] == result['element_pattern']


def _record_children(monkeypatch):
    original = sparse_process.subprocess.Popen
    children = []

    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append((child, args, kwargs))
        return child

    monkeypatch.setattr(sparse_process.subprocess, 'Popen', launch)
    return children


def test_cancel_during_real_child_progress_reaps_and_removes_private_files(monkeypatch, tmp_path):
    children = _record_children(monkeypatch)
    monkeypatch.setattr(sparse_process.tempfile, 'tempdir', str(tmp_path))
    cancel = [False]
    cancel_time = [None]

    def progress(message):
        if cancel_time[0] is None:
            cancel_time[0] = time.monotonic()
        cancel[0] = True

    with pytest.raises(ComputationCancelled):
        sparse_process.run_sparse_process(Config(nx=20, ny=20), progress=progress, cancelled=lambda: cancel[0])
    assert cancel_time[0] is not None and time.monotonic() - cancel_time[0] < 2
    assert len(children) == 1 and children[0][0].poll() is not None
    assert not list(tmp_path.glob('arraylab-sparse-*'))
    assert children[0][2]['close_fds'] is True
    if os.name == 'nt':
        assert children[0][2]['creationflags'] & subprocess.CREATE_NO_WINDOW


def test_pre_cancel_never_starts_child(monkeypatch, small_config):
    children = _record_children(monkeypatch)
    with pytest.raises(ComputationCancelled):
        sparse_process.run_sparse_process(small_config, cancelled=lambda: True)
    assert children == []


def test_progress_callback_failure_still_reaps_child(monkeypatch):
    children = _record_children(monkeypatch)

    def fail(message):
        raise RuntimeError('observer failure')

    with pytest.raises(RuntimeError, match='observer failure'):
        sparse_process.run_sparse_process(Config(nx=20, ny=20), progress=fail)
    assert len(children) == 1 and children[0][0].poll() is not None


def test_child_error_message_has_bounded_tail_and_no_pipe_deadlock(monkeypatch, small_config):
    script = "import sys;sys.stderr.write('x'*300000);sys.stderr.write(' tail-marker');sys.exit(7)"
    monkeypatch.setattr(sparse_process, '_cli_command', lambda: ([sys.executable, '-c', script], Path.cwd()))
    with pytest.raises(RuntimeError, match='退出码 7') as caught:
        sparse_process.run_sparse_process(small_config)
    assert 'tail-marker' in str(caught.value)
    assert len(str(caught.value)) < 8300


def test_success_exit_without_valid_bundle_is_rejected_and_child_reaped(monkeypatch, small_config):
    children = _record_children(monkeypatch)
    monkeypatch.setattr(sparse_process, '_cli_command', lambda: ([sys.executable, '-c', 'pass'], Path.cwd()))
    with pytest.raises(FileNotFoundError):
        sparse_process.run_sparse_process(small_config)
    assert children[0][0].poll() == 0


@pytest.mark.parametrize('reported_version', [None, '0.0.0-mismatched'])
def test_child_result_with_missing_or_different_version_is_rejected(monkeypatch, small_config, reported_version):
    original_load = sparse_process.load_bundle

    def altered_version(path):
        result = original_load(path)
        if reported_version is None:
            result.pop('integration', None)
        else:
            result['integration']['arraylab_version'] = reported_version
        return result

    monkeypatch.setattr(sparse_process, 'load_bundle', altered_version)
    with pytest.raises(RuntimeError, match='完整解压同一版本'):
        sparse_process.run_sparse_process(small_config, workers=1)


def test_frozen_missing_cli_never_relaunches_gui(monkeypatch, tmp_path):
    monkeypatch.setattr(sparse_process.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sparse_process.sys, 'executable', str(tmp_path / 'ArrayLab.exe'))
    with pytest.raises(RuntimeError, match='ArrayLab-cli.exe 缺失'):
        sparse_process._cli_command()
    (tmp_path / 'ArrayLab-cli.exe').touch()
    command, cwd = sparse_process._cli_command()
    assert command == [str(tmp_path / 'ArrayLab-cli.exe'), '--headless', '--sparse']
    assert cwd == tmp_path


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object behavior')
def test_parent_process_death_kills_child_job(tmp_path):
    # This separate supervisor is killed deliberately. Keep an OS child handle to
    # verify process termination without confusing a later PID reuse.
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    state = tmp_path / 'child.json'
    script = '''
import json,sys,time,subprocess
from pathlib import Path
from arraylab.sparse_process import _WindowsJob
job=_WindowsJob()
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],creationflags=subprocess.CREATE_NO_WINDOW)
job.attach(child)
Path(sys.argv[1]).write_text(json.dumps({'pid':child.pid}))
time.sleep(60)
'''
    parent = subprocess.Popen([sys.executable, '-c', script, str(state)],
                              cwd=_PROJECT_ROOT, creationflags=subprocess.CREATE_NO_WINDOW)
    child_handle = None
    try:
        deadline = time.monotonic() + 15
        while not state.exists() and time.monotonic() < deadline and parent.poll() is None:
            time.sleep(.05)
        assert state.exists(), 'test supervisor did not launch its child'
        child_handle = api.OpenProcess(0x00100000, False, json.loads(state.read_text())['pid'])
        assert child_handle
        assert api.WaitForSingleObject(child_handle, 0) == 0x102  # WAIT_TIMEOUT: still running
        parent.kill()
        parent.wait(timeout=5)
        assert api.WaitForSingleObject(child_handle, 3000) == 0  # WAIT_OBJECT_0: exited
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=5)
        if child_handle:
            api.CloseHandle(child_handle)


@pytest.mark.skipif(os.name != 'nt', reason='Windows Job Object memory bound')
def test_job_memory_budget_rejects_oversized_native_allocation():
    job = sparse_process._WindowsJob(memory_limit_bytes=64 * 1024**2)
    script = "import sys\ntry: data=bytearray(128*1024**2)\nexcept MemoryError: sys.exit(23)\nsys.exit(0)"
    child = subprocess.Popen([sys.executable, '-c', script],
        creationflags=subprocess.CREATE_NO_WINDOW | 0x00000004)
    try:
        job.attach(child)
        sparse_process._resume_child(child)
        assert child.wait(timeout=10) == 23
    finally:
        sparse_process._stop(child)
        job.close()
