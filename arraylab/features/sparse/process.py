"""Run sparse jobs through the matching CLI, keeping native solvers off the GUI process."""
from dataclasses import asdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from arraylab import __version__
from arraylab.framework.parallel import check_cancelled, requested_workers
from arraylab.framework.process import (temporary_directory as _temporary_directory, WindowsJob as _WindowsJob,
    resume_child as _resume_child, stop_child as _stop, ProgressReader as _ProgressReader)
from arraylab.features.sparse.engine.core import Config
from arraylab.features.sparse.engine.headless import dumps, load_bundle, normalize_request, save_bundle
from arraylab.features.sparse.engine.resources import ensure_resources


_POLL_SECONDS = .05
_TAIL_BYTES = 8192






def _cli_command():
    """Never use a frozen desktop executable as a Python interpreter."""
    if getattr(sys, 'frozen', False):
        executable = Path(sys.executable).resolve().with_name('ArrayLab-cli.exe')
        if not executable.is_file():
            raise RuntimeError('稀疏计算程序 ArrayLab-cli.exe 缺失，请完整解压 ArrayLab 便携包。')
        return [str(executable), '--headless', '--sparse'], executable.parent
    from arraylab.framework.resources import source_root
    root = source_root()
    return [sys.executable, '-m', 'arraylab', '--headless', '--sparse'], root








def _tail(path):
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - _TAIL_BYTES))
        return stream.read(_TAIL_BYTES).decode('utf-8', errors='replace').strip()


def _failure(process, stdout_path, stderr_path):
    output = _tail(stdout_path)
    try:
        document = json.loads(output)
        detail = document.get('error', {}).get('message', '') if isinstance(document, dict) else ''
    except ValueError:
        detail = ''
    detail = detail or _tail(stderr_path) or output or '无错误信息'
    return RuntimeError(f'稀疏计算子进程失败（退出码 {process.returncode}）：{detail[-_TAIL_BYTES:]}')


def run_sparse_process(config=None, weighting=None, *, workers=0, progress=None, cancelled=None):
    """Return only a completed, validated bundle; cancellation never publishes partial output.

    ``weighting`` is the same (bundle, row index, Weighting) tuple accepted by
    SparseWorker. All files belong to one TemporaryDirectory, never a user output folder.
    """
    workers = requested_workers(workers)
    check_cancelled(cancelled)
    command, cwd = _cli_command()
    if config is not None and weighting is not None:
        raise ValueError('综合参数与稀疏后加权任务不能同时指定。')
    task_config = (Config() if config is None else config) if weighting is None else Config(**weighting[0]['config'])
    task_config.validate()
    operation = 'run' if weighting is None else ('weight' if weighting[2].method == 'minimax' else 'evaluate')
    resources = ensure_resources(task_config, workers=workers if weighting is None else 1, operation=operation)
    with _temporary_directory() as temporary:
        folder = Path(temporary)
        output = folder / 'result'
        if weighting is None:
            config = Config() if config is None else config
            original_element_file = config.element_file
            request = normalize_request({'schema_version': 1, 'config': asdict(config), 'weighting': []}, Path.cwd())
            input_file = folder / 'config.json'
            input_file.write_text(dumps(request) + '\n', encoding='utf-8')
            arguments = ['run', '--config', str(input_file)]
        else:
            bundle, index, settings = weighting
            if type(index) is not int or not 0 <= index < len(bundle['results']):
                raise ValueError('请选择一项已计算的布局。')
            settings.validate(Config(**bundle['config']))
            original_element_file = bundle['config']['element_file']
            source = folder / 'source'
            save_bundle(bundle, source, report=False)
            settings_file = folder / 'weighting.json'
            settings_file.write_text(dumps(asdict(settings)) + '\n', encoding='utf-8')
            arguments = ['weight', '--input', str(source), '--index', str(index), '--settings', str(settings_file)]
        arguments += ['--output', str(output), '--workers', str(workers)]
        check_cancelled(cancelled)
        env = os.environ.copy()
        env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        if getattr(sys, 'frozen', False):
            env['PYINSTALLER_RESET_ENVIRONMENT'] = '1'
        else:
            env['PYTHONPATH'] = str(cwd) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        stdout_path, stderr_path = folder / 'stdout.log', folder / 'stderr.log'
        process = None
        job = _WindowsJob(memory_limit_bytes=resources['budget_bytes'])
        try:
            with stdout_path.open('wb') as stdout, stderr_path.open('wb') as stderr:
                process = subprocess.Popen(command + arguments, cwd=cwd, env=env,
                    stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                    creationflags=(subprocess.CREATE_NO_WINDOW | 0x00000004) if os.name == 'nt' else 0,
                    close_fds=True)
                job.attach(process)
                _resume_child(process)
                with stderr_path.open('rb') as progress_file:
                    reader = _ProgressReader(progress_file, progress)
                    while process.poll() is None:
                        check_cancelled(cancelled)
                        reader.read()
                        time.sleep(_POLL_SECONDS)
                    check_cancelled(cancelled)
                    while reader.read():
                        check_cancelled(cancelled)
            if process.returncode != 0:
                raise _failure(process, stdout_path, stderr_path)
            if progress:
                progress('读取并校验计算结果')
            check_cancelled(cancelled)
            result = load_bundle(output)
            integration = result.get('integration')
            version = integration.get('arraylab_version') if isinstance(integration, dict) else None
            if version != __version__:
                raise RuntimeError('稀疏计算程序与界面版本不一致或缺少版本信息，请完整解压同一版本的 ArrayLab 便携包。')
            # The child needs absolute imported-element paths; the GUI retains the user's snapshot.
            result['config']['element_file'] = original_element_file
            check_cancelled(cancelled)
            return result
        finally:
            try:
                if process is not None:
                    _stop(process)
            finally:
                job.close()
