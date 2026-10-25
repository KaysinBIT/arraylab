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
from arraylab.features.sparse.engine.core import Config
from arraylab.features.sparse.engine.headless import dumps, load_bundle, normalize_request, save_bundle
from arraylab.features.sparse.engine.resources import ensure_resources


_POLL_SECONDS = .05
_TAIL_BYTES = 8192


@contextmanager
def _temporary_directory():
    temporary = tempfile.TemporaryDirectory(prefix='arraylab-sparse-')
    try:
        yield temporary.name
    finally:
        # Windows can report process termination shortly before releasing its
        # redirected file handles. Retry only this task's owned directory.
        deadline = time.monotonic() + 1
        while True:
            try:
                temporary.cleanup()
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(.02)


class _WindowsJob:
    """A noninherited Job handle kills its child even if the GUI process crashes."""
    def __init__(self, memory_limit_bytes=None):
        self.handle = None
        if os.name != 'nt':
            return
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [('PerProcessUserTimeLimit', ctypes.c_longlong),
                        ('PerJobUserTimeLimit', ctypes.c_longlong),
                        ('LimitFlags', wintypes.DWORD),
                        ('MinimumWorkingSetSize', ctypes.c_size_t),
                        ('MaximumWorkingSetSize', ctypes.c_size_t),
                        ('ActiveProcessLimit', wintypes.DWORD),
                        ('Affinity', ctypes.c_size_t),
                        ('PriorityClass', wintypes.DWORD),
                        ('SchedulingClass', wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(key, ctypes.c_ulonglong) for key in
                        ('ReadOperationCount', 'WriteOperationCount', 'OtherOperationCount',
                         'ReadTransferCount', 'WriteTransferCount', 'OtherTransferCount')]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [('BasicLimitInformation', BasicLimits), ('IoInfo', IoCounters),
                        ('ProcessMemoryLimit', ctypes.c_size_t), ('JobMemoryLimit', ctypes.c_size_t),
                        ('PeakProcessMemoryUsed', ctypes.c_size_t), ('PeakJobMemoryUsed', ctypes.c_size_t)]

        class Accounting(ctypes.Structure):
            _fields_ = [('TotalUserTime', ctypes.c_longlong), ('TotalKernelTime', ctypes.c_longlong),
                        ('ThisPeriodTotalUserTime', ctypes.c_longlong), ('ThisPeriodTotalKernelTime', ctypes.c_longlong),
                        ('TotalPageFaultCount', wintypes.DWORD), ('TotalProcesses', wintypes.DWORD),
                        ('ActiveProcesses', wintypes.DWORD), ('TotalTerminatedProcesses', wintypes.DWORD)]

        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype = wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.api.SetInformationJobObject.restype = wintypes.BOOL
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype = wintypes.BOOL
        self.api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.TerminateJobObject.restype = wintypes.BOOL
        self.api.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                     wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        self.api.QueryInformationJobObject.restype = wintypes.BOOL
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.accounting_type = Accounting
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if memory_limit_bytes is not None:
            # Bound aggregate committed memory, including venv launcher children.
            limits.BasicLimitInformation.LimitFlags |= 0x0200  # JOB_OBJECT_LIMIT_JOB_MEMORY
            limits.JobMemoryLimit = int(memory_limit_bytes)
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def attach(self, process):
        if self.handle is None:
            return
        import ctypes
        # AssignProcessToJobObject requires PROCESS_SET_QUOTA and PROCESS_TERMINATE.
        child = self.api.OpenProcess(0x0100 | 0x0001, False, process.pid)
        if not child:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if not self.api.AssignProcessToJobObject(self.handle, child):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            self.api.CloseHandle(child)

    def close(self):
        if self.handle is not None:
            import ctypes
            try:
                # Windows venv launchers can have a second Python process. Wait
                # for every job member to release its files before deleting them.
                if not self.api.TerminateJobObject(self.handle, 1):
                    raise ctypes.WinError(ctypes.get_last_error())
                deadline = time.monotonic() + 3
                while True:
                    info = self.accounting_type()
                    if not self.api.QueryInformationJobObject(self.handle, 1, ctypes.byref(info), ctypes.sizeof(info), None):
                        raise ctypes.WinError(ctypes.get_last_error())
                    if info.ActiveProcesses == 0:
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError('稀疏计算子进程未能及时退出。')
                    time.sleep(.01)
            finally:
                self.api.CloseHandle(self.handle)
                self.handle = None


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


def _resume_child(process):
    """Resume only after Job assignment, before a venv launcher can create descendants."""
    if os.name != 'nt':
        return
    import ctypes
    from ctypes import wintypes

    class ThreadEntry(ctypes.Structure):
        _fields_ = [('dwSize', wintypes.DWORD), ('cntUsage', wintypes.DWORD),
                    ('th32ThreadID', wintypes.DWORD), ('th32OwnerProcessID', wintypes.DWORD),
                    ('tpBasePri', wintypes.LONG), ('tpDeltaPri', wintypes.LONG), ('dwFlags', wintypes.DWORD)]

    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    api.Thread32First.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
    api.Thread32First.restype = wintypes.BOOL
    api.Thread32Next.argtypes = [wintypes.HANDLE, ctypes.POINTER(ThreadEntry)]
    api.Thread32Next.restype = wintypes.BOOL
    api.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenThread.restype = wintypes.HANDLE
    api.ResumeThread.argtypes = [wintypes.HANDLE]
    api.ResumeThread.restype = wintypes.DWORD
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    snapshot = api.CreateToolhelp32Snapshot(0x00000004, 0)  # TH32CS_SNAPTHREAD
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = ThreadEntry()
        entry.dwSize = ctypes.sizeof(entry)
        exists = api.Thread32First(snapshot, ctypes.byref(entry))
        while exists:
            if entry.th32OwnerProcessID == process.pid:
                handle = api.OpenThread(0x0002, False, entry.th32ThreadID)  # THREAD_SUSPEND_RESUME
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    if api.ResumeThread(handle) == 0xffffffff:
                        raise ctypes.WinError(ctypes.get_last_error())
                    return
                finally:
                    api.CloseHandle(handle)
            exists = api.Thread32Next(snapshot, ctypes.byref(entry))
        raise RuntimeError('无法启动稀疏计算进程的主线程。')
    finally:
        api.CloseHandle(snapshot)


def _stop(process):
    """Reap the owned child before temporary files or the worker are released."""
    if process.poll() is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    else:
        process.wait()


class _ProgressReader:
    def __init__(self, stream, callback):
        self.stream, self.callback, self.pending = stream, callback, b''

    def read(self):
        # File redirection cannot block the child when the GUI is busy. Bound each poll.
        chunk = self.stream.read(262144)
        if not chunk:
            return False
        lines = (self.pending + chunk).split(b'\n')
        self.pending = lines.pop()[-65536:]
        for line in lines:
            if self.callback is None:
                continue
            try:
                item = json.loads(line.decode('utf-8'))
            except (ValueError, UnicodeError):
                continue
            if (isinstance(item, dict) and item.get('schema_version') == 1
                    and item.get('event') == 'progress' and isinstance(item.get('message'), str)):
                self.callback(item['message'])
        return True


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
