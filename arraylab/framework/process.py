"""Owned child-process lifetime primitives, shared by numerical features."""
from contextlib import contextmanager
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from .parallel import check_cancelled

@contextmanager
def temporary_directory(prefix='arraylab-sparse-'):
    temporary = tempfile.TemporaryDirectory(prefix=prefix)
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


class WindowsJob:
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


def resume_child(process):
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


def stop_child(process):
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


class ProgressReader:
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


def run_owned(command, *, cwd, environment, folder, memory_limit_bytes=None, progress=None, cancelled=None):
    """Supervise an owned command, with cancellation and crash cleanup.

    Command construction and result decoding belong to the application or
    feature. No concrete business module is imported by this runner.
    """
    folder=Path(folder);stdout_path=folder/'stdout.log';stderr_path=folder/'stderr.log'
    process=None;job=WindowsJob(memory_limit_bytes=memory_limit_bytes)
    try:
        check_cancelled(cancelled)
        with stdout_path.open('wb') as stdout,stderr_path.open('wb') as stderr:
            process=subprocess.Popen(command,cwd=cwd,env=environment,stdin=subprocess.DEVNULL,
                stdout=stdout,stderr=stderr,close_fds=True,
                creationflags=(subprocess.CREATE_NO_WINDOW|0x00000004) if os.name=='nt' else 0)
            job.attach(process);resume_child(process)
            with stderr_path.open('rb') as stream:
                reader=ProgressReader(stream,progress)
                while process.poll() is None:
                    check_cancelled(cancelled);reader.read();time.sleep(.05)
                check_cancelled(cancelled)
                while reader.read():check_cancelled(cancelled)
        if process.returncode:
            def tail(path):
                with path.open('rb') as stream:
                    stream.seek(0,os.SEEK_END);stream.seek(max(0,stream.tell()-8192));return stream.read().decode('utf-8',errors='replace')
            raw=tail(stdout_path)
            try:message=json.loads(raw).get('error',{}).get('message','')
            except (ValueError,AttributeError):message=''
            message=message or tail(stderr_path) or raw or '没有可用错误信息。'
            raise RuntimeError(f'计算子进程退出（{process.returncode}）：{message[-8192:]}')
        return process.returncode
    finally:
        try:
            if process is not None:stop_child(process)
        finally:job.close()
