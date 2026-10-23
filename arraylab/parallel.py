"""Bounded NumPy task parallelism; no GUI, processes, or nested BLAS pools."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import os
import threading

from threadpoolctl import threadpool_limits

_preference = 0
_numerical_lock = threading.Lock()
AUTO_LIMIT = 4
MAX_WORKERS = 8
AUTO_MIN_WORK = 2_000_000


class ComputationCancelled(Exception):
    """The caller must retain its previous successful result."""


def requested_workers(value=None):
    value = _preference if value is None else value
    if type(value) is not int or not 0 <= value <= MAX_WORKERS:
        raise ValueError("计算并行数须为0–8的整数；0为自动，1为单核。")
    return value


def set_workers(value):
    global _preference
    _preference = requested_workers(value)


def check_cancelled(cancelled):
    if cancelled and cancelled():
        raise ComputationCancelled("计算已取消；原结果保留。")


def execution_plan(count, work_units, workers=None):
    requested = requested_workers(workers)
    cpus = max(1, (getattr(os, 'process_cpu_count', os.cpu_count)() or 1))
    limit = requested or min(AUTO_LIMIT, max(1, cpus // 2))
    actual = max(1, min(limit, cpus, count))
    reason = 'requested_limit' if requested else 'auto_bounded'
    if count < 2:
        actual, reason = 1, 'one_task'
    elif requested == 0 and work_units < AUTO_MIN_WORK:
        actual, reason = 1, 'small_workload'
    return {'requested_workers':requested, 'workers':actual, 'task_count':count,
            'backend':'threads' if actual > 1 else 'serial', 'reason':reason,
            'native_blas_threads':1, 'auto_limit':AUTO_LIMIT}


def ordered_map(function, items, *, workers=None, work_units=0, progress=None, cancelled=None):
    """Private outputs per task; progress on caller; at most workers tasks in flight.

    NumPy's native loops release the GIL. Inputs must only be read by tasks.
    The lock coordinates process-wide BLAS settings between overlapping jobs.
    Cancel/error waits for active tasks, cancels pending work, then restores BLAS.
    """
    items = list(items)
    plan = execution_plan(len(items), work_units, workers)
    check_cancelled(cancelled)
    while not _numerical_lock.acquire(timeout=.05):
        check_cancelled(cancelled)
    try:
        with threadpool_limits(limits=1, user_api='blas'):
            if plan['workers'] == 1:
                results=[]
                for item in items:
                    check_cancelled(cancelled)
                    results.append(function(item))
                    if progress: progress(len(results), len(items))
                check_cancelled(cancelled)
                return results, plan
            results=[None]*len(items);pending={};next_index=0;finished=0
            pool=ThreadPoolExecutor(max_workers=plan['workers'],thread_name_prefix='ArrayLab-calc')
            try:
                while next_index < len(items) or pending:
                    check_cancelled(cancelled)
                    while next_index < len(items) and len(pending) < plan['workers']:
                        pending[pool.submit(function,items[next_index])]=next_index;next_index+=1
                    done,_=wait(pending,timeout=.05,return_when=FIRST_COMPLETED)
                    for future in sorted(done,key=lambda f:pending[f]):
                        results[pending.pop(future)]=future.result();finished+=1
                        if progress:progress(finished,len(items))
                check_cancelled(cancelled)
            finally:
                for future in pending:future.cancel()
                pool.shutdown(wait=True,cancel_futures=True)
            return results, plan
    finally:
        _numerical_lock.release()
