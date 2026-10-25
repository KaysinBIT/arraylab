"""Source/frozen acceptance of bounded numerical concurrency, UI and performance."""
import copy
import gc
import hashlib
import json
import os
import threading
import time

from arraylab.features.linear.model import ArrayParameters
from arraylab.features.feed.model import FeedSettings
from arraylab.features.planar.model import PlanarParameters, design_planar
from arraylab.features.studies.model import StudySettings, run_study
from arraylab.features.sparse.engine.core import Config, run
from arraylab.features.sparse.engine.headless import jsonable


def numerical(value):
    if isinstance(value,dict):return {key:numerical(item) for key,item in value.items() if key not in ('execution','seconds')}
    if isinstance(value,list):return [numerical(item) for item in value]
    return value


def performance_cases():
    """Record end-to-end compute time; no promised speedup threshold."""
    cases={
        'scan_200_elements_121_points':lambda workers:run_study(ArrayParameters(elements=200,step_deg=.05),
            FeedSettings(topology='symmetric_series',phase_bits=4),StudySettings(count=121,start_deg=-60,stop_deg=60),workers=workers),
        'sparse_64_candidates_two_cases':lambda workers:run(Config(nx=8,ny=8,k=16,iterations=20,opt_grid=25,
            eval_grid=121,scan_angles='0,0;30,40',phase_bits=3),workers=workers),
    }
    measurements=[]
    for name,function in cases.items():
        reference=None;rows=[]
        for workers in (1,4):
            gc.collect();samples=[];stop=threading.Event()
            def sample():
                while not stop.is_set():
                    value=working_set_mib()
                    if value is not None:samples.append(value)
                    stop.wait(.02)
            baseline=working_set_mib();sampler=threading.Thread(target=sample,name='ArrayLab-memory-sample');sampler.start()
            cpu=time.process_time();start=time.perf_counter()
            try:result=function(workers)
            finally:
                elapsed=time.perf_counter()-start;cpu=time.process_time()-cpu;stop.set();sampler.join()
            actual=hashlib.sha256(json.dumps(numerical(jsonable(result)),sort_keys=True,allow_nan=False).encode()).hexdigest()
            if reference is None:reference=actual
            else:assert actual==reference,name+' numerical mismatch'
            rows.append({'requested_workers':workers,'actual_workers':result['execution']['workers'],
                         'wall_seconds':elapsed,'cpu_seconds':cpu,'cpu_over_wall':cpu/elapsed,
                         'working_set_before_mib':baseline,'sampled_peak_working_set_mib':max(samples) if samples else None})
            del result
            assert not any(t.name.startswith('ArrayLab-calc') for t in threading.enumerate())
        measurements.append({'case':name,'measurements':rows,'speedup':rows[0]['wall_seconds']/rows[1]['wall_seconds'],
                             'numerically_identical':True})
    return measurements


def working_set_mib():
    """Windows process RSS sampled by the acceptance helper, not a memory guarantee."""
    if os.name!='nt':return None
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD),
                  *[(name,ctypes.c_size_t) for name in ('peak','working','peak_paged','paged','peak_nonpaged','nonpaged','pagefile','peak_pagefile')]]
    data=Counters();data.cb=ctypes.sizeof(data)
    get_process=ctypes.windll.kernel32.GetCurrentProcess;get_process.restype=wintypes.HANDLE
    get_memory=ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];get_memory.restype=wintypes.BOOL
    return data.working/1024**2 if get_memory(get_process(),ctypes.byref(data),data.cb) else None


def exercise_multicore(app, window, output, report, check):
    from PySide6.QtCore import QEventLoop, QTimer, Qt
    from PySide6.QtTest import QTest
    from arraylab.features.studies.desktop import ScanStudyDialog
    from arraylab.framework.parallel import requested_workers, ordered_map
    from threadpoolctl import threadpool_info,threadpool_limits
    with threadpool_limits(limits=3,user_api='blas'):
        native,_=ordered_map(lambda _: [(p['prefix'],p['num_threads']) for p in threadpool_info() if p['user_api']=='blas'],range(3),workers=3)
        restored=[p['num_threads'] for p in threadpool_info() if p['user_api']=='blas']
    check(all(row and all(count==1 for _,count in row) for row in native),'runtime_native_BLAS_pools_limited_inside_parallel_workers')
    check(restored and all(count==3 for count in restored),'runtime_native_BLAS_limits_restored_after_parallel_job')
    selected=window.planar_panel.report
    window.worker_choice.setCurrentIndex(window.worker_choice.findData(2))
    check(requested_workers()==2 and window.planar_panel.report is selected,
          'resource_selector_changes_execution_only_not_design_snapshot')
    source=design_planar(PlanarParameters(nx=2,ny=2),workers=2)
    dialog=ScanStudyDialog(source,window);dialog.count.setValue(7);dialog.show();app.processEvents()
    ticks=[];heartbeat=QTimer();heartbeat.setInterval(5);heartbeat.timeout.connect(lambda:ticks.append(1))
    def wait_finished():
        loop=QEventLoop();poll=QTimer();poll.setInterval(10)
        poll.timeout.connect(lambda:loop.quit() if dialog.worker is None else None)
        timeout=QTimer();timeout.setSingleShot(True);timeout.timeout.connect(loop.quit)
        poll.start();timeout.start(30000);loop.exec();poll.stop();timeout.stop()
        check(dialog.worker is None,'parallel_Qt_scan_worker_finishes_and_joins')
    try:
        QTest.mouseClick(dialog.run_button,Qt.MouseButton.LeftButton);wait_finished()
        check(dialog.report is not None and dialog.report['execution']['workers']==2,'Qt_scan_uses_captured_resource_limit')
        dialog.save_path(output/'parallel-study.json')
        report['parallel_study_result']=dialog.report
        previous=dialog.report
        dialog.parameters=ArrayParameters(elements=200,step_deg=.05);dialog.feed_settings=FeedSettings()
        dialog.count.setValue(121);dialog.update_source_label();heartbeat.start()
        QTest.mouseClick(dialog.run_button,Qt.MouseButton.LeftButton)
        # Change preference after launch: this job retains its captured value.
        window.worker_choice.setCurrentIndex(window.worker_choice.findData(1))
        QTimer.singleShot(40,dialog.cancel);wait_finished();heartbeat.stop()
        check(dialog.report is previous and '取消' in dialog.status.text(),'parallel_cancel_keeps_previous_result')
        check(len(ticks)>=2,'Qt_heartbeat_continues_during_parallel_scan_and_cancellation')
        check(not any(t.name.startswith('ArrayLab-calc') for t in threading.enumerate()),'cancelled_parallel_job_leaves_no_worker_threads')
    finally:
        heartbeat.stop();dialog.close();dialog.deleteLater()
        window.worker_choice.setCurrentIndex(window.worker_choice.findData(0))
    report['multicore_performance']=performance_cases()
    check(all(case['numerically_identical'] for case in report['multicore_performance']),
          'frozen_or_source_serial_parallel_representative_results_identical')
