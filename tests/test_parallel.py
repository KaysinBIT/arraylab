import copy
import json
import threading
import time

import numpy as np
import pytest
from threadpoolctl import threadpool_info,threadpool_limits

from arraylab import parallel
from arraylab.core import ArrayParameters
from arraylab.feed import FeedSettings
from arraylab.planar import PlanarParameters, design_planar
from arraylab.study import StudySettings, run_study, StudyCancelled
from arraylab.sparse2d.core import Config, run
from arraylab.sparse2d.headless import jsonable, reevaluate
from arraylab.sparse2d.weighting import add_weighting, Weighting
from test_cli import run_cli


def no_workers():
    assert not [t.name for t in threading.enumerate() if t.name.startswith('ArrayLab-calc')]


def numerical(value):
    value=copy.deepcopy(jsonable(value))
    def remove(item):
        if isinstance(item,dict):return {k:remove(v) for k,v in item.items() if k not in ('execution','seconds','headless','runtime','integration')}
        if isinstance(item,list):return [remove(v) for v in item]
        return item
    return remove(value)


def test_auto_bounds_and_small_job(monkeypatch):
    monkeypatch.setattr(parallel.os,'process_cpu_count',lambda:16,raising=False)
    assert parallel.execution_plan(30,1,0)['workers']==1
    assert parallel.execution_plan(30,10**8,0)['workers']==4
    assert parallel.execution_plan(3,10**8,8)['workers']==3
    monkeypatch.setattr(parallel.os,'process_cpu_count',lambda:1)
    assert parallel.execution_plan(30,10**8,8)['workers']==1


@pytest.mark.parametrize('value',[-1,9,True,2.5,'4'])
def test_invalid_resource_limit(value):
    with pytest.raises(ValueError):parallel.requested_workers(value)


@pytest.fixture
def three_blas_threads():
    with threadpool_limits(limits=3,user_api='blas'):
        yield


def test_order_progress_caller_blas_and_cleanup(three_blas_threads):
    caller=threading.get_ident();progress=[];seen=[]
    before={x['prefix']:x['num_threads'] for x in threadpool_info() if x['user_api']=='blas'}
    assert before and all(count==3 for count in before.values())
    def task(i):
        seen.append(threading.get_ident())
        assert all(x['num_threads']==1 for x in threadpool_info() if x['user_api']=='blas')
        time.sleep(.002*(8-i));return int(np.dot(np.arange(i+1),np.arange(i+1)))
    rows,meta=parallel.ordered_map(task,range(8),workers=3,progress=lambda done,total:progress.append((threading.get_ident(),done,total)))
    assert rows==[sum(x*x for x in range(i+1)) for i in range(8)]
    expected=parallel.execution_plan(8,0,3)['workers']
    assert len(set(seen))==expected and meta['workers']==expected
    assert progress==[(caller,i,8) for i in range(1,9)]
    assert before=={x['prefix']:x['num_threads'] for x in threadpool_info() if x['user_api']=='blas'}
    no_workers()


def test_cancel_stops_submission_and_releases_threads():
    stopped=threading.Event();started=[]
    def task(i):started.append(i);time.sleep(.02);return i
    with pytest.raises(StudyCancelled):
        parallel.ordered_map(task,range(50),workers=2,cancelled=stopped.is_set,
                             progress=lambda *_:stopped.set())
    assert len(started)<=2
    no_workers()
    assert parallel.ordered_map(lambda x:x,[1,2],workers=2)[0]==[1,2]


def test_failure_is_not_hidden_and_leaves_no_workers():
    def fail(i):raise ValueError('numerical failure')
    with pytest.raises(ValueError,match='numerical failure'):
        parallel.ordered_map(fail,range(20),workers=4)
    no_workers()


def test_overlapping_jobs_share_a_bounded_budget():
    active=0;maximum=0;guard=threading.Lock();errors=[]
    def task(i):
        nonlocal active,maximum
        with guard:active+=1;maximum=max(active,maximum)
        time.sleep(.01)
        with guard:active-=1
        return i
    def coordinator():
        try:assert parallel.ordered_map(task,range(6),workers=2)[0]==list(range(6))
        except Exception as exc:errors.append(exc)
    jobs=[threading.Thread(target=coordinator) for _ in range(2)]
    for t in jobs:t.start()
    for t in jobs:t.join(timeout=10)
    assert not errors and not any(t.is_alive() for t in jobs) and maximum==parallel.execution_plan(6,0,2)['workers']
    no_workers()


@pytest.mark.parametrize('p', [ArrayParameters(elements=12,step_deg=2),PlanarParameters(nx=3,ny=4,step_deg=2,scan_phi_deg=27)])
def test_scan_serial_parallel_exact_results_seed_and_order(p):
    s=FeedSettings(topology='series',phase_bits=3,amplitude_error_db=.5,phase_error_deg=2,seed=43,failed_ports=(2,))
    study=StudySettings(count=7,start_deg=0,stop_deg=55)
    serial=run_study(p,s,study,workers=1);multi=run_study(p,s,study,workers=4)
    assert numerical(serial)==numerical(multi)
    assert multi['execution']['workers']==parallel.execution_plan(7,0,4)['workers']
    assert [r['index'] for r in multi['records']]==list(range(1,8))
    no_workers()


@pytest.mark.parametrize('geometry',['rectangular','arbitrary'])
def test_planar_serial_parallel_exact(geometry):
    p=PlanarParameters(geometry=geometry,nx=2,ny=2,scan_theta_deg=34,scan_phi_deg=-40,step_deg=2,
        coordinates=((-1,-.2),(-.3,.4),(.3,-.4),(1,.2)) if geometry=='arbitrary' else ())
    s=FeedSettings(topology='symmetric_series',phase_bits=3)
    assert numerical(design_planar(p,s,workers=1))==numerical(design_planar(p,s,workers=3))
    no_workers()


def test_sparse_serial_parallel_quantized_and_cached_element():
    c=Config(nx=6,ny=6,k=8,iterations=12,opt_grid=25,eval_grid=81,phase_bits=3,
             scan_angles='0,0;25,30',element_model='dipole',element_dipole_axis='x')
    serial=run(c,workers=1);multi=run(c,workers=4)
    assert numerical(serial)==numerical(multi)
    assert multi['execution']['workers']==parallel.execution_plan(len(multi['results']),0,4)['workers']
    restored=reevaluate(serial,workers=3)
    assert numerical(restored)==numerical(serial)
    weighted=add_weighting(serial,0,Weighting(method='uniform'),workers=4)
    assert weighted['execution']['workers']==1 and weighted['execution']['scope']=='sparse_weighted_validation'
    assert len(serial['results'])+1==len(weighted['results'])
    no_workers()


def test_cancel_before_calculation_keeps_inputs():
    with pytest.raises(StudyCancelled):run_study(ArrayParameters(),FeedSettings(),cancelled=lambda:True,workers=4)
    with pytest.raises(StudyCancelled):run(Config(),cancelled=lambda:True,workers=4)
    no_workers()


def test_cli_resource_limits_and_serial_option():
    for value in ('-1','9'):
        result=run_cli('--planar','--workers',value)
        assert result.returncode==2 and json.loads(result.stderr)['error']['type']=='input'
    result=run_cli('--planar','--nx','2','--ny','2','--workers','1')
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['planar']['execution']['backend']=='serial'
    result=run_cli('--study-config','examples/scan-linear.json','--study-count','3','--workers','2')
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['study']['execution']['workers']==parallel.execution_plan(3,0,2)['workers']
