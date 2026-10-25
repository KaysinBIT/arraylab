"""Saved sparse runs are lazy immutable snapshots, not new synthesis jobs."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from arraylab import sparse_history_store as history
from arraylab.sparse_history_store import SparseHistoryStore, SparseHistoryError, normalize_view_state
from arraylab.sparse2d.core import Config, run
from arraylab.sparse2d.headless import jsonable
from arraylab.sparse2d.weighting import Weighting, add_weighting


@pytest.fixture(scope='module')
def bundle():
    return run(Config(nx=4, ny=4, k=8, iterations=2, opt_grid=25, eval_grid=81,
                      phase_bits=2, scan_angles='0,0;20,30'), workers=1)


def payload_bytes(path, row_id):
    with sqlite3.connect(path) as connection:
        return connection.execute('SELECT payload FROM sparse_results WHERE id=?', (row_id,)).fetchone()[0]


def test_exact_results_weights_and_metadata_round_trip_without_solver(bundle, tmp_path, monkeypatch):
    original = deepcopy(bundle)
    original['custom_metadata'] = {'description': '完整记录 Ωλ', 'precision': 1.0000000000000002}
    path = tmp_path/'中文 稀疏历史.sqlite3'
    store = SparseHistoryStore(path)
    state = dict(result_index=2, case_index=1, view='cuts', tab='plot', pattern_view='array_factor', phase_view='quantized')
    meta = store.append(original, view_state=state)
    before = jsonable(original)
    def no_solver(*args, **kwargs): pytest.fail('History restoration must not optimize or reevaluate')
    for target in ('arraylab.sparse2d.core.run', 'arraylab.sparse2d.headless.run',
                   'arraylab.sparse2d.headless.reevaluate', 'arraylab.sparse2d.core.evaluate_result',
                   'arraylab.sparse2d.weighting.add_weighting'):
        monkeypatch.setattr(target, no_solver)
    restored = SparseHistoryStore(path).load_record(meta['id'])
    assert jsonable(restored['bundle']) == before == jsonable(original)
    assert restored['view_state'] == state
    for a, b in zip(original['results'], restored['bundle']['results']):
        for key in ('indices', 'integer', 'xy', 'weights'):
            assert isinstance(b[key], np.ndarray)
            np.testing.assert_array_equal(a[key], b[key])


def test_navigation_reads_only_metadata_and_selected_payload_is_lazy(bundle, tmp_path, monkeypatch):
    store = SparseHistoryStore(tmp_path/'lazy.sqlite3')
    ids = [store.append(bundle)['id'] for _ in range(7)]
    store.select(ids[1], {'view': 'coarray', 'tab': 'ports'})
    monkeypatch.setattr(history, 'snapshot_from_bytes', lambda *args: pytest.fail('List must not decode scientific snapshots'))
    rows, selected = SparseHistoryStore(store.path).list_records()
    assert len(rows) == 7 and selected == ids[1]
    assert [r['id'] for r in rows] == ids
    assert all('bundle' not in row and 'payload' not in row for row in rows)
    assert rows[1]['view_state']['tab'] == 'ports'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows logical file-I/O counters')
def test_metadata_listing_does_not_read_large_blob_overflow_pages(bundle, tmp_path):
    store = SparseHistoryStore(tmp_path/'metadata-read-budget.sqlite3')
    entry = store.append(bundle)
    with sqlite3.connect(store.path) as connection:
        # Deliberately invalid synthetic payload: listing must not deserialize it.
        # This checks actual reads, not just the table layout or Python decoder.
        connection.execute('UPDATE sparse_results SET payload=zeroblob(?) WHERE id=?',
                           (8*1024*1024, entry['id']))
    child = subprocess.run([sys.executable, '-c', """
import ctypes,json,sys
from arraylab.sparse_history_store import SparseHistoryStore
class IO(ctypes.Structure):
    _fields_=[(name,ctypes.c_ulonglong) for name in
        ('reads','writes','other','read_bytes','write_bytes','other_bytes')]
api=ctypes.WinDLL('kernel32',use_last_error=True)
api.GetCurrentProcess.restype=ctypes.c_void_p
api.GetProcessIoCounters.argtypes=[ctypes.c_void_p,ctypes.POINTER(IO)]
def counters():
    value=IO()
    if not api.GetProcessIoCounters(api.GetCurrentProcess(),ctypes.byref(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    return value
before=counters(); store=SparseHistoryStore(sys.argv[1])
rows,selected=store.list_records(); after=counters()
print(json.dumps({'rows':len(rows),'selected':selected,
                  'read_bytes':after.read_bytes-before.read_bytes}))
""", str(store.path)], cwd=Path(__file__).resolve().parents[1], capture_output=True,
        text=True, encoding='utf-8', timeout=30, check=True)
    result = json.loads(child.stdout)
    assert result['rows'] == 1 and result['selected'] == entry['id']
    assert result['read_bytes'] < 1024*1024, result


def test_selection_update_never_changes_snapshot_and_can_restore_every_view(bundle, tmp_path):
    store = SparseHistoryStore(tmp_path/'view.sqlite3')
    row = store.append(bundle)
    original = payload_bytes(store.path, row['id'])
    for view in ('overview','layout','pattern','cuts','coarray','cases','weights','quantization','element','comparison','structures'):
        state = normalize_view_state(bundle, {'view':view,'tab':2,'result_index':1,'case_index':1})
        store.update_view_state(row['id'], state)
        assert store.load_record(row['id'])['view_state'] == state
        assert payload_bytes(store.path, row['id']) == original


def test_weighted_run_keeps_old_full_run_and_both_snapshots(bundle, tmp_path):
    weighted = add_weighting(bundle, 2, Weighting(method='gaussian'), workers=1)
    store = SparseHistoryStore(tmp_path/'weight.sqlite3')
    first = store.append(bundle)
    second = store.append(weighted, kind='weight')
    assert first['kind']=='synthesis' and second['kind']=='weight'
    assert second['result_count']==first['result_count']+1
    assert jsonable(store.load_record(first['id'])['bundle']) == jsonable(bundle)
    assert jsonable(store.load_record(second['id'])['bundle']) == jsonable(weighted)
    assert second['view_state']['result_index'] == weighted['last_weighted_index']


@pytest.mark.parametrize('state',[{'result_index':999},{'case_index':-1},{'result_index':True},
    {'view':'unknown'},{'tab':False},{'view':[]},{'phase_view':'random'},{'extra':0}])
def test_bad_view_state_does_not_change_archive(bundle, tmp_path, state):
    store=SparseHistoryStore(tmp_path/'state.sqlite3');entry=store.append(bundle)
    before=store.path.read_bytes()
    with pytest.raises(SparseHistoryError): store.select(entry['id'],state)
    assert store.path.read_bytes()==before


@pytest.mark.parametrize('damage',['nan','missing_metrics','scenario','coordinates','quantization','duplicate_indices'])
def test_incomplete_snapshot_is_rejected_without_touching_existing_data(bundle,tmp_path,damage):
    store=SparseHistoryStore(tmp_path/'bad.sqlite3');store.append(bundle);before=store.path.read_bytes()
    broken=deepcopy(bundle)
    if damage=='nan':broken['results'][0]['metrics']['validation_psl_db']=float('nan')
    elif damage=='missing_metrics':broken['results'][0].pop('scenario_metrics')
    elif damage=='scenario':broken['scenarios'][0]['theta']=50
    elif damage=='coordinates':broken['results'][0]['xy'][0,0]+=.1
    elif damage=='quantization':broken['results'][0]['phase_quantization']['bits']=1
    else:broken['results'][0]['indices'][1]=broken['results'][0]['indices'][0]
    with pytest.raises(SparseHistoryError):store.append(broken)
    assert store.path.read_bytes()==before


def test_unrelated_corrupt_and_future_files_are_never_rebuilt(bundle,tmp_path):
    corrupt=tmp_path/'corrupt.sqlite3';corrupt.write_bytes(b'not a database')
    with pytest.raises(SparseHistoryError):SparseHistoryStore(corrupt)
    assert corrupt.read_bytes()==b'not a database'
    future=tmp_path/'future.sqlite3';store=SparseHistoryStore(future);store.append(bundle)
    with sqlite3.connect(future) as connection:connection.execute('PRAGMA user_version=99')
    before=future.read_bytes()
    with pytest.raises(SparseHistoryError):SparseHistoryStore(future)
    assert future.read_bytes()==before


def test_corrupted_payload_is_not_read_during_listing_and_rejected_on_load(bundle,tmp_path):
    store=SparseHistoryStore(tmp_path/'checksum.sqlite3');entry=store.append(bundle)
    with sqlite3.connect(store.path) as connection:
        connection.execute('UPDATE sparse_results SET payload=? WHERE id=?',(b'{"corrupt":true}',entry['id']))
    before=store.path.read_bytes()
    assert store.list_records()[0][0]['id']==entry['id']
    with pytest.raises(SparseHistoryError,match='摘要'):store.load_record(entry['id'])
    assert store.path.read_bytes()==before


def test_duplicate_json_keys_and_nonfinite_values_are_rejected():
    for raw in (b'{"config":{},"config":{}}',b'{"a":NaN}',b'{"a":Infinity}'):
        with pytest.raises(SparseHistoryError):history.snapshot_from_bytes(raw)


def test_concurrent_append_delete_and_empty_restart_never_reuse_ids(bundle,tmp_path):
    path=tmp_path/'concurrent.sqlite3';SparseHistoryStore(path)
    with ThreadPoolExecutor(max_workers=4) as pool:
        entries=list(pool.map(lambda _:SparseHistoryStore(path).append(bundle),range(12)))
    ids=[entry['id'] for entry in entries]
    assert len(set(ids))==12
    store=SparseHistoryStore(path); rows,_=store.list_records();assert len(rows)==12
    store.remove(ids[2],ids[4]); assert len(store.list_records()[0])==11
    assert store.load_record(ids[4])['bundle']['config']==bundle['config']
    for row in store.list_records()[0]:store.remove(row['id'])
    assert store.list_records()==([],-1)
    assert SparseHistoryStore(path).append(bundle)['id']>max(ids)


def test_failed_commit_rolls_back_payload_and_selected_metadata(bundle,tmp_path,monkeypatch):
    store=SparseHistoryStore(tmp_path/'failure.sqlite3');store.append(bundle)
    before=store.path.read_bytes(); real_connect=store._connect
    class Connection:
        def __init__(self,connection):self.connection=connection
        def __getattr__(self,name):return getattr(self.connection,name)
        def commit(self):raise sqlite3.OperationalError('simulated disk full at commit')
    @contextmanager
    def fail_commit():
        with real_connect() as connection:yield Connection(connection)
    monkeypatch.setattr(store,'_connect',fail_commit)
    with pytest.raises(SparseHistoryError,match='自动保存失败'):store.append(bundle)
    assert store.path.read_bytes()==before


def test_hot_journal_recovers_last_committed_snapshot(bundle,tmp_path):
    store=SparseHistoryStore(tmp_path/'crash.sqlite3');entry=store.append(bundle)
    before=store.list_records()
    child=subprocess.run([sys.executable,'-c',"""
import os,sqlite3,sys
c=sqlite3.connect(sys.argv[1]);c.execute('PRAGMA cache_size=8');c.execute('BEGIN IMMEDIATE')
c.execute('UPDATE sparse_results SET payload=? WHERE id=?',(b'x'*2000000,int(sys.argv[2])))
c.execute("UPDATE history_meta SET value='9999' WHERE key='selected_id'")
os._exit(77)
""",str(store.path),str(entry['id'])],timeout=20)
    assert child.returncode==77 and Path(str(store.path)+'-journal').exists()
    recovered=SparseHistoryStore(store.path)
    assert recovered.list_records()==before
    assert jsonable(recovered.load_record(entry['id'])['bundle'])==jsonable(bundle)


def test_record_budget_does_not_evict_or_truncate(bundle,tmp_path,monkeypatch):
    store=SparseHistoryStore(tmp_path/'budget.sqlite3');store.append(bundle);before=store.path.read_bytes()
    monkeypatch.setattr(history,'MAX_BYTES',100)
    with pytest.raises(SparseHistoryError,match='256 MiB'):store.append(bundle)
    assert store.path.read_bytes()==before


def test_imported_element_is_restored_from_embedded_snapshot(tmp_path,monkeypatch):
    csv=tmp_path/'单元.csv'
    csv.write_text('theta_deg,response_db\n0,0\n30,-2\n60,-8\n90,-30\n',encoding='utf-8')
    bundle=run(Config(nx=4,ny=4,k=8,iterations=1,opt_grid=25,eval_grid=81,
        element_model='imported',element_file=str(csv),element_import_mode='axisymmetric'),workers=1)
    store=SparseHistoryStore(tmp_path/'imported.sqlite3');entry=store.append(bundle)
    csv.rename(tmp_path/'source-no-longer-available.csv')
    monkeypatch.setattr('arraylab.sparse2d.elements._import_csv',lambda *args:pytest.fail('Must use embedded element snapshot'))
    restored=store.load_record(entry['id'])['bundle']
    assert jsonable(restored)==jsonable(bundle)


@pytest.mark.parametrize('changes', [dict(nx=3,ny=3,k=1),
    dict(nx=2,ny=1,k=2,spacing=1,theta=30,phase_bits=1)])
def test_single_element_and_quantization_null_metrics_are_preserved(tmp_path,changes):
    bundle=run(Config(iterations=1,opt_grid=25,eval_grid=81,**changes),workers=1)
    store=SparseHistoryStore(tmp_path/'nullable.sqlite3');entry=store.append(bundle)
    assert jsonable(store.load_record(entry['id'])['bundle'])==jsonable(bundle)
