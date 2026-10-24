from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from arraylab.sparse2d.core import Config, run, field, candidates
from arraylab.sparse2d.headless import compute, jsonable, save_bundle, source_index
from arraylab.sparse_bridge import planar_from_sparse, export_planar_handoff
from arraylab.planar import design_planar, load_planar_configuration
from arraylab.sparse2d.runtime_paths import resource_path, data_directory


@pytest.fixture(scope='module')
def sparse_bundle():
    return run(Config(nx=4,ny=4,k=8,iterations=8,opt_grid=25,eval_grid=81,theta=30,phi=35,
                      reference_ghz=10,frequencies_ghz='9,10,11',phase_bits=3))


def test_bridge_keeps_coordinates_amplitude_and_ideal_array_factor(sparse_bundle):
    row_index=source_index(sparse_bundle,'annealing');row=sparse_bundle['results'][row_index]
    before=jsonable(sparse_bundle)
    for case_index,case in enumerate(sparse_bundle['scenarios']):
        p,s,origin=planar_from_sparse(sparse_bundle,row_index,case_index)
        np.testing.assert_allclose(np.array(p.coordinates),row['xy']*299.792458/10,rtol=0,atol=1e-12)
        np.testing.assert_allclose(s.custom_amplitudes,np.array(row['weights'])/max(row['weights']),rtol=0,atol=1e-12)
        assert s.phase_bits==0 and origin['source_phase_bits']==3
        assert p.frequency_ghz==case['frequency_ghz'] and s.topology=='corporate'
        output=design_planar(p,s)['electrical']['patterns']['output']
        angles=np.deg2rad(output['cut_angles_deg']);phi=np.deg2rad(p.cut_phi_deg)
        uv=np.column_stack([np.sin(angles)*np.cos(phi),np.sin(angles)*np.sin(phi)])
        steering=np.sin(np.deg2rad(case['theta']))*np.array([np.cos(np.deg2rad(case['phi'])),np.sin(np.deg2rad(case['phi']))])
        reference=abs(field(row['xy']*case['frequency_ratio'],row['weights'],uv,steering))
        np.testing.assert_allclose(10**(np.array(output['cut_db'])/20),reference,rtol=0,atol=1e-12)
        assert origin['candidate_indices']==row['indices'].tolist()
    assert jsonable(sparse_bundle)==before


@pytest.mark.parametrize('row,case',[(True,0),(-1,0),(999,0),(0,True),(0,-1),(0,999)])
def test_bridge_indices_rejected(sparse_bundle,row,case):
    with pytest.raises(ValueError):planar_from_sparse(sparse_bundle,row,case)


def test_bridge_rejects_non_point_source_and_damaged_coordinates(sparse_bundle):
    bad=copy.deepcopy(sparse_bundle);bad['results'][0]['xy'][0,0]+=.1
    with pytest.raises(ValueError):planar_from_sparse(bad,0,0)
    patch=run(Config(nx=4,ny=4,k=8,iterations=1,opt_grid=25,eval_grid=81,element_model='patch'))
    with pytest.raises(ValueError,match='单元模型'):planar_from_sparse(patch,0,0)


def test_bridge_never_truncates_or_changes_excessive_layout(sparse_bundle):
    source=copy.deepcopy(sparse_bundle)
    config=Config(nx=20,ny=20,k=8,spacing=2,frequencies_ghz='40')
    integer,xy,_=candidates(config)
    source['config']=asdict(config)
    def select(indices):
        source['results']=[{'name':'checked synthetic boundary','indices':np.array(indices),
                           'xy':xy[indices],'integer':integer[indices],'weights':np.ones(len(indices))/len(indices)}]
    select(list(range(400)))
    with pytest.raises(ValueError,match='200'):planar_from_sparse(source,0,0)
    select([0,1,20,21,378,379,398,399])
    with pytest.raises(ValueError,match='40'):planar_from_sparse(source,0,0)


def test_sparse_batch_retains_success_and_records_failure(tmp_path):
    request={'schema_version':1,'config':{'nx':4,'ny':4,'k':8,'iterations':1,'opt_grid':25,'eval_grid':81}}
    bad={'schema_version':1,'config':{'nx':0}}
    manifest=tmp_path/'batch.json'
    manifest.write_text(json.dumps({'schema_version':1,'jobs':[{'id':'bad','request':bad},{'id':'good','request':request}]}))
    r=sparse_command('batch','--manifest',manifest,'--output',tmp_path/'batch')
    assert r.returncode==4,r.stdout+r.stderr
    doc=json.loads(r.stdout);assert [item['status'] for item in doc['jobs']]==['failed','completed']
    assert (tmp_path/'batch/good/summary.json').is_file()
    r=sparse_command('batch','--manifest',manifest,'--output',tmp_path/'stop','--fail-fast')
    assert r.returncode==4 and [item['status'] for item in json.loads(r.stdout)['jobs']]==['failed','skipped']


def test_bridge_export_configuration_and_provenance(sparse_bundle,tmp_path):
    target=tmp_path/'中文 联动';row=source_index(sparse_bundle,'annealing')
    origin=export_planar_handoff(target,sparse_bundle,row,1)
    p,s=load_planar_configuration(target/'planar-config.json')
    assert p.coordinate_unit=='mm' and p.elements==8 and s.phase_bits==0
    assert json.loads((target/'source.json').read_text(encoding='utf-8'))==origin
    assert len((target/'coordinates-mm.csv').read_text().splitlines())==9
    keep=(target/'planar-config.json').read_bytes()
    with pytest.raises(FileExistsError):export_planar_handoff(target,sparse_bundle,row,1)
    assert (target/'planar-config.json').read_bytes()==keep


def sparse_command(*args):
    script='''
import importlib.abc,runpy,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(('PySide','PyQt','arraylab.sparse_widgets','arraylab.sparse2d.gui','arraylab.sparse2d.detail_view','sparse2d_demo')):
   raise AssertionError('Unexpected GUI or independent demo dependency: '+fullname)
sys.meta_path.insert(0,Guard())
sys.argv=['arraylab','--headless','--sparse',*sys.argv[1:]]
runpy.run_module('arraylab',run_name='__main__')
'''
    return subprocess.run([sys.executable,'-c',script,*map(str,args)],cwd=Path(__file__).resolve().parents[1],
        capture_output=True,text=True,encoding='utf-8',timeout=90,
        env={**os.environ,'QT_QPA_PLATFORM':'invalid-platform','MPLBACKEND':'QtAgg','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','PYTHONIOENCODING':'cp936'})


def test_unified_headless_run_weight_evaluate_and_planar(sparse_bundle,tmp_path):
    request={'schema_version':1,'config':{'nx':4,'ny':4,'k':8,'iterations':5,'opt_grid':25,'eval_grid':81,'phase_bits':3},'weighting':[]}
    config=tmp_path/'稀疏 参数.json';config.write_text(json.dumps(request),encoding='utf-8')
    output=tmp_path/'稀疏 结果';r=sparse_command('run','--config',config,'--output',output)
    assert r.returncode==0,r.stdout+r.stderr
    doc=json.loads(r.stdout);assert doc['ok'] and len(doc['results'])>=6
    bundle=json.loads((output/'summary.json').read_text(encoding='utf-8'))
    assert bundle['integration']['source_engine']=='Sparse2D 1.3.0'
    direct=compute(request)
    for a,b in zip(bundle['results'],direct['results']):
        assert a['indices']==b['indices'].tolist()
        np.testing.assert_allclose(a['weights'],b['weights'],rtol=0,atol=1e-12)
        assert a['metrics']==b['metrics']
    row=source_index(direct,'annealing')
    r=sparse_command('planar','--input',output,'--index',row,'--case-index',0,'--output',tmp_path/'handoff')
    assert r.returncode==0,r.stdout+r.stderr
    assert json.loads(r.stdout)['source']['notes'][0].startswith('保持选中方案')
    assert load_planar_configuration(tmp_path/'handoff/planar-config.json')[0].elements==8
    settings=tmp_path/'weight.json';settings.write_text('{"method":"gaussian","dr_db":10,"sigma":0.5}')
    weighted=tmp_path/'weighted';r=sparse_command('weight','--input',output,'--source','annealing','--settings',settings,'--output',weighted)
    assert r.returncode==0,r.stdout+r.stderr
    r=sparse_command('evaluate','--input',weighted,'--phase-bits',2,'--output',tmp_path/'evaluated','--report')
    assert r.returncode==0,r.stdout+r.stderr
    assert (tmp_path/'evaluated/report.html').exists()
    keep=(output/'summary.json').read_bytes()
    assert sparse_command('run','--config',config,'--output',output).returncode==3
    assert (output/'summary.json').read_bytes()==keep
    assert sparse_command('planar','--input',output,'--index',999,'--output',tmp_path/'bad').returncode==2


def test_resources_and_writable_directory_are_separate(monkeypatch,tmp_path):
    assert resource_path('MANUAL.md').is_file() and resource_path('RESEARCH.md').is_file()
    assert resource_path('examples/element_grid_multifrequency.csv').is_file()
    monkeypatch.setenv('ARRAYLAB_SPARSE_DATA_DIR',str(tmp_path/'writable'))
    assert data_directory()==tmp_path/'writable'
    assert data_directory()!=resource_path('')
