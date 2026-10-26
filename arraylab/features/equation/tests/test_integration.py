from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from arraylab.features.equation import service, io, config, statistics, metadata, presentation
from arraylab.shared.simulation_io import read_npz


def small(kind='fmcw',mode='simo',tx=1):
    p=service.default_configuration(kind,10)
    w=p['waveform']
    if kind=='fmcw':
        w.update(fs_hz=1e6,samples=64,range_fft=64,bandwidth_hz=10e6,ramp_s=80e-6,
                 adc_start_s=10e-6,pri_s=100e-6,rx_bandwidth_hz=.5e6,range_max_m=500.)
        distance=4*w['fs_hz']/w['samples']*config.C/(2*w['bandwidth_hz']/w['ramp_s'])
    else:
        w.update(fs_hz=4e6,samples=128,range_fft=128,pulse_width_s=2e-6,bandwidth_hz=1e6,
                 pri_s=40e-6,rx_bandwidth_hz=4e6,recovery_s=.25e-6,range_max_m=config.C*127/(2*4e6))
        distance=24*config.C/(2*w['fs_hz'])
    w.update(pulses=16,doppler_fft=16)
    p['array'].update(mode=mode,tx_count=tx,rx_count=2,tx_positions_m=[],rx_positions_m=[],tx_weights=[],rx_weights=[])
    p['scene']['targets'][0].update(range_m=distance,height_m=1.,velocity_mps=0.)
    p['clutter'].update(range_min_m=distance*.8,range_max_m=distance*1.2)
    p['detection'].update(method='fixed',training=[2,1],guard=[1,0])
    p['experiment'].update(trials=3,cut_range_m=distance,cut_velocity_mps=0.,
        distance_min_m=distance*.9,distance_max_m=distance*1.1,distance_count=2,roc_points=3)
    return service.normalize_configuration(p)


@pytest.fixture(scope='module')
def simulation_result():
    return service.execute('simulation',small())


def test_six_numerical_kernels_keep_frozen_source_hashes():
    root=Path(service.__file__).parent
    manifest=json.loads((root/'data/source.json').read_text(encoding='utf-8'))
    assert len(manifest['kernel_normalized_sha256'])==6
    for name,expected in manifest['kernel_normalized_sha256'].items():
        assert hashlib.sha256((root/name).read_text(encoding='utf-8').encode('utf-8')).hexdigest()==expected


def test_imports_and_budget_do_not_require_qt_or_web_server():
    code="""
import sys
from arraylab.features.equation import calculate_budget,default_configuration
from arraylab.features.equation import service
assert calculate_budget(default_configuration())['valid']
assert callable(calculate_budget) and callable(service.budget)
assert not any(n.startswith(('PySide6','fastapi','uvicorn','pydantic')) for n in sys.modules)
"""
    root=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
    process=subprocess.run([sys.executable,'-c',code],cwd=root,env={**os.environ,'QT_QPA_PLATFORM':'invalid-test-platform'},capture_output=True,timeout=20)
    assert process.returncode==0,process.stderr


def test_presets_and_metadata_cover_all_editable_domains_without_mutation():
    entries=service.presets();assert len(entries)==14
    original=deepcopy(entries)
    for row in entries:
        assert {'id','name','configuration'}==set(row)
        assert service.validate_configuration(row['configuration'])['valid']
    assert entries==original and len({row['id'] for row in entries})==14
    assert set(service.default_configuration())-{'schema_version','name'}==set(metadata.GROUPS)
    assert set(k for k,_ in metadata.ACTIONS)==set(service.ACTIONS)


@pytest.mark.parametrize('kind',['fmcw','rect','lfm'])
def test_budget_input_and_inverse_remain_source_contract(kind):
    p=small(kind);before=deepcopy(p);r,arrays=service.execute('budget',p)
    assert p==before and arrays=={} and r['kind']=='budget' and r['valid']
    assert r['config_hash']==config.config_hash(p)
    assert r['inverse']['required_power_w']>0
    assert r['source']['version']=='1.0.0' and r['algorithm_version']=='1.0.0'


def test_unreachable_budget_retains_kind_and_can_be_exported(tmp_path):
    p=small();p['array']['rx_weights']=[[1.,0.],[-1.,0.]]
    result,arrays=service.execute('budget',p)
    assert result['kind']=='budget' and result['valid'] is False and result['config_valid'] is True
    assert result['inverse'] is None
    output=io.export_result(tmp_path/'unreachable-budget',result,arrays)
    assert '不可达或未定义' in (output/'report.html').read_text(encoding='utf-8')


@pytest.mark.parametrize('action',['simulation','snr','calibration','statistics','distance','roc'])
def test_all_source_experiments_work_through_native_service(action):
    p=small();before=deepcopy(p);progress=[]
    r,a=service.execute(action,p,progress=lambda fraction,message:progress.append((fraction,message)))
    assert p==before and r['config']==p and r['config_hash']==config.config_hash(p)
    assert r['kind']==action and r['budget']['kind']=='budget'
    assert a['raw'].ndim==3 and a['rd_power'].shape==a['noise_power'].shape
    assert r['resource_guard']['trial_record_estimated_bytes']>=0
    if action in ('snr','statistics'):assert not r['validated']
    if action=='calibration':assert not r['ready'] and not r['verified']
    assert all(0<=fraction<=1 and isinstance(message,str) for fraction,message in progress)


def test_resource_guard_rejects_large_trial_output_before_engine(monkeypatch):
    p=small();p['experiment'].update(trials=1000000,memory_mib=64)
    monkeypatch.setattr(statistics,'execute',lambda *args,**kw:pytest.fail('Oversized trial records must not run'))
    with pytest.raises(ValueError,match='逐次统计记录'):service.execute('statistics',p)
    assert service.execute('budget',p)[0]['valid']


@pytest.mark.parametrize('action',['budget','simulation','snr','calibration','statistics','distance','roc'])
def test_cancel_precedes_experiment_allocation(action,monkeypatch):
    monkeypatch.setattr(statistics,'execute',lambda *a,**k:pytest.fail('Must not run'))
    with pytest.raises(InterruptedError):service.execute(action,small(),cancel=lambda:True)


def test_configuration_files_accept_original_and_native_formats_without_running(tmp_path,monkeypatch):
    p=small();source=tmp_path/'原方程工程.json';source.write_text(json.dumps(p),encoding='utf-8')
    monkeypatch.setattr(statistics,'execute',lambda *a,**k:pytest.fail('Loading must not calculate'))
    assert io.load_configuration(source)==p
    target=tmp_path/'native.json';io.save_configuration(target,p)
    assert io.load_configuration(target)==p
    assert json.loads(target.read_text(encoding='utf-8'))['format']=='arraylab.equation'
    target.write_text('{"schema_version":1,"schema_version":1}',encoding='utf-8')
    with pytest.raises(ValueError,match='重复'):io.load_configuration(target)


@pytest.mark.parametrize('version',[True,2,'1'])
def test_malformed_or_unknown_native_schema_is_rejected(version):
    with pytest.raises(ValueError):service.normalize_configuration({'schema_version':version})


def test_apply_calibration_requires_sufficient_count_and_matching_cut():
    from arraylab.features.equation.detection import calibration_fingerprint
    p=small();p['detection']['pfa']=.25
    r={'kind':'calibration','ready':True,'verified':False,'trials':80,'pfa':.25,'factor':3.,'calibration_hash':calibration_fingerprint(p)}
    original=deepcopy(p);updated=service.apply_calibration(p,r)
    assert p==original and updated['detection']['threshold_mode']=='calibrated'
    assert updated['detection']['calibration_factor']==3 and r['verified'] is False
    changed=deepcopy(p);changed['experiment']['cut_range_m']+=1
    with pytest.raises(ValueError,match='不匹配'):service.apply_calibration(changed,r)
    with pytest.raises(ValueError,match='尾部'):service.apply_calibration(p,{**r,'trials':1})
    with pytest.raises(ValueError):service.apply_calibration(p,{**r,'ready':False})


def test_explicit_array_reset_rebuilds_counts_without_mutating_inputs():
    from arraylab.features.equation import reset_array, LIST_DEFAULTS
    p=small(mode='phased',tx=2);p['array']['tx_count']=3
    original=deepcopy(p)
    rebuilt=reset_array(p)
    assert p==original and len(p['array']['tx_positions_m'])==2
    assert len(rebuilt['array']['tx_positions_m'])==len(rebuilt['array']['tx_weights'])==3
    assert rebuilt['array']['mode']=='phased'
    assert set(LIST_DEFAULTS)=={'scene.targets','propagation.custom_paths'}


def test_export_json_npz_csv_html_and_replay_without_solver(tmp_path,simulation_result,monkeypatch):
    result,arrays=simulation_result;before={key:value.copy() for key,value in arrays.items()}
    output=io.export_result(tmp_path/'完整 结果',result,arrays)
    assert {p.name for p in output.iterdir()}=={'result.json','arrays.npz','parameters.json','results.csv','report.html'}
    saved,loaded=read_npz(output/'arrays.npz')
    assert saved['config_hash']==result['config_hash']
    assert loaded.keys()==arrays.keys()
    for key in arrays:
        np.testing.assert_array_equal(arrays[key],before[key])
        np.testing.assert_array_equal(loaded[key],arrays[key])
    monkeypatch.setattr(statistics,'execute',lambda *a,**k:pytest.fail('Restore must not calculate'))
    assert io.load_configuration(output/'result.json')==result['config']
    assert io.load_configuration(output/'arrays.npz')==result['config']
    assert 'ArrayLab' in (output/'report.html').read_text(encoding='utf-8')
    assert '实际距离' in (output/'report.html').read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):io.export_result(output,result,arrays)


def test_all_views_render_saved_values_only(tmp_path,simulation_result,monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    monkeypatch.setattr(statistics,'execute',lambda *a,**k:pytest.fail('View must not calculate'))
    result,arrays=simulation_result
    with matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans']}):
        for view,_ in metadata.VIEWS:
            figure=Figure(figsize=(10,6));FigureCanvasAgg(figure)
            presentation.draw(figure,result,arrays,view);figure.canvas.draw()
            columns,rows,note=presentation.table(result,arrays,view)
            assert columns and isinstance(rows,list) and note
            figure.savefig(tmp_path/(view+'.png'))
            assert (tmp_path/(view+'.png')).read_bytes().startswith(b'\x89PNG')
