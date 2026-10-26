"""Only the FMCW display adapter and its backward-compatible record extension."""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from arraylab.features.waveform import model, io, scene


@pytest.fixture(scope='module')
def report():
    config=model.default_waveform_configuration()
    config['params'].update(samples=64,rangeFft=128,loops=8,dopplerFft=16,targetRange=12,targetVelocity=-3)
    config['profileB'].update(samples=64,rangeFft=128,loops=8,dopplerFft=16)
    config['abOptions'].update(enabled=True)
    config['geometry']={'tx':[[0,0],[0,.5],[0,1]],'rx':[[0,0],[.5,0],[1,0],[1.5,0]],
                        'steerAzDeg':20,'steerElDeg':10,'referenceFrequencyGHz':77}
    return model.calculate_waveform(config,workers=1)


def test_scene_uses_saved_profile_without_recalculating_or_changing_inputs(report,monkeypatch):
    before=deepcopy(report)
    from arraylab.features.waveform import antenna,core
    def fail(*args,**kwargs):pytest.fail('Display must not run a numerical solver')
    monkeypatch.setattr(model,'calculate_waveform',fail);monkeypatch.setattr(antenna,'calculate',fail);monkeypatch.setattr(core,'simulate',fail)
    payload=scene.scene_payload(report,'A',2)
    assert report==before and payload['grid'] is None
    assert payload['project']['scene']['terrain']=={'enabled':False}
    target=payload['project']['scene']['targets'][0]
    assert target['position_m']==[12.,0.,0.] and target['velocity_mps']==[0.,0.,0.]
    assert len(payload['project']['scene']['targets'])==1
    assert 'v=-3 m/s' in target['id'] and '正值表示拍频增加' in payload['note']
    assert payload['project']['array']['pointing_deg']==[20.,10.]
    json.dumps(payload,allow_nan=False)


def test_fixed_physical_array_stays_fixed_across_ab_and_uses_physical_channels(report):
    a=scene.scene_payload(report,'A');b=scene.scene_payload(report,'B')
    for side,count in [('tx',3),('rx',4)]:
        pa=a['project']['array'][side+'_positions_m'];pb=b['project']['array'][side+'_positions_m']
        assert len(pa)==len(pb)==count
        np.testing.assert_allclose(pa,pb,rtol=1e-14,atol=0)
        assert all(p[0]==0 for p in pa)
    assert a['project']['scene']['targets']==b['project']['scene']['targets']
    assert a['project']['array']['rx_positions_m'][1][1]==pytest.approx(.5*scene.C/77e9)


def test_display_angles_and_explicit_motion_do_not_infer_velocity_sign(report):
    options={'target_az_deg':30,'target_el_deg':20,'motion':'approaching'}
    payload=scene.scene_payload(report,'A',100,options)
    unit=np.array([math.cos(math.radians(20))*math.cos(math.radians(30)),
                   math.cos(math.radians(20))*math.sin(math.radians(30)),math.sin(math.radians(20))])
    target=payload['project']['scene']['targets'][0]
    np.testing.assert_allclose(target['position_m'],unit*12)
    np.testing.assert_allclose(target['velocity_mps'],unit*-3)
    assert payload['max_time_s']==4 and payload['time_s']==4
    np.testing.assert_allclose(np.array(target['position_m'])+4*np.array(target['velocity_mps']),0,atol=1e-14)
    outward=scene.scene_payload(report,'A',1,{**options,'motion':'receding'})
    np.testing.assert_allclose(outward['project']['scene']['targets'][0]['velocity_mps'],unit*3)


def test_missing_beam_width_and_legacy_array_hide_beam_instead_of_inventing(report):
    missing=deepcopy(report);missing['antenna']['A']['beamwidthElDeg']=None
    payload=scene.scene_payload(missing,'A')
    assert payload['project']['array']['beam_visible'] is False
    assert payload['project']['array']['pointing_axis'] is True
    assert payload['project']['array']['beamwidth_deg']==[0.,0.]
    assert '隐藏波束锥' in payload['note']
    del missing['antenna']
    legacy=scene.scene_payload(missing,'A')
    assert legacy['project']['array']['tx_positions_m']==[]
    assert legacy['project']['array']['pointing_axis'] is False


@pytest.mark.parametrize('value',[
    {'time_s':float('nan')},{'time_s':-1},{'target_az_deg':181},{'target_el_deg':91},
    {'motion':'auto'},{'target_az_deg':True},{'platform_height':30},
    {'camera':{'position':[1,2,float('inf')],'target':[0,0,0],'zoom':1}},
    {'camera':{'position':[1,2,3],'target':[0,0,0],'zoom':0}},
])
def test_scene_state_rejects_invalid_assumptions(value):
    with pytest.raises(ValueError):scene.normalize_scene_state(value)


def test_schema_one_default_unchanged_and_schema_two_scene_roundtrip(report,tmp_path):
    old=tmp_path/'old.json';io.save_waveform_record(old,report,profile='B')
    old_doc=json.loads(old.read_text(encoding='utf-8'))
    assert old_doc['schema_version']==1 and 'scene_state' not in old_doc
    loaded=io.load_waveform_record(old);assert 'scene_state' not in loaded and loaded['report']==report
    state=scene.normalize_scene_state({'time_s':1.2,'target_az_deg':25,'motion':'receding',
              'camera':{'position':[100,200,300],'target':[10,20,30],'zoom':2}})
    new=tmp_path/'new.json';io.save_waveform_record(new,report,view='scene',profile='B',scene_state=state)
    new_doc=json.loads(new.read_text(encoding='utf-8'));assert new_doc['schema_version']==2
    loaded=io.load_waveform_record(new)
    assert loaded['scene_state']==state and loaded['profile']=='B' and loaded['view']=='scene'
    assert loaded['report']==report and report['schema_version']==2
    new_doc['scene_state']['target_el_deg']=40;new.write_text(json.dumps(new_doc),encoding='utf-8')
    with pytest.raises(ValueError,match='摘要'):io.load_waveform_record(new)


def test_scene_export_keeps_png_and_original_report_and_old_defaults(report,tmp_path):
    # Fixed, valid 1x1 PNG supplied by the display caller; export does not render it.
    import base64
    png=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/b9sAAAAASUVORK5CYII=')
    target=tmp_path/'scene'
    io.export_waveform_directory(target,report,plot='none',scene_png=png,scene_state={'time_s':1},profile='B')
    assert (target/'scene.png').read_bytes()==png
    view=json.loads((target/'scene-view.json').read_text(encoding='utf-8'))
    assert view['profile']=='B' and view['scene_state']['time_s']==1
    loaded=io.load_waveform_record(target/'waveform-record.json')
    assert loaded['view']=='scene' and loaded['scene_state']==view['scene_state'] and loaded['report']==report
    assert json.loads((target/'summary.json').read_text(encoding='utf-8'))==report
    old=tmp_path/'plain';io.export_waveform_directory(old,report,plot='none')
    assert not (old/'scene-view.json').exists() and not (old/'scene.png').exists()
    assert json.loads((old/'waveform-record.json').read_text(encoding='utf-8'))['schema_version']==1
    with pytest.raises(FileExistsError):io.export_waveform_directory(target,report,plot='none')


def test_failed_scene_export_does_not_publish_partial_directory(report,tmp_path,monkeypatch):
    original=io._write
    def write(path,value):
        if Path(path).name=='scene-view.json':raise OSError('injected write failure')
        return original(path,value)
    monkeypatch.setattr(io,'_write',write)
    target=tmp_path/'failed'
    with pytest.raises(OSError,match='injected'):
        io.export_waveform_directory(target,report,plot='none',scene_state={})
    assert not target.exists() and not list(tmp_path.glob('.arraylab-waveform-*'))
