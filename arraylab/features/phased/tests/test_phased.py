from copy import deepcopy
from pathlib import Path
import json,math,os,subprocess,sys
import numpy as np
import pytest
from arraylab.framework.parallel import ComputationCancelled
from arraylab.features.phased import engine as E,linear_optimizer as L,planar_optimizer as P,service as S

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())

def small_planar(mode='grid'):
    o=P.defaults();o.update(count=8,gridNx=5,gridNy=5,width=3,height=3,scans=[{'x':0,'y':0}],frequencies=[10],iterations=30,
        targetSll=-5,maxHpbwX=60,maxHpbwY=60,mainlobeRadiusU=.5,mainlobeRadiusV=.5,mode=mode)
    return o

def provided():
    points=[]
    for x,y,a in [(.35,.8,1),(1.1,.3,.8),(1.4,-.75,.65),(.4,-1.3,.5),(.9,1.2,.7),(1.7,.8,.4)]:
        points.extend([dict(x=x,y=y,amplitude=a,phaseDeg=0),dict(x=-x,y=-y,amplitude=a,phaseDeg=0)])
    return {**P.continuous_defaults(),'count':12,'width':4,'height':4,'initialization':'provided','initialPositions':points,
            'scans':[dict(x=0,y=0)],'frequencies':[10],'targetSll':-8,'maxHpbwX':60,'maxHpbwY':60,
            'mainlobeRadiusU':.35,'mainlobeRadiusV':.35,'iterations':40}

@pytest.mark.parametrize('change',[{'frequencyGHz':101},{'steerX':60,'steerY':60},{'phaseBits':True},{'geometry':'linear','nx':2,'window':'hann'},
    {'geometry':'custom','custom':[]},{'geometry':'custom','custom':[dict(x=0,y=0,amplitude=0,phaseDeg=0)]*2}])
def test_invalid_array_inputs_are_rejected(change):
    with pytest.raises(ValueError):E.evaluate(change)

def test_uniform_ula_matches_closed_form_complex_sum():
    c=E.normalized_config(dict(geometry='linear',nx=16));els=E.build_elements(c)
    angles=np.array([-38,-15,0,12,31]);u=np.sin(np.radians(angles));delta=np.pi*.5*u
    amp=np.divide(np.sin(16*delta),16*np.sin(delta),out=np.ones_like(delta),where=abs(delta)>1e-15)
    np.testing.assert_allclose(E.field_values(c,els,u,0),np.maximum(-80,20*np.log10(np.maximum(abs(amp),1e-4))),atol=1e-11)

def test_fixed_phase_squint_and_ttd_keep_physical_aperture():
    c=dict(geometry='linear',nx=32,steerX=40)
    base=E.evaluate(c);squint=E.evaluate({**c,'frequencyGHz':12});ttd=E.evaluate({**c,'frequencyGHz':12,'phaseMode':'ttd'})
    assert squint['metrics']['peakAngleX']==pytest.approx(math.degrees(math.asin(math.sin(math.radians(40))/1.2)),abs=.13)
    assert ttd['metrics']['peakAngleX']==pytest.approx(40,abs=.13)
    assert base['metrics']['widthMm']==squint['metrics']['widthMm']==ttd['metrics']['widthMm']
    assert ttd['metrics']['pointingPowerDb']==pytest.approx(0,abs=1e-11)

def test_quantization_tie_rounding_and_custom_weights_do_not_get_extra_taper():
    c=E.normalized_config(dict(geometry='custom',window='hann',phaseBits=2,custom=[dict(x=-.5,y=0,amplitude=.2,phaseDeg=-45),dict(x=.5,y=0,amplitude=1,phaseDeg=45)]))
    e=E.build_elements(c)
    assert [p['phaseDeg'] for p in e]==[0,90]
    assert [p['amplitude'] for p in e]==[.2,1]
    assert E.phase_rms(c,e)==pytest.approx(45)

def test_periodic_grating_lobe_candidates_are_not_used_for_custom_geometry():
    c=E.normalized_config(dict(geometry='linear',nx=8,dx=1,steerX=30));r=E.evaluate(c)
    assert any(abs(g['xDeg']+30)<1e-8 for g in r['metrics']['gratingLobes'])
    custom={**c,'geometry':'custom','custom':[dict(x=i,y=0,amplitude=1,phaseDeg=0) for i in range(8)]}
    assert not E.evaluate(custom)['metrics']['gratingLobes']

def test_scan_step_empty_status_and_ideal_sidelobe_flatness():
    c=dict(geometry='linear',nx=32)
    assert [r['angle'] for r in E.scan_study(c,dict(maxAngle=31,step=10),workers=1)]==[-31,-30,-20,-10,0,10,20,30,31]
    rows=E.scan_study(c,dict(maxAngle=40,step=10),workers=1)
    assert all(r['resolved'] for r in rows)
    assert max(r['sll'] for r in rows)-min(r['sll'] for r in rows)<.06
    assert rows[0]['hpbw']>rows[4]['hpbw']
    empty=E.scan_study(dict(geometry='linear',nx=2,dx=.1),dict(maxAngle=0,step=5),workers=1)[0]
    assert empty['status']=='incomplete_beam' and empty['hpbw'] is None

def test_planar_visibility_and_vertical_line_cannot_claim_x_scan():
    out=E.scan_study(dict(geometry='rectangular',nx=64,ny=8,frequencyGHz=9,steerY=40),dict(maxAngle=40,step=10),workers=1)[-1]
    assert out['status']=='outside_visible' and out['hpbw'] is out['sll'] is None
    c=dict(geometry='custom',custom=[dict(x=0,y=i*.5,amplitude=1,phaseDeg=0) for i in range(8)],elementPower=2)
    row=E.scan_study(c,dict(maxAngle=0,step=5),workers=1)[0]
    assert row['status']=='incomplete_beam' and row['hpbw'] is None

def test_linear_optimization_preserves_fixed_endpoints_symmetry_and_feasible_gaps():
    o={**L.DEFAULTS,'iterations':40};r=L.optimize(dict(geometry='linear',nx=16),o)
    xs=np.array([p['x'] for p in r['config']['custom']]);np.testing.assert_allclose(xs,-xs[::-1])
    assert xs[-1]-xs[0]==pytest.approx(o['aperture'])
    assert min(np.diff(xs))>=o['minSpacing']-1e-9
    assert L.score(r['after'],o)<=L.score(r['before'],o)+1e-8
    assert r['feasible']==(r['after']['resolved'] and r['after']['worstSll']<=o['targetSll']+1e-8 and r['after']['hpbw']<=o['maxHpbw']+1e-8)

@pytest.mark.parametrize('change',[{'count':7},{'aperture':1,'minSpacing':1},{'iterations':0},{'seed':-1}])
def test_linear_invalid_constraints(change):
    assert not L.validate(change)['ok']

def test_full_disk_detects_diagonal_grating_lobe_missed_by_cuts():
    points=[dict(x=(i-2)*.8,y=(j-2)*.8,amplitude=1,phaseDeg=0) for j in range(5) for i in range(5) if (i+j)%2==0 and (i,j)!=(2,2)]
    o={**small_planar(),'count':12,'dx':.8,'dy':.8,'minSpacing':.8,'mainlobeRadiusU':.25,'mainlobeRadiusV':.25}
    r=P.evaluate_layout(E.defaults(),o,points,workers=1);case=r['cases'][0]
    assert case['sll2D']==pytest.approx(0,abs=.12)
    assert abs(case['sidelobeU'])>.5 and abs(case['sidelobeV'])>.5 and not r['feasible']

def test_grid_search_reproducible_and_final_objective_never_worsens():
    o=small_planar();a=P.optimize(E.defaults(),o,workers=1);b=P.optimize(E.defaults(),o,workers=2)
    assert a==b
    P.verify_positions(a['config']['custom'],o)
    assert P.objective(a['after'],o)<=P.objective(a['before'],o)+1e-8
    assert a['after']==P.evaluate_layout(E.defaults(),o,a['config']['custom'],workers=1)

@pytest.mark.parametrize('position,amplitude',[(True,False),(False,True),(True,True)])
def test_continuous_freezing_and_joint_constraints(position,amplitude):
    o=provided();o.update(optimizePositions=position,optimizeAmplitudes=amplitude)
    r=P.optimize(E.defaults(),o,workers=1);P.verify_positions(r['config']['custom'],o)
    assert P.objective(r['after'],o)<=P.objective(r['before'],o)+1e-8
    if not position:assert r['diagnostics']['coordinateMovedCount']==0
    if not amplitude:assert r['diagnostics']['amplitudeChangedCount']==0
    for row in r['diagnostics']['pairChanges']:
        if not position:assert row['before']['x']==row['after']['x'] and row['before']['y']==row['after']['y']
        if not amplitude:assert row['before']['amplitude']==row['after']['amplitude']

def test_provided_zero_budget_keeps_full_precision_and_no_hidden_normalization():
    o=provided();o.update(iterations=0,optimizePositions=False,optimizeAmplitudes=False,gridNx=1,gridNy=1,dx=999,dy=999)
    old=deepcopy(o);r=P.optimize(E.defaults(),o,workers=1)
    assert o==old and r['before']==r['after'] and r['verification']['baselineFallback']
    assert sorted((p['x'],p['y'],p['amplitude']) for p in r['config']['custom'])==sorted((p['x'],p['y'],p['amplitude']) for p in o['initialPositions'])

@pytest.mark.parametrize('change',[{'count':5},{'frequencies':[10,10]},{'scans':[{'x':70,'y':70}]},{'width':121},{'optimizePositions':'yes'}])
def test_planar_invalid_bounds(change):
    assert not P.validate(E.defaults(),{**provided(),**change})['ok']

def test_invalid_initial_weights_symmetry_and_case_budget_rejected():
    o=provided();o['initialPositions'][0]['amplitude']=.9
    assert not P.validate(E.defaults(),o)['ok']
    o=provided();o['frequencies']=[9,10,11,12,13,14];o['scans']=[dict(x=i,y=0) for i in range(5)]
    assert not P.validate(E.defaults(),o)['ok']

@pytest.mark.parametrize('kind',['analysis','scan','optimize1d','optimize2d'])
def test_cancellation_and_invalid_worker_budget_happen_before_work(kind):
    with pytest.raises(ComputationCancelled):S.execute(kind,cancel=lambda:True)
    with pytest.raises(ValueError):S.execute(kind,workers=99)

def test_source_schema_versions_and_no_implicit_optimization():
    for version in (1,2,3):
        cfg=S.normalize_configuration(dict(format='array-studio',schemaVersion=version,config={**E.defaults(),'geometry':'linear'}))
        assert cfg['array']['geometry']=='linear'
    source=dict(format='array-studio',schemaVersion=2,config=E.defaults(),synthesis={'type':'planar2d','options':provided()})
    with pytest.raises(ValueError):S.normalize_configuration(source)

def test_service_case_arrays_snapshot_and_safe_roundtrip(tmp_path):
    from arraylab.shared.simulation_io import clean_json,write_npz,read_npz
    cfg=S.default_configuration();cfg['planar']=small_planar();cfg['planar']['frequencies']=[9,11];cfg['planar']['iterations']=0
    report,arrays=S.execute('optimize2d',cfg,workers=1)
    assert len(report['caseAnalyses'])==2 and arrays['case_0_uv_db'].shape==(81,81)
    target=tmp_path/'result.npz';write_npz(target,report,arrays);saved,data=read_npz(target)
    assert saved==clean_json(report)
    for key in arrays:np.testing.assert_equal(data[key],arrays[key])
    snapshot=S.array_snapshot(report)
    assert snapshot['coordinate_convention'].startswith('x-y')
    assert len(snapshot['coordinates_m'])==8

def test_cli_is_qt_free_and_exports_without_overwriting(tmp_path):
    cfg=S.presets()[0]['configuration'];cfg['mapSize']=21
    source=tmp_path/'input.json';source.write_text(json.dumps(cfg),encoding='utf-8')
    output=tmp_path/'输出';env={**os.environ,'QT_QPA_PLATFORM':'invalid-test','MPLBACKEND':'QtAgg'}
    args=[sys.executable,'-m','arraylab.features.phased.cli','--config',str(source),'--output-dir',str(output),'--workers','1']
    p=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,encoding='utf-8');assert p.returncode==0,p.stderr
    assert json.loads(p.stdout)['status']=='ok'
    assert (output/'arrays.npz').is_file() and (output/'elements.csv').is_file()
    p=subprocess.run(args,cwd=ROOT,env=env,capture_output=True,encoding='utf-8');assert p.returncode==3
    p=subprocess.run([sys.executable,'-c',"import sys;from arraylab.features.phased import default_configuration;default_configuration();assert not any(x.startswith('PySide6') for x in sys.modules)"],cwd=ROOT,env=env,capture_output=True)
    assert p.returncode==0,p.stderr


def test_saved_views_case_selection_and_candidate_application_do_not_compute(monkeypatch,tmp_path):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from arraylab.features.phased import draw,table,apply_selected_case,array_snapshot,export_result
    cfg=S.default_configuration();cfg['planar']=small_planar();cfg['planar'].update(iterations=0,frequencies=[9,11]);cfg['mapSize']=21
    result,arrays=S.execute('optimize2d',cfg,workers=1)
    def fail(*args,**kwargs):pytest.fail('Saved-result browsing/export must not evaluate scientific fields')
    for owner,name in [(E,'evaluate'),(E,'field_values'),(P,'optimize'),(P,'evaluate_layout'),(L,'optimize')]:monkeypatch.setattr(owner,name,fail)
    with matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans']}):
        for view in ('layout','cuts','uv','scan','optimization','cases','elements'):
            figure=Figure(figsize=(10,7));FigureCanvasAgg(figure);draw(figure,result,arrays,view,selection=1);figure.canvas.draw()
            assert figure.axes
            if view=='cuts':np.testing.assert_equal(figure.axes[0].lines[0].get_ydata(),arrays['case_1_cut_x'][:,1])
            assert len(table(result,arrays,view))==3
            figure.clear()
    applied=apply_selected_case(result,1)
    assert applied['array']['frequencyGHz']==11 and applied['array']['custom']==result['synthesis']['config']['custom']
    a=array_snapshot(result,0);b=array_snapshot(result,1)
    assert a['coordinates_m']==b['coordinates_m'] and a['frequency_ghz']==9 and b['frequency_ghz']==11
    export_result(tmp_path/'rendered',result,arrays,plot='cuts')
    assert (tmp_path/'rendered/cuts.png').read_bytes().startswith(b'\x89PNG')


def test_shipped_source_examples_import_without_search(monkeypatch):
    from arraylab.features.phased import load_configuration
    monkeypatch.setattr(P,'optimize',lambda *a,**kw:pytest.fail('Import ran optimization'))
    paths=sorted((ROOT/'examples').glob('phased-*.json'))
    assert len(paths)>=5
    for path in paths:
        configuration=load_configuration(path)
        assert E.validate(configuration['array'])['ok']


def test_progress_only_completes_after_saved_case_arrays_exist():
    cfg=S.default_configuration();cfg['mapSize']=21;cfg['planar']=small_planar();cfg['planar']['iterations']=0
    messages=[];result,arrays=S.execute('optimize2d',cfg,workers=1,progress=messages.append)
    values=[m['progress'] for m in messages]
    assert values==sorted(values) and values[-1]==1 and all(v<1 for v in values[:-1])
    assert 'case_0_uv_db' in arrays


def test_element_table_reads_selected_saved_case_phase_without_calculating(monkeypatch):
    from arraylab.features.phased import table
    def fail(*args,**kwargs):pytest.fail('Saved-case table must not evaluate scientific fields')
    for owner,name in [(E,'evaluate'),(E,'field_values'),(P,'optimize'),(P,'evaluate_layout'),(L,'optimize')]:
        monkeypatch.setattr(owner,name,fail)
    first={'config':{'referenceGHz':10.,'frequencyGHz':9.,'steerX':0.,'steerY':0.},
           'elements':[{'x':-.5,'y':0.,'amplitude':1.,'phaseDeg':0.},
                       {'x':.5,'y':0.,'amplitude':.8,'phaseDeg':0.}]}
    second=deepcopy(first);second['config'].update(frequencyGHz=11.,steerX=20.,steerY=-5.)
    second['elements'][0]['phaseDeg']=67.721;second['elements'][1]['phaseDeg']=-67.721
    result={'analysis':first,'caseAnalyses':[first,second]};before=deepcopy(result)
    columns,rows,note=table(result,{},'elements',selection=1)
    assert columns[-1]=='馈相/°' and [row[-1] for row in rows]==[67.721,-67.721]
    assert all(text in note for text in ('工况 2','参考 10 GHz','分析 11 GHz','X 20°','Y -5°'))
    assert [row[-1] for row in table(result,{},'elements')[1]]==[0.,0.]
    assert result==before
