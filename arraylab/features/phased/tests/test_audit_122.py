"""Only ArrayStudio 1.2.2 numerical fixes and their affected output contracts."""
from copy import deepcopy
import csv,math
import numpy as np
import pytest
from arraylab.features.phased import engine as E,linear_optimizer as L,planar_optimizer as P,service as S


def line(**options):return {**E.defaults(),'geometry':'linear','nx':16,'ny':1,**options}
def grid(nx,ny,spacing=.5):
    return [dict(x=(i-(nx-1)/2)*spacing,y=(j-(ny-1)/2)*spacing,amplitude=1,phaseDeg=0) for j in range(ny) for i in range(nx)]
def options(nx,ny,spacing=.5,**overrides):
    return {**P.defaults(),'count':nx*ny,'gridNx':nx,'gridNy':ny,'dx':spacing,'dy':spacing,
            'frequencies':[10],'scans':[dict(x=0,y=0)],'mainlobeRadiusU':.3,'mainlobeRadiusV':.3,
            'targetSll':0,'maxHpbwX':90,'maxHpbwY':90,'iterations':0,**overrides}


def test_subnormal_common_voltage_scale_preserves_field_efficiency_and_rms():
    c=line(geometry='custom',steerX=17.3,phaseBits=3,custom=[
        dict(x=-.25,y=0,amplitude=1e-200,phaseDeg=0),dict(x=.25,y=0,amplitude=.5e-200,phaseDeg=0)])
    ref=deepcopy(c)
    for p in ref['custom']:p['amplitude']/=1e-200
    a,b=E.evaluate(c),E.evaluate(ref)
    assert a['metrics']['taperEfficiency']==pytest.approx(.9,abs=1e-12)
    assert a['metrics']['coherentGainDb']==pytest.approx(10*math.log10(1.8),abs=1e-12)
    assert a['metrics']['phaseRmsDeg']==pytest.approx(b['metrics']['phaseRmsDeg'],abs=1e-10)
    assert [p['amplitude'] for p in a['elements']]==[1e-200,.5e-200]
    np.testing.assert_allclose(E.field_values(c,a['elements'],[-.7,0,.3,.8],0),E.field_values(ref,b['elements'],[-.7,0,.3,.8],0),atol=1e-10)


def test_large_integral_phase_is_wrapped_before_steering():
    c=line(geometry='custom',steerX=30,custom=[dict(x=x,y=0,amplitude=1,phaseDeg=360e15) for x in (0,.5)])
    e=E.build_elements(c)
    assert [p['phaseDeg'] for p in e]==pytest.approx([0,-90],abs=1e-10)
    assert E.pattern_at_raw(c,e,.5,0)==pytest.approx(0,abs=1e-10)


@pytest.mark.parametrize('bits,u',[(2,.5),(3,.25)])
@pytest.mark.parametrize('sign',[-1,1])
def test_half_code_ties_keep_the_same_field_in_both_scan_directions(bits,u,sign):
    c=line(nx=6,phaseBits=bits,steerX=sign*math.degrees(math.asin(u)))
    actual=E.evaluate(c)['metrics'];ideal=E.evaluate({**c,'phaseBits':0})['metrics']
    assert actual['pointingPowerDb']==pytest.approx(0,abs=1e-10)
    assert actual['phaseRmsDeg']==pytest.approx(180/2**bits,abs=1e-10)
    assert actual['sllX']==pytest.approx(ideal['sllX'],abs=1e-7)
    assert actual['hpbwX']==pytest.approx(ideal['hpbwX'],abs=1e-7)
    assert E.quantized_phase(45-1e-6,90)==0 and E.quantized_phase(45+1e-6,90)==90


def test_display_floor_never_changes_deep_cancellation_beam_metrics_or_csv(tmp_path):
    base=line(nx=32,window='taylor');reference=E.evaluate(base)
    custom=[]
    for p in reference['elements']:
        custom.extend([{**p,'phaseDeg':0},{**p,'amplitude':p['amplitude']*.998,'phaseDeg':180}])
    c={**base,'geometry':'custom','custom':custom};actual=E.evaluate(c)
    for key in ('sllX','hpbwX'):assert actual['metrics'][key]==pytest.approx(reference['metrics'][key],abs=1e-7)
    assert actual['metrics']['pointingPowerDb']==pytest.approx(20*math.log10(.002/1.998),abs=1e-8)
    assert E.pattern_at(c,actual['elements'],.4,0)==-80
    assert E.pattern_at_raw(c,actual['elements'],.4,0)<-90
    from arraylab.features.phased.io import export_result
    result,arrays=S.execute('analysis',{'array':c,'mapSize':21},workers=1)
    export_result(tmp_path/'out',result,arrays)
    with (tmp_path/'out/cut-x.csv').open(encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    assert min(float(row['relative_power_db']) for row in rows)<-90


@pytest.mark.parametrize('span',[30,100,300])
def test_narrow_two_element_beam_matches_closed_form_half_power_roots(span):
    angle=20.1;u=math.sin(math.radians(angle))
    c=line(geometry='custom',steerX=angle,custom=[dict(x=x,y=0,amplitude=1,phaseDeg=0) for x in (-span/2,span/2)])
    m=E.evaluate(c)['metrics'];width=math.degrees(math.asin(u+1/(4*span))-math.asin(u-1/(4*span)))
    assert m['peakAngleX']==pytest.approx(angle,abs=1e-6)
    assert m['sllX']==pytest.approx(0,abs=1e-6)
    assert m['hpbwX']==pytest.approx(width,abs=1e-6)


def test_inactive_axes_are_symmetric_and_ignore_zero_weight_positions():
    xs=(np.arange(16)-7.5)*.5
    a=E.evaluate(line(geometry='custom',elementPower=2,custom=[dict(x=float(x),y=0,amplitude=1,phaseDeg=0) for x in xs]))['metrics']
    b=E.evaluate(line(geometry='custom',elementPower=2,custom=[dict(x=0,y=float(x),amplitude=1,phaseDeg=0) for x in xs]))['metrics']
    assert a['hpbwX']==pytest.approx(b['hpbwY'],abs=1e-9)
    assert a['sllX']==pytest.approx(b['sllY'],abs=1e-9)
    assert a['hpbwY'] is None and b['hpbwX'] is None
    c=line(geometry='custom',elementPower=2,custom=[dict(x=0,y=0,amplitude=1,phaseDeg=0),dict(x=.5,y=1,amplitude=0,phaseDeg=0)])
    m=E.evaluate(c)['metrics'];assert m['hpbwX'] is None and m['hpbwY'] is None
    assert m['coherentGainDb']==pytest.approx(0,abs=1e-12)


def test_floor_plateau_and_mutable_external_excitation_have_no_stale_peak():
    points=[dict(angle=x,db=-300 if x else 0) for x in (-2,-1,0,1,2)]
    assert E.analyze_cut(points)['sll'] is None
    assert E.analyze_cut([dict(angle=x,db=-300) for x in (-2,-1,0,1,2)])['peakAngle'] is None
    c=line(nx=2);external=grid(2,1)
    assert E.pattern_at_raw(c,external,0,0)==pytest.approx(0)
    external[1]['phaseDeg']=180;assert E.pattern_at_raw(c,external,0,0)<=-299
    external[0]['amplitude']=0;assert E.pattern_at_raw(c,external,.4,0)==pytest.approx(0,abs=1e-10)


def test_feasible_linear_baseline_is_not_exchanged_for_excess_width():
    o=dict(count=8,aperture=3.5,minSpacing=.15,scanMax=30,targetSll=-12,maxHpbw=14.84035378025589,iterations=500,seed=42)
    result=L.optimize(line(),o)
    assert L.is_feasible(result['before'],o)
    assert result['feasible'] and result['after']['hpbw']<=o['maxHpbw']+1e-8
    assert result['baselineFallback']
    assert result['after']['hpbw']==pytest.approx(14.8356114343,abs=1e-7)


def test_linear_coarse_and_fine_use_the_same_half_code_quantization():
    result=L.optimize(line(phaseBits=2),dict(count=6,aperture=2.5,minSpacing=.5,scanMax=30,targetSll=-1,maxHpbw=90,iterations=20))
    assert result['history'][0]['worstSll']==pytest.approx(-12.42553665,abs=.002)
    assert result['before']['worstSll']==pytest.approx(-12.42553665,abs=.002)


def test_invisible_linear_main_order_is_explicit_and_cannot_be_replaced():
    result=L.optimize(line(frequencyGHz=5),dict(count=12,aperture=8,minSpacing=.45,scanMax=60,targetSll=-1,maxHpbw=90,iterations=20))
    assert not result['feasible'] and result['after']['hpbw'] is None and result['after']['worstSll'] is None
    for row in result['scanResults']:
        if row['angle']:
            assert row['status']=='outside_visible' and row['expectedAngle'] is None and row['peakAngle'] is None
            assert row['sll'] is None and row['hpbw'] is None and not row['resolved']


def test_planar_small_sll_ellipse_cannot_hide_connected_half_power_truncation():
    angle=math.degrees(math.asin(.63));points=grid(8,8)
    o=options(8,8,scans=[dict(x=angle,y=angle)],mainlobeRadiusU=.1,mainlobeRadiusV=.1,targetSll=-2)
    row=P.evaluate_layout(E.defaults(),o,points,workers=1)['cases'][0]
    assert row['hpbwX'] is not None and row['hpbwY'] is not None
    assert not row['resolved'] and not row['feasible']
    assert any('连通' in issue and '截断' in issue for issue in row['issues'])


def test_disconnected_horizon_grating_lobe_is_not_main_beam_truncation():
    row=P.evaluate_layout(E.defaults(),options(4,4,1),grid(4,4,1),workers=1)['cases'][0]
    assert row['resolved'] and row['feasible']
    assert row['sll2D']==pytest.approx(0,abs=.001)
    assert not any('截断' in issue for issue in row['issues'])


def test_current_and_old_record_advice_preserves_exact_saved_feed(monkeypatch):
    result,arrays=S.execute('analysis',{'array':line(nx=6,phaseBits=2,steerX=30),'mapSize':21},workers=1)
    assert S.record_compatibility(result)['current']
    old=deepcopy(result);old['source']['version']='1.2.1';old['analysis']['elements'][0]['phaseDeg']=123.456
    before=deepcopy(old)
    monkeypatch.setattr(E,'build_elements',lambda *_:pytest.fail('Snapshot must not regenerate feed phases'))
    assert S.array_snapshot(old)['phases_deg'][0]==123.456
    assert not S.record_compatibility(old)['current'] and '重新计算' in S.record_compatibility(old)['message']
    assert not S.record_compatibility({**old,'source':None})['current']
    assert old==before
