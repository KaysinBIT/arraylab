"""Snapshot-only radar handoffs with explicit coordinate and timing semantics."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from arraylab.application.bridges.radar import import_array,import_waveform,apply_phased_to_planar,C
from arraylab.features import equation,pulse,phased,waveform


class Choice:
    def __init__(self,value):self.value=value
    def currentData(self):return self.value


def panel(config=None,report=None):
    return SimpleNamespace(_busy=False,thread=None,report=report,
                           configuration=lambda:deepcopy(config),selection=Choice(0))


def host(source=None):
    source=source or {'positions_lambda':np.array([[-.5,0.],[.5,0.]]),'weights':np.ones(2),
                      'frequency_ghz':3.,'element_model':'isotropic','origin':'已保存阵列'}
    return SimpleNamespace(selected_linear_array_snapshot=lambda:deepcopy(source),
        sparse_panel=SimpleNamespace(_busy=False,thread=None,selected_array_snapshot=lambda:deepcopy(source)),
        pulse_panel=panel(pulse.default_configuration()),equation_panel=panel(equation.default_configuration()))


@pytest.mark.parametrize('destination',['pulse','equation'])
@pytest.mark.parametrize('source',['linear','sparse'])
def test_geometry_mapping_in_metres_keeps_frequency_and_other_side(destination,source):
    h=host();target=getattr(h,destination+'_panel');before=target.configuration()
    result,note=import_array(h,destination,'rx',source)
    expected=np.array([[0,-.5*C/3e9,0],[0,.5*C/3e9,0]])
    np.testing.assert_allclose(result['array']['rx_positions_m'],expected,rtol=1e-14,atol=1e-14)
    assert result['waveform']['fc_hz']==before['waveform']['fc_hz']
    assert result['radar']==before['radar'] and result['array']['pointing_deg']==before['array']['pointing_deg']
    if destination=='pulse':
        prior=pulse.validate_configuration(before)['metrics']['tx_positions_m']
        assert result['array']['tx_positions_m']==prior and result['array']['geometry']=='custom'
    else:assert result['array']['tx_positions_m']==before['array']['tx_positions_m']
    assert target.configuration()==before and '(z,x,y)' in note


def test_only_equation_accepts_nonuniform_amplitudes_and_calibration_invalidates():
    snap={'positions_lambda':[[0,0],[1,2]],'weights':[1.,.25],'frequency_ghz':10.,'element_model':'isotropic','origin':'加权'}
    h=host(snap);configuration=h.equation_panel.configuration()
    configuration['detection'].update(threshold_mode='calibrated',calibration_factor=13.,calibration_hash='old')
    h.equation_panel=panel(configuration)
    imported,note=import_array(h,'equation','rx','sparse')
    assert imported['array']['rx_weights']==[[1.,0.],[.25,0.]]
    assert imported['detection']['calibration_factor'] is None and imported['detection']['calibration_hash'] is None
    assert imported['detection']['threshold_mode']=='calibrated' and '重新校准' in note
    with pytest.raises(ValueError,match='等幅'):import_array(h,'pulse','rx','sparse')


@pytest.mark.parametrize('mutation',[{'element_model':'cosine'},{'weights':[0.,0.]},{'weights':[1.,-1.]},
    {'weights':[1+1j,1]},{'positions_lambda':[[0,0],[float('nan'),0]]}])
def test_unknown_models_or_invalid_arrays_are_not_silently_cast(mutation):
    snap={'positions_lambda':[[0,0],[1,0]],'weights':[1.,1.],'frequency_ghz':10.,'element_model':'isotropic'}
    snap.update(mutation)
    with pytest.raises(ValueError):import_array(host(snap),'equation','rx','linear')


def test_simo_tx_count_is_not_implicitly_reinterpreted():
    h=host()
    with pytest.raises(ValueError,match='SIMO'):import_array(h,'equation','tx','linear')
    config=h.equation_panel.configuration();config['array']['mode']='phased';h.equation_panel=panel(config)
    imported,_=import_array(h,'equation','tx','linear')
    assert imported['array']['mode']=='phased' and imported['array']['tx_count']==2


def test_source_mimo_uses_physical_selected_side_not_virtual_channels():
    h=host();h.mimo_panel=panel(report={'format':'arraylab.mimo.result','parameters':{'frequency_ghz':30},
        'geometry':{'tx':{'positions_lambda':[[0,0],[2,0]],'weights':[.5,.5]},
                    'rx':{'positions_lambda':[[0,0],[.5,0],[1,0]],'weights':[.2,.3,.5]}}})
    imported,note=import_array(h,'equation','rx','mimo')
    assert imported['array']['rx_count']==3 and imported['array']['rx_weights']==[[.4,0],[.6,0],[1.,0]]
    assert '物理阵列' in note


@pytest.fixture(scope='module')
def phased_report():
    config=phased.default_configuration();config['mapSize']=21
    config['array'].update(geometry='linear',nx=4,ny=1,referenceGHz=10.,frequencyGHz=12.,steerX=30.,phaseBits=2)
    return phased.execute('analysis',config,workers=1)[0]


def test_phased_reference_frequency_and_selected_actual_phase(phased_report):
    h=host();h.phased_panel=panel(report={'format':'arraylab.simulation.result','feature':'phased','result':deepcopy(phased_report)})
    snap=phased.array_snapshot(phased_report)
    imported,_=import_array(h,'equation','rx','phased')
    np.testing.assert_allclose(imported['array']['rx_positions_m'],np.asarray(snap['coordinates_m'])[:,[2,0,1]])
    prepared=apply_phased_to_planar(h)
    parameters,settings=prepared['parameters'],prepared['settings']
    assert parameters.frequency_ghz==12 and parameters.coordinate_unit=='mm'
    xy=parameters.positions();u,v=snap['steer_u'],snap['steer_v']
    actual=(-360*(xy[:,0]*u+xy[:,1]*v)+np.asarray(settings.custom_phases_deg)+180)%360-180
    np.testing.assert_allclose(actual,snap['phases_deg'],atol=1e-10)
    assert settings.topology=='corporate' and settings.phase_bits==0 and prepared['provenance']['actual_phase_preserved']


def test_busy_missing_source_and_unsupported_destination_are_rejected():
    h=host();h.pulse_panel._busy=True
    with pytest.raises(ValueError,match='计算'):import_array(h,'pulse','rx','linear')
    with pytest.raises(ValueError):import_array(h,'other','rx','linear')
    h.mimo_panel=panel()
    with pytest.raises(ValueError):import_array(h,'equation','rx','mimo')


@pytest.fixture(scope='module')
def waveform_report():
    config=waveform.default_waveform_configuration()
    for key in ('params','profileB'):
        config[key].update(samples=64,rangeFft=128,loops=8,dopplerFft=16,targetRange=12.,targetVelocity=1.)
    config['abOptions'].update(enabled=True,layout='blocks')
    config['antennaDesign']['gainOptions'].update(temperatureK=270.,noiseFigureDb=6.,txPowerDbm=13.,rcsDbsm=3.)
    return waveform.calculate_waveform(config,workers=1)


def wave_host(saved):
    h=host();h.waveform_panel=panel(report=deepcopy(saved));h.waveform_panel.result_profile=Choice('B')
    cfg=h.equation_panel.configuration();cfg['radar'].update(tx_gain_dbi=20.,rx_gain_dbi=15.)
    cfg['scene']['targets'][0].update(az_deg=17.,height_m=2.)
    cfg['array']['pointing_deg']=[5.,3.]
    cfg['detection'].update(threshold_mode='calibrated',calibration_factor=3.,calibration_hash='old')
    h.equation_panel=panel(cfg)
    return h


def test_selected_uniform_waveform_group_timing_power_and_noise(waveform_report):
    h=wave_host(waveform_report);before=deepcopy(h.waveform_panel.report);destination=h.equation_panel.configuration()
    config,note=import_waveform(h);p=before['analysis']['profiles']['B']['params']
    assert config['waveform']['pulses']==p['loops']*p['txCount']
    assert config['waveform']['pri_s']==pytest.approx((p['rampUs']+p['idleUs'])*1e-6)
    assert config['waveform']['fc_hz']==pytest.approx(p['startGHz']*1e9+p['slopeMHzUs']*p['rampUs']*1e6/2)
    assert config['waveform']['bandwidth_hz']==pytest.approx(p['slopeMHzUs']*p['rampUs']*1e6)
    assert config['radar']['noise_mode']=='tsys' and config['radar']['system_temperature_k']==pytest.approx(270*10**.6)
    assert config['radar']['power_w']==pytest.approx(10**((13-30)/10))
    assert config['radar']['tx_gain_dbi']==20 and config['radar']['rx_gain_dbi']==15
    assert config['scene']['targets'][0]['az_deg']==17 and config['scene']['targets'][0]['height_m']==2
    assert config['array']['pointing_deg']==[5.,3.]
    assert config['detection']['threshold_mode']=='calibrated' and config['detection']['calibration_hash'] is None
    assert h.waveform_panel.report==before and h.equation_panel.configuration()==destination
    assert 'B组' in note and 'ADC窗中心' in note and 'Tsys=T×F' in note and '另一组能量' in note


@pytest.mark.parametrize('failure',['alternating','missing_group','bad_gain','missing_geometry','busy'])
def test_waveform_incompatibility_is_explicit(waveform_report,failure):
    h=wave_host(waveform_report)
    if failure=='alternating':h.waveform_panel.report['analysis']['options']['layout']='alternating'
    elif failure=='missing_group':h.waveform_panel.report['analysis']['profiles']['B']=None
    elif failure=='bad_gain':h.waveform_panel.report['gain_budget']['valid']=False
    elif failure=='missing_geometry':h.waveform_panel.report.pop('antenna')
    else:h.waveform_panel._busy=True
    with pytest.raises(ValueError):import_waveform(h)


def test_handoff_never_invokes_any_solver(waveform_report,monkeypatch):
    def forbidden(*args,**kwargs):pytest.fail('Handoff invoked a solver')
    monkeypatch.setattr(equation,'execute',forbidden);monkeypatch.setattr(pulse,'execute',forbidden)
    monkeypatch.setattr(phased,'execute',forbidden);monkeypatch.setattr(waveform,'calculate_waveform',forbidden)
    h=wave_host(waveform_report)
    import_array(h,'equation','rx','linear');import_waveform(h)


@pytest.mark.parametrize('destination',['pulse','equation'])
def test_import_repairs_pending_count_coordinate_mismatch_without_early_validation(destination):
    h=host();target=getattr(h,destination+'_panel');raw=target.configuration()
    if destination=='pulse':
        metrics=pulse.validate_configuration(raw)['metrics']
        raw['array'].update(geometry='custom',tx_positions_m=metrics['tx_positions_m'],rx_positions_m=metrics['rx_positions_m'])
    previous_other=deepcopy(raw['array']['tx_positions_m'])
    raw['array']['rx_count']=17
    target.parameter_widget=SimpleNamespace(value=lambda:deepcopy(raw))
    target.configuration=lambda:(_ for _ in ()).throw(AssertionError('Do not validate mismatched destination before replacing selected side'))
    result,note=import_array(h,destination,'rx','linear')
    assert result['array']['rx_count']==2 and len(result['array']['rx_positions_m'])==2
    assert result['array']['tx_positions_m']==previous_other
    assert raw['array']['rx_count']==17


def test_pulse_metric_rows_use_chinese_labels_and_explicit_units_without_mutating_values():
    result={'kind':'budget','metrics':{'wavelength_m':.03,'pulse_width_actual_s':2e-6,
        'noise_psd_w_hz':1.2e-20,'pulse_response_measurement':'actual sampled waveform and receive/matched filter; zero Doppler; 8x grid interpolation'}}
    original=deepcopy(result)
    headers,rows,note=pulse.table(result,{},'metrics')
    assert rows[0][0]=='参考波长 / m' and rows[1][0]=='实际采样脉宽 / s'
    assert rows[2][0]=='输入噪声功率谱密度 / W/Hz' and '实际采样' in rows[3][1]
    assert result==original
