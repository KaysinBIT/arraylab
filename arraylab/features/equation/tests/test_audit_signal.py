"""Independent RF-waveform and matched-vector regressions from the 2026-10 audit."""
import copy
import math
import numpy as np
import pytest

from arraylab.features.equation.config import default_config, checked_config
from arraylab.features.equation.engine import simulate, synthesize_raw

C=299792458.


def setup(kind="fmcw", fc=10e9):
    p=default_config(kind,fc/1e9)
    p["array"].update(tx_count=1,rx_count=1,tx_positions_m=[[0.,0.,0.]],rx_positions_m=[[0.,0.,0.]],tx_weights=[[1.,0.]],rx_weights=[[1.,0.]])
    p["waveform"].update(pulses=8,doppler_fft=8)
    if kind=="fmcw":
        p["waveform"].update(fs_hz=1e6,samples=64,range_fft=64,bandwidth_hz=10e6,ramp_s=80e-6,adc_start_s=10e-6,pri_s=100e-6,rx_bandwidth_hz=.5e6,range_max_m=500.)
        r=50.123
    else:
        p["waveform"].update(fs_hz=4e6,samples=128,range_fft=509,bandwidth_hz=3.9e6,pulse_width_s=8e-6,pri_s=40e-6,rx_bandwidth_hz=4e6,recovery_s=0.,range_max_m=C*127/(8e6))
        r=48.5*C/(8e6)
    p["scene"]["targets"][0].update(range_m=r,height_m=1.,az_deg=0.,velocity_mps=0.)
    return checked_config(p)


def chirp_minus(t,w):
    """Integrate f(t)=fc-B/2+Bt/T, then evaluate exp(-j phase)."""
    return np.exp(-2j*np.pi*((w["fc_hz"]-w["bandwidth_hz"]/2)*t+w["bandwidth_hz"]/w["ramp_s"]*t*t/2))


def prefactor(p):
    r=p["radar"];w=p["waveform"]
    return math.sqrt(r["power_w"]*10**((r["tx_gain_dbi"]+r["rx_gain_dbi"]-r["loss_db"])/10)*(C/w["fc_hz"])**2/(4*np.pi)**3)


def test_static_displacement_and_receding_motion_have_consistent_phase():
    p=setup();w=p["waveform"];r=p["scene"]["targets"][0]["range_m"];dr=C/w["fc_hz"]/16
    x,_,op=synthesize_raw(p,noise=False,clutter=False)
    q=copy.deepcopy(p);q["scene"]["targets"][0]["range_m"]+=dr
    y,_,_=synthesize_raw(q,noise=False,clutter=False)
    t=op["input_times_s"]
    oracle=chirp_minus(t-2*(r+dr)/C,w)/chirp_minus(t-2*r/C,w)
    actual=y[0,0]/x[0,0];actual/=abs(actual)
    np.testing.assert_allclose(actual,oracle,atol=3e-9,rtol=0)
    q=copy.deepcopy(p);q["scene"]["targets"][0]["velocity_mps"]=dr/w["pri_s"]
    moving,_,_=synthesize_raw(q,noise=False,clutter=False)
    motion=np.angle(moving[1,0]/moving[0,0])
    # Fixed-envelope narrowband Doppler is the declared motion approximation;
    # its error relative to a fully delayed chirp is bounded by swept B/fc.
    assert np.all(np.angle(actual)>0)
    np.testing.assert_allclose(motion,np.pi/4,atol=1e-14)
    assert np.max(abs(np.angle(actual)-motion))<np.pi/4*w["bandwidth_hz"]/w["fc_hz"]


@pytest.mark.parametrize("mode,nt",[("simo",1),("phased",2),("tdm",2),("ddm",2)])
def test_off_axis_array_matches_explicit_delayed_rf_and_event_codes(mode,nt):
    p=setup();p["array"].update(mode=mode,tx_count=nt,rx_count=2,tx_positions_m=[],rx_positions_m=[],tx_weights=[],rx_weights=[],pointing_deg=[17.,0.])
    p["scene"]["targets"][0]["az_deg"]=23.
    p=checked_config(p);w=p["waveform"];r=p["scene"]["targets"][0]["range_m"]
    actual,_,op=synthesize_raw(p,noise=False,clutter=False)
    times=op["input_times_s"]
    target=np.array([math.cos(math.radians(23)),math.sin(math.radians(23)),0.])
    look=np.array([math.cos(math.radians(17)),math.sin(math.radians(17)),0.])
    expected=np.zeros_like(actual)
    for e in range(w["pulses"]):
        for tx,position in enumerate(np.array(p["array"]["tx_positions_m"])):
            code=(float(e%nt==tx) if mode=="tdm" else np.exp(2j*np.pi*e*tx/4) if mode=="ddm" else 1.)
            steering=np.exp(2j*np.pi*w["fc_hz"]/C*np.dot(position,look))
            for rx,receive_position in enumerate(np.array(p["array"]["rx_positions_m"])):
                delay=(2*r-np.dot(position+receive_position,target))/C
                transmitted=chirp_minus(times,w)
                received=prefactor(p)/r**2*chirp_minus(times-delay,w)*steering*code
                expected[e,rx]+=received*np.conj(transmitted)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<3e-9


@pytest.mark.parametrize("fc,bandwidth,extra",[(1e9,800e6,1.1e-9),(77e9,4e9,.1e-9)])
def test_coherent_two_paths_match_independent_centered_upchirp_mixer(fc,bandwidth,extra):
    p=setup(fc=fc);w=p["waveform"]
    w.update(bandwidth_hz=bandwidth,range_max_m=7.)
    if fc==77e9:
        w.update(fs_hz=5e6,samples=256,range_fft=256,ramp_s=64e-6,adc_start_s=4e-6,pri_s=80e-6,rx_bandwidth_hz=2.5e6)
    p["scene"]["targets"][0]["range_m"]=3.
    p["propagation"].update(model="custom",custom_paths=[{"extra_delay_s":extra,"attenuation_db":0.,"phase_deg":180.,"doppler_hz":0.,"departure_deg":[0.,0.],"arrival_deg":[0.,0.]}])
    actual,_,op=synthesize_raw(p,noise=False,clutter=False)
    t=op["input_times_s"];r=3.
    received=prefactor(p)/r**2*chirp_minus(t-2*r/C,w)-prefactor(p)/(r+C*extra/2)**2*chirp_minus(t-2*r/C-extra,w)
    expected=received*np.conj(chirp_minus(t,w))
    assert np.linalg.norm(actual[0,0]-expected)/np.linalg.norm(expected)<5e-9


def test_four_ground_paths_use_negative_time_fresnel_and_coherent_phase():
    p=setup();p["scene"]["targets"][0]["height_m"]=2.
    p["propagation"].update(model="two_ray",conductivity_s_m=.4,epsilon_r=4.,polarization="horizontal")
    actual,_,op=synthesize_raw(p,noise=False,clutter=False)
    w=p["waveform"];r=p["scene"]["targets"][0]["range_m"];rr=math.sqrt(r*r+8.)
    sine=3/rr;t=op["input_times_s"];expected=np.zeros(len(t),complex)
    for rt,rx,bounces in [(r,r,0),(rr,r,1),(r,rr,1),(rr,rr,2)]:
        delay=(rt+rx)/C;u=t-delay
        frequency=w["fc_hz"]+w["bandwidth_hz"]/w["ramp_s"]*(u-w["ramp_s"]/2)
        # Derive lossy-dielectric Fresnel in exp(-j omega t), independently
        # of propagation.py's exp(+j omega t) helper.
        epsilon=4.+1j*.4/(2*np.pi*frequency*8.8541878128e-12)
        root=np.sqrt(epsilon-1+sine*sine)
        gamma=(sine-root)/(sine+root)
        expected+=prefactor(p)/(rt*rx)*gamma**bounces*chirp_minus(u,w)*np.conj(chirp_minus(t,w))
    assert np.linalg.norm(actual[0,0]-expected)/np.linalg.norm(expected)<3e-9


@pytest.mark.parametrize("fraction",[0.,.25,.5,.75])
def test_fractional_lfm_delay_matches_actual_sample_vector_without_peak_bias(fraction):
    p=setup("lfm");w=p["waveform"];tau=(48+fraction)/w["fs_hz"]
    p["scene"]["targets"][0]["range_m"]=C*tau/2
    summary,a=simulate(p,noise=False,clutter=False)
    ri=int(np.argmin(abs(a["ranges_m"]-C*tau/2)));vi=4
    assert abs(a["ranges_m"][ri]-C*tau/2)<1e-9
    t=np.arange(w["samples"])/w["fs_hz"];u=t-tau
    envelope=np.exp(-1j*np.pi*w["bandwidth_hz"]/w["pulse_width_s"]*(u-w["pulse_width_s"]/2)**2)*((u>=0)&(u<w["pulse_width_s"]))
    raw=a["raw"][0,0];amplitude=np.vdot(envelope,raw)/np.vdot(envelope,envelope)
    np.testing.assert_allclose(raw,amplitude*envelope,rtol=1e-12,atol=1e-25)
    expected=w["pulses"]*np.vdot(raw,raw).real/summary["noise"]["independent_input_sample_power_w"]
    actual=a["rd_power"][vi,ri]/a["noise_power"][vi,ri]
    assert actual==pytest.approx(expected,rel=2e-13)
    assert np.argmax(a["rd_power"][vi])==ri


@pytest.mark.parametrize("kind",["rect","lfm"])
def test_hann_efficiency_is_continuous_when_receiver_filter_becomes_identity(kind):
    p=setup(kind);w=p["waveform"]
    w.update(pulse_width_s=2e-6,bandwidth_hz=1e6,range_fft=128,recovery_s=.25e-6)
    p["scene"]["targets"][0]["range_m"]=32*C/(2*w["fs_hz"])
    p["processing"]["fast_window"]="hann"
    _,base=simulate(p,noise=False,clutter=False)
    q=copy.deepcopy(p);q["waveform"]["rx_bandwidth_hz"]*=1-1e-10
    _,filtered=simulate(q,noise=False,clutter=False)
    vi,ri=4,32
    snr=base["rd_power"][vi,ri]/base["noise_power"][vi,ri]
    got=filtered["rd_power"][vi,ri]/filtered["noise_power"][vi,ri]
    assert got==pytest.approx(snr,rel=2e-9)
    q=copy.deepcopy(p);q["processing"]["fast_window"]="rect"
    _,unwindowed=simulate(q,noise=False,clutter=False)
    reference=unwindowed["rd_power"][vi,ri]/unwindowed["noise_power"][vi,ri]
    window=np.hanning(8)
    assert snr/reference==pytest.approx(window.sum()**2/(8*np.vdot(window,window).real),rel=1e-12)


def test_zero_envelope_does_not_evaluate_extrapolated_rf_outside_itu_band():
    p=setup("lfm",1.001e9);w=p["waveform"]
    w.update(pulse_width_s=2e-6,bandwidth_hz=1e6,range_fft=128,recovery_s=.25e-6)
    p["scene"]["targets"][0]["range_m"]=24*C/(2*w["fs_hz"])
    p["propagation"]["atmosphere"]["enabled"]=True
    _,a=simulate(p,noise=False,clutter=False)
    assert np.any(abs(a["raw"])>0)
    q=copy.deepcopy(p);q["waveform"]["fc_hz"]=1e9
    with pytest.raises(ValueError,match="frequency|频率|1.*1000"):
        simulate(q,noise=False,clutter=False)


def test_large_known_channel_aperture_near_target_is_outside_model_coverage():
    p=setup();p["array"].update(mode="phased",tx_count=2,tx_positions_m=[[0.,-.5,0.],[0.,.5,0.]],tx_weights=[])
    p["scene"]["targets"][0].update(range_m=10.,az_deg=30.)
    summary,_=simulate(p,noise=False,clutter=False)
    observation=summary["observability"][0];far=observation["far_field"]
    independent_required=2/(C/(p["waveform"]["fc_hz"]+p["waveform"]["bandwidth_hz"]/2))
    assert far["tx_coordinate_extent_m"]==1.
    assert far["tx_required_distance_m"]==pytest.approx(independent_required)
    assert far["tx_path_distance_m"]==pytest.approx(10.)
    assert far["known_geometry_only"]
    assert not far["valid"]
    assert not observation["observable"]
    assert not observation["physical_observable"]
    assert any("2D²/λ" in warning for warning in summary["warnings"])


def test_normal_small_array_passes_known_geometry_check_without_claiming_element_aperture():
    p=setup();p["array"].update(mode="phased",tx_count=2,rx_count=2,tx_positions_m=[],rx_positions_m=[],tx_weights=[],rx_weights=[])
    summary,_=simulate(p,noise=False,clutter=False)
    observation=summary["observability"][0]
    assert observation["observable"]
    assert observation["far_field"]["valid"]
    assert observation["far_field"]["known_geometry_only"]


def test_coordinate_reference_offset_is_included_in_far_field_check():
    p=setup();p["array"]["tx_positions_m"]=[[0.,1.,0.]]
    p["scene"]["targets"][0]["range_m"]=10.
    summary,_=simulate(p,noise=False,clutter=False)
    observation=summary["observability"][0]
    assert observation["far_field"]["tx_coordinate_extent_m"]==1.
    assert not observation["observable"]
