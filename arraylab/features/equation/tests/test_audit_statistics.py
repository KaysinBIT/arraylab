"""Adversarial acceptance checks using independent configurations and real signals."""
import copy
import math

import numpy as np
import pytest

from arraylab.features.equation.config import C, checked_config, default_config
from arraylab.features.equation.detection import run_detector
from arraylab.features.equation.engine import simulate
from arraylab.features.equation.statistics import (_seed, _truth_gates, apparent_range, calibrate, distance_scan,
                                    roc, snr_experiment, statistics)


def audit_case(trials=128):
    p=default_config("fmcw",10)
    w=p["waveform"]
    w.update(fs_hz=1e6,samples=32,range_fft=32,bandwidth_hz=10e6,ramp_s=40e-6,
             adc_start_s=5e-6,pri_s=50e-6,rx_bandwidth_hz=.5e6,range_max_m=280.,
             pulses=16,doppler_fft=16)
    r=4*w["fs_hz"]/w["samples"]*C/(2*w["bandwidth_hz"]/w["ramp_s"])
    p["array"].update(mode="simo",tx_count=1,rx_count=1,tx_positions_m=[],rx_positions_m=[],tx_weights=[],rx_weights=[])
    p["scene"]["targets"][0].update(range_m=r,height_m=1.,velocity_mps=0.,rcs_m2=1.)
    p["processing"].update(fast_window="rect",slow_window="rect",looks=1)
    p["detection"].update(method="fixed",fixed_snr_db=13.,training=[2,1],guard=[1,0])
    p["experiment"].update(trials=trials,cut_range_m=r,cut_velocity_mps=0.)
    p["requirements"].update(pd=.9,pfa=.05)
    return checked_config(p)


def narrow_clutter_case():
    p=audit_case()
    r=p["scene"]["targets"][0]["range_m"]
    p["scene"]["targets"][0]["rcs_m2"]=1e-12
    p["experiment"]["cut_velocity_mps"]=C/10e9/(2*16*50e-6)*4
    p["clutter"].update(kind="gaussian",cnr_db=25.,scatterers=1,range_min_m=r*.9999,range_max_m=r*1.0001,
                        doppler_hz=0.,doppler_width_hz=0.,az_span_deg=0.)
    return p


def test_quiet_h0_cut_cannot_validate_clutter_as_a_target_at_another_cut():
    p=narrow_clutter_case()
    result,arrays=statistics(p)
    assert not result["validated"]
    assert not result["cut_covers_primary_target"]
    assert result["fixed_cut_pd"]["count"]==0
    assert result["pfa"]["count"]==0
    assert result["association_pd"]["count"]==p["experiment"]["trials"]
    assert np.count_nonzero(arrays["trial_results"][:,1])==0
    p["experiment"]["cut_velocity_mps"]=0.
    colocated,_=statistics(p)
    assert colocated["cut_covers_primary_target"]
    assert colocated["pfa"]["count"]==p["experiment"]["trials"]
    assert not colocated["validated"]


def test_roc_keeps_h1_at_same_quiet_predeclared_cut_as_h0():
    p=narrow_clutter_case();p["experiment"]["trials"]=32
    result,arrays=roc(p)
    # Quiet CUT samples stay at the thermal scale in BOTH branches despite a
    # very strong clutter peak at the target position.
    assert np.mean(arrays["h0_scores"])<2.
    assert np.mean(arrays["h1_scores"])<2.
    assert result["cut"]["velocity_mps"]==p["experiment"]["cut_velocity_mps"]


def test_fmcw_gate_uses_beat_fft_range_spacing_not_pulsed_delay_spacing():
    p=checked_config(default_config())
    expected=C/(2*(p["waveform"]["bandwidth_hz"]/p["waveform"]["ramp_s"])*(p["waveform"]["samples"]/p["waveform"]["fs_hz"]))
    assert _truth_gates(p)[0]==pytest.approx(expected,rel=1e-14)
    assert _truth_gates(p)[0]<2.


def test_unmodelled_fmcw_range_migration_cannot_pass_statistical_acceptance():
    p=audit_case()
    p["waveform"].update(bandwidth_hz=1e9,range_max_m=2.8)
    target=p["scene"]["targets"][0]
    target["range_m"]/=100
    target["velocity_mps"]=C/10e9/(2*16*50e-6)*4
    p["experiment"].update(cut_velocity_mps=target["velocity_mps"],
                           cut_range_m=apparent_range(p,target["range_m"],target["velocity_mps"]))
    result,_=statistics(p)
    assert result["pd"]["lower"]>.9 and result["pfa"]["upper"]<.05
    assert not result["motion_within_model"]
    assert not result["validated"]
    assert any("距离徙动" in warning for warning in result["warnings"])
    p["experiment"]["trials"]=2
    snr,_=snr_experiment(p)
    assert not snr["motion_within_model"]
    assert not snr["energy_equation_passed"]


def test_known_32bit_seed_collisions_and_calibration_leakage_are_removed():
    # These exact pairs collided in the old single-uint32 derivation.
    assert _seed(42,310,6600,0)!=_seed(42,310,20528,0)
    assert _seed(42,310,58019,0)!=_seed(42,310,64157,0)
    assert _seed(42,210,63598)!=_seed(42,310,28382,0)
    assert _seed(42,210,63598)>2**32
    seeds={_seed(42,purpose,i,branch,stream) for purpose in (210,310,320,510)
           for i in range(64) for branch in (0,1) for stream in (0,1,2)}
    assert len(seeds)==4*64*2*3


def test_distance_scan_aligns_both_cut_coordinates_and_records_distinct_streams():
    p=audit_case()
    target=p["scene"]["targets"][0]
    target["velocity_mps"]=C/10e9/(2*16*50e-6)*2
    r=target["range_m"]
    p["experiment"].update(distance_min_m=r*.999,distance_max_m=r*1.001,distance_count=2,
                           cut_velocity_mps=-target["velocity_mps"])
    p["detection"].update(threshold_mode="calibrated",pfa=.2)
    result,_=distance_scan(p)
    assert len(result["points"])==2
    for point in result["points"]:
        assert point["actual_cut"]["velocity_mps"]==pytest.approx(target["velocity_mps"])
        assert point["cut_covers_primary_target"]
        assert point["random_stream"]["seed"]==42
        assert point["random_stream"]["derived_seed_bits"]==128
    assert [point["random_stream"]["stream"] for point in result["points"]]==[1,2]


def test_noncoherent_power_ratio_uses_per_look_reference_without_fake_loss():
    p=audit_case(16);p["processing"]["looks"]=2;p["waveform"]["doppler_fft"]=8
    result,_=snr_experiment(p)
    assert result["energy_reference_applicable"]
    assert result["energy_equation_passed"]
    assert result["discrete_loss_db"]==pytest.approx(0.,abs=1e-10)
    assert result["total_energy_reference_snr_db"]-result["power_ratio_reference_snr_db"]==pytest.approx(10*math.log10(2),abs=1e-10)
    assert result["theoretical_snr_db"]==pytest.approx(result["power_ratio_reference_snr_db"],abs=1e-10)


def test_cell_total_power_ratio_has_explicit_name_and_legacy_alias():
    p=audit_case(1)
    _,arrays=simulate(p)
    result=run_detector(p,arrays)
    assert result["detections"]
    for point in result["detections"]:
        iv,ir=point["velocity_index"],point["range_index"]
        expected=10*math.log10(arrays["rd_power"][iv,ir]/arrays["noise_power"][iv,ir])
        assert point["cell_power_to_noise_db"]==pytest.approx(expected)
        assert point["cell_snr_db"]==point["cell_power_to_noise_db"]
    assert "total measured cell power" in result["cell_power_ratio_definition"]


def test_calibration_statistics_roc_replay_exactly_with_disjoint_actual_streams(monkeypatch):
    import arraylab.features.equation.statistics as module
    original=module._simulate
    calls=[]
    def observed(p,seed,*args,**kwargs):
        calls.append(seed)
        return original(p,seed,*args,**kwargs)
    monkeypatch.setattr(module,"_simulate",observed)
    p=audit_case();p["detection"]["pfa"]=.2
    first,a=calibrate(p,_stream=2);calibration_seeds=set(calls);calls.clear()
    repeat,b=calibrate(p,_stream=2)
    assert first==repeat
    np.testing.assert_array_equal(a["calibration_scores"],b["calibration_scores"])
    assert first["ready"]
    p["detection"].update(threshold_mode="calibrated",calibration_factor=first["factor"],calibration_hash=first["calibration_hash"])
    calls.clear();first,a=statistics(p,_stream=2);statistics_seeds=set(calls);calls.clear()
    repeat,b=statistics(p,_stream=2)
    assert first==repeat
    np.testing.assert_array_equal(a["trial_results"],b["trial_results"])
    np.testing.assert_array_equal(a["raw"],b["raw"])
    calls.clear();first,a=roc(p);roc_seeds=set(calls);calls.clear()
    repeat,b=roc(p)
    assert first==repeat
    for key in ("h0_scores","h1_scores"):np.testing.assert_array_equal(a[key],b[key])
    assert len(calibration_seeds)==128
    assert len(statistics_seeds)==len(roc_seeds)==256
    assert calibration_seeds.isdisjoint(statistics_seeds|roc_seeds)
    assert statistics_seeds.isdisjoint(roc_seeds)


def test_strong_coincident_other_target_cannot_validate_a_negligible_primary_target(monkeypatch):
    import arraylab.features.equation.statistics as module
    original=module._simulate;calls=[]
    def observed(p,seed,*args,**kwargs):
        calls.append(seed)
        return original(p,seed,*args,**kwargs)
    monkeypatch.setattr(module,"_simulate",observed)
    p=audit_case()
    second=copy.deepcopy(p["scene"]["targets"][0]);second["name"]="strong interfering target"
    p["scene"]["targets"][0]["rcs_m2"]=1e-12
    p["scene"]["targets"].append(second)
    result,arrays=statistics(p)
    assert result["cut_covers_primary_target"]
    assert result["pd"]["count"]==result["association"]["count"]==128
    assert result["pfa"]["count"]==0
    assert result["background_pfa"]["count"]==128
    assert result["background_pfa"]["lower"]>.95
    assert result["additional_background_trials"]==128
    assert not result["background_pfa_aliases_classic"]
    assert result["background_random_stream"]["purpose"]==320
    assert int(arrays["background_trial_events"].sum())==128
    assert not result["validated"]
    assert len(calls)==len(set(calls))==384
    expected={_seed(42,purpose,i,branch) for i in range(128)
              for purpose,branch in ((310,0),(310,1),(320,0))}
    assert set(calls)==expected


def test_single_target_background_pfa_alias_does_not_add_trials():
    p=audit_case(4)
    result,arrays=statistics(p)
    assert result["background_pfa"]==result["pfa"]
    assert result["additional_background_trials"]==0
    assert result["background_pfa_aliases_classic"]
    assert result["background_random_stream"] is None
    np.testing.assert_array_equal(arrays["background_trial_events"],arrays["trial_results"][:,2])
