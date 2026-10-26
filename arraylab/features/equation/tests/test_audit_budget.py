"""Regression probes from the independent formula/receiver-boundary audit."""
import copy
import math

import numpy as np
import pytest

from arraylab.features.equation.budget import calculate, observability
from arraylab.features.equation.config import C, checked_config, complex_weights, default_config, event_plan, normalize_config, resource_estimate, validate_config
from arraylab.features.equation.engine import simulate


@pytest.mark.parametrize("mode,kind,tx,looks,window,minimum", [
    ("simo", "fmcw", 1, 1, "rect", 2),
    ("simo", "fmcw", 1, 2, "rect", 2),
    ("tdm", "fmcw", 4, 1, "rect", 4),
    ("cdm", "rect", 4, 2, "hann", 24),
    ("ddm", "fmcw", 3, 2, "hann", 24),
])
def test_inverse_minimum_is_an_executable_event_count(mode, kind, tx, looks, window, minimum):
    p = default_config(kind)
    p["array"].update(mode=mode, tx_count=tx)
    p["radar"]["power_w"] = 1e6
    p["waveform"].update(pulses=24, doppler_fft=24)
    p["processing"].update(looks=looks, slow_window=window)
    p = checked_config(p)
    result = calculate(p)
    assert result["inverse"]["required_events"] == minimum
    p["waveform"]["pulses"] = result["inverse"]["required_events"]
    assert validate_config(p) == []
    assert calculate(p)["snr_margin_db"] >= 0


def test_fir_history_overlap_is_rejected_by_configuration_and_budget():
    p = normalize_config(default_config("rect"))
    p["waveform"]["samples"] = 800  # nominal 40 us fits PRI; 32 FIR samples do not
    assert any("FIR" in error for error in validate_config(p))
    assert not calculate(p)["valid"]
    p["waveform"]["rx_bandwidth_hz"] = p["waveform"]["fs_hz"]
    assert validate_config(p) == []


def test_slow_receiver_quadratic_allocation_is_rejected_without_allocating_it():
    p = default_config()
    p["array"].update(rx_count=1)
    p["waveform"].update(samples=2, range_fft=2, pulses=4096, doppler_fft=4096)
    p["experiment"]["memory_mib"] = 64
    p = normalize_config(p)
    # One complex [velocity,event,Rx] array alone exceeds the user's limit.
    unavoidable_bytes = 4096*4096*16
    estimate = resource_estimate(p)
    assert estimate["components"]["slow_operator_workspace_bytes"] >= unavoidable_bytes
    assert estimate["estimated_bytes"] >= unavoidable_bytes
    assert any("内存" in error for error in validate_config(p))
    assert not calculate(p)["valid"]


@pytest.mark.parametrize("kind", ["fmcw", "rect", "lfm"])
def test_default_projects_fit_the_conservative_memory_contract(kind):
    p = normalize_config(default_config(kind))
    assert resource_estimate(p)["estimated_mib"] < p["experiment"]["memory_mib"]
    assert validate_config(p) == []


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_weight_scale_invariance_survives_finite_float_extremes(scale):
    values = np.array([[1., .2], [2., -.3]])
    np.testing.assert_allclose(complex_weights(values*scale, 2), complex_weights(values, 2), rtol=2e-15)
    p = default_config()
    p["array"].update(mode="tdm", tx_count=2, tx_weights=(values*scale).tolist())
    p["radar"]["power_basis"] = "total"
    p = checked_config(p)
    plan = event_plan(p)
    np.testing.assert_allclose(plan["per_tx_power_w"]*np.sum(abs(plan["weights"])**2, axis=1), p["radar"]["power_w"])
    assert calculate(p)["valid"]


def fmcw_grid():
    p = default_config("fmcw", 10)
    p["waveform"].update(fs_hz=1e6, samples=64, range_fft=64, bandwidth_hz=10e6,
        ramp_s=80e-6, adc_start_s=10e-6, pri_s=100e-6, rx_bandwidth_hz=.5e6,
        range_max_m=500., pulses=24, doppler_fft=24)
    p["scene"]["targets"][0]["range_m"] = 4*1e6/64*C/(2*10e6/80e-6)
    p["requirements"]["range_m"] = p["scene"]["targets"][0]["range_m"]
    return checked_config(p)


def test_noncoherent_output_power_reference_matches_actual_sum_without_false_loss():
    p = fmcw_grid()
    p["processing"].update(looks=2, slow_window="hann", fast_window="hann")
    p["array"].update(mode="tdm", tx_count=3, tx_weights=[[1., .2], [2., -.3], [.4, .7]], tx_positions_m=[])
    p = checked_config(p)
    budget = calculate(p)
    _, arrays = simulate(p, noise=False, clutter=False)
    ri = int(np.argmin(abs(arrays["ranges_m"]-p["requirements"]["range_m"])))
    vi = int(np.argmin(abs(arrays["velocities_mps"])))
    actual = 10*math.log10(arrays["rd_power"][vi, ri]/arrays["noise_power"][vi, ri])
    assert actual == pytest.approx(budget["summed_power_snr_db"], abs=2e-10)
    assert budget["energy_snr_db"]-actual == pytest.approx(10*math.log10(2), abs=2e-10)
    assert budget["output_snr_db"] == budget["energy_snr_db"]


def test_sampling_grid_coverage_is_distinct_from_physical_observability():
    p = fmcw_grid()
    # Last included FFT center is 26 bins. 499 m is within display/IF limits,
    # but beyond the actual final center at about 487.16 m.
    result = observability(p, 499., 0.)
    assert result["physical_observable"]
    assert not result["within_processed_grid"]
    assert not result["observable"]
    assert any("校核网格" in reason for reason in result["reasons"])
    assert result["if_sampling_limit_m"] > result["processed_range_max_m"]
    # The negative Nyquist endpoint exists; the positive one aliases to it.
    vmax = C/(4*p["waveform"]["fc_hz"]*p["waveform"]["pri_s"])
    assert observability(p, 200., -vmax)["unambiguous_doppler"]
    assert not observability(p, 200., vmax)["unambiguous_doppler"]


def test_budget_and_engine_share_the_known_geometry_far_field_guard():
    p = fmcw_grid()
    p["array"].update(rx_count=2, rx_positions_m=[[0., -2., 0.], [0., 2., 0.]], rx_weights=[])
    p["scene"]["targets"][0]["range_m"] = 100.
    p["requirements"]["range_m"] = 100.
    p = checked_config(p)
    budget = calculate(p)["observability"]
    summary, _ = simulate(p, noise=False, clutter=False)
    actual = summary["observability"][0]
    # At the highest instantaneous RF, 2*D^2/lambda = 1067.9388 m.
    expected_limit = 2*4**2*(10e9+5e6)/C
    assert budget["far_field"]["rx_required_distance_m"] == pytest.approx(expected_limit)
    assert budget["far_field"] == actual["far_field"]
    assert budget["within_processed_grid"]
    assert budget["physical_observable"] == actual["physical_observable"] == False
    assert budget["observable"] == actual["observable"] == False
    assert any("远场" in reason for reason in budget["reasons"])


@pytest.mark.parametrize("kind", ["fmcw", "rect"])
def test_budget_and_engine_require_the_same_filter_history(kind):
    p = fmcw_grid() if kind == "fmcw" else checked_config(default_config("rect"))
    w = p["waveform"]
    if kind == "fmcw":
        w.update(adc_start_s=1e-6, rx_bandwidth_hz=.4e6)
        delay = .5e-6
    else:
        delay = w["pulse_width_s"]+w["recovery_s"]+.1e-6
    p["scene"]["targets"][0]["range_m"] = C*delay/2
    p["requirements"]["range_m"] = C*delay/2
    expected = observability(p, C*delay/2)
    summary, _ = simulate(p, noise=False, clutter=False)
    actual = summary["observability"][0]
    assert expected["complete_echo"] and not expected["filter_settled"]
    assert expected["observable"] == actual["observable"] == False
    assert expected["filter_settled"] == actual["filter_settled"]


@pytest.mark.parametrize("kind", ["fmcw", "lfm"])
def test_actual_chirp_band_must_be_positive_and_inside_enabled_atmosphere_support(kind):
    p = default_config(kind, 1)
    p["waveform"].update(bandwidth_hz=2e9, fs_hz=4e9, pulse_width_s=1e-9)
    assert any("下边缘" in error for error in validate_config(normalize_config(p)))
    p = default_config(kind, 1)
    p["propagation"]["atmosphere"]["enabled"] = True
    assert any("完整实际射频频带" in error for error in validate_config(normalize_config(p)))
    p = default_config(kind, 1000)
    p["propagation"]["atmosphere"]["enabled"] = True
    assert any("完整实际射频频带" in error for error in validate_config(normalize_config(p)))
    # A rectangular pulse has no swept center-frequency offset in this model.
    p = default_config("rect", 1)
    p["propagation"]["atmosphere"]["enabled"] = True
    assert validate_config(normalize_config(p)) == []
