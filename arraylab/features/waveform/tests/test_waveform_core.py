"""Independent FMCW timing, FFT and boundary checks for the new core only."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

import arraylab.features.waveform.core as wave


def test_adc_bandwidth_and_centered_wavelength_have_independent_units():
    r = wave.calculate()
    m = r['metrics']
    assert r['valid']
    assert m['adcUs'] == pytest.approx(512 / 20)
    assert m['bandwidthMHz'] == pytest.approx(408)
    assert m['observedBandwidthMHz'] == pytest.approx(307.2)
    assert m['centerGHz'] == pytest.approx(77.1896)
    assert m['lambdaMm'] == pytest.approx(299.792458 / 77.1896)
    assert m['rangeResolutionM'] == pytest.approx(299792458 / (2 * 307.2e6))
    assert m['txIntervalUs'] == 114
    assert m['chirpsPerFrame'] == 384
    assert m['frameSamples'] == 512 * 384 * 4


@pytest.mark.parametrize('name', ['lrr', 'mrr', 'srr'])
def test_frozen_source_presets_meet_basic_planning_checks(name):
    r = wave.calculate(wave.PRESETS[name]['params'])
    assert r['valid']
    checks = {row['key']: row['status'] for row in r['checks']}
    assert all(checks[k] == 'pass' for k in ('adc', 'frame', 'if', 'velocity'))


def test_tdm_and_ddm_use_different_revisits_without_claiming_decoding():
    one = wave.calculate({'txCount': 1})['metrics']
    tdm = wave.calculate({'txCount': 3})['metrics']
    ddm = wave.calculate({'txCount': 3, 'mode': 'ddm', 'ddmSlots': 4})['metrics']
    assert tdm['maxVelocityMps'] == pytest.approx(one['maxVelocityMps'] / 3)
    assert tdm['coherentMs'] == pytest.approx(3 * one['coherentMs'])
    assert ddm['txIntervalUs'] == one['txIntervalUs']
    assert ddm['rawMaxVelocityMps'] == one['maxVelocityMps']
    assert ddm['maxVelocityMps'] == pytest.approx(one['maxVelocityMps'] / 4)
    assert ddm['totalTxDutyPercent'] == 3 * ddm['perTxDutyPercent']
    assert tdm['totalTxDutyPercent'] == tdm['dutyPercent']


def test_zero_padding_changes_bins_but_not_physical_resolution():
    a = wave.calculate({'rangeFft': 512, 'dopplerFft': 128})['metrics']
    b = wave.calculate({'rangeFft': 2048, 'dopplerFft': 512})['metrics']
    for key in ('rangeResolutionM', 'velocityResolutionMps', 'coherentMs'):
        assert a[key] == b[key]
    assert a['rangeBinM'] == 4 * b['rangeBinM']
    assert a['velocityBinMps'] == 4 * b['velocityBinMps']


def test_fft_peak_matches_an_independently_bin_centered_if_signal():
    p = {**wave.BASE, 'window': 'rect', 'samples': 128, 'rangeFft': 128,
         'loops': 64, 'dopplerFft': 64, 'targetVelocity': 0, 'txCount': 1}
    target_bin = 17
    beat_hz = target_bin * p['sampleRateMsps'] * 1e6 / p['rangeFft']
    p['targetRange'] = beat_hz * 299792458 / (2 * p['slopeMHzUs'] * 1e12)
    r = wave.simulate(p)
    assert r['valid'] and r['detected']['rangeM'] == pytest.approx(p['targetRange'])
    peak = r['rangeProfile'][target_bin]
    assert peak['y'] == pytest.approx(0, abs=1e-11)
    assert max(row['y'] for i, row in enumerate(r['rangeProfile']) if i != target_bin) <= -150


def test_ddm_phase_coded_samples_produce_three_peaks_and_one_empty_slot():
    p = {**wave.BASE, 'mode': 'ddm', 'window': 'rect', 'loops': 64, 'dopplerFft': 64, 'targetVelocity': 0}
    r = wave.simulate(p); y = np.array([row['y'] for row in r['dopplerProfile']])
    np.testing.assert_allclose(y[[0, 32, 48]], 0, atol=1e-11)
    assert y[16] <= -150
    assert wave.calculate(p)['metrics']['ddmPhasesDeg'] == [0, 90, 180]
    assert '非 DDM 盲解码' in r['detected']['estimationLabel']


def test_velocity_alias_sign_and_target_outside_if_do_not_fabricate_range():
    p = {'window': 'rect', 'targetVelocity': -50, 'targetRange': 400}
    m = wave.calculate(p)['metrics']; r = wave.simulate(p)
    wrapped = ((p['targetVelocity'] + m['rawMaxVelocityMps']) % (2 * m['rawMaxVelocityMps'])) - m['rawMaxVelocityMps']
    assert abs(r['detected']['velocityMps'] - wrapped) <= m['velocityBinMps'] / 2 + 1e-10
    assert not r['detected']['rangeValid']
    assert r['detected']['rangeM'] is None and r['detected']['correctedRangeM'] is None


def test_hann_reduces_far_sidelobes_of_fractional_bin_target():
    p = {**wave.BASE, 'samples': 128, 'rangeFft': 1024, 'targetVelocity': 0, 'window': 'rect'}
    p['targetRange'] = 23.37 * (p['sampleRateMsps'] * 1e6 / p['samples']) * 299792458 / (2 * p['slopeMHzUs'] * 1e12)
    rect = np.array([row['y'] for row in wave.simulate(p)['rangeProfile']])
    hann = np.array([row['y'] for row in wave.simulate({**p, 'window': 'hann'})['rangeProfile']])
    mask = np.abs(np.arange(len(rect)) - np.argmax(rect)) > 40
    assert hann[mask].max() < rect[mask].max() - 20


@pytest.mark.parametrize('change', [
    {'adcStartUs': 40}, {'frameMs': 1}, {'samples': True}, {'loops': 4.5},
    {'rangeFft': 16384}, {'dopplerFft': 100}, {'samples': 1024, 'rangeFft': 512},
    {'startGHz': math.nan}, {'startGHz': None}, {'txCount': 0},
    {'mode': 'ddm', 'ddmSlots': 2}, {'window': 'unknown'}, {'mode': 'fdm'},
    {'targetRange': 1e308}, {'frameMs': 1e-308}, {'samples': 10**1000}, {'unexpected': 1},
    {'slopeMHzUs': 1e-200, 'rampUs': 1e240, 'idleUs': 0, 'frameMs': 1e250, 'targetVelocity': 1e80},
])
def test_invalid_or_excessive_parameters_are_bounded_and_json_safe(change):
    r = wave.calculate(change)
    assert not r['valid'] and r['errors']
    json.dumps(r, allow_nan=False)
    spectrum = wave.simulate(change)
    assert not spectrum['valid'] and spectrum['rangeProfile'] == []
    json.dumps(spectrum, allow_nan=False)


def test_external_slow_time_context_changes_only_slow_time_quantities():
    a = wave.calculate()['metrics']
    b = wave.calculate({}, {'slowTimeIntervalUs': 286})['metrics']
    assert b['txIntervalUs'] == 286
    assert b['velocityResolutionMps'] == pytest.approx(a['velocityResolutionMps'] * 114 / 286)
    for key in ('burstMs', 'chirpUs', 'frameSamples', 'rangeResolutionM', 'rangeBinM', 'dataRateMBps'):
        assert a[key] == b[key]
    assert not wave.calculate({}, {'slowTimeIntervalUs': 100})['valid']


def test_maximum_allowed_fft_is_finite_and_heatmap_is_bounded():
    p = {'samples': 4096, 'rangeFft': 4096, 'loops': 2048, 'dopplerFft': 4096,
         'rampUs': 220, 'slopeMHzUs': 1, 'frameMs': 2000}
    r = wave.simulate(p)
    assert r['valid']
    assert len(r['rangeProfile']) == 2048 and len(r['dopplerProfile']) == 4096
    assert r['rangeDoppler']['width'] <= 128 and r['rangeDoppler']['height'] <= 64
    assert len(r['rangeDoppler']['values']) <= 8192
    json.dumps(r, allow_nan=False)


def test_input_and_presets_are_not_mutated():
    p = deepcopy(wave.PRESETS['mrr']['params']); original = deepcopy(p)
    r = wave.calculate(p); r['params']['startGHz'] = 79
    assert p == original == wave.BASE
    assert wave.PRESETS['mrr']['params']['startGHz'] == 77
