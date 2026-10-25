"""Real-ADC folding observations, inverse aliases and physical passband bounds."""
from copy import deepcopy
import json
import math

import pytest

import arraylab.features.waveform.ambiguity as adc
from arraylab.features.waveform.core import BASE

C = 299792458


def exact(distance, fs_mhz, slope=12, doppler=0, error=1e-5):
    # Independent phase-fraction construction of a real-sampling observation.
    beat = 2*slope*1e12*distance/C + doppler
    fraction = beat/(fs_mhz*1e6) % 1
    return {'sampleRateMsps': fs_mhz, 'slopeMHzUs': slope,
            'foldedBeatMHz': min(fraction, 1-fraction)*fs_mhz,
            'knownDopplerHz': doppler, 'uncertaintyHz': error}


@pytest.fixture
def scene():
    return {**BASE, 'sampleRateMsps': 4, 'samples': 64, 'rangeFft': 1024,
            'targetRange': 60, 'targetVelocity': 0, 'ifBandwidthMHz': 10}


@pytest.fixture
def options():
    return {'method': 'dual_rate', 'secondSampleRateMsps': 5, 'secondSlopeMHzUs': 9,
            'searchMaxM': 100, 'knownVelocityMps': 0}


def test_two_rate_inverse_uses_signed_alias_branches_not_diagnostic_truth():
    observations = [exact(60, 4), exact(60, 5)]
    result = adc.resolve(observations, {'searchMaxM': 100})
    assert result['status'] == 'resolved' and result['rangeM'] == pytest.approx(60, abs=1e-7)
    assert [(result['candidates'][0][f'foldOrder{i}'], result['candidates'][0][f'sign{i}']) for i in (1, 2)] == [(1, 1), (1, -1)]
    poisoned = adc.resolve([{**m, 'trueRangeM': 999, 'trueBeatMHz': 999} for m in observations], {'searchMaxM': 100})
    assert poisoned['candidates'] == result['candidates']


def test_large_search_retains_periodic_and_mirror_candidates():
    result = adc.resolve([exact(60, 4), exact(60, 5)], {'searchMaxM': 600})
    period = C*20e6/(2*12e12)
    assert result['status'] == 'ambiguous' and result['rangeM'] is result['resolvedRangeM'] is None
    assert [row['rangeM'] for row in result['candidates']] == pytest.approx([60, period-60, period+60, 2*period-60, 2*period+60])


def test_two_slope_inverse_and_coprime_rate_mirror_ambiguity():
    assert adc.resolve([exact(60, 4, 12), exact(60, 4, 9)], {'searchMaxM': 100})['rangeM'] == pytest.approx(60)
    observations = [{'sampleRateMsps': fs, 'slopeMHzUs': 12, 'foldedBeatMHz': .5, 'uncertaintyHz': 1e-5} for fs in (4, 5)]
    result = adc.resolve(observations, {'searchMaxM': 100})
    assert result['status'] == 'ambiguous'
    assert [row['rangeM'] for row in result['candidates']] == pytest.approx([C*.5e6/(2*12e12), C*4.5e6/(2*12e12)])


def test_uncertainty_intersection_retains_separate_regions():
    observations = [exact(60, 4, error=400), exact(60, 5, error=600)]
    observations[0]['foldedBeatMHz'] += .0002; observations[1]['foldedBeatMHz'] -= .0003
    result = adc.resolve(observations, {'searchMaxM': 100})
    assert result['status'] == 'resolved'
    assert result['candidates'][0]['minRangeM'] < 60 < result['candidates'][0]['maxRangeM']
    assert result['candidates'][0]['residualHz'] < 600
    assert len(adc.resolve(observations, {'searchMaxM': 600})['candidates']) == 5


def test_wrong_search_or_observation_or_velocity_prior_does_not_choose_truth():
    observations = [exact(60, 4), exact(60, 5)]
    observations[1]['foldedBeatMHz'] += .05
    assert adc.resolve(observations, {'searchMaxM': 100})['status'] == 'inconsistent'
    assert adc.resolve([exact(60, 4), exact(60, 5)], {'searchMaxM': 30})['status'] == 'inconsistent'
    shifted = [exact(60, 4, doppler=21000), exact(60, 5, doppler=23000)]
    assert adc.resolve(shifted, {'searchMaxM': 100})['rangeM'] == pytest.approx(60)
    assert adc.resolve([{**m, 'knownDopplerHz': 0} for m in shifted], {'searchMaxM': 100})['status'] == 'inconsistent'


def test_negative_beat_dc_and_nyquist_are_not_double_counted():
    zero = adc.resolve([exact(0, 4, doppler=-100000), exact(0, 5, doppler=-120000)], {'searchMaxM': 10})
    assert zero['status'] == 'resolved' and zero['rangeM'] == pytest.approx(0)
    for distance, maximum in [(0, 10), (C*2e6/(2*12e12), 30)]:
        result = adc.resolve([exact(distance, 4), exact(distance, 5)], {'searchMaxM': maximum})
        assert result['status'] == 'resolved' and len(result['candidates']) == 1
        assert result['rangeM'] == pytest.approx(distance)


def test_invalid_redundant_or_unbounded_inverse_parameters_are_blocked():
    first = exact(60, 4)
    for second in [first, {**first, 'uncertaintyHz': math.inf}, {**first, 'foldedBeatMHz': 3},
                   {**first, 'uncertaintyHz': -1}, {**first, 'sampleRateHz': True},
                   {**first, 'slopeHzPerSec': 1e308}]:
        result = adc.resolve([first, second], {'searchMaxM': 100})
        assert result['status'] == 'blocked'
        json.dumps(result, allow_nan=False)
    excessive = {'sampleRateHz': 1, 'slopeHzPerSec': 1e15, 'foldedBeatHz': .1, 'uncertaintyHz': .001}
    assert adc.resolve([excessive, exact(60, 5)], {'searchMaxM': 10000})['status'] == 'blocked'
    assert adc.resolve([first, exact(60, 5)], {'searchMaxM': 10001})['status'] == 'blocked'


def test_real_adc_fft_demo_and_two_slope_measurements(scene, options):
    for method in ('dual_rate', 'dual_slope'):
        result = adc.simulate(scene, {**options, 'method': method})
        assert result['status'] == 'resolved' and result['rangeM'] == pytest.approx(60, abs=.02)
        for measurement in result['measurements']:
            assert len(measurement['observedSpectrum']) == 513
            assert all(math.isfinite(row['x']) and math.isfinite(row['y']) for row in measurement['observedSpectrum'])
            assert 0 <= measurement['foldedBeatMHz'] <= measurement['nyquistMHz']
        json.dumps(result, allow_nan=False)


def test_analog_filter_removes_unobservable_echo(scene, options):
    result = adc.simulate({**scene, 'ifBandwidthMHz': 4}, options)
    assert result['status'] == 'blocked' and result['rangeM'] is None
    assert '模拟IF' in result['reason'] and len(result['measurements']) == 2
    assert all(not row['observable'] for row in result['measurements'])


@pytest.mark.parametrize('bad', [{'secondSampleRateMsps': 1}, {'secondSampleRateMsps': 4},
    {'method': 'dual_slope', 'secondSlopeMHzUs': 12}, {'method': 'dual_slope', 'secondSlopeMHzUs': 200},
    {'knownVelocityMps': math.nan}, {'toleranceBins': 0}, {'searchMaxM': -1}, {'secondSampleRateMsps': 0},
    {'searchMaxM': 10**500}, {'knownVelocityMps': True}])
def test_invalid_second_observation_is_blocked_json_safe(scene, options, bad):
    result = adc.simulate(scene, {**options, **bad})
    assert result['status'] == 'blocked'
    json.dumps(result, allow_nan=False)


def test_simulated_nonuniqueness_and_edge_frequency_gate(scene, options):
    ambiguous = adc.simulate({**scene, 'ifBandwidthMHz': 25}, {**options, 'searchMaxM': 600})
    assert ambiguous['status'] == 'ambiguous' and len(ambiguous['candidates']) == 3
    assert ambiguous['rangeM'] is None
    edge = adc.simulate({**scene, 'targetRange': 0}, options)
    assert all(row['edgeLimited'] and row['uncertaintyHz'] >= row['fftBinHz'] for row in edge['measurements'])
    assert any('DC' in note for note in edge['warnings'])


def test_all_alias_candidates_obey_each_analog_passband():
    observations = [exact(60, 4), exact(60, 5)]
    narrow = adc.resolve([{**m, 'analogBandwidthMHz': 10} for m in observations], {'searchMaxM': 600})
    assert narrow['status'] == 'resolved' and narrow['rangeM'] == pytest.approx(60)
    wide = adc.resolve([{**m, 'analogBandwidthMHz': 25} for m in observations], {'searchMaxM': 600})
    assert wide['status'] == 'ambiguous' and len(wide['candidates']) == 3
    assert all(2*12e12*row['maxRangeM']/C < 25e6 for row in wide['candidates'])
    mismatched = [{**exact(60, 4, slope), 'analogBandwidthMHz': bandwidth} for slope, bandwidth in [(12, 10), (9, 3)]]
    assert adc.resolve(mismatched, {'searchMaxM': 100})['status'] == 'inconsistent'


def test_exact_analog_cutoff_is_excluded_even_with_zero_tolerance():
    distance = C*4.5e6/(2*12e12)
    observations = [{**exact(distance, fs, error=0), 'analogBandwidthHz': 4.5e6} for fs in (4, 5)]
    result = adc.resolve(observations, {'searchMaxM': 100})
    assert all(2*12e12*row['rangeM']/C < 4.5e6 for row in result['candidates'])
    assert all(abs(row['rangeM']-distance) > 1e-10 for row in result['candidates'])


def test_off_does_not_validate_truth_and_tdm_ddm_fast_time_is_same(scene, options):
    off = adc.simulate({'targetRange': math.nan}, {'method': 'off'})
    assert off['status'] == 'off' and not off['measurements']
    tdm, ddm = adc.simulate(scene, options), adc.simulate({**scene, 'mode': 'ddm', 'ddmSlots': 4}, options)
    assert tdm['rangeM'] == pytest.approx(ddm['rangeM'])


def test_input_and_options_are_not_mutated(scene, options):
    original, settings = deepcopy(scene), deepcopy(options)
    adc.simulate(scene, options)
    assert scene == original and options == settings
