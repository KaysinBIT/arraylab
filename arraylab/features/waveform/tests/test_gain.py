"""Physical gain/energy budget checks for the new WaveLab 1.4 capability."""
from copy import deepcopy
import json
import math

import pytest

from arraylab.features.waveform import ab, core, gain


def single(params=None, acquisition=None):
    return [{'name': 'A', 'params': {**core.BASE, **(params or {})}, 'acquisition': acquisition or {}}]


def test_independent_linear_radar_energy_equation():
    result = gain.calculate(single())
    wavelength = 299792458 / ((77 + 12 * (3 + 512 / 20 / 2) / 1000) * 1e9)
    fast_efficiency = 2 * 511 / (3 * 512)
    slow_efficiency = 2 * 127 / (3 * 128)
    energy_s = 512 / 20e6 * 128
    snr = (0.01 * wavelength ** 2 * energy_s * fast_efficiency * slow_efficiency * 4
           / ((4 * math.pi) ** 3 * 100 ** 4 * 1.380649e-23 * 290 * 10 ** 1.2 * 10 ** 0.6))
    expected = (16 - 10 * math.log10(snr)) / 2
    assert result['valid'] and result['status'] == 'designed'
    assert result['requiredTxGainDbi'] == pytest.approx(expected, abs=1e-11)
    assert result['requiredRxGainDbi'] == pytest.approx(16.80257474, abs=1e-8)
    assert result['profiles'][0]['deliveredSnrDb'] == pytest.approx(16)
    assert result['profiles'][0]['energyTimeMs'] == pytest.approx(3.2768)


@pytest.mark.parametrize('options,expected', [
    ({'rangeM': 200}, 40 * math.log10(2)),
    ({'rcsDbsm': 10}, -10), ({'txPowerDbm': 20}, -10),
    ({'noiseFigureDb': 15}, 3), ({'systemLossDb': 9}, 3),
    ({'requiredSnrDb': 16}, 3), ({'marginDb': 6}, 3),
    ({'temperatureK': 580}, 10 * math.log10(2)),
])
def test_energy_budget_scaling_laws(options, expected):
    reference = gain.calculate(single())
    result = gain.calculate(single(), options)
    assert result['requiredGainSumDb'] - reference['requiredGainSumDb'] == pytest.approx(expected)


def test_no_hidden_transmit_gain_and_explicit_receive_combining():
    reference = gain.calculate(single())
    one_tx = gain.calculate(single({'txCount': 1}))
    one_rx = gain.calculate(single(), {'combining': 'single_rx'})
    assert one_tx['requiredGainSumDb'] == pytest.approx(reference['requiredGainSumDb'])
    assert one_rx['requiredGainSumDb'] - reference['requiredGainSumDb'] == pytest.approx(10 * math.log10(4))
    assert one_rx['profiles'][0]['rxCombiningFactor'] == 1


def test_total_active_tx_power_divides_only_simultaneous_ddm_channels():
    tdm = gain.calculate(single(), {'powerBasis': 'total_active'})
    reference = gain.calculate(single())
    ddm_per = gain.calculate(single({'mode': 'ddm'}))
    ddm_total = gain.calculate(single({'mode': 'ddm'}), {'powerBasis': 'total_active'})
    assert tdm['requiredGainSumDb'] == pytest.approx(reference['requiredGainSumDb'])
    assert ddm_per['requiredGainSumDb'] == pytest.approx(reference['requiredGainSumDb'])
    assert ddm_total['requiredGainSumDb'] - ddm_per['requiredGainSumDb'] == pytest.approx(10 * math.log10(3))
    assert ddm_total['profiles'][0]['powerPerTxW'] == pytest.approx(.01 / 3)


@pytest.mark.parametrize('count', [4, 128, 512, 4096])
def test_finite_symmetric_hann_efficiency(count):
    assert gain.window_efficiency(count, 'hann') == pytest.approx(2 * (count - 1) / (3 * count))
    assert gain.window_efficiency(count, 'rect') == 1


@pytest.mark.parametrize('count,window', [(2, 'hann'), (0, 'rect'), (16385, 'hann'), (1.5, 'rect'),
                                         (True, 'rect'), (float('inf'), 'rect'), (32, 'unknown')])
def test_window_invalid_input_is_rejected(count, window):
    with pytest.raises(ValueError):
        gain.window_efficiency(count, window)


def test_no_energy_gain_from_padding_idle_or_revisit_gap():
    reference = gain.calculate(single())
    cases = [single({'rangeFft': 4096, 'dopplerFft': 1024}),
             single({'idleUs': 30, 'frameMs': 100}),
             single({'frameMs': 100}, {'slowTimeIntervalUs': 500})]
    for case in cases:
        result = gain.calculate(case)
        assert result['valid']
        assert result['requiredGainSumDb'] == pytest.approx(reference['requiredGainSumDb'])
        assert result['profiles'][0]['energyTimeMs'] == reference['profiles'][0]['energyTimeMs']


def test_actual_samples_and_chirp_count_determine_energy():
    reference = gain.calculate(single({'window': 'rect'}))
    twice = gain.calculate(single({'window': 'rect', 'loops': 256, 'dopplerFft': 256, 'frameMs': 100}))
    half_samples = gain.calculate(single({'window': 'rect', 'samples': 256}))
    assert twice['requiredGainSumDb'] - reference['requiredGainSumDb'] == pytest.approx(-10 * math.log10(2))
    ratio = half_samples['profiles'][0]['lambdaMm'] / reference['profiles'][0]['lambdaMm']
    assert half_samples['requiredGainSumDb'] - reference['requiredGainSumDb'] == pytest.approx(10 * math.log10(2) - 20 * math.log10(ratio))


@pytest.mark.parametrize('mode,fixed_key,solved_key', [('solve_rx', 'requiredTxGainDbi', 'requiredRxGainDbi'),
                                                     ('solve_tx', 'requiredRxGainDbi', 'requiredTxGainDbi')])
def test_known_channel_gain_solves_other_side(mode, fixed_key, solved_key):
    result = gain.calculate(single(), {'gainMode': mode, 'fixedGainDbi': 12})
    assert result[fixed_key] == 12
    assert result[solved_key] == pytest.approx(result['requiredGainSumDb'] - 12)


def test_effective_aperture_and_efficiency_do_not_increase_snr():
    result = gain.calculate(single())
    less_efficient = gain.calculate(single(), {'apertureEfficiency': .3})
    profile = result['profiles'][0]
    area = 10 ** (result['requiredTxGainDbi'] / 10) * (profile['lambdaMm'] / 1000) ** 2 / (4 * math.pi)
    assert profile['txEffectiveAreaM2'] == pytest.approx(area)
    assert profile['txPhysicalAreaM2'] == pytest.approx(area / .6)
    assert profile['txEquivalentSquareSideMm'] == pytest.approx(math.sqrt(area / .6) * 1000)
    assert less_efficient['profiles'][0]['txPhysicalAreaM2'] == pytest.approx(2 * profile['txPhysicalAreaM2'])
    assert less_efficient['requiredGainSumDb'] == result['requiredGainSumDb']


def test_ab_common_gain_uses_limiting_profile_without_energy_fusion():
    b = {**core.BASE, 'slopeMHzUs': 18, 'rampUs': 42, 'idleUs': 6, 'loops': 64}
    analysis = ab.calculate(core.BASE, b, {'enabled': True, 'layout': 'blocks', 'switchGapUs': 5})
    assert analysis['valid']
    profiles = [{'name': name, 'params': profile['params'], 'acquisition': profile['acquisition']}
                for name, profile in analysis['profiles'].items()]
    result = gain.calculate(profiles, {'rangeM': 50})
    independent = [gain.calculate([profile], {'rangeM': 50}) for profile in profiles]
    assert result['requiredGainSumDb'] == max(row['requiredGainSumDb'] for row in independent)
    assert result['limitingProfile'] == 'B'
    assert all(profile['deliveredSnrDb'] >= 16 - 1e-10 for profile in result['profiles'])
    assert result['profiles'][0]['budgetSlackDb'] > 2.9


def test_if_failure_does_not_hide_mathematical_budget_and_velocity_is_warning():
    far = gain.calculate(single(), {'rangeM': 500})
    assert far['valid'] and far['status'] == 'waveform_limited'
    assert not far['profiles'][0]['observable']
    assert math.isfinite(far['requiredTxGainDbi'])
    assert any(check['key'] == 'A.if' and check['status'] == 'fail' for check in far['checks'])
    moving = gain.calculate(single(), {'rangeM': 50, 'radialVelocityMps': 50})
    assert moving['status'] == 'designed'
    assert any('速度' in warning for warning in moving['warnings'])


@pytest.mark.parametrize('options', [
    {'rangeM': 0}, {'rcsDbsm': float('nan')}, {'txPowerDbm': float('inf')},
    {'noiseFigureDb': -1}, {'systemLossDb': -1}, {'temperatureK': 0},
    {'apertureEfficiency': 0}, {'powerBasis': 'eirp'}, {'combining': 'virtual_channels'},
    {'gainMode': 'unknown'}, {'rangeM': True}, {'rangeM': '100'}, [],
])
def test_invalid_options_return_no_gain(options):
    result = gain.calculate(single(), options)
    assert not result['valid'] and result['status'] == 'invalid'
    assert result['requiredTxGainDbi'] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('profiles', [[], None, [{'name': 'A'}], [{'name': 'Z', 'params': {}}],
                                     [{'name': 'A', 'params': {}}, {'name': 'A', 'params': {}}]])
def test_invalid_profiles_are_rejected(profiles):
    assert gain.calculate(profiles)['status'] == 'invalid'


def test_invalid_waveform_and_acquisition_have_no_gain_result():
    invalid = gain.calculate(single({'samples': 4096}))
    assert invalid['status'] == 'invalid' and invalid['requiredGainSumDb'] is None
    context = gain.calculate(single(acquisition={'slowTimeIntervalUs': float('inf')}))
    assert context['status'] == 'invalid' and context['requiredGainSumDb'] is None


def test_bounded_extremes_remain_finite_json():
    result = gain.calculate(single(), {'rangeM': 100000, 'rcsDbsm': -100, 'txPowerDbm': -100,
        'noiseFigureDb': 60, 'systemLossDb': 100, 'requiredSnrDb': 80, 'marginDb': 60, 'temperatureK': 5000})
    assert result['valid'] and result['status'] == 'waveform_limited'
    json.dumps(result, allow_nan=False)
    assert math.isfinite(result['requiredTxGainDbi'])


def test_unbounded_waveform_frequency_overflow_returns_invalid_without_throwing():
    result = gain.calculate(single({'startGHz': 1e250}), {'gainMode': 'solve_tx'})
    assert not result['valid'] and result['requiredTxGainDbi'] is None
    assert '有限数值' in result['errors'][0]
    json.dumps(result, allow_nan=False)


def test_inputs_and_default_metadata_are_not_mutated():
    profiles = single()
    options = deepcopy(gain.DEFAULTS)
    before = deepcopy((profiles, options, gain.DEFAULTS))
    result = gain.calculate(profiles, options)
    result['assumptions'].clear()
    result['sources'][0]['title'] = 'changed'
    assert (profiles, options, gain.DEFAULTS) == before
    assert gain.ASSUMPTIONS and gain.SOURCES[0]['title'] != 'changed'


def test_layout_warning_uses_fixed_reference_frequency_without_changing_gain():
    result = gain.calculate(single())
    before = deepcopy(result)
    # 5 lambda at 77 GHz fits the equivalent square; at 160 GHz it does not.
    geometry = {'tx': [[0, 0], [5, 0]], 'rx': [[0, 0]]}
    assert not gain.layout_warnings(result, geometry)
    warnings = gain.layout_warnings(result, {**geometry, 'referenceFrequencyGHz': 160})
    assert len(warnings) == 1 and 'TX' in warnings[0]
    assert result == before


def test_layout_warning_without_reference_tracks_profile_wavelength_and_both_kinds():
    result = gain.calculate(single())
    geometry = {'tx': [[0, 0], [.5, 0]], 'rx': [[0, 0], [0, .5]]}
    warnings = gain.layout_warnings(result, geometry)
    assert len(warnings) == 2
    assert any('TX' in line for line in warnings) and any('RX' in line for line in warnings)
    assert gain.layout_warnings({'valid': False}, geometry) == []


@pytest.mark.parametrize('reference', [0, -1, True, float('inf')])
def test_invalid_layout_reference_is_rejected(reference):
    with pytest.raises(ValueError):
        gain.layout_warnings(gain.calculate(single()), {'tx': [[0, 0]], 'rx': [[0, 0]], 'referenceFrequencyGHz': reference})
