"""Independent delay-residue examples and bounded inverse-solver behavior."""
import json

import pytest

import arraylab.features.waveform.pri as pri


def pair(first=15, second=10, tolerance=0):
    return [{'periodUs': 38, 'delayUs': first, 'toleranceNs': tolerance},
            {'periodUs': 43, 'delayUs': second, 'toleranceNs': tolerance}]


def test_known_53_microsecond_delay_and_observation_only_input():
    observations = pair()
    result = pri.resolve(observations, 20000)
    assert result['status'] == 'resolved'
    assert result['rangeM'] == pytest.approx(299792458 * 53e-6 / 2, abs=1e-7)
    assert result['candidates'][0]['order1'] == result['candidates'][0]['order2'] == 1
    poisoned = pri.resolve({'measurements': observations, 'searchMaxM': 20000, 'targetRangeM': 123})
    assert poisoned['candidates'] == result['candidates']


def test_quantized_demo_contains_truth_and_matches_direct_inverse():
    result = pri.simulate()
    assert result['status'] == 'resolved'
    assert result['candidates'][0]['minM'] <= 8000 <= result['candidates'][0]['maxM']
    assert result['rangeM'] == pytest.approx(299792458 * 53370e-9 / 2, abs=1e-7)
    assert pri.resolve(result['measurements'], 20000)['candidates'] == result['candidates']


def test_large_search_preserves_periodic_aliases_and_same_period_warning():
    result = pri.resolve(pair(), 300000)
    assert result['status'] == 'ambiguous' and result['rangeM'] is None
    assert len(result['candidates']) == 2
    assert result['candidates'][1]['rangeM'] - result['candidates'][0]['rangeM'] == pytest.approx(299792458 * 1634e-6 / 2)
    identical = pri.simulate({'period1Us': 38, 'period2Us': 38})
    assert identical['status'] == 'ambiguous' and len(identical['candidates']) == 4
    assert any('周期相同' in note for note in identical['warnings'])


def test_residue_wrap_interval_and_negative_alias_order_at_origin():
    distance = (38000-.2) * pri.METRES_PER_NS
    result = pri.simulate({'targetRangeM': distance, 'timingResolutionNs': 2})
    assert result['measurements'][0]['foldedDelayUs'] == 0
    assert result['candidates'][0]['minM'] <= distance <= result['candidates'][0]['maxM']
    zero = pri.resolve([{'periodUs': 38, 'delayUs': 37.9998, 'toleranceNs': 1},
                        {'periodUs': 43, 'delayUs': 0, 'toleranceNs': 1}], 10)
    assert zero['status'] == 'resolved'
    assert zero['candidates'][0]['order1'] == -1
    assert zero['candidates'][0]['minM'] == 0
    assert zero['candidates'][0]['maxM'] == pytest.approx(.8 * pri.METRES_PER_NS, abs=1e-8)


def test_inconsistent_observations_and_excluding_prior_do_not_fall_back_to_truth():
    assert pri.resolve(pair(second=10.1), 20000)['status'] == 'inconsistent'
    result = pri.simulate({'targetRangeM': 8000, 'searchMaxM': 1000})
    assert result['status'] == 'inconsistent' and result['rangeM'] is None
    assert result['candidates'] == []


def test_off_mode_supplies_observations_without_inference():
    result = pri.simulate({'method': 'off'})
    assert result['status'] == 'off' and len(result['measurements']) == 2
    assert result['rangeM'] is None and not result['candidates']


@pytest.mark.parametrize('bad', [{'period1Us': 0}, {'period2Us': float('nan')},
    {'timingResolutionNs': 0}, {'timingResolutionNs': 1e-320}, {'timingResolutionNs': 4000},
    {'targetRangeM': -1}, {'searchMaxM': float('inf')}, {'method': 'crt_magic'},
    {'targetRangeM': 10**500}, {'period1Us': True}])
def test_invalid_parameters_are_blocked_without_nonfinite_json(bad):
    result = pri.simulate(bad)
    assert result['status'] == 'blocked'
    json.dumps(result, allow_nan=False)


def test_enumeration_budget_and_invalid_observation_shape():
    result = pri.simulate({'period1Us': .1, 'period2Us': .11, 'timingResolutionNs': 1, 'searchMaxM': 1e7})
    assert result['status'] == 'blocked' and '10000' in result['reason']
    assert pri.resolve(pair(first=38), 20000)['status'] == 'blocked'
    assert pri.resolve([None, {}], 20000)['status'] == 'blocked'


def test_fractional_period_alias_and_half_up_quantization():
    result = pri.simulate({'method': 'staggered_pri', 'period1Us': 38.125, 'period2Us': 43.375})
    assert result['status'] == 'resolved' and result['method'] == 'dual_period'
    assert pri.resolve({'measurements': result['measurements'], 'searchMaxM': 20000})['candidates'] == result['candidates']
    half = pri.simulate({'targetRangeM': 5 * pri.METRES_PER_NS, 'timingResolutionNs': 2})
    assert half['measurements'][0]['delayUs'] == pytest.approx(.006)


def test_touching_intervals_and_search_boundaries():
    result = pri.resolve([{'periodUs': 38, 'delayUs': 1, 'toleranceNs': 1},
                          {'periodUs': 43, 'delayUs': 1.002, 'toleranceNs': 1}], 1000)
    assert result['status'] == 'resolved'
    assert result['candidates'][0]['minM'] == pytest.approx(1001 * pri.METRES_PER_NS)
    assert result['candidates'][0]['maxM'] == pytest.approx(1001 * pri.METRES_PER_NS)
    zero = pri.simulate({'targetRangeM': 0})
    assert zero['status'] == 'resolved' and zero['candidates'][0]['minM'] == 0


def test_inputs_and_default_options_are_not_mutated():
    config = {'method': 'staggered_pri'}
    pri.simulate(config)
    assert config == {'method': 'staggered_pri'} and pri.DEFAULTS['method'] == 'dual_period'
