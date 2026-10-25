"""Only the newly integrated physical Tx/Rx synthesis and DOA contracts."""
from copy import deepcopy
import math

import pytest

from arraylab.features.waveform import sparse_design as s
from arraylab.framework.parallel import ComputationCancelled

TX, RX = [0, 2, 4], [0, .5, 1, 1.5]
COMMON = {'targetResolutionDeg': 4, 'scanMaxDeg': 20, 'maxSidelobeDb': -10,
          'iterations': 100, 'trials': 2, 'snrDb': 20, 'snapshots': 128, 'seed': 42}


def test_two_element_cosine_and_grating_lobe_rejection():
    result = s.evaluate([0], [0, .5], {'criterion': 'hpbw', 'targetResolutionDeg': 40,
                                    'scanMaxDeg': 0, 'maxApertureLambda': 2, 'iterations': 100})
    expected = 2*math.degrees(math.asin(math.acos(math.sqrt(10**(-3/10)))/(math.pi*.5)))
    assert result['metrics']['hpbwDeg'] == pytest.approx(expected, abs=1e-7)
    assert next(p['y'] for p in result['pattern'] if abs(p['x']-30) < 1e-8) == pytest.approx(-3.01029995664)
    grating = s.evaluate([0], [0, 2], {'criterion': 'hpbw', 'targetResolutionDeg': 20,
                                     'scanMaxDeg': 0, 'maxSidelobeDb': -10, 'iterations': 100})
    assert grating['metrics']['hpbwDeg'] < 20
    assert grating['metrics']['worstSidelobeDb'] == pytest.approx(0, abs=1e-7)
    assert grating['status'] == 'unmet'


def test_source_reference_sparse_search_has_realizable_positions():
    result = s.design({'targetResolutionDeg': 12, 'scanMaxDeg': 20, 'maxSidelobeDb': -8,
                       'iterations': 1500, 'seed': 42})
    assert result['status'] == 'met'
    sums = {tx[0]+rx[0] for tx in result['geometry']['tx'] for rx in result['geometry']['rx']}
    assert result['metrics']['uniqueCount'] == len(sums) == 12
    span = max(sums)-min(sums)
    assert result['metrics']['sparseFillRatio'] == pytest.approx(len(sums)/(round(span/.5)+1))
    assert result['metrics']['sparseFillRatio'] < 1
    assert result['search']['globalOptimality'] is False
    assert result['metrics']['achievedResolutionDeg'] is None
    # The published source candidate is evaluated independently of the seeded
    # annealing trajectory: floating ties may select another equivalent basin.
    reference = s.evaluate([0, .5, 4.5], [0, 1.5, 2.5, 3.5],
                           {'targetResolutionDeg': 12, 'scanMaxDeg': 20, 'maxSidelobeDb': -8})
    assert reference['metrics']['worstHpbwDeg'] == pytest.approx(6.584567048751128, abs=1e-8)
    assert reference['metrics']['worstPairDipDb'] == pytest.approx(7.911803191317311, abs=1e-8)
    assert reference['metrics']['worstSidelobeDb'] == pytest.approx(-11.214234469148412, abs=1e-8)


def test_hpbw_and_two_target_are_distinct_and_intermediate_angles_matter():
    common = {'targetResolutionDeg': 8, 'scanMaxDeg': 20, 'maxSidelobeDb': -8, 'iterations': 100}
    width = s.evaluate([0, 2, 4], [0, .5, 1, 3.5], {**common, 'criterion': 'hpbw'})
    pair = s.evaluate([0, 2, 4], [0, .5, 1, 3.5], {**common, 'criterion': 'two_target'})
    assert width['status'] == 'met'
    assert pair['status'] == 'unmet'
    result = s.evaluate([0, 12, 13.5], [0, .5, 2.5, 4.5],
                        {'criterion': 'two_target', 'targetResolutionDeg': 7, 'scanMaxDeg': 36,
                         'maxSidelobeDb': -1, 'iterations': 100})
    assert [p['centerDeg'] for p in result['scanChecks']] == [0, 12, 24, 36]
    assert result['scanChecks'][0]['resolved'] and result['scanChecks'][-1]['resolved']
    assert not result['scanChecks'][2]['resolved']
    assert result['status'] == 'unmet'


def test_seed_reproducibility_translation_axis_and_single_rx():
    options = {'criterion': 'hpbw', 'targetResolutionDeg': 10, 'scanMaxDeg': 20, 'iterations': 100, 'seed': 711}
    original = deepcopy(options)
    first, second = s.design(options), s.design(options)
    assert options == original and first == second
    rotated = s.design({**options, 'axis': 'elevation'})
    assert rotated['pattern'] == first['pattern']
    assert rotated['geometry']['tx'] == [p[::-1] for p in first['geometry']['tx']]
    shifted = s.evaluate([10], [100, 100.5], {'criterion': 'hpbw', 'targetResolutionDeg': 40, 'scanMaxDeg': 0})
    assert shifted['geometry']['rx'] == [[0, 0], [.5, 0]]
    single = s.design({'txCount': 4, 'rxCount': 1, 'criterion': 'hpbw', 'targetResolutionDeg': 40,
                       'scanMaxDeg': 0, 'maxApertureLambda': 8, 'iterations': 200})
    assert single['status'] == 'met' and single['metrics']['uniqueCount'] == 4


@pytest.mark.parametrize('options', [{'txCount': 12, 'rxCount': 16, 'maxApertureLambda': 1},
                                    {'targetResolutionDeg': math.nan}, {'txCount': 1.5}, {'rxCount': 0},
                                    {'gridLambda': 0}, {'iterations': 99}, {'seed': -1},
                                    {'scanMaxDeg': 70, 'targetResolutionDeg': 20},
                                    {'txCount': 1, 'rxCount': 1}, {'criterion': 'music', 'snapshots': 0},
                                    {'criterion': 'music', 'trials': 1.5}, {'criterion': 'music', 'snrDb': math.inf},
                                    {'criterion': 'dml', 'smoothing': 'spatial'}])
def test_invalid_bounds_do_not_fabricate_a_geometry(options):
    result = s.design(options)
    assert result['status'] == 'invalid' and result['valid'] is False
    assert result['geometry'] is None and result['errors']


@pytest.mark.parametrize('method', s.SUPER_METHODS)
def test_each_actual_algorithm_and_spectrum_semantics(method):
    result = s.evaluate(TX, RX, {**COMMON, 'criterion': method, 'scanMaxDeg': 0}, workers=1)
    assert result['status'] == 'met'
    sr = result['superresolution']
    assert sr['valid'] and sr['status'] == 'evaluated' and sr['method'] == method
    assert len(sr['representative']['estimates']) == 2
    if method in ('esprit', 'dml'):
        assert result['twoTargetPattern'] == []
    else:
        assert any(p['y'] < -3 for p in result['twoTargetPattern'])


def test_holdout_uses_disjoint_seeds_and_all_signed_centres():
    result = s.design({**COMMON, 'criterion': 'music', 'scanMaxDeg': 21, 'trials': 3}, workers=1)
    sr = result['superresolution']
    assert [row['centerDeg'] for row in sr['validation']] == [0, -21, -14, -7, 7, 14, 21]
    assert result['search']['holdoutUsedForSelection'] is False
    assert 1 <= result['search']['shortlistCount'] <= 4
    validation_seeds = [trial['seed'] for row in sr['validation'] for trial in row['details']]
    screening_seeds = {(seed+t*0x9e3779b9) & 0xffffffff for candidate in result['search']['screening']
                       for seed in candidate['seeds'] for t in (1, 2)}
    assert len(set(validation_seeds)) == 21
    assert not screening_seeds.intersection(validation_seeds)
    assert sr['successRate'] == min(row['successRate'] for row in sr['validation'])


def test_coherent_noise_failure_and_explicit_smoothed_or_dml_success():
    options = {**COMMON, 'snrDb': 30, 'snapshots': 256, 'sourceCorrelation': 1, 'scanMaxDeg': 0, 'trials': 3}
    plain = s.evaluate(TX, RX, {**options, 'criterion': 'music'}, workers=1)
    assert plain['status'] == 'unmet'
    music = s.evaluate(TX, RX, {**options, 'criterion': 'music', 'smoothing': 'spatial'}, workers=1)
    dml = s.evaluate(TX, RX, {**options, 'criterion': 'dml'}, workers=1)
    assert music['status'] == dml['status'] == 'met'
    assert music['superresolution']['usedChannelCount'] < 12
    assert dml['superresolution']['usedChannelCount'] == 12


def test_unsupported_structures_resource_bounds_and_duplicate_positions():
    irregular = s.evaluate([0], [0, .7, 2.1, 3.5], {**COMMON, 'criterion': 'esprit', 'scanMaxDeg': 0,
                             'gridLambda': .1, 'minSpacingLambda': .1, 'maxSidelobeDb': -.1, 'trials': 1})
    assert irregular['status'] == 'unmet' and irregular['superresolution']['status'] == 'unsupported'
    duplicate = s.evaluate([0, .5, 1], [0, .5, 1, 1.5], {**COMMON, 'criterion': 'music', 'targetResolutionDeg': 14,
                            'scanMaxDeg': 0, 'maxSidelobeDb': -.1, 'trials': 1})
    assert duplicate['metrics']['channelCount'] == 12
    assert duplicate['superresolution']['uniqueChannelCount'] == duplicate['superresolution']['usedChannelCount'] == 6
    large = s.evaluate([i*4 for i in range(8)], [i*.5 for i in range(8)],
                        {**COMMON, 'criterion': 'music', 'scanMaxDeg': 0, 'maxApertureLambda': 40, 'trials': 1})
    assert large['superresolution']['status'] == 'unsupported'
    assert large['superresolution']['usedChannelCount'] == 64


def test_parallel_validation_matches_serial_and_cancel_is_not_invalid():
    options = {**COMMON, 'criterion': 'music', 'trials': 2}
    serial = s.evaluate(TX, RX, options, workers=1)
    parallel = s.evaluate(TX, RX, options, workers=2)
    for result in (serial, parallel):
        result['superresolution'].pop('execution')
    assert parallel == serial
    serial_design = s.design(options, workers=1)
    parallel_design = s.design(options, workers=2)
    for result in (serial_design, parallel_design):
        result['superresolution'].pop('execution')
    assert parallel_design == serial_design
    with pytest.raises(ComputationCancelled):
        s.design({'iterations': 100}, cancelled=lambda: True)
    calls = [0]
    def cancelled():
        calls[0] += 1
        return calls[0] > 10
    with pytest.raises(ComputationCancelled):
        s.design({'iterations': 5000}, cancelled=cancelled)


def test_pooled_rmse_and_nonoverlapping_matching_tolerance():
    result = s.evaluate([0], [0, .5, 1], {**COMMON, 'criterion': 'mvdr', 'snrDb': -10, 'snapshots': 8,
                                         'trials': 6, 'maxSidelobeDb': -.1}, workers=1)
    sr = result['superresolution']
    complete = [trial for row in sr['validation'] for trial in row['details'] if trial['estimationValid']]
    errors = [error for trial in complete for error in trial['errors']]
    assert sr['rmseDeg'] == pytest.approx(math.sqrt(sum(e*e for e in errors)/len(errors)), abs=1e-10)
    assert sr['estimatedTrialsTotal'] == len(complete)
    assert sr['totalTrials'] == sr['requestedTrials'] == 42
    result = s.evaluate(TX, RX, {**COMMON, 'criterion': 'music', 'targetResolutionDeg': .5,
                                'angleToleranceDeg': 5, 'scanMaxDeg': 0, 'trials': 1})
    assert result['superresolution']['effectiveToleranceDeg'] == .2
