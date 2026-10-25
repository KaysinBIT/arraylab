"""Focused physical/algorithm contracts for the new WaveLab 1.4 DOA port."""
import json
import math

import numpy as np
import pytest

from arraylab.features.waveform import doa, doa_dml
from arraylab.framework.parallel import ComputationCancelled

ULA = [i*.5 for i in range(12)]


def unpack(cov):
    return (np.asarray(cov['re'])+1j*np.asarray(cov['im'])).reshape(cov['n'], cov['n'])


def pack(matrix):
    return dict(n=len(matrix), re=matrix.real.ravel().tolist(), im=matrix.imag.ravel().tolist())


def fixture(positions, angles, coherent=False, noise=0):
    atoms = np.exp(2j*np.pi*np.asarray(positions)[:, None]*np.sin(np.radians(angles))[None, :])
    t = np.arange(32)
    first = np.exp(.71j*t)
    second = (.65+.25j)*first if coherent else .6*np.exp(1.31j*t)
    x = atoms@np.vstack([first, second])
    return pack(x@x.conj().T/x.shape[1]+np.eye(len(positions))*noise), x


def test_eigenpairs_reconstruct_complex_hermitian_matrix():
    matrix = np.array([[1, 1j], [-1j, 2]])
    result = doa.hermitian_eigen(pack(matrix))
    values = np.array(result['values'])
    vectors = (np.array(result['re'])+1j*np.array(result['im'])).reshape(2, 2)
    np.testing.assert_allclose(values, [(3+math.sqrt(5))/2, (3-math.sqrt(5))/2], atol=1e-12)
    np.testing.assert_allclose(matrix@vectors, vectors*values, atol=1e-12)


def test_covariance_is_finite_snapshot_hermitian_and_seed_reproducible():
    a = doa.generate_covariance(ULA, [-2, 2], dict(snapshots=128), 123)
    assert a == doa.generate_covariance(ULA, [-2, 2], dict(snapshots=128), 123)
    assert a != doa.generate_covariance(ULA, [-2, 2], dict(snapshots=128), 124)
    matrix = unpack(a)
    np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-14)
    assert np.linalg.eigvalsh(matrix).min() > 0


@pytest.mark.parametrize('method', ['music', 'mvdr', 'esprit', 'dml'])
def test_four_estimators_resolve_actual_high_snr_data(method):
    report = doa.evaluate(ULA, dict(method=method, trials=3))
    assert report['valid'] and report['successRate'] == 1
    assert report['rmseDeg'] < .1 and report['usedChannelCount'] == 12
    assert report['representative']['estimates'] == report['details'][0]['estimates']
    assert report['trialResults'] == report['details']
    assert bool(report['spectrum']) == (method in ('music', 'mvdr'))
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('method', ['music', 'mvdr', 'esprit'])
def test_low_snr_short_observation_is_not_theoretical_resolution(method):
    report = doa.evaluate(ULA, dict(method=method, snrDb=-20, snapshots=8, trials=4))
    assert report['valid'] and report['successRate'] == 0
    assert report['rmseDeg'] > 1


def test_nonuniform_manifold_and_explicit_esprit_subset():
    positions = [0, .5, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 6, 7, 8]
    music = doa.evaluate(positions, dict(method='music', trials=2))
    esprit = doa.evaluate(positions, dict(method='esprit', trials=2))
    assert music['successRate'] == 1 and music['usedChannelCount'] == 12
    assert esprit['subset']['positions'] == [1.5, 2, 2.5, 3, 3.5, 4, 4.5]
    assert esprit['usedChannelCount'] == 7


@pytest.mark.parametrize('positions', [[0, .7, 1.6, 2.8], [0, 1, 2, 3]])
def test_unsupported_ula_does_not_fallback_to_another_algorithm(positions):
    assert doa.select_ula(positions) is None
    assert doa.evaluate(positions, dict(method='esprit'))['status'] == 'unsupported'
    assert doa.evaluate(positions, dict(smoothing='spatial'))['status'] == 'unsupported'


@pytest.mark.parametrize('method', ['music', 'esprit'])
def test_coherent_failure_and_explicit_spatial_smoothing(method):
    basic = doa.evaluate(ULA, dict(method=method, sourceCorrelation=1, snrDb=30, trials=3))
    smoothed = doa.evaluate(ULA, dict(method=method, sourceCorrelation=1, snrDb=30,
                                    separationDeg=8, smoothing='spatial', trials=3))
    assert basic['successRate'] == 0
    assert smoothed['successRate'] == 1
    assert smoothed['usedChannelCount'] == 7 and smoothed['smoothing']['subarrayCount'] == 6


def test_smoothing_lengths_and_dml_preprocessing_not_silently_changed():
    assert doa.evaluate(ULA, dict(smoothing='spatial', subarrayLength=12))['status'] == 'unsupported'
    assert doa.evaluate([0, .5, 1], dict(smoothing='spatial'))['status'] == 'unsupported'
    assert doa.evaluate(ULA, dict(method='dml', smoothing='spatial'))['status'] == 'unsupported'


@pytest.mark.parametrize('method', ['music', 'mvdr', 'esprit', 'dml'])
def test_estimator_never_uses_supplied_truth_fields(method):
    cov = doa.generate_covariance(ULA, [23, 31], dict(snrDb=30), 144)
    first = doa.estimate(cov, ULA, dict(method=method, centerDeg=-40, separationDeg=1))
    second = doa.estimate(cov, ULA, dict(method=method, centerDeg=60, separationDeg=40))
    assert first['estimates'] == second['estimates']
    np.testing.assert_allclose(first['estimates'], [23, 31], atol=.15)


def test_population_identity_has_no_fictitious_peaks():
    covariance = pack(np.eye(12, dtype=complex))
    assert doa.estimate(covariance, ULA)['estimates'] == []
    assert doa_dml.estimate(covariance, ULA)['status'] == 'unsupported'


def test_tolerance_cap_prevents_two_estimates_clustered_at_center_passing():
    report = doa.evaluate(ULA, dict(separationDeg=.2, angleToleranceDeg=5, snrDb=-20,
                                    snapshots=4, trials=2))
    assert report['effectiveToleranceDeg'] == pytest.approx(.08)
    assert report['successRate'] == 0


@pytest.mark.parametrize('options', [dict(method='unknown'), dict(snrDb=float('nan')),
    dict(snapshots=0), dict(trials=65), dict(powerDifferenceDb=-1), dict(sourceCorrelation=1.1),
    dict(gridStepDeg=0), dict(diagonalLoading=-1), dict(seed=-1), dict(centerDeg=79, separationDeg=4),
    dict(snapshots=True)])
def test_invalid_numerical_inputs_rejected(options):
    assert doa.evaluate(ULA, options)['valid'] is False


def test_duplicate_or_excess_channels_rejected_without_truncation():
    assert doa.evaluate([0, .5, .5, 1])['status'] == 'invalid'
    assert doa.evaluate([i*.5 for i in range(49)])['status'] == 'unsupported'


def test_mvdr_loading_changes_rank_deficient_covariance():
    cov = doa.generate_covariance(ULA, [-2, 2], dict(snapshots=4), 234)
    assert doa.estimate(cov, ULA, dict(method='mvdr', diagonalLoading=0))['status'] == 'failed'
    loaded = doa.estimate(cov, ULA, dict(method='mvdr', diagonalLoading=.01))
    assert loaded['valid'] and loaded['loadingPower'] > 0
    assert all(math.isfinite(p['y']) for p in loaded['spectrum'])


def test_dml_projection_agrees_with_independent_qr_sample_residual():
    positions, angles = [0, .5, 1, 2, 3.5, 5], [-6, 12]
    cov, x = fixture(positions, [-9, 15], coherent=True)
    atoms = np.exp(2j*np.pi*np.asarray(positions)[:, None]*np.sin(np.radians(angles))[None, :])
    q, _ = np.linalg.qr(atoms)
    residual = x-q@(q.conj().T@x)
    expected = float(np.sum(abs(residual)**2)/x.shape[1])
    report = doa_dml.projection_objective(cov, positions, angles)
    assert report['residualPower'] == pytest.approx(expected, abs=1e-11)
    assert doa_dml.projection_objective(cov, positions, [-9, 15])['residualPower'] < 1e-10


def test_dml_supports_coherent_close_pair_without_rank_two_sources():
    cov, _ = fixture(ULA, [-.5, .5], coherent=True, noise=1e-6)
    result = doa_dml.estimate(cov, ULA, dict(scanMinDeg=-40, scanMaxDeg=40, dmlRefineLevels=4))
    assert result['valid']
    np.testing.assert_allclose(result['estimates'], [-.5, .5], atol=.08)
    assert result['residualPower'] < 1e-4 and result['spectrum'] == []


def test_dml_translation_invariance():
    positions = [0, .5, 1.5, 2.5, 4, 4.5, 6, 7.5]
    cov, _ = fixture(positions, [-13.25, 22.5], coherent=True, noise=.001)
    a = doa_dml.estimate(cov, positions, dict(dmlCoarseStepDeg=1))
    b = doa_dml.estimate(cov, [p+100 for p in positions], dict(dmlCoarseStepDeg=1))
    assert a['estimates'] == b['estimates']
    np.testing.assert_allclose(a['estimates'], [-13.25, 22.5], atol=.05)


def test_spatial_alias_competitors_are_not_marked_resolved():
    result = doa.evaluate([0, 1, 2, 3, 4, 5], dict(method='dml', separationDeg=20,
                          snrDb=35, snapshots=256, trials=2))
    assert result['valid'] and result['ambiguousTrials'] == 2 and result['successRate'] == 0


def test_indefinite_and_nonhermitian_covariance_rejected():
    matrix = np.full((3, 3), -.9, complex)
    np.fill_diagonal(matrix, 1)
    assert doa_dml.estimate(pack(matrix), [0, .5, 1])['status'] == 'invalid'
    assert doa.estimate(pack(matrix), [0, .5, 1])['status'] == 'invalid'
    matrix[0, 1] += 1j
    assert doa.estimate(pack(matrix), [0, .5, 1])['status'] == 'invalid'


@pytest.mark.parametrize('value', ['1', True, float('inf')])
def test_covariance_entries_are_numeric_not_coerced(value):
    covariance = pack(np.eye(3, dtype=complex))
    covariance['re'][0] = value
    assert doa.estimate(covariance, [0, .5, 1])['status'] == 'invalid'
    assert doa_dml.estimate(covariance, [0, .5, 1])['status'] == 'invalid'


def test_explicit_cancellation_is_not_converted_to_algorithm_failure():
    with pytest.raises(ComputationCancelled):
        doa.evaluate(ULA, cancelled=lambda: True)
    cov, _ = fixture(ULA, [-2, 2])
    with pytest.raises(ComputationCancelled):
        doa_dml.estimate(cov, ULA, cancelled=lambda: True)
