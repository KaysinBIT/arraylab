import numpy as np

from arraylab.planar_pattern import integrated_power as reference_integral
from arraylab.sparse2d.aperture_metrics import integrated_power, maximum_distance


def test_blocked_diameter_matches_full_pairwise_and_single_point():
    rng = np.random.default_rng(71)
    points = rng.normal(size=(1300, 2))
    expected = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2).max()
    assert maximum_distance(points) == expected
    assert maximum_distance(np.zeros((1, 2))) == 0.


def test_large_sinc_integral_matches_original_with_complex_phases():
    rng = np.random.default_rng(32)
    points = rng.uniform(-12, 12, (600, 2))
    weights = rng.uniform(.1, 1, 600) * np.exp(1j * rng.uniform(-np.pi, np.pi, 600))
    weights /= np.sum(abs(weights))
    np.testing.assert_allclose(integrated_power(points, weights), reference_integral(points, weights), rtol=1e-13)


def test_large_linear_halfwave_uniform_array_exact_integral():
    points = np.column_stack((np.arange(1000)*.5, np.zeros(1000)))
    weights = np.ones(1000) / 1000
    np.testing.assert_allclose(integrated_power(points, weights), .001, rtol=1e-12)


def test_narrow_beam_search_stops_at_first_crossing_with_same_root():
    from arraylab.pattern_metrics import spherical_beamwidths
    visited = []
    sigma = .0001
    def response(xyz):
        visited.append(len(xyz))
        return np.exp(-xyz[:,0]**2/(2*sigma*sigma)).astype(complex)
    summary = spherical_beamwidths(response, 0, 0, .001, flat_cuts=('transverse',))
    expected = 2*np.rad2deg(np.arcsin(sigma*np.sqrt(np.log(2))))
    np.testing.assert_allclose(summary['orthogonal_beamwidths_deg'][0], expected, rtol=1e-8)
    assert summary['orthogonal_beamwidths_deg'][1] is None
    assert sum(visited) < 1000


def test_single_axis_sparse_summary_has_flat_transverse_cut():
    from dataclasses import asdict
    from arraylab.sparse2d.core import Config, candidates
    from arraylab.sparse2d.elements import make_snapshot
    from arraylab.sparse2d.scenarios import scenarios
    from arraylab.sparse2d.pattern_metrics import sparse_pattern_summary
    config = Config(nx=513, ny=1, k=513, opt_grid=25, eval_grid=81)
    points = candidates(config)[1]
    result = {'xy': points, 'weights': np.ones(len(points))/len(points)}
    bundle = {'config': asdict(config), 'scenarios': scenarios(config), 'element_pattern': make_snapshot(config)}
    summary = sparse_pattern_summary(bundle, result)
    assert summary['orthogonal_beamwidths_deg'][0] > 0
    assert summary['orthogonal_beamwidths_deg'][1] is None
    np.testing.assert_allclose(summary['directivity_dbi'], 10*np.log10(513), atol=1e-10)
