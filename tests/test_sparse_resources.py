"""Allocation-free preflight and exact bounded geometry for arbitrary grids."""
from dataclasses import replace

import numpy as np
import pytest

from arraylab.sparse2d import constraints, geometry, resources
from arraylab.sparse2d.core import Config


def memory_with_budget(budget):
    return {'available_physical_bytes': budget * 2, 'available_commit_bytes': budget * 4,
            'budget_bytes': budget, 'source': '测试内存快照', 'fallback': False}


@pytest.fixture(autouse=True)
def deterministic_memory(monkeypatch):
    snapshot = memory_with_budget(4 * resources.GIB)
    monkeypatch.setattr(resources, '_memory_state', lambda: snapshot.copy())
    monkeypatch.setattr(geometry, '_memory_state', lambda: snapshot.copy())


def legacy_coarray(points):
    differences = (points[:, None, :] - points[None, :, :]).reshape(-1, 2)
    lags, multiplicity = np.unique(differences, axis=0, return_counts=True)
    available = set(map(tuple, lags))
    mx, my = np.max(np.abs(lags), axis=0)
    best = (1, 0, 0)
    for rx in range(int(mx) + 1):
        for ry in range(int(my) + 1):
            area = (2 * rx + 1) * (2 * ry + 1)
            if area > best[0] and all((x, y) in available for x in range(-rx, rx+1) for y in range(-ry, ry+1)):
                best = area, rx, ry
    return {'unique_lags': len(lags), 'bounding_holes': int((2*mx+1)*(2*my+1)-len(lags)),
            'central_rectangle_lags': best[0], 'central_half_widths': [best[1], best[2]],
            'lags': lags, 'multiplicity': multiplicity}


@pytest.mark.parametrize('seed', range(8))
def test_coarray_matches_previous_counts_rectangle_and_tie_order(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    grid = np.array([(x, y) for x in range(7) for y in range(5)])
    points = grid[rng.choice(len(grid), 15, replace=False)]
    monkeypatch.setattr(geometry, 'FIELD_WORKSPACE_BYTES', 1024)
    expected, actual = legacy_coarray(points), geometry.coarray(points)
    for name in ('lags', 'multiplicity'):
        np.testing.assert_array_equal(actual[name], expected[name])
    for name in ('unique_lags', 'bounding_holes', 'central_rectangle_lags', 'central_half_widths'):
        assert actual[name] == expected[name]


def test_large_full_rectangle_coarray_has_exact_triangular_multiplicity():
    nx, ny = 42, 35
    points = np.array([(x, y) for x in range(nx) for y in range(ny)])
    actual = geometry.coarray(points)
    lags = np.array([(x, y) for x in range(1-nx, nx) for y in range(1-ny, ny)])
    expected = (nx - abs(lags[:, 0])) * (ny - abs(lags[:, 1]))
    np.testing.assert_array_equal(actual['lags'], lags)
    np.testing.assert_array_equal(actual['multiplicity'], expected)
    assert int(actual['multiplicity'].sum()) == (nx*ny) ** 2
    assert actual['bounding_holes'] == 0
    assert actual['central_rectangle_lags'] == (2*nx-1)*(2*ny-1)
    assert actual['central_half_widths'] == [nx-1, ny-1]


@pytest.mark.parametrize('points', [np.array([[0, 0]]), np.array([[0, 0], [1, 2]], dtype=np.uint8),
                                   np.array([[0, 0], [0, 0], [1, 0]])])
def test_coarray_singleton_unsigned_and_repeated_points(points):
    expected, actual = legacy_coarray(points.astype(np.int64)), geometry.coarray(points)
    np.testing.assert_array_equal(actual['lags'], expected['lags'])
    np.testing.assert_array_equal(actual['multiplicity'], expected['multiplicity'])
    assert actual['central_half_widths'] == expected['central_half_widths']


@pytest.mark.parametrize('minimum', [0., .5, .5000000001, .8, 1.5])
def test_blocked_pair_conflicts_and_spacing_match_full_distance_tensor(minimum, monkeypatch):
    xy = np.array([(x, y) for x in range(21) for y in range(23)], dtype=float) * .5
    pairs = np.column_stack((np.arange(len(xy)//2), np.arange(len(xy)-1, len(xy)//2, -1)))
    points = xy[pairs]
    expected = np.min(np.linalg.norm(points[:, None, :, None, :] - points[None, :, None, :, :], axis=-1), axis=(2, 3)) < minimum-1e-10
    np.fill_diagonal(expected, False)
    monkeypatch.setattr(constraints, 'FIELD_WORKSPACE_BYTES', 2048)
    np.testing.assert_array_equal(constraints.pair_conflicts(xy, pairs, minimum), expected)
    distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    values = distances[np.triu_indices(len(xy), 1)]
    actual = constraints.spacing_stats(xy, minimum)
    assert actual['min_distance_lambda_ref'] == float(values.min())
    assert actual['spacing_violations'] == int(np.count_nonzero(values < minimum-1e-10))


def test_single_point_spacing_and_empty_pair_set():
    assert constraints.spacing_stats(np.array([[0., 0.]]), 1.) == {
        'min_distance_lambda_ref': None, 'spacing_violations': 0}
    assert constraints.pair_conflicts(np.empty((0, 2)), np.empty((0, 2), dtype=int), 1.).shape == (0, 0)


def test_above_twenty_and_odd_dimensions_are_estimated_without_grid_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('preflight must not allocate a candidate grid')
    for name in ('zeros', 'ones', 'empty', 'arange', 'meshgrid'):
        monkeypatch.setattr(np, name, forbidden)
    result = resources.ensure_resources(Config(nx=31, ny=26, k=2), workers=1)
    assert result['candidate_count'] == 806
    assert result['group_count'] == 403
    assert result['within_budget']
    assert result['workers'] == 1


def test_target_k_does_not_hide_full_reference_cost():
    config = Config(nx=50, ny=51, k=2)
    small = resources.estimate_resources(config, workers=1)
    full = resources.estimate_resources(replace(config, k=config.nx*config.ny), workers=1)
    assert small['estimated_peak_bytes'] == full['estimated_peak_bytes']


def test_huge_python_integer_rejected_without_float_overflow_or_grid_allocation(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('preflight must not allocate arrays')
    for name in ('zeros', 'ones', 'empty', 'arange', 'meshgrid'):
        monkeypatch.setattr(np, name, forbidden)
    config = Config(nx=10**1000, ny=10**1000, k=2)
    estimate = resources.estimate_resources(config, workers=1)
    assert not estimate['within_budget']
    assert type(estimate['estimated_peak_bytes']) is int
    with pytest.raises(ValueError, match='超过当前计算预算') as error:
        resources.ensure_resources(config, workers=1)
    assert len(str(error.value)) < 350


def test_budget_boundary_and_restore_estimate_do_not_include_solver(monkeypatch):
    config = Config(nx=32, ny=33, k=2)
    result = resources.estimate_resources(config, workers=1)
    required = result['estimated_peak_bytes']
    monkeypatch.setattr(resources, '_memory_state', lambda: memory_with_budget(required))
    assert resources.ensure_resources(config, workers=1)['within_budget']
    monkeypatch.setattr(resources, '_memory_state', lambda: memory_with_budget(required-1))
    assert not resources.estimate_resources(config, workers=1)['within_budget']
    with pytest.raises(ValueError, match='参数可以保存'):
        resources.ensure_resources(config, workers=1)
    restored = resources.ensure_resources(config, workers=1, operation='restore')
    assert restored['components']['lp_workspaces'] == 0
    assert restored['components']['amplitude_lp_workspaces'] == 0
    assert restored['components']['conflict_matrix'] == 0


def test_joint_amplitude_and_parallel_geometry_costs_are_reserved(monkeypatch):
    monkeypatch.setattr('arraylab.parallel.os.cpu_count', lambda: 16)
    if hasattr(resources.os, 'process_cpu_count'):
        monkeypatch.setattr(resources.os, 'process_cpu_count', lambda: 16)
    base = Config(nx=51, ny=50, scan_angles='0,0;30,0')
    serial = resources.estimate_resources(base, workers=1)
    parallel = resources.estimate_resources(base, workers=4)
    assert serial['components']['amplitude_lp_workspaces'] > 0
    assert parallel['components']['field_workspace'] == 4 * serial['components']['field_workspace']
    assert parallel['components']['coarray_and_geometry'] == 4 * serial['components']['coarray_and_geometry']
    assert parallel['scenario_count'] == 2
    spacing = resources.estimate_resources(replace(base, min_spacing=.8), workers=1)
    assert spacing['components']['spacing_feasibility_workspaces'] > 0


def test_evaluate_reserves_actual_workers_and_new_sampling_without_solver(monkeypatch):
    monkeypatch.setattr('arraylab.parallel.os.cpu_count', lambda: 16)
    if hasattr(resources.os, 'process_cpu_count'):
        monkeypatch.setattr(resources.os, 'process_cpu_count', lambda: 16)
    config = Config(nx=51, ny=50, scan_angles='0,0;30,0', min_spacing=.8)
    restored = resources.estimate_resources(config, workers=4, operation='restore')
    evaluated = resources.estimate_resources(config, workers=4, operation='evaluate')
    denser = resources.estimate_resources(replace(config, eval_grid=401), workers=4, operation='evaluate')
    assert restored['workers'] == 1
    assert evaluated['workers'] == 4
    assert evaluated['components']['field_workspace'] == 4 * restored['components']['field_workspace']
    for name in ('lp_workspaces', 'amplitude_lp_workspaces', 'conflict_matrix', 'spacing_feasibility_workspaces'):
        assert evaluated['components'][name] == 0
    assert denser['components']['sample_domains'] > evaluated['components']['sample_domains']
    assert denser['estimated_peak_bytes'] > evaluated['estimated_peak_bytes'] > restored['estimated_peak_bytes']
    expected = sum(evaluated['components'][name] for name in
        ('runtime_reserve', 'coordinates_and_results', 'sample_domains', 'coarray_and_geometry', 'field_workspace'))
    assert evaluated['estimated_peak_bytes'] == expected
    monkeypatch.setattr(resources, '_memory_state', lambda: memory_with_budget(restored['estimated_peak_bytes']))
    resources.ensure_resources(config, workers=4, operation='restore')
    with pytest.raises(ValueError, match='超过当前计算预算'):
        resources.ensure_resources(config, workers=4, operation='evaluate')


def test_geometry_checks_memory_before_allocating_large_lag_histogram(monkeypatch):
    points = np.array([[0, 0], [1000000, 1000000]])
    def forbidden(*args, **kwargs):
        raise AssertionError('large histogram should be rejected first')
    monkeypatch.setattr(np, 'zeros', forbidden)
    with pytest.raises(ValueError, match='共阵几何预计需要'):
        geometry.coarray(points)
