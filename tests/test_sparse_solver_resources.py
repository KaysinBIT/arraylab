"""Large-array solver accuracy, responsive stage cancellation, and resource ownership."""
import threading

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from arraylab.parallel import ComputationCancelled
from arraylab.sparse2d import core
from arraylab.sparse2d.constraints import feasible_groups
from arraylab.sparse2d.numerics import bounded_call, minimax_solver


def test_minimax_policy_and_legacy_config():
    assert core.Config().lp_solver == 'auto'
    assert minimax_solver('auto', 127) == 'highs'
    assert minimax_solver('auto', 128) == 'highs-ipm'
    assert minimax_solver('highs', 200) == 'highs'
    assert minimax_solver('highs-ipm', 8) == 'highs-ipm'
    for invalid in ('simplex', '', None, True, []):
        with pytest.raises(ValueError):
            core.Config(lp_solver=invalid).validate()


def test_small_auto_retains_existing_highs_result_and_five_rounds():
    matrix = core.joint_matrix(core.Config(nx=6, ny=6, k=12, opt_grid=25))
    auto, ai = core.reweighted_lp(matrix)
    legacy, li = core.reweighted_lp(matrix, lp_solver='highs')
    np.testing.assert_array_equal(auto, legacy)
    assert ai['minimax_linear'] == li['minimax_linear']
    assert ai['rounds'] == li['rounds']
    assert ai['minimax_solver'] == 'highs'
    assert ai['reweighted_solver'] == 'highs'
    assert len(ai['solver_diagnostics']) == 6
    assert all(row['native_threads'] == 1 for row in ai['solver_diagnostics'])


def test_20_by_20_auto_minimax_matches_explicit_legacy_objective():
    matrix = core.joint_matrix(core.Config(nx=20, ny=20))
    assert matrix.shape == (2547, 200)
    auto, info = core.reweighted_lp(matrix, rounds=0)
    legacy, previous = core.reweighted_lp(matrix, rounds=0, lp_solver='highs')
    assert info['minimax_solver'] == 'highs-ipm'
    assert previous['minimax_solver'] == 'highs'
    assert abs(info['minimax_linear'] - previous['minimax_linear']) < 1e-9
    for weights, metadata in ((auto, info), (legacy, previous)):
        assert abs(weights.sum() - 1) < 1e-9
        assert weights.min() >= -1e-9
        assert np.max(abs(matrix @ weights)) <= metadata['minimax_linear'] + 1e-8
        assert metadata['solver_diagnostics'][0]['max_inequality_violation'] < 1e-8
        assert metadata['solver_diagnostics'][0]['normalization_residual'] < 1e-8


def test_large_fixed_support_uses_ipm_with_full_dynamic_range_constraints():
    matrix = core.joint_matrix(core.Config(nx=16, ny=16, opt_grid=25))
    weights, info = core.amplitude_minimax(matrix, 10)
    assert info['minimax_solver'] == 'highs-ipm'
    assert len(weights) == 128 and weights.min() > 0
    assert abs(weights.sum() - 1) < 1e-9
    assert 20*np.log10(weights.max()/weights.min()) <= 10 + 1e-7
    assert np.max(abs(matrix @ weights)) <= info['lp_peak_linear'] + 1e-8
    assert info['solver_diagnostics']['max_inequality_violation'] < 1e-8


def test_explicit_ipm_controls_minimax_only(monkeypatch):
    methods = []
    original = core.linprog
    def record(*args, **kwargs):
        methods.append(kwargs['method'])
        return original(*args, **kwargs)
    monkeypatch.setattr(core, 'linprog', record)
    core.reweighted_lp(core.joint_matrix(core.Config(nx=4, ny=4, k=8, opt_grid=25)),
                       rounds=2, lp_solver='highs-ipm')
    assert methods == ['highs-ipm', 'highs', 'highs']


def test_cancel_between_lp_rounds_prevents_next_native_solve(monkeypatch):
    stopped = threading.Event(); methods = []
    original = core.linprog
    def record(*args, **kwargs):
        methods.append(kwargs['method'])
        return original(*args, **kwargs)
    monkeypatch.setattr(core, 'linprog', record)
    def progress(message):
        if message == '重加权 LP：2/5': stopped.set()
    with pytest.raises(ComputationCancelled):
        core.reweighted_lp(core.joint_matrix(core.Config(nx=4, ny=4, k=8, opt_grid=25)),
                           progress=progress, cancelled=stopped.is_set)
    assert methods == ['highs', 'highs']


def test_cancel_during_greedy_anneal_and_packing_loops():
    matrix = core.joint_matrix(core.Config(nx=4, ny=4, k=8, opt_grid=25))
    for method in ('exchange', 'anneal', 'packing'):
        stopped = threading.Event()
        def progress(_): stopped.set()
        with pytest.raises(ComputationCancelled):
            if method == 'exchange':
                core.exchange(matrix, np.arange(4), progress=progress, cancelled=stopped.is_set)
            elif method == 'anneal':
                core.anneal(matrix, np.arange(4), np.random.default_rng(1), 10000,
                            progress=progress, cancelled=stopped.is_set)
            else:
                conflicts = np.eye(8, k=1, dtype=bool); conflicts |= conflicts.T
                feasible_groups(conflicts, 4, np.random.default_rng(1),
                                progress=progress, cancelled=stopped.is_set)


def test_run_exposes_meaningful_stage_and_round_progress():
    messages = []
    result = core.run(core.Config(nx=4, ny=4, k=8, iterations=20, opt_grid=25, eval_grid=81),
                      progress=messages.append, workers=1)
    assert any(x.startswith('构造优化矩阵：工况') for x in messages)
    assert any(x.startswith('贪心交换：第') for x in messages)
    assert '模拟退火：20/20' in messages
    assert any(x.startswith('首次最小最大 LP：') for x in messages)
    assert [x for x in messages if x.startswith('重加权 LP：')] == [f'重加权 LP：{i}/5' for i in range(1, 6)]
    assert messages[-1] == f"验证布局：{len(result['results'])}/{len(result['results'])}"


@pytest.mark.parametrize('failure', [False, True])
def test_solver_limits_and_restores_native_resources(monkeypatch, failure):
    original = core.linprog
    observed = []
    def record(*args, **kwargs):
        observed.extend(x['num_threads'] for x in threadpool_info() if x['user_api'] == 'blas')
        assert kwargs['options'] == {'threads': 1}
        if failure: raise RuntimeError('intentional solver failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(core, 'linprog', record)
    matrix = core.joint_matrix(core.Config(nx=4, ny=4, k=8, opt_grid=25))
    with threadpool_limits(limits=3, user_api='blas'):
        if failure:
            with pytest.raises(RuntimeError, match='intentional'):
                core.reweighted_lp(matrix, rounds=0)
        else:
            core.reweighted_lp(matrix, rounds=0)
        assert observed and set(observed) == {1}
        assert all(x['num_threads'] == 3 for x in threadpool_info() if x['user_api'] == 'blas')
        assert bounded_call(lambda: 42) == 42
    assert not [t for t in threading.enumerate() if t.name.startswith('ArrayLab-calc')]
