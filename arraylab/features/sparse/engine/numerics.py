"""Serialized native numerical stages with a bounded BLAS/HiGHS budget."""
import warnings

import numpy as np
from scipy.optimize import OptimizeWarning

from arraylab.framework.parallel import check_cancelled, ordered_map


MINIMAX_IPM_GROUPS = 128
LP_SOLVERS = ('auto', 'highs', 'highs-ipm')


def minimax_solver(requested, group_count):
    if type(requested) is not str or requested not in LP_SOLVERS:
        raise ValueError('LP 求解器须为 auto、highs 或 highs-ipm。')
    return ('highs-ipm' if group_count >= MINIMAX_IPM_GROUPS else 'highs') if requested == 'auto' else requested


def bounded_call(function, *args, cancelled=None, **kwargs):
    """One caller-thread stage. Never call this from inside ordered_map tasks."""
    rows, _ = ordered_map(lambda _: function(*args, **kwargs), [None], workers=1, cancelled=cancelled)
    return rows[0]


def solve_lp(solver, objective, *, method, cancelled=None, **kwargs):
    """Keep native calls bounded; cancellation is cooperative around each solve.

    GUI jobs own a separate process to interrupt an in-flight native solve.
    Only SciPy's notice for the explicitly forwarded threads option is hidden.
    """
    check_cancelled(cancelled)
    def solve():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                message=r"Unrecognized options detected: \{'threads': 1\}\. These will be passed to HiGHS verbatim\.",
                category=OptimizeWarning)
            solution = solver(objective, method=method, options={'threads': 1}, **kwargs)
        check_cancelled(cancelled)
        diagnostics = {'solver': method, 'native_threads': 1, 'success': bool(solution.success)}
        if solution.success:
            values = np.asarray(solution.x)
            inequality = np.asarray(kwargs['A_ub']) @ values - kwargs['b_ub']
            equality = np.asarray(kwargs['A_eq']) @ values - kwargs['b_eq']
            diagnostics.update(objective=float(solution.fun),
                max_inequality_violation=float(max(0., np.max(inequality))),
                normalization_residual=float(np.max(np.abs(equality))),
                max_nonnegative_bound_violation=float(max(0., -np.min(values))),
                iterations=int(solution.nit), crossover_iterations=int(getattr(solution, 'crossover_nit', 0)))
            if not np.all(np.isfinite(values)) or max(diagnostics['max_inequality_violation'],
                    diagnostics['normalization_residual'], diagnostics['max_nonnegative_bound_violation']) > 1e-6:
                raise RuntimeError('LP 返回结果未通过完整约束残差检查。')
        return solution, diagnostics
    return bounded_call(solve, cancelled=cancelled)


def solve_milp(solver, *args, cancelled=None, **kwargs):
    check_cancelled(cancelled)
    def solve():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                message=r"Unrecognized options detected: \{'threads'\}\. These will be passed to HiGHS verbatim\.",
                category=RuntimeWarning)
            result = solver(*args, options={'time_limit': 10., 'presolve': True, 'threads': 1}, **kwargs)
        check_cancelled(cancelled)
        return result
    return bounded_call(solve, cancelled=cancelled)
