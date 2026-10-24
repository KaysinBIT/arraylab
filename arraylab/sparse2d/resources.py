"""Allocation-free preflight for sparse jobs; estimates are not solver guarantees.

The limits follow currently available memory rather than an array-side limit.
Saving a configuration must use only ``estimate_resources`` (or no preflight).
"""
import ctypes
import os
from pathlib import Path

from ..parallel import execution_plan
from .scenarios import scenarios


MIB = 1024 ** 2
GIB = 1024 ** 3
FIELD_WORKSPACE_BYTES = 32 * MIB
FALLBACK_AVAILABLE_BYTES = 2 * GIB


def format_bytes(value):
    """Format even extremely large Python integers without float conversion."""
    value = max(0, int(value))
    for unit, divisor in (('GiB', GIB), ('MiB', MIB), ('KiB', 1024)):
        if value >= divisor:
            whole, fraction = divmod(value * 10 // divisor, 10)
            if whole.bit_length() > 50:
                return f'超过 10^{(whole.bit_length()-1)*301//1000} {unit}'
            return f'{whole}.{fraction} {unit}'
    return f'{value} B'


def _memory_state():
    """Read free physical/commit bytes with standard-library platform fallbacks."""
    physical = commit = None
    source = '保守后备值（无法读取本机可用内存）'
    if os.name == 'nt':
        try:
            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                            *[(name, ctypes.c_ulonglong) for name in
                              ('total_physical', 'available_physical', 'total_pagefile',
                               'available_pagefile', 'total_virtual', 'available_virtual',
                               'available_extended_virtual')]]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                physical = int(status.available_physical)
                commit = int(status.available_pagefile)
                source = 'Windows 当前可用物理内存及可用提交量'
        except (AttributeError, OSError, ValueError):
            pass
    else:
        try:
            fields = {}
            for line in Path('/proc/meminfo').read_text(encoding='ascii').splitlines():
                key, rest = line.split(':', 1)
                fields[key] = int(rest.split()[0]) * 1024
            physical = fields.get('MemAvailable')
            if physical is None:
                physical = sum(fields.get(k, 0) for k in ('MemFree', 'Buffers', 'Cached'))
            if 'CommitLimit' in fields and 'Committed_AS' in fields:
                commit = max(0, fields['CommitLimit'] - fields['Committed_AS'])
                # Linux overcommit modes 0/1 do not enforce CommitLimit. The
                # physical-plus-swap figure remains meaningful in those modes.
                mode = Path('/proc/sys/vm/overcommit_memory').read_text(encoding='ascii').strip()
                if mode != '2':
                    commit = physical + fields.get('SwapFree', 0)
            source = 'Linux 当前可用物理内存及提交/交换空间'
        except (OSError, ValueError, IndexError):
            try:
                physical = int(os.sysconf('SC_AVPHYS_PAGES')) * int(os.sysconf('SC_PAGE_SIZE'))
                if physical <= 0:
                    physical = None
                else:
                    source = '系统可用物理页（提交量未提供）'
            except (AttributeError, OSError, ValueError):
                pass
    available = [value for value in (physical, commit) if value is not None]
    fallback = not available
    budget = min(available) // 2 if available else FALLBACK_AVAILABLE_BYTES // 2
    return {'available_physical_bytes': physical, 'available_commit_bytes': commit,
            'budget_bytes': max(0, budget), 'source': source, 'fallback': fallback,
            'budget_fraction': '1/2 of each available memory limit'}


def estimate_resources(c, *, workers=None, operation='run'):
    """Estimate peak bytes using integer arithmetic only; never create a grid.

    Dense reference and L1 supports can contain every candidate, regardless of K.
    LP's 96 bytes per coefficient reserves dense copies and native work arrays;
    native factorization fill and platform libraries can still exceed estimates.
    """
    if operation not in ('run', 'weight', 'evaluate', 'restore'):
        raise ValueError('未知稀疏资源预检操作。')
    for name in ('nx', 'ny', 'opt_grid', 'eval_grid'):
        if type(getattr(c, name)) is not int or getattr(c, name) <= 0:
            raise ValueError(f'{name} 必须为正整数。')
    count = c.nx * c.ny
    groups = (count + 1) // 2
    cases = len(scenarios(c))
    optimization_rows = cases * (c.opt_grid ** 2 + 12 * c.opt_grid + 9)
    validation_rows = cases * sum(q ** 2 + 12 * q + 9 for q in (c.eval_grid, c.eval_grid + 2))
    task_count = 1 if operation in ('weight', 'restore') else 9
    plan = execution_plan(task_count, count * validation_rows * task_count, workers)
    concurrent = plan['workers']
    lag_cells = (2 * c.nx - 1) * (2 * c.ny - 1)
    components = {
        'runtime_reserve': 128 * MIB,
        'coordinates_and_results': 512 * count,
        'sample_domains': 64 * (optimization_rows + validation_rows),
        'coarray_and_geometry': concurrent * (48 * lag_cells + FIELD_WORKSPACE_BYTES),
        'field_workspace': concurrent * FIELD_WORKSPACE_BYTES,
        'conflict_matrix': groups ** 2 if operation == 'run' else 0,
        'spacing_feasibility_workspaces': (48 * groups ** 2
            if operation == 'run' and c.min_spacing-1e-10 > c.spacing else 0),
        'lp_workspaces': 96 * optimization_rows * groups if operation in ('run', 'weight') else 0,
        'amplitude_lp_workspaces': (48 * groups ** 2
            if operation == 'weight' or operation == 'run' and (c.hole_shape != 'none' or cases > 1) else 0),
    }
    # The numerical and result-evaluation phases are sequential. The full
    # optimization matrix remains alive during evaluation, so include it there.
    shared = sum(components[key] for key in ('runtime_reserve', 'coordinates_and_results', 'sample_domains'))
    optimization = sum(components[key] for key in ('conflict_matrix', 'spacing_feasibility_workspaces',
                                                  'lp_workspaces', 'amplitude_lp_workspaces'))
    evaluation = components['coarray_and_geometry'] + components['field_workspace']
    if operation in ('run', 'weight'):
        evaluation += 8 * optimization_rows * groups + components['conflict_matrix']
    estimated = shared + max(optimization, evaluation)
    memory = _memory_state()
    return {'operation': operation, 'candidate_count': count, 'group_count': groups,
            'scenario_count': cases, 'optimization_rows_upper_bound': optimization_rows,
            'workers': concurrent, 'estimated_peak_bytes': estimated,
            'budget_bytes': memory['budget_bytes'], 'within_budget': estimated <= memory['budget_bytes'],
            'components': components, 'memory': memory,
            'estimate_note': '按完整候选参考阵及完整采样估计；原生求解器工作区可能变化，并非硬内存上界。'}


def ensure_resources(c, *, workers=None, operation='run'):
    """Reject a job before candidate allocation, while retaining its parameters."""
    estimate = estimate_resources(c, workers=workers, operation=operation)
    if not estimate['within_budget']:
        names = {'lp_workspaces': '线性规划矩阵及求解器', 'amplitude_lp_workspaces': '幅度优化约束',
                 'coarray_and_geometry': '共阵及几何统计', 'field_workspace': '并行方向图',
                 'conflict_matrix': '间距冲突矩阵', 'spacing_feasibility_workspaces': '间距整数可行性检查',
                 'coordinates_and_results': '候选及结果',
                 'sample_domains': '角度采样', 'runtime_reserve': '运行基础内存'}
        dominant = max(estimate['components'], key=estimate['components'].get)
        raise ValueError(
            f"预计峰值内存 {format_bytes(estimate['estimated_peak_bytes'])}，超过当前计算预算 "
            f"{format_bytes(estimate['budget_bytes'])}（可用物理/提交内存的各一半；"
            f"{estimate['memory']['source']}）。主要占用：{names[dominant]}。"
            '参数可以保存；请减少候选规模、联合工况或采样，或降低并行数后重试。')
    return estimate
