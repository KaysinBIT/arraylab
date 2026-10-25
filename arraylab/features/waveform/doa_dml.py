"""Bounded two-source deterministic maximum likelihood from WaveLab 1.4.

The estimator sees covariance and array geometry only, never source truth.
NumPy performs linear algebra; the coarse grid and multi-basin refinement retain
the reference search policy. This module has no GUI or JavaScript dependency.
"""
from __future__ import annotations

import math
import numpy as np

from arraylab.framework.parallel import check_cancelled

DEFAULTS = dict(scanMinDeg=-80, scanMaxDeg=80, gridStepDeg=.2,
                dmlCoarseStepDeg=2, dmlRefineLevels=3,
                dmlMinSeparationDeg=.1, dmlBasins=8)
defaults = DEFAULTS.copy()
SOURCES = [
    {'title': 'Nehorai, Starer & Stoica (1991) · DOA estimation with multipath and few snapshots',
     'url': 'https://www.ese.wustl.edu/~nehorai/paper/direction.pdf', 'doi': '10.1007/BF01187549'},
    {'title': 'Fan et al. (2019) · Deterministic ML DOA objective, §2',
     'url': 'https://onlinelibrary.wiley.com/doi/10.1155/2019/6035870', 'doi': '10.1155/2019/6035870'}]
ASSUMPTIONS = [
    '双目标确定性最大似然（DML/CML）：已知目标数为2，源复幅度为未知确定性参数；不要求源互不相关。',
    '输入为有限快拍样本协方差R=XXᴴ/N；假设空间白噪声、已校准的窄带远场流形。估计器不接收真角。',
    '最小化tr[(I−A(AᴴA)⁻¹Aᴴ)R]，完整视场联合搜索两个角，再细化多个候选盆地。',
    '有界网格与局部细化不保证连续参数空间全局最优；最小搜索间隔是数值约束，不是物理分辨率。',
    'DML输出两角和二维投影残差，不伪造一维谱；混叠、低SNR、少快拍及阵列误差可导致错误或非唯一估计。']
sources = SOURCES
assumptions = ASSUMPTIONS


def _finite(value):
    return isinstance(value, (int, float, np.number)) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _failure(message, status='invalid'):
    return dict(valid=False, status=status, estimates=[], spectrum=[], objective=None,
                residualPower=None, explainedPower=None, condition=None, evaluations=0,
                warnings=[message], errors=[message], assumptions=ASSUMPTIONS.copy(),
                sources=SOURCES.copy(), globalOptimality=False, convergence='not_run', ambiguous=False)


class _Unsupported(ValueError):
    pass


def _checked(cov, positions):
    if not isinstance(cov, dict) or type(cov.get('n')) is not int:
        raise ValueError('DML需要{n,re,im}形式的协方差。')
    n = cov['n']
    if not 3 <= n <= 128:
        raise _Unsupported('双目标DML支持3–128个观测通道。')
    if not isinstance(positions, (list, tuple, np.ndarray)) or len(positions) != n or any(not _finite(v) or abs(v) > 1e6 for v in positions):
        raise ValueError('DML位置须为与协方差维数一致的有限波长坐标。')
    if len({math.floor(float(v)*1e9+.5) for v in positions}) < 3:
        raise _Unsupported('双目标DML至少需要三个不同的空间位置。')
    try:
        if any(not isinstance(cov.get(k), (list, tuple, np.ndarray)) or any(not _finite(v) for v in cov[k]) for k in ('re', 'im')):
            raise ValueError('DML协方差须为有限数值数组。')
        re, im = np.asarray(cov['re'], dtype=float), np.asarray(cov['im'], dtype=float)
        if re.ndim != 1 or im.ndim != 1 or re.size != n*n or im.size != n*n:
            raise ValueError('DML协方差尺寸无效。')
        matrix = (re + 1j*im).reshape(n, n).copy()
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('DML协方差尺寸无效。') from exc
    if not np.isfinite(matrix).all():
        raise ValueError('DML协方差包含非有限数值。')
    scale = max(float(np.abs(matrix.diagonal().real).max()), 1e-200)
    tol = scale*1e-8
    if np.any(matrix.diagonal().real < -tol) or np.any(np.abs(matrix.diagonal().imag) > tol):
        raise ValueError('DML协方差对角须为非负实数。')
    if np.max(np.abs((matrix-matrix.conj().T).real)) > tol or np.max(np.abs((matrix-matrix.conj().T).imag)) > tol:
        raise ValueError('DML需要Hermitian样本协方差。')
    matrix = (matrix+matrix.conj().T)/2
    matrix[np.diag_indices(n)] = np.maximum(0, matrix.diagonal().real)
    trace = float(np.trace(matrix).real)
    if not math.isfinite(trace) or trace <= 0:
        raise ValueError('DML协方差总功率须为有限正数。')
    diagonal = matrix.diagonal().real
    if np.any(abs(matrix)**2 > diagonal[:, None]*diagonal[None, :]+scale*scale*1e-7):
        raise ValueError('DML输入不满足协方差半正定条件。')
    # Reference semidefinite Cholesky also accepts coherent rank-deficient data.
    factor = np.zeros_like(matrix)
    psd_tolerance = scale*1e-9
    for j in range(n):
        residual = float(matrix[j, j].real-np.vdot(factor[j, :j], factor[j, :j]).real)
        if residual < -psd_tolerance:
            raise ValueError('DML输入协方差不是半正定矩阵。')
        root = math.sqrt(residual) if residual > psd_tolerance else 0.
        factor[j, j] = root
        for i in range(j+1, n):
            entry = matrix[i, j]-np.vdot(factor[j, :j], factor[i, :j])
            if root:
                factor[i, j] = entry/root
            elif abs(entry) > math.sqrt(psd_tolerance*scale)*2:
                raise ValueError('DML输入协方差不是半正定矩阵。')
    return matrix, np.asarray(positions, dtype=float)-positions[0], trace


def _options(values):
    if not isinstance(values, dict):
        raise ValueError('DML搜索参数须为对象。')
    o = {**DEFAULTS, **values}
    if any(not _finite(o[k]) for k in DEFAULTS):
        raise ValueError('DML搜索参数须为有限数值。')
    if o['scanMinDeg'] < -90 or o['scanMaxDeg'] > 90 or o['scanMinDeg'] >= o['scanMaxDeg']:
        raise ValueError('DML视场须为−90°至90°内递增区间。')
    if not .005 <= o['gridStepDeg'] <= 2 or not .25 <= o['dmlCoarseStepDeg'] <= 5:
        raise ValueError('DML显示精度须为0.005–2°，粗网格须为0.25–5°。')
    if not .005 <= o['dmlMinSeparationDeg'] <= 10 or o['dmlMinSeparationDeg'] >= o['scanMaxDeg']-o['scanMinDeg']:
        raise ValueError('DML最小搜索间隔无效。')
    for key, low, high in [('dmlRefineLevels', 0, 5), ('dmlBasins', 1, 16)]:
        if type(o[key]) is not int or not low <= o[key] <= high:
            raise ValueError('DML细化层数须为0–5，候选盆地数须为1–16。')
    return o


class _Evaluator:
    def __init__(self, matrix, positions, trace):
        self.matrix, self.positions, self.trace = matrix, positions, trace
        self.n = len(positions)
        self.cache, self.evaluations, self.rejected = {}, 0, 0

    def atom(self, angle):
        key = f'{angle:.10f}'
        if key not in self.cache:
            a = np.exp(2j*np.pi*self.positions*math.sin(math.radians(angle)))
            y = self.matrix@a
            self.cache[key] = a, y, float(np.vdot(a, y).real)
        return self.cache[key]

    def evaluate(self, first, second):
        self.evaluations += 1
        a, _, ha = self.atom(first)
        b, yb, hb = self.atom(second)
        g, h = np.vdot(a, b), np.vdot(a, yb)
        magnitude = abs(g)
        denominator = self.n*self.n-magnitude*magnitude
        if denominator <= self.n*self.n*1e-10:
            self.rejected += 1
            return None
        captured = float((self.n*(ha+hb)-2*(g.conjugate()*h).real)/denominator)
        if not math.isfinite(captured) or captured < -self.trace*1e-7 or captured > self.trace*(1+1e-7):
            self.rejected += 1
            return None
        captured = min(self.trace, max(0, captured))
        return dict(estimates=[first, second], explainedPower=captured,
                    residualPower=max(0, self.trace-captured), condition=float((self.n+magnitude)/(self.n-magnitude)))

    def stats(self):
        return dict(evaluations=self.evaluations, rejected=self.rejected, steeringVectors=len(self.cache))


def projection_objective(covariance, positions, angles):
    try:
        matrix, positions, trace = _checked(covariance, positions)
        if not isinstance(angles, (list, tuple)) or len(angles) != 2 or any(not _finite(a) or abs(a) > 90 for a in angles):
            raise ValueError('DML目标函数需要两个有效角度。')
        value = _Evaluator(matrix, positions, trace).evaluate(*sorted(angles))
        if value is None:
            return _failure('两个导向向量线性相关或投影病态。', 'unsupported')
        return dict(valid=True, **value, objective=value['residualPower'], totalPower=trace)
    except (ValueError, TypeError) as exc:
        return _failure(str(exc), 'unsupported' if isinstance(exc, _Unsupported) else 'invalid')


def estimate(covariance, positions, options=None, *, cancelled=None):
    try:
        matrix, p, trace = _checked(covariance, positions)
        o = _options({} if options is None else options)
    except (ValueError, TypeError) as exc:
        return _failure(str(exc), 'unsupported' if isinstance(exc, _Unsupported) else 'invalid')
    evaluator = _Evaluator(matrix, p, trace)
    top = []
    worst, best_score = math.inf, -math.inf

    def retain(candidate):
        nonlocal worst, best_score
        if candidate is None:
            return
        score = candidate['explainedPower']
        best_score, worst = max(best_score, score), min(worst, score)
        if len(top) >= 128 and score <= top[-1]['explainedPower']:
            return
        lo, hi = 0, len(top)
        while lo < hi:
            mid = (lo+hi)//2
            if top[mid]['explainedPower'] >= score:
                lo = mid+1
            else:
                hi = mid
        top.insert(lo, candidate)
        if len(top) > 128:
            top.pop()

    span = o['scanMaxDeg']-o['scanMinDeg']
    count = math.ceil(span/o['dmlCoarseStepDeg'])
    coarse = span/count
    angles = [o['scanMinDeg']+i*coarse for i in range(count+1)]
    for i, first in enumerate(angles):
        check_cancelled(cancelled)
        for second in angles[i+1:]:
            if second-first >= o['dmlMinSeparationDeg']-1e-10:
                retain(evaluator.evaluate(first, second))
    separations = list(dict.fromkeys(v for v in [o['dmlMinSeparationDeg'], .2, .5, 1, coarse/2]
                                    if o['dmlMinSeparationDeg'] <= v < coarse))
    for center in angles:
        for separation in separations:
            first, second = center-separation/2, center+separation/2
            if first >= o['scanMinDeg'] and second <= o['scanMaxDeg']:
                retain(evaluator.evaluate(first, second))
    if not top:
        return {**_failure('指定视场内没有数值稳定的双目标导向矩阵。', 'unsupported'), **evaluator.stats()}
    if best_score-worst <= trace*1e-9:
        return {**_failure('联合投影目标函数近乎平坦，不能确定两个方向。', 'unsupported'), **evaluator.stats(), 'totalPower': trace}
    distance = lambda a, b: max(abs(a[0]-b[0]), abs(a[1]-b[1]))
    basins = []
    for candidate in top:
        if all(distance(c['estimates'], candidate['estimates']) >= coarse*1.25 for c in basins):
            basins.append(candidate)
            if len(basins) >= o['dmlBasins']:
                break
    for candidate in top:
        if len(basins) >= o['dmlBasins']:
            break
        if not any(candidate is other for other in basins):
            basins.append(candidate)
    refined = []
    local_tolerance = max(1e-6, min(1e-4, o['gridStepDeg']/100))

    def nearby(center, step, offsets, *, skip_center=False):
        for i in offsets:
            for j in offsets:
                if skip_center and i == 0 and j == 0:
                    continue
                a, b = center[0]+i*step, center[1]+j*step
                if a < o['scanMinDeg']-1e-10 or b > o['scanMaxDeg']+1e-10 or b-a < o['dmlMinSeparationDeg']-1e-10:
                    continue
                yield evaluator.evaluate(max(o['scanMinDeg'], a), min(o['scanMaxDeg'], b))

    for basin in basins:
        check_cancelled(cancelled)
        best, step = basin, coarse
        for _ in range(o['dmlRefineLevels']):
            center, step = best['estimates'][:], step/4
            for candidate in nearby(center, step, range(-4, 5)):
                if candidate and candidate['explainedPower'] > best['explainedPower']+trace*1e-14:
                    best = candidate
        local_step = step
        for _ in range(90):
            check_cancelled(cancelled)
            if local_step <= local_tolerance:
                break
            updated = best
            for candidate in nearby(best['estimates'][:], local_step, range(-1, 2), skip_center=True):
                if candidate and candidate['explainedPower'] > updated['explainedPower']+trace*1e-14:
                    updated = candidate
            if updated is best:
                local_step /= 2
            else:
                best = updated
        refined.append(best)
    refined.sort(key=lambda c: -c['explainedPower'])
    best = refined[0]
    final_step = coarse/4**o['dmlRefineLevels']
    alternatives = []
    for candidate in refined:
        if distance(candidate['estimates'], best['estimates']) <= max(.2, 2*final_step):
            continue
        if best['explainedPower']-candidate['explainedPower'] <= trace*1e-8 and all(distance(c, candidate['estimates']) > .2 for c in alternatives):
            alternatives.append(candidate['estimates'])
    warnings = ['DML为全视场粗网格加多个盆地局部细化，不保证连续空间全局最优。']
    if best['condition'] > 1e6:
        warnings.append('估计的双目标投影矩阵病态，结果可能不稳定。')
    if alternatives:
        warnings.append('发现多个相距较远且残差近乎相同的候选，存在非唯一方向解释。')
    if len(p) < 4:
        warnings.append('观测通道少于四个，相干双源唯一性尤其依赖阵列及场景条件。')
    stats = evaluator.stats()
    search = dict(scanMinDeg=o['scanMinDeg'], scanMaxDeg=o['scanMaxDeg'], coarseStepDeg=coarse,
                  refineLevels=o['dmlRefineLevels'], finalGridStepDeg=final_step, localToleranceDeg=local_tolerance,
                  minimumSeparationDeg=o['dmlMinSeparationDeg'], basins=len(basins))
    return dict(valid=True, status='estimated', estimates=best['estimates'], spectrum=[],
                spectrumKind='two_dimensional_joint_likelihood', objective=best['residualPower'],
                objectiveType='tr((I-P_A)R)', residualPower=best['residualPower'],
                explainedPower=best['explainedPower'], totalPower=trace, condition=best['condition'],
                **stats, search=search, diagnostics=dict(conditionNumber=best['condition'], totalPower=trace,
                    residualPower=best['residualPower'], explainedPower=best['explainedPower'], **stats, search=search),
                warnings=warnings, assumptions=ASSUMPTIONS.copy(), sources=SOURCES.copy(),
                globalOptimality=False, convergence='grid_limited', ambiguous=bool(alternatives),
                alternatives=alternatives, candidates=[{k: c[k] for k in ('estimates', 'residualPower', 'condition')} for c in refined])
