"""Finite-snapshot MUSIC, MVDR, LS-ESPRIT and DML, ported from WaveLab 1.4.

All coordinates are in wavelengths and every unique virtual position contributes
one separated, calibrated channel. Estimators are independent of truth; only
``evaluate`` synthesizes known two-target experiments and scores their errors.
"""
from __future__ import annotations

import math
import numpy as np

from arraylab.framework.parallel import check_cancelled
from . import doa_dml

MAX_SENSORS = 48
DEFAULTS = dict(method='music', separationDeg=4, centerDeg=0, snrDb=20, snapshots=128,
                powerDifferenceDb=0, sourceCorrelation=0, trials=8, seed=42,
                angleToleranceDeg=.5, scanMinDeg=-80, scanMaxDeg=80, gridStepDeg=.2,
                diagonalLoading=.001, smoothing='none', subarrayLength=0)
defaults = DEFAULTS.copy()
SOURCES = [
    {'title': 'MathWorks · MUSIC Super-Resolution DOA Estimation', 'url': 'https://www.mathworks.com/help/phased/ug/music-super-resolution-doa-estimation.html'},
    {'title': 'MathWorks · ESPRIT DOA estimation', 'url': 'https://www.mathworks.com/help/phased/ref/espritdoa.html'},
    {'title': 'MathWorks · MVDR beamformer and diagonal loading', 'url': 'https://www.mathworks.com/help/phased/ref/phased.mvdrbeamformer-system-object.html'}]
ASSUMPTIONS = [
    '实际有限快拍的复高斯双目标与独立复白噪声，计算样本协方差XXᴴ/N；不是无限快拍总体协方差。',
    '目标数K=2已知；强目标单独立通道SNR、功率差及源相关系数由用户给定。无模型阶数估计、互耦、阵元误差、杂波或接收链校准。',
    '相关系数为0–1实数；完全相干为同相源模型，不遍历全部相干相位或目标起伏。',
    '每个独立虚拟位置只取一个已分离、校相通道；重合位置先去重，不隐含额外合并增益。TDM运动补偿及DDM解码须先完成。',
    'MUSIC/MVDR在整个搜索域选两个最强局部峰；ESPRIT/DML输出两个角。估计器不接收真角，真角只用于事后评价。',
    'ESPRIT只使用连续、等间距且d≤0.5λ的ULA子阵；其他位置不参与。无合适子阵时明确不适用。',
    '前向空间平滑须为连续ULA且至少两个重叠子阵，各子阵长度>K；损失有效孔径。DML不使用此预处理。',
    '成功要求两估计均在不重叠真角邻域内，有效容差=min(用户容差,0.4×间隔)，且无竞争模糊解。固定种子的有限经验通过率不是硬件保证。',
    '基本MUSIC/ESPRIT可能因相干源失效，增加快拍不自动恢复秩；可显式选择空间平滑。']
sources, assumptions = SOURCES, ASSUMPTIONS
_ESTIMATOR_KEYS = ('method', 'scanMinDeg', 'scanMaxDeg', 'gridStepDeg', 'diagonalLoading', 'smoothing', 'subarrayLength')


def _finite(x):
    return isinstance(x, (int, float, np.number)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(x)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _issue(status, reason):
    return dict(valid=False, status=status, reason=reason, errors=[reason], warnings=[],
                assumptions=ASSUMPTIONS.copy(), sources=SOURCES.copy(), successRate=None,
                rmseDeg=None, worstErrorDeg=None, estimates=[], spectrum=[], eigenvalues=[], subset=None, details=[])


def read_positions(values):
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) < 3:
        raise ValueError('至少需要3个独立虚拟位置来估计已知的两个目标。')
    if len(values) > 256:
        raise ValueError('输入位置数量超出有界结构检查规模。')
    if any(not _finite(x) or abs(x) > 200 for x in values):
        raise ValueError('虚拟位置须为±200λ内的有限数值。')
    if len({math.floor(float(x)*1e9+.5) for x in values}) != len(values):
        raise ValueError('输入包含重合虚拟位置；请先去重，不隐含重复通道收益。')
    return [float(x) for x in values]


def estimator_options(options=None):
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError('测角参数须为对象。')
    o = {k: options.get(k, DEFAULTS[k]) for k in _ESTIMATOR_KEYS}
    if o['method'] not in ('music', 'mvdr', 'esprit', 'dml'):
        raise ValueError('未知的高分辨测角算法。')
    if o['smoothing'] not in ('none', 'spatial'):
        raise ValueError('未知的空间平滑设置。')
    if not _finite(o['scanMinDeg']) or not _finite(o['scanMaxDeg']) or o['scanMinDeg'] < -89.9 or o['scanMaxDeg'] > 89.9 or o['scanMinDeg'] >= o['scanMaxDeg']:
        raise ValueError('搜索角域须为−89.9°至89.9°内递增区间。')
    if not _finite(o['gridStepDeg']) or not .02 <= o['gridStepDeg'] <= 2 or math.ceil((o['scanMaxDeg']-o['scanMinDeg'])/o['gridStepDeg']) > 10000:
        raise ValueError('谱步长须为0.02–2°，总格点不超过10001。')
    if not _finite(o['diagonalLoading']) or not 0 <= o['diagonalLoading'] <= 1:
        raise ValueError('对角加载系数须在0–1内。')
    if not _integer(o['subarrayLength']) or not 0 <= o['subarrayLength'] <= MAX_SENSORS:
        raise ValueError('平滑子阵长度须为0或不超过48的整数。')
    return o


def evaluation_options(options=None):
    if options is None:
        options = {}
    o = {**DEFAULTS, **options, **estimator_options(options)}
    for key, low, high in [('separationDeg', .02, 80), ('centerDeg', -89, 89), ('snrDb', -30, 60),
                           ('powerDifferenceDb', 0, 40), ('sourceCorrelation', 0, 1), ('angleToleranceDeg', .02, 5)]:
        if not _finite(o[key]) or not low <= o[key] <= high:
            raise ValueError(key+'超出允许范围。')
    for key, low, high in [('snapshots', 4, 4096), ('trials', 1, 64), ('seed', 0, 0xffffffff)]:
        if not _integer(o[key]) or not low <= o[key] <= high:
            raise ValueError(key+'须为允许范围内整数。')
    if o['centerDeg']-o['separationDeg']/2 < o['scanMinDeg'] or o['centerDeg']+o['separationDeg']/2 > o['scanMaxDeg']:
        raise ValueError('双目标真角须位于声明的搜索角域内。')
    return o


def _rng(seed):
    state = int(seed) & 0xffffffff
    def random():
        nonlocal state
        state = (state+0x6d2b79f5) & 0xffffffff
        t = state
        t = ((t ^ (t >> 15))*(t | 1)) & 0xffffffff
        t ^= (t+(((t ^ (t >> 7))*(t | 61)) & 0xffffffff)) & 0xffffffff
        return ((t ^ (t >> 14)) & 0xffffffff)/4294967296
    return random


def _normal(random):
    radius = math.sqrt(-math.log(max(1e-15, random())))
    phase = 2*math.pi*random()
    return complex(radius*math.cos(phase), radius*math.sin(phase))


def _pack(matrix):
    return {'n': len(matrix), 're': matrix.real.ravel().tolist(), 'im': matrix.imag.ravel().tolist()}


def _matrix(cov, n):
    if not isinstance(cov, dict) or cov.get('n') != n:
        raise ValueError('协方差维度与位置数量不一致。')
    try:
        if any(not isinstance(cov.get(k), (list, tuple, np.ndarray)) or any(not _finite(v) for v in cov[k]) for k in ('re', 'im')):
            raise ValueError('协方差须为有限数值数组。')
        re, im = np.asarray(cov['re'], dtype=float), np.asarray(cov['im'], dtype=float)
        if re.ndim != 1 or im.ndim != 1 or re.size != n*n or im.size != n*n:
            raise ValueError('协方差维度与位置数量不一致。')
        m = (re+1j*im).reshape(n, n)
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError('协方差维度与位置数量不一致。') from exc
    if not np.isfinite(m).all():
        raise ValueError('协方差包含非有限数值。')
    scale = max(float(np.abs(m.real).max()), float(np.abs(m.imag).max()))
    tol = max(1e-12, scale*1e-9)
    if np.any(m.diagonal().real < -tol) or np.any(abs(m.diagonal().imag) > tol):
        raise ValueError('协方差对角项无效。')
    difference = m-m.conj().T
    if np.any(abs(difference.real) > tol) or np.any(abs(difference.imag) > tol):
        raise ValueError('输入不是Hermitian协方差。')
    if np.trace(m).real <= 0:
        raise ValueError('协方差总能量必须为正。')
    return m


def generate_covariance(positions, angles, options=None, seed=None, *, cancelled=None):
    p = read_positions(positions)
    o = {**DEFAULTS, **({} if options is None else options)}
    n = len(p)
    if n > MAX_SENSORS:
        raise ValueError('协方差计算最多支持48个独立位置。')
    if not isinstance(angles, (list, tuple, np.ndarray)) or len(angles) != 2 or any(not _finite(a) or abs(a) >= 90 for a in angles):
        raise ValueError('正向合成需要两个有效真角。')
    if not _integer(o['snapshots']) or not 4 <= o['snapshots'] <= 4096 or any(not _finite(o[k]) or not low <= o[k] <= high for k, low, high in [('snrDb', -30, 60), ('powerDifferenceDb', 0, 40), ('sourceCorrelation', 0, 1)]):
        raise ValueError('快拍合成参数无效。')
    random = _rng(o['seed'] if seed is None else seed)
    steering = np.exp(2j*np.pi*np.asarray(p)[:, None]*np.sin(np.radians(angles))[None, :])
    ratio, rho = 10**(-o['powerDifferenceDb']/20), o['sourceCorrelation']
    independent, noise = math.sqrt(max(0, 1-rho*rho)), 10**(-o['snrDb']/20)
    matrix = np.zeros((n, n), dtype=complex)
    for t in range(o['snapshots']):
        if t % 32 == 0:
            check_cancelled(cancelled)
        u, v = _normal(random), _normal(random)
        signal = steering@np.array([u, ratio*(rho*u+independent*v)])
        x = np.array([noise*_normal(random) for _ in range(n)])+signal
        matrix += np.outer(x, x.conj())/o['snapshots']
    matrix[np.diag_indices(n)] = matrix.diagonal().real
    return _pack(matrix)


def hermitian_eigen(cov):
    if not isinstance(cov, dict) or not _integer(cov.get('n')) or not 1 <= cov['n'] <= MAX_SENSORS:
        raise ValueError('特征分解维度须为1–48。')
    matrix = _matrix(cov, cov['n'])
    values, vectors = np.linalg.eigh(matrix)
    values, vectors = values[::-1], vectors[:, ::-1]
    residual = matrix@vectors-vectors*values
    return dict(n=cov['n'], values=values.tolist(), re=vectors.real.ravel().tolist(),
                im=vectors.imag.ravel().tolist(), converged=True,
                maxEigenResidual=float(np.max(abs(residual))), solver='numpy.linalg.eigh')


def select_ula(positions):
    try:
        p = read_positions(positions)
    except (ValueError, TypeError):
        return None
    ordered = sorted((x, i) for i, x in enumerate(p))
    lookup = {math.floor(x*1e8+.5): (x, i) for x, i in ordered}
    spacings = {}
    for i, (x, _) in enumerate(ordered):
        for y, _ in ordered[i+1:]:
            d = y-x
            if d > .5+1e-8:
                break
            if d >= 1e-5:
                spacings[math.floor(d*1e8+.5)] = None
    best = None
    for key in spacings:
        d = key/1e8
        for start, _ in ordered:
            seq = []
            for k in range(len(ordered)):
                point = lookup.get(math.floor((start+k*d)*1e8+.5))
                if point is None or abs(point[0]-start-k*d) > 1e-7:
                    break
                seq.append(point)
            length = len(seq)
            if length < 3:
                continue
            quality = length*(length*length-1)*d*d
            if best is None or quality > best['quality']+1e-10 or (abs(quality-best['quality']) < 1e-10 and length > best['count']):
                best = dict(positions=[v[0] for v in seq], indices=[v[1] for v in seq], count=length,
                            spacing=d, aperture=(length-1)*d, quality=quality)
    return best


def _structure(p, o):
    if len(p) > MAX_SENSORS:
        return _issue('unsupported', '最多验证48个独立位置，不自动删减阵列。')
    if o['method'] == 'dml' and o['smoothing'] != 'none':
        return _issue('unsupported', 'DML使用原始协方差，不隐式施加空间平滑。')
    if o['method'] == 'esprit' or o['smoothing'] == 'spatial':
        subset = select_ula(p)
        if subset is None:
            return _issue('unsupported', '没有至少3个连续且间距≤0.5λ的ULA位置；ESPRIT/平滑不适用。')
        length = (o['subarrayLength'] or subset['count']//2+1) if o['smoothing'] == 'spatial' else subset['count']
        if o['smoothing'] == 'spatial' and (length < 3 or length >= subset['count']):
            return _issue('unsupported', '平滑要求子阵长度≥3且至少两个重叠子阵。')
        return dict(valid=True, subset=subset, length=length)
    return dict(valid=True, subset=dict(positions=p[:], indices=list(range(len(p))), count=len(p),
                                       spacing=None, aperture=max(p)-min(p)), length=len(p))


def _prepare(matrix, o, info):
    subset = info['subset'].copy()
    indices = subset['indices']
    matrix = matrix[np.ix_(indices, indices)]
    p, smoothing = subset['positions'][:], None
    if o['smoothing'] == 'spatial':
        n = info['length']
        blocks = len(matrix)-n+1
        matrix = sum((matrix[k:k+n, k:k+n]/blocks for k in range(blocks)), np.zeros((n, n), complex))
        p = [i*subset['spacing'] for i in range(n)]
        smoothing = dict(method='forward spatial smoothing', subarrayLength=n, subarrayCount=blocks)
    subset.update(processedCount=len(p), processedAperture=max(p)-min(p))
    return matrix, p, smoothing, subset


def _peaks(spectrum, step):
    if max(p['y'] for p in spectrum)-min(p['y'] for p in spectrum) < 1e-6:
        return []
    peaks = []
    for i in range(1, len(spectrum)-1):
        left, center, right = (spectrum[j]['y'] for j in (i-1, i, i+1))
        if center > left and center >= right:
            denominator = left-2*center+right
            offset = min(.5, max(-.5, .5*(left-right)/denominator)) if abs(denominator) > 1e-12 else 0
            peaks.append((spectrum[i]['x']+offset*step, center))
    peaks.sort(key=lambda p: -p[1])
    chosen = []
    for x, _ in peaks:
        if all(abs(x-q) >= 2*step-1e-10 for q in chosen):
            chosen.append(x)
        if len(chosen) == 2:
            break
    return sorted(chosen)


def estimate(cov, positions, options=None, *, cancelled=None):
    try:
        p = read_positions(positions)
        o = estimator_options(options)
        info = _structure(p, o)
        if not info['valid']:
            return info
        matrix, used, smoothing, subset = _prepare(_matrix(cov, len(p)), o, info)
    except (ValueError, TypeError) as exc:
        return _issue('invalid', str(exc))
    check_cancelled(cancelled)
    n, warnings = len(matrix), []
    if subset['count'] < len(p):
        warnings.append(f"仅使用{subset['count']}/{len(p)}个位置组成连续ULA，未使用其余位置。")
    if smoothing:
        warnings.append(f"前向空间平滑：{smoothing['subarrayCount']}个长度{n}子阵，处理孔径缩短。")
    base = dict(valid=True, status='estimated', estimates=[], spectrum=[], eigenvalues=[], subset=subset,
                smoothing=smoothing, warnings=warnings, assumptions=ASSUMPTIONS.copy(), sources=SOURCES.copy())
    if o['method'] == 'dml':
        result = doa_dml.estimate(_pack(matrix), used, o, cancelled=cancelled)
        return {**base, **result, 'subset': subset, 'warnings': warnings+result.get('warnings', []), 'sources': SOURCES+result.get('sources', [])}
    vectors = None
    if o['method'] == 'mvdr':
        extra = o['diagonalLoading']*float(np.trace(matrix).real)/n
        try:
            factor = np.linalg.cholesky(matrix+np.eye(n)*extra)
            if np.min(abs(factor.diagonal())**2) <= float(np.trace(matrix).real)/n*1e-12:
                raise np.linalg.LinAlgError()
        except np.linalg.LinAlgError:
            return {**base, 'valid': False, 'status': 'failed', 'reason': '样本协方差非正定；请增加快拍或对角加载。'}
        base['loadingPower'] = extra
    else:
        try:
            values, vectors = np.linalg.eigh(matrix)
        except np.linalg.LinAlgError:
            return {**base, 'valid': False, 'status': 'failed', 'reason': 'Hermitian特征分解未收敛。'}
        values, vectors = values[::-1], vectors[:, ::-1]
        base.update(eigenvalues=values.tolist(), eigenConverged=True)
        if values[-1] < -max(1e-12, values[0]*1e-8):
            return _issue('invalid', '输入Hermitian矩阵不是半正定协方差。')
        if values[1] < values[0]*1e-9:
            warnings.append('第二信号特征值接近数值秩阈值，双目标模型可能退化。')
    if o['method'] == 'esprit':
        first, second = vectors[:-1, :2], vectors[1:, :2]
        gram = first.conj().T@first
        if np.linalg.det(gram).real <= 1e-10:
            return {**base, 'valid': False, 'status': 'failed', 'reason': 'ESPRIT子阵最小二乘退化。'}
        roots = np.linalg.eigvals(np.linalg.solve(gram, first.conj().T@second))
        sine = np.angle(roots)/(2*np.pi*subset['spacing'])
        if not np.isfinite(sine).all() or np.any(abs(sine) > 1+1e-8):
            return {**base, 'valid': False, 'status': 'failed', 'reason': 'ESPRIT旋转特征值超出物理范围。'}
        angles = sorted(np.degrees(np.arcsin(np.clip(sine, -1, 1))).tolist())
        base.update(estimates=[a for a in angles if o['scanMinDeg'] <= a <= o['scanMaxDeg']],
                    rotationEigenvalues=[[float(v.real), float(v.imag)] for v in roots])
        return base
    count = math.ceil((o['scanMaxDeg']-o['scanMinDeg'])/o['gridStepDeg'])
    step = (o['scanMaxDeg']-o['scanMinDeg'])/count
    angles = o['scanMinDeg']+np.arange(count+1)*step
    atoms = np.exp(2j*np.pi*np.asarray(used)[:, None]*np.sin(np.radians(angles))[None, :])
    check_cancelled(cancelled)
    if o['method'] == 'mvdr':
        solved = np.linalg.solve(factor, atoms)
        power = 1/np.maximum(1e-20, np.sum(abs(solved)**2, axis=0))
    else:
        projection = np.sum(abs(vectors[:, :2].conj().T@atoms)**2, axis=0)
        power = 1/np.maximum(n*1e-12, n-projection)
        # A scalar multiple of identity has no preferred direction. Its
        # degenerate eigenspace may differ by BLAS, so do not turn that arbitrary
        # basis into fictitious MUSIC peaks.
        if np.ptp(values) <= max(1e-12, abs(values[0])*1e-12):
            power[:] = power[0]
    db = 10*np.log10(np.maximum(1e-30, power))
    db = np.maximum(-120, db-db.max())
    base['spectrum'] = [{'x': float(a), 'y': float(v)} for a, v in zip(angles, db)]
    base['estimates'] = _peaks(base['spectrum'], step)
    return base


def evaluate(positions, options=None, *, cancelled=None):
    try:
        p, o = read_positions(positions), evaluation_options(options)
        info = _structure(p, o)
        if not info['valid']:
            return info
    except (ValueError, TypeError) as exc:
        return _issue('invalid', str(exc))
    truth = [o['centerDeg']-o['separationDeg']/2, o['centerDeg']+o['separationDeg']/2]
    tolerance = min(o['angleToleranceDeg'], .4*o['separationDeg'])
    estimator = estimator_options(o)
    details, representative = [], None
    successes = estimated = 0
    squared, worst = 0., 0.
    for trial in range(o['trials']):
        check_cancelled(cancelled)
        seed = (o['seed']+(trial+1)*0x9e3779b9) & 0xffffffff
        cov = generate_covariance(p, truth, o, seed, cancelled=cancelled)
        answer = estimate(cov, p, estimator, cancelled=cancelled)
        if answer['status'] in ('unsupported', 'invalid'):
            return {**answer, 'trial': trial}
        if representative is None:
            representative = answer
        angles = sorted(a for a in answer.get('estimates', []) if _finite(a))
        complete = bool(answer['valid'] and len(angles) == 2 and angles[1]-angles[0] > 1e-6)
        ambiguous = bool(answer.get('ambiguous') or answer['status'] == 'ambiguous')
        errors = [a-t for a, t in zip(angles, truth)] if complete else []
        max_error = max(map(abs, errors)) if complete else None
        rmse = math.sqrt(sum(e*e for e in errors)/2) if complete else None
        success = bool(complete and not ambiguous and max_error <= tolerance)
        successes += success
        if complete:
            estimated += 1
            squared += sum(e*e for e in errors)
            worst = max(worst, max_error)
        details.append(dict(trial=trial+1, seed=seed, estimates=angles, errors=errors, success=success,
                            ambiguous=ambiguous, estimationValid=complete, algorithmStatus=answer['status'],
                            rmseDeg=rmse, maxErrorDeg=max_error, reason=answer.get('reason')))
    warnings = representative.get('warnings', [])[:]
    if o['sourceCorrelation'] > .95 and o['smoothing'] == 'none' and o['method'] in ('music', 'esprit'):
        warnings.append('高相关源未做空间平滑，基本子空间方法秩条件可能失效。')
    if tolerance < o['angleToleranceDeg']:
        warnings.append(f'为防止两个估计挤在一起仍成功，有效误差容差收紧为{tolerance:.4f}°。')
    if o['gridStepDeg'] > o['separationDeg']/2 and o['method'] in ('music', 'mvdr'):
        warnings.append('谱网格相对目标间隔较粗；不会依据真角自动细化。')
    if estimated < o['trials']:
        warnings.append(f"{o['trials']-estimated}次未输出两个有效角；RMSE只统计完整估计，成功率分母为全部试验。")
    example = dict(estimates=representative.get('estimates', []), truth=truth[:], errors=details[0]['errors'],
                   success=details[0]['success'], ambiguous=details[0]['ambiguous'], spectrum=representative.get('spectrum', []),
                   subset=representative['subset'], eigenvalues=representative.get('eigenvalues', []), diagnostics=representative.get('diagnostics'))
    return dict(valid=True, status='evaluated', method=o['method'], options=o, truthAngles=truth,
                effectiveToleranceDeg=tolerance, successRate=successes/o['trials'], successes=successes,
                trials=o['trials'], estimatedTrials=estimated, unresolvedTrials=o['trials']-estimated,
                ambiguousTrials=sum(t['ambiguous'] for t in details), rmseDeg=math.sqrt(squared/(2*estimated)) if estimated else None,
                worstErrorDeg=worst if estimated else None, usedChannelCount=representative['subset']['processedCount'],
                estimates=example['estimates'], spectrum=example['spectrum'], eigenvalues=example['eigenvalues'],
                subset=representative['subset'], smoothing=representative['smoothing'], representative=example,
                algorithmValidity='conditional finite-snapshot trials; known K=2', warnings=warnings,
                assumptions=ASSUMPTIONS.copy(), sources=representative.get('sources', SOURCES.copy()),
                details=details, trialResults=details)
