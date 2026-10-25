"""Bounded, reproducible WaveLab 1.4 physical Tx/Rx coordinate synthesis.

Geometry search is serial to preserve its random stream. Independent noisy
scan-centre experiments may use the application's bounded thread scheduler.
"""
from copy import deepcopy
import math

import numpy as np

from arraylab.features.waveform import antenna
from arraylab.framework.parallel import check_cancelled, ordered_map

RAD = math.pi/180
SUPER_METHODS = ('music', 'mvdr', 'esprit', 'dml')
METHOD_NAMES = {'music': 'MUSIC', 'mvdr': 'Capon / MVDR', 'esprit': 'LS-ESPRIT', 'dml': 'DML'}
DEFAULTS = {'txCount': 3, 'rxCount': 4, 'targetResolutionDeg': 14, 'criterion': 'two_target',
            'axis': 'azimuth', 'scanMaxDeg': 30, 'maxApertureLambda': 20,
            'minSpacingLambda': .5, 'gridLambda': .5, 'maxSidelobeDb': -10,
            'iterations': 800, 'seed': 42, 'snrDb': 20, 'snapshots': 128,
            'powerDifferenceDb': 0, 'sourceCorrelation': 0, 'trials': 8,
            'angleToleranceDeg': .5, 'requiredSuccessRate': .8, 'diagonalLoading': .001,
            'smoothing': 'none', 'subarrayLength': 0, 'scanGridStepDeg': .2}
ASSUMPTIONS = [
    '综合实际 Tx/Rx 的一维设计轴坐标，虚拟位置为 Tx+Rx；不直接优化无法分解为 Tx/Rx 的虚拟位置。',
    '方位设计时 Tx/Rx 放在平行水平行，俯仰设计时放在平行垂直行；两行相距最小阵元间距。孔径上限只约束设计轴，非完整二维封装尺寸。',
    '按等幅、已校相、已正确分离 Tx 的远场单频阵列因子评估。TDM 运动相位补偿和 DDM 解码由外部处理完成。',
    '双目标判据使用两个等功率、不相干、无噪声点目标的常规波束形成功率和，要求各目标附近存在局部峰且中间谷比弱峰低至少 3 dB。该判据不是通用硬件角分辨率。',
    '经典角度约束检查 0°、扫描半角的 1/3、2/3 和端点，负角在经典等功率无噪声模型中对称等价；超分辨另行检查正负中心及端点。离散角检查不保证所有连续扫描角。',
    'HPBW、常规双目标和选定超分辨算法分别评估；经典判据不包含噪声，超分辨判据使用独立设置的有限角度快拍和含噪实验。真实单元方向图、互耦、安装遮挡、宽带误差均未纳入。',
    '搜索采用固定种子的有界启发式，并以更细网格及局部极值细化复核最佳候选。未达标表示本次搜索未找到满足全部约束的方案，不是不存在可行阵列的证明。',
    '虚拟网格占用率是独立虚拟位置数除以其设计轴网格跨度的可用格点数；重复通道仍按重数参与常规阵列因子，不算新的空间位置。超分辨实验保守地每个独立位置只保留一路，不额外计重合通道的合并收益。',
    '超分辨搜索先以导向矢量相关性和几何旁瓣筛选至多四组候选，再用所选算法和筛选噪声比较；最终验证使用另一组独立随机种子，不根据验证结果重新挑选阵列。',
    '统计快拍数独立于波形 loops，表示已完成距离/多普勒处理、Tx 已正确分离和校相后的角度数据；已知目标数固定为两个，验证通过率不是检测概率 Pd 或硬件分辨率保证。',
    'ESPRIT 或空间平滑只使用满足结构要求的连续均匀子阵；不适用时明确未通过，不退回常规波束形成。DML 为有界双目标联合似然搜索，不保证连续角域全局最优。']
SOURCES = [
    {'title': 'TI · MIMO Radar / virtual array', 'url': 'https://www.ti.com/lit/pdf/SWRA554'},
    {'title': 'MathWorks · Grating lobes', 'url': 'https://www.mathworks.com/help/antenna/ug/grating-lobes.html'}]


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _integer(value):
    return _finite(value) and float(value).is_integer()


def _round(value):
    return math.floor(value+.5)


def _clamp(x, a, b):
    return max(a, min(b, x))


def _db(power):
    return 10*math.log10(max(1e-12, power))


def normalize(input=None):
    if input is not None and type(input) is not dict:
        raise ValueError('稀疏设计参数须为对象。')
    o = {**DEFAULTS, **(input or {})}
    if not _integer(o['txCount']) or not 1 <= o['txCount'] <= 12 or not _integer(o['rxCount']) or not 1 <= o['rxCount'] <= 16:
        raise ValueError('Tx 须为 1–12，Rx 须为 1–16 的整数。')
    if o['txCount']*o['rxCount'] < 2:
        raise ValueError('1 Tx × 1 Rx 只有一个虚拟位置，无法通过阵列孔径满足角度分辨设计；请增加至少一类通道数。')
    for key in ('targetResolutionDeg', 'scanMaxDeg', 'maxApertureLambda', 'minSpacingLambda', 'gridLambda', 'maxSidelobeDb'):
        if not _finite(o[key]):
            raise ValueError(key+' 必须为有限数值。')
    if not .2 <= o['targetResolutionDeg'] <= 40:
        raise ValueError('角度目标须为 0.2–40°。')
    if not 0 <= o['scanMaxDeg'] <= 70 or o['scanMaxDeg']+o['targetResolutionDeg'] > 85:
        raise ValueError('扫描半角须在 0–70°，且半角加分辨目标不得超过 85°。')
    if not 0 < o['maxApertureLambda'] <= 50:
        raise ValueError('设计轴物理孔径上限须在 (0,50] λ。')
    if not .1 <= o['minSpacingLambda'] <= 10 or not .05 <= o['gridLambda'] <= 2:
        raise ValueError('最小间距须为 0.1–10 λ，格点间距须为 0.05–2 λ。')
    if not -40 <= o['maxSidelobeDb'] < 0:
        raise ValueError('旁瓣上限须在 [-40,0) dB。')
    if o['criterion'] not in ('two_target', 'hpbw', *SUPER_METHODS) or o['axis'] not in ('azimuth', 'elevation'):
        raise ValueError('未知的角度判据或设计轴。')
    if o['criterion'] in SUPER_METHODS:
        bounds = {'snrDb': (-30, 60), 'snapshots': (4, 4096), 'powerDifferenceDb': (0, 40),
                  'sourceCorrelation': (0, 1), 'trials': (1, 64), 'angleToleranceDeg': (.02, 5),
                  'requiredSuccessRate': (.01, 1), 'diagonalLoading': (0, 1),
                  'subarrayLength': (0, 48), 'scanGridStepDeg': (.05, 1)}
        for key, (low, high) in bounds.items():
            if not _finite(o[key]) or not low <= o[key] <= high:
                raise ValueError(f'{key} 须为 {low}–{high} 的有限数值。')
        for key in ('snapshots', 'trials', 'subarrayLength'):
            if not _integer(o[key]):
                raise ValueError(key+' 必须是整数。')
            o[key] = int(o[key])
        if o['subarrayLength'] != 0 and o['subarrayLength'] < 3:
            raise ValueError('空间平滑子阵长度须为 0（自动）或至少 3。')
        if o['smoothing'] not in ('none', 'spatial'):
            raise ValueError('空间平滑仅支持 none 或 spatial。')
        if o['criterion'] == 'dml' and o['smoothing'] != 'none':
            raise ValueError('DML 使用原始样本协方差，本实现不在 DML 前应用空间平滑。')
        if o['txCount']*o['rxCount'] < 3:
            raise ValueError('双目标超分辨实验至少需要三个独立观测位置。')
    if not _integer(o['iterations']) or not 100 <= o['iterations'] <= 5000 or not _integer(o['seed']) or not 0 <= o['seed'] <= 0xffffffff:
        raise ValueError('搜索次数须为 100–5000，种子须为 0–4294967295 的整数。')
    for key in ('txCount', 'rxCount', 'iterations', 'seed'):
        o[key] = int(o[key])
    o['maxSlot'] = math.floor(o['maxApertureLambda']/o['gridLambda']+1e-9)
    o['minStep'] = math.ceil(o['minSpacingLambda']/o['gridLambda']-1e-9)
    if (max(o['txCount'], o['rxCount'])-1)*o['minStep'] > o['maxSlot']:
        raise ValueError('当前孔径、格点和最小间距无法容纳给定通道数。')
    return o


def _random(seed):
    state = int(seed) & 0xffffffff
    def draw():
        nonlocal state
        state = (state+0x6d2b79f5) & 0xffffffff
        t = ((state^(state >> 15))*(state | 1)) & 0xffffffff
        t ^= (t+((t^(t >> 7))*(t | 61) & 0xffffffff)) & 0xffffffff
        return ((t^(t >> 14)) & 0xffffffff)/4294967296
    return draw


def _power_function(tx, rx):
    def scalar(d):
        value = 1.
        for positions in (tx, rx):
            real = imaginary = 0.
            for x in positions:
                phase = math.tau*x*d
                real += math.cos(phase)
                imaginary += math.sin(phase)
            value *= (real*real+imaginary*imaginary)/(len(positions)**2)
        return _clamp(value, 0, 1)
    def vector(d):
        d = np.asarray(d)
        value = np.ones_like(d, dtype=float)
        for positions in (tx, rx):
            real = np.zeros_like(d, dtype=float)
            imaginary = np.zeros_like(d, dtype=float)
            for x in positions:
                phase = math.tau*x*d
                real += np.cos(phase)
                imaginary += np.sin(phase)
            value *= (real*real+imaginary*imaginary)/(len(positions)**2)
        return np.clip(value, 0, 1)
    scalar.vector = vector
    return scalar


def _extrema(fn, a, b, maximum, steps=24):
    ratio = (math.sqrt(5)-1)/2
    x, y = b-ratio*(b-a), a+ratio*(b-a)
    fx, fy = fn(x), fn(y)
    for _ in range(steps):
        if (fx < fy) if maximum else (fx > fy):
            a, x, fx = x, y, fy
            y = a+ratio*(b-a)
            fy = fn(y)
        else:
            b, y, fy = y, x, fx
            x = b-ratio*(b-a)
            fx = fn(x)
    point = (a+b)/2
    return {'x': point, 'y': fn(point)}


def _shape(tx, rx, o, fine):
    fn = _power_function(tx, rx)
    span = tx[-1]-tx[0]+rx[-1]-rx[0]
    limit = 1+math.sin(o['scanMaxDeg']*RAD)
    samples = max(1200 if fine else 180, math.ceil(limit*max(1, span)*(70 if fine else 10)))
    step = limit/samples
    powers = fn.vector(np.arange(samples+1)*step)
    powers[0] = 1
    first_min, half_high = -1, None
    for i in range(1, samples+1):
        if half_high is None and powers[i] <= 10**(-3/10):
            half_high = i*step
        if i >= 2 and first_min < 0 and powers[i-1] <= powers[i-2] and powers[i-1] < powers[i]:
            first_min = i-1
    half = None
    if half_high is not None:
        low, high = half_high-step, half_high
        for _ in range(38 if fine else 16):
            mid = (low+high)/2
            if fn(mid) > 10**(-3/10):
                low = mid
            else:
                high = mid
        half = (low+high)/2
    sidelobe, first_minimum = 1., None
    if first_min > 0:
        minimum = _extrema(fn, (first_min-1)*step, (first_min+1)*step, False, 28 if fine else 10)
        first_minimum = minimum['x']
        sidelobe = max(fn(limit), float(powers[first_min+1:].max()))
        if fine:
            for i in range(first_min+1, samples):
                if powers[i] >= powers[i-1] and powers[i] >= powers[i+1]:
                    sidelobe = max(sidelobe, _extrema(fn, (i-1)*step, (i+1)*step, True, 25)['y'])
    def width(center):
        u = math.sin(center*RAD)
        return (math.asin(u+half)-math.asin(u-half))/RAD if half is not None and u+half <= 1 and u-half >= -1 else None
    return {'fn': fn, 'span': span, 'halfU': half, 'firstMinimumU': first_minimum,
            'hpbwDeg': width(0), 'worstHpbwDeg': width(o['scanMaxDeg']),
            'worstSidelobeDb': _db(sidelobe), 'sampleCount': samples}


def _pair_response(fn, center, separation, fine=False, return_pattern=False):
    t1, t2 = center-separation/2, center+separation/2
    u1, u2 = math.sin(t1*RAD), math.sin(t2*RAD)
    n = 400 if fine else 80
    low, high = max(-89.9, center-separation), min(89.9, center+separation)
    step = (high-low)/n
    def signal(theta):
        u = math.sin(theta*RAD)
        return fn(u-u1)+fn(u-u2)
    xs = low+np.arange(n+1)*step
    us = np.sin(xs*RAD)
    ys = fn.vector(us-u1)+fn.vector(us-u2)
    peaks = []
    for i in range(1, n):
        if ys[i] > ys[i-1] and ys[i] >= ys[i+1]:
            peaks.append(_extrema(signal, float(xs[i-1]), float(xs[i+1]), True, 28) if fine else
                         {'x': float(xs[i]), 'y': float(ys[i])})
    def near(target, side):
        values = [p for p in peaks if abs(p['x']-target) <= separation*.35 and
                  (p['x'] < center if side < 0 else p['x'] > center)]
        return min(values, key=lambda p: abs(p['x']-target)) if values else None
    left, right = near(t1, -1), near(t2, 1)
    dip, valley = 0., None
    if left and right:
        minimum = {'x': center, 'y': signal(center)}
        for x, y in zip(xs, ys):
            if left['x'] < x < right['x'] and y < minimum['y']:
                minimum = {'x': float(x), 'y': float(y)}
        valley = (_extrema(signal, max(left['x'], minimum['x']-step), min(right['x'], minimum['x']+step), False, 28)
                  if fine else minimum)
        dip = max(0, _db(min(left['y'], right['y'])/max(1e-12, valley['y'])))
    peak = float(ys.max())
    pattern = [{'x': float(x), 'y': _db(float(y)/peak)} for x, y in zip(xs, ys)] if return_pattern else []
    return {'resolved': bool(left and right and dip >= 3-1e-6), 'dipDb': dip,
            'leftPeak': left['x'] if left else None, 'rightPeak': right['x'] if right else None,
            'valleyDeg': valley['x'] if valley else None, 'pattern': pattern}


def unique_positions(tx, rx):
    values = {}
    for t in tx:
        for r in rx:
            values.setdefault(_round((t+r)*1e9), t+r)
    ordered = sorted(values.values())
    return [x-ordered[0] for x in ordered]


def _super_surrogate(positions, o):
    from . import doa
    used = positions
    penalty = 100 if not 3 <= len(positions) <= 48 else 0
    if o['criterion'] == 'esprit' or o['smoothing'] == 'spatial':
        subset = doa.select_ula(positions)
        if not subset:
            penalty += 100
        else:
            used = subset['positions']
            if o['smoothing'] == 'spatial':
                length = o['subarrayLength'] or len(used)//2+1
                if length < 3 or length >= len(used):
                    penalty += 100
                else:
                    used = used[:length]
    worst = 0.
    for center in (0, o['scanMaxDeg']):
        delta = math.sin((center+o['targetResolutionDeg']/2)*RAD)-math.sin((center-o['targetResolutionDeg']/2)*RAD)
        re = sum(math.cos(math.tau*x*delta) for x in used)
        im = sum(math.sin(math.tau*x*delta) for x in used)
        correlation = _clamp((re*re+im*im)/len(used)**2, 0, 1)
        worst = max(worst, correlation/max(.002, 1-correlation)/math.sqrt(len(used)))
    return {'penalty': worst+penalty, 'usedChannelCount': len(used)}


def _score(tx, rx, o, fine=False):
    is_super = o['criterion'] in SUPER_METHODS
    s = _shape(tx, rx, o, fine)
    centers = list(dict.fromkeys([0, o['scanMaxDeg']/3, 2*o['scanMaxDeg']/3, o['scanMaxDeg']] if fine else [0, o['scanMaxDeg']]))
    pairs = ([{'centerDeg': center, **_pair_response(s['fn'], center, o['targetResolutionDeg'], fine, fine and center == 0)}
              for center in centers] if not is_super and (o['criterion'] == 'two_target' or fine) else [])
    pair_dip = min(p['dipDb'] for p in pairs) if pairs else None
    resolution_pass = False if is_super else (s['worstHpbwDeg'] is not None and s['worstHpbwDeg'] <= o['targetResolutionDeg']+1e-6
                                              if o['criterion'] == 'hpbw' else bool(pairs) and all(p['resolved'] for p in pairs))
    width_penalty = 25 if s['worstHpbwDeg'] is None else max(0, s['worstHpbwDeg']/o['targetResolutionDeg']-1)
    dip_penalty = 0 if pair_dip is None else max(0, 3-pair_dip)/3
    sl_penalty = max(0, s['worstSidelobeDb']-o['maxSidelobeDb'])/6
    unique = {_round((t+r)/o['gridLambda']) for t in tx for r in rx}
    aperture = max(tx[-1]-tx[0], rx[-1]-rx[0])
    proxy = _super_surrogate(unique_positions(tx, rx), o) if is_super else None
    loss = 10*(proxy['penalty'] if is_super else width_penalty if o['criterion'] == 'hpbw' else dip_penalty)+6*sl_penalty+.08*aperture/o['maxApertureLambda']+.08*(1-len(unique)/(len(tx)*len(rx)))
    return {**s, 'pair0': pairs[0] if pairs else None, 'pairEdge': pairs[-1] if pairs else None,
            'pairs': pairs, 'worstPairDipDb': pair_dip, 'resolutionPass': resolution_pass,
            'loss': loss, 'uniqueCount': len(unique), 'physicalApertureLambda': aperture}


def _legal(slots, count, o):
    return len(slots) == count and all(_integer(v) and 0 <= v <= o['maxSlot'] and
                                     (i == 0 or v-slots[i-1] >= o['minStep']) for i, v in enumerate(slots))


def _scaled(values, scale, o):
    items = sorted(_round(v*scale/o['gridLambda']) for v in values)
    return [v-items[0] for v in items]


def _random_array(count, cap, o, rng):
    cap = max((count-1)*o['minStep'], min(cap, o['maxSlot']))
    slack = cap-(count-1)*o['minStep']
    extra = sorted(math.floor(rng()*(slack+1)) for _ in range(count))
    result = [e+i*o['minStep'] for i, e in enumerate(extra)]
    return [v-result[0] for v in result]


def _geometry(tx, rx, o):
    row = o['minSpacingLambda']
    if o['axis'] == 'azimuth':
        return {'tx': [[x, row] for x in tx], 'rx': [[x, 0] for x in rx]}
    return {'tx': [[row, z] for z in tx], 'rx': [[0, z] for z in rx]}


def _experiment_seed(seed, block_offset, index):
    return (seed+((block_offset+index)*128*0x9e3779b9)) & 0xffffffff


def _algorithm_experiment(tx, rx, o, phase='validation', *, workers=None, cancelled=None, progress=None):
    from . import doa
    positions = unique_positions(tx, rx)
    angle = o['scanMaxDeg']
    centers = list(dict.fromkeys([0, -angle, angle] if phase == 'screening' else
                                 [0, -angle, -2*angle/3, -angle/3, angle/3, 2*angle/3, angle]))
    trials = min(2, o['trials']) if phase == 'screening' else o['trials']
    salt = 0 if phase == 'screening' else 16
    tolerance = min(o['angleToleranceDeg'], .4*o['targetResolutionDeg'])
    options = {'method': o['criterion'], 'separationDeg': o['targetResolutionDeg'],
               **{k: o[k] for k in ('snrDb', 'snapshots', 'powerDifferenceDb', 'sourceCorrelation',
                                   'angleToleranceDeg', 'diagonalLoading', 'smoothing', 'subarrayLength')},
               'trials': trials, 'scanMinDeg': -80, 'scanMaxDeg': 80, 'gridStepDeg': o['scanGridStepDeg']}
    notes = [
        '每个独立虚拟位置只使用一路角度观测，重复位置未额外提供合并收益。',
        '统计快拍独立于波形 loops，已知源数固定为两个；估计器不接收真角来挑峰，真角仅用于事后匹配。',
        '验证分别运行所列正负中心角，有限噪声、强弱目标和相关源的逐次结果不按镜像对称替代。',
        '通过率仅适用于给定 SNR、快拍、强弱比、源相关性和有限重复次数，不是 Pd 或连续视场的保证。',
        '验证种子与几何筛选种子独立；最终验证失败时不换另一候选继续试到通过。']
    def failure(message):
        return {'method': o['criterion'], 'valid': False, 'status': 'unsupported', 'successRate': None,
                'rmseDeg': None, 'worstErrorDeg': None, 'estimatedTrialsTotal': 0, 'totalTrials': 0,
                'requestedTrials': trials*len(centers), 'validation': [], 'representative': None,
                'usedChannelCount': len(positions), 'uniqueChannelCount': len(positions),
                'effectiveToleranceDeg': tolerance, 'options': options, 'notes': notes,
                'warnings': [message], 'phase': phase}
    if not 3 <= len(positions) <= 48:
        return failure('双目标超分辨实验支持 3–48 个独立观测位置；未截断或替换通道。')
    if len(positions)**2*o['snapshots']*trials*len(centers) > 2e8:
        return failure('当前通道、快拍和重复次数超出工作台计算预算，请缩小实验规模。')
    def run(item):
        index, center = item
        check_cancelled(cancelled)
        seed = _experiment_seed(o['seed'], salt, index)
        result = doa.evaluate(positions, {**options, 'centerDeg': center, 'seed': seed}, cancelled=cancelled)
        return {**result, 'centerDeg': center, 'seed': seed,
                'truthAngles': [center-o['targetResolutionDeg']/2, center+o['targetResolutionDeg']/2]}
    validation, execution = ordered_map(run, enumerate(centers), workers=workers,
                                       work_units=len(positions)**2*o['snapshots']*trials*len(centers),
                                       cancelled=cancelled,
                                       progress=(lambda done, total: progress(f'{phase}：{done}/{total} 个扫描中心')) if progress else None)
    valid = all(r.get('valid') and r.get('status') == 'evaluated' and _finite(r.get('successRate')) for r in validation)
    estimated = sum(r.get('estimatedTrials', 0) for r in validation)
    total = sum(r.get('trials', 0) for r in validation)
    complete = [r for r in validation if r.get('estimatedTrials', 0) > 0]
    errors = [r['worstErrorDeg'] for r in validation if _finite(r.get('worstErrorDeg'))]
    rmse = math.sqrt(sum(r['rmseDeg']**2*r['estimatedTrials'] for r in complete)/estimated) if estimated > 0 and all(_finite(r.get('rmseDeg')) for r in complete) else None
    first = validation[0] if validation else None
    representative = {**first, **(first.get('representative') or {}), 'centerDeg': first['centerDeg'],
                      'seed': first['seed'], 'truthAngles': first['truthAngles']} if first else None
    counts = [r.get('usedChannelCount', (r.get('subset') or {}).get('count', len(positions))) for r in validation]
    warnings = [w for r in validation for w in (r.get('warnings') or [])]
    warnings += [w for r in validation if not r.get('valid') for w in (r.get('errors') or [r.get('reason') or '所选算法在此阵列上不适用。'])]
    sources = {s['url']: s for r in validation for s in (r.get('sources') or [])}
    return {'method': o['criterion'], 'valid': valid,
            'status': 'evaluated' if valid else 'unsupported' if any(r.get('status') == 'unsupported' for r in validation) else 'invalid',
            'successRate': min(r['successRate'] for r in validation) if valid else None,
            'rmseDeg': rmse, 'worstErrorDeg': max(errors) if errors else None,
            'estimatedTrialsTotal': estimated, 'totalTrials': total, 'requestedTrials': trials*len(centers),
            'validation': validation, 'representative': representative, 'usedChannelCount': min(counts),
            'uniqueChannelCount': len(positions), 'effectiveToleranceDeg': tolerance,
            'options': options, 'notes': notes, 'warnings': list(dict.fromkeys(warnings)),
            'sources': list(sources.values()), 'phase': phase, 'execution': execution}


def _report(tx, rx, o, search, algorithm_result=None, *, workers=None, cancelled=None, progress=None):
    check_cancelled(cancelled)
    s = _score(tx, rx, o, True)
    geometry = _geometry(tx, rx, o)
    a = antenna.calculate({**geometry, 'steerAzDeg': 0, 'steerElDeg': 0}, cancelled=cancelled)
    is_super = o['criterion'] in SUPER_METHODS
    sr = (algorithm_result if algorithm_result is not None else
          _algorithm_experiment(tx, rx, o, workers=workers, cancelled=cancelled, progress=progress)) if is_super else None
    resolution_pass = bool(sr['valid'] and sr['successRate'] >= o['requiredSuccessRate']-1e-12) if is_super else s['resolutionPass']
    status = 'met' if resolution_pass and s['worstSidelobeDb'] <= o['maxSidelobeDb']+1e-6 else 'unmet'
    pattern = [{'x': -90+i*.1, 'y': _db(s['fn'](math.sin((-90+i*.1)*RAD)))} for i in range(1801)]
    if is_super:
        detail = (f"目标间隔 {o['targetResolutionDeg']}°，各检查角最低通过率 {100*sr['successRate']:.1f}% / 要求 {100*o['requiredSuccessRate']:.1f}%；有效角误差限 {sr['effectiveToleranceDeg']:.3f}°，每角 {o['trials']} 次独立验证。"
                  if sr['valid'] else '所选 '+METHOD_NAMES[o['criterion']]+' 未通过适用性或数值检查：'+' '.join(sr['warnings']))
    elif o['criterion'] == 'hpbw':
        width = '不可定' if s['worstHpbwDeg'] is None else f"{s['worstHpbwDeg']:.3f}°"
        detail = f"最差 HPBW {width} / 目标 {o['targetResolutionDeg']}°。"
    else:
        centers = '/'.join(f"{p['centerDeg']:.2f}" for p in s['pairs'])
        detail = f"目标间隔 {o['targetResolutionDeg']}°，扫描角 {centers}°（负角对称等价），最差峰谷差 {s['worstPairDipDb']:.3f} dB / 要求 ≥3 dB。"
    checks = [
        {'label': '物理通道数与最小间距', 'status': 'pass',
         'detail': f"{o['txCount']} Tx / {o['rxCount']} Rx；同类阵元间距不小于 {o['minSpacingLambda']} λ，设计轴坐标在 {o['gridLambda']} λ 格点上。"},
        {'label': '设计轴物理孔径', 'status': 'pass' if s['physicalApertureLambda'] <= o['maxApertureLambda']+1e-9 else 'fail',
         'detail': f"{s['physicalApertureLambda']:.3f} λ / 上限 {o['maxApertureLambda']} λ。"},
        {'label': METHOD_NAMES[o['criterion']]+' 有限快拍验证' if is_super else '扫描端点 HPBW 约束' if o['criterion'] == 'hpbw' else '采样扫描角的双目标 3 dB 谷判据',
         'status': 'pass' if resolution_pass else 'fail', 'detail': detail},
        {'label': '常规阵列因子的全可见旁瓣 / 栅瓣' if is_super else '全可见角域旁瓣 / 栅瓣',
         'status': 'pass' if s['worstSidelobeDb'] <= o['maxSidelobeDb']+1e-6 else 'fail',
         'detail': f"最差 {s['worstSidelobeDb']:.2f} dB / 上限 {o['maxSidelobeDb']} dB；主瓣由最近局部极小值界定。"}]
    sites = _round(s['span']/o['gridLambda'])+1
    fill = s['uniqueCount']/sites
    warnings = []
    if status == 'unmet':
        warnings.append('本次有界搜索未找到同时满足所有指标的候选；请增加通道或搜索预算、放宽分辨角/旁瓣要求，或调整孔径和扫描范围。不能据此证明问题无解。')
    if s['uniqueCount'] < o['txCount']*o['rxCount']:
        warnings.append('部分 Tx–Rx 通道形成重合虚拟位置；通道数不等于独立空间采样数。')
    if fill >= 1-1e-9:
        warnings.append('此候选的虚拟格点连续占满；物理 Tx/Rx 可以稀疏，但虚拟阵列本身不是抽稀格点阵。')
    warnings.append('仅对所列正负离散扫描中心运行有限重复的含噪角度实验，不保证连续视场或实际硬件分辨率；真角仅用于事后评价，未用于挑峰。' if is_super else
                    '仅离散复核中心、两个中间扫描角和端点，负角由对称性等价；不保证连续扫描域，噪声、互耦及真实天线方向图可能改变分辨能力。')
    if is_super:
        warnings.extend(sr['warnings'])
        scan_checks = [{'centerDeg': p['centerDeg'], 'resolved': bool(p.get('valid') and p.get('successRate') is not None and p['successRate'] >= o['requiredSuccessRate']),
                        'successRate': p.get('successRate'), 'rmseDeg': p.get('rmseDeg'),
                        'worstErrorDeg': p.get('worstErrorDeg'), 'estimates': p.get('estimates') or [],
                        'algorithmStatus': p.get('status')} for p in sr['validation']]
    else:
        scan_checks = []
        for p in s['pairs']:
            u, half = math.sin(p['centerDeg']*RAD), s['halfU']
            width = (math.asin(u+half)-math.asin(u-half))/RAD if half is not None and abs(u+half) <= 1 and abs(u-half) <= 1 else None
            scan_checks.append({'centerDeg': p['centerDeg'], 'hpbwDeg': width,
                                'pairDipDb': p['dipDb'], 'resolved': p['resolved']})
    sources = list({source['url']: source for source in [*SOURCES, *((sr or {}).get('sources') or [])]}.values())
    out = {'valid': True, 'status': status, 'options': deepcopy(o), 'geometry': geometry,
           'metrics': {'channelCount': o['txCount']*o['rxCount'], 'uniqueCount': s['uniqueCount'],
                       'physicalApertureLambda': s['physicalApertureLambda'], 'virtualApertureLambda': s['span'],
                       'hpbwDeg': s['hpbwDeg'], 'worstHpbwDeg': s['worstHpbwDeg'],
                       'achievedResolutionDeg': None, 'targetResolutionPass': resolution_pass,
                       'worstPairDipDb': None if is_super else s['worstPairDipDb'],
                       'worstSidelobeDb': s['worstSidelobeDb'], 'sparseFillRatio': fill,
                       'firstMinimumDirectionCosine': s['firstMinimumU']},
           'checks': checks, 'warnings': warnings, 'assumptions': list(ASSUMPTIONS), 'sources': deepcopy(sources),
           'search': {**search, 'verificationSamples': s['sampleCount']+1}, 'pattern': pattern,
           'twoTargetPattern': (sr.get('representative') or {}).get('spectrum') or [] if is_super else s['pair0']['pattern'],
           'scanChecks': scan_checks, 'antennaSummary': {key: a[key] for key in ('beamwidthAzDeg', 'beamwidthElDeg')}}
    if is_super:
        out['superresolution'] = sr
    check_cancelled(cancelled)
    return out


def _invalid(error):
    return {'valid': False, 'status': 'invalid', 'geometry': None, 'metrics': None,
            'checks': [], 'warnings': [], 'errors': [str(error)], 'assumptions': list(ASSUMPTIONS),
            'sources': deepcopy(SOURCES), 'pattern': [], 'twoTargetPattern': []}


def design(input=None, *, workers=None, cancelled=None, progress=None):
    """Search physical positions; held-out noise never selects the geometry."""
    check_cancelled(cancelled)
    try:
        o = normalize(input)
    except (ValueError, TypeError, OverflowError) as exc:
        return _invalid(exc)
    is_super = o['criterion'] in SUPER_METHODS
    rng = _random(o['seed'])
    evaluations = 0
    def to_positions(slots):
        return [x*o['gridLambda'] for x in slots]
    base_tx = [i*o['rxCount']*.5 for i in range(o['txCount'])]
    base_rx = [i*.5 for i in range(o['rxCount'])]
    regular_seed = {'tx': _scaled(base_tx, 1, o), 'rx': _scaled(base_rx, 1, o)}
    seeds = []
    for factor in (.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3):
        tx, rx = _scaled(base_tx, factor, o), _scaled(base_rx, factor, o)
        if _legal(tx, o['txCount'], o) and _legal(rx, o['rxCount'], o):
            seeds.append({'tx': tx, 'rx': rx})
    seeds.append({key: [i*o['minStep'] for i in range(o[key+'Count'])] for key in ('tx', 'rx')})
    suggested = _clamp((1.45 if o['criterion'] == 'two_target' else .7 if is_super else 1)*.886/
                       (o['targetResolutionDeg']*RAD*math.cos(o['scanMaxDeg']*RAD)), 1, 2*o['maxApertureLambda'])
    for _ in range(18):
        cap = min(o['maxSlot'], _round(suggested/o['gridLambda']*(.3+.8*rng())))
        seeds.append({key: _random_array(o[key+'Count'], cap, o, rng) for key in ('tx', 'rx')})
    short_pool = []
    def geometry_key(candidate):
        return ','.join(f'{x:.8f}' for x in unique_positions(to_positions(candidate['tx']), to_positions(candidate['rx'])))
    def retain(candidate):
        if not is_super:
            return
        key = geometry_key(candidate)
        existing = next((i for i, item in enumerate(short_pool) if item['key'] == key), None)
        if existing is not None:
            if short_pool[existing]['candidate']['analysis']['loss'] <= candidate['analysis']['loss']:
                return
            short_pool.pop(existing)
        short_pool.append({'key': key, 'candidate': candidate})
        short_pool.sort(key=lambda item: item['candidate']['analysis']['loss'])
        del short_pool[12:]
    def assess(candidate):
        nonlocal evaluations
        check_cancelled(cancelled)
        evaluations += 1
        result = {**candidate, 'analysis': _score(to_positions(candidate['tx']), to_positions(candidate['rx']), o)}
        retain(result)
        return result
    best = regular = None
    for seed in seeds:
        candidate = assess(seed)
        if seed == regular_seed:
            regular = candidate
        if best is None or candidate['analysis']['loss'] < best['analysis']['loss']:
            best = candidate
    current = best
    for iteration in range(o['iterations']):
        check_cancelled(cancelled)
        if progress and iteration % 25 == 0:
            progress(f"几何搜索：{iteration}/{o['iterations']}")
        if iteration > 0 and iteration % 120 == 0:
            current = best if iteration % 240 == 0 else assess({key: _random_array(o[key+'Count'], _round((.25+.75*rng())*o['maxSlot']), o, rng) for key in ('tx', 'rx')})
        candidate = {key: current[key].copy() for key in ('tx', 'rx')}
        key = 'rx' if o['txCount'] == 1 else 'tx' if o['rxCount'] == 1 else 'tx' if rng() < .5 else 'rx'
        count, items = o[key+'Count'], candidate[key]
        j = math.floor(rng()*len(items))
        if rng() < .8:
            items[j] += _round((-1 if rng() < .5 else 1)*(1+math.floor(rng()*max(2, o['maxSlot']*.15))))
        else:
            items[j] = math.floor(rng()*(o['maxSlot']+1))
        items.sort()
        offset = items[0]
        candidate[key] = [v-offset for v in items]
        if not _legal(candidate[key], count, o):
            continue
        evaluated = assess(candidate)
        delta = evaluated['analysis']['loss']-current['analysis']['loss']
        temperature = .05+.8*(1-iteration/o['iterations'])
        if delta <= 0 or rng() < math.exp(-delta/temperature):
            current = evaluated
        if evaluated['analysis']['loss'] < best['analysis']['loss']:
            best = evaluated
    search = {'iterations': o['iterations'], 'seed': o['seed'], 'evaluations': evaluations,
              'algorithm': 'manifold-conditioned geometry search + selected-algorithm screening + independent held-out validation' if is_super else 'bounded seeded coordinate annealing',
              'globalOptimality': False}
    if not is_super:
        return _report(to_positions(best['tx']), to_positions(best['rx']), o, search,
                       workers=workers, cancelled=cancelled, progress=progress)
    chosen = [x['candidate'] for x in short_pool[:3]]
    keys = {geometry_key(c) for c in chosen}
    if regular is not None and geometry_key(regular) not in keys:
        chosen.append(regular)
        keys.add(geometry_key(regular))
    for entry in short_pool:
        if len(chosen) >= 4:
            break
        if entry['key'] not in keys:
            chosen.append(entry['candidate'])
            keys.add(entry['key'])
    selected, screening = None, []
    for index, candidate in enumerate(chosen):
        check_cancelled(cancelled)
        if progress:
            progress(f'算法筛选候选：{index+1}/{len(chosen)}')
        tx, rx = to_positions(candidate['tx']), to_positions(candidate['rx'])
        physical = _shape(tx, rx, o, True)
        experiment = _algorithm_experiment(tx, rx, o, 'screening', workers=workers, cancelled=cancelled, progress=progress)
        penalty = 1000+100*(physical['worstSidelobeDb']-o['maxSidelobeDb']) if physical['worstSidelobeDb'] > o['maxSidelobeDb']+1e-6 else 0
        error_rank = experiment['rmseDeg']/max(.02, experiment['effectiveToleranceDeg']) if _finite(experiment['rmseDeg']) else 100
        rank = penalty+(100*(1-experiment['successRate'])+min(100, error_rank) if experiment['valid'] else 1e6)+candidate['analysis']['loss']*.001
        screening.append({'index': index, 'surrogateLoss': candidate['analysis']['loss'],
                          'worstSidelobeDb': physical['worstSidelobeDb'], 'algorithmStatus': experiment['status'],
                          'successRate': experiment['successRate'], 'rmseDeg': experiment['rmseDeg'],
                          'seeds': [v['seed'] for v in experiment['validation']]})
        if selected is None or rank < selected['rank']:
            selected = {'candidate': candidate, 'index': index, 'rank': rank}
    best = selected['candidate']
    validation = _algorithm_experiment(to_positions(best['tx']), to_positions(best['rx']), o, 'validation', workers=workers, cancelled=cancelled, progress=progress)
    search.update(shortlistCount=len(chosen), selectedScreeningIndex=selected['index'], screening=screening,
                  validationSeeds=[v['seed'] for v in validation['validation']], holdoutUsedForSelection=False)
    return _report(to_positions(best['tx']), to_positions(best['rx']), o, search, validation,
                   workers=workers, cancelled=cancelled, progress=progress)


def evaluate(tx, rx, input=None, *, workers=None, cancelled=None, progress=None):
    """Check a given physical layout against all selected design constraints."""
    check_cancelled(cancelled)
    try:
        if not isinstance(tx, (list, tuple)) or not isinstance(rx, (list, tuple)) or any(not _finite(x) for x in [*tx, *rx]):
            raise ValueError('评估坐标须为设计轴的一维有限数值数组。')
        o = normalize({**(input or {}), 'txCount': len(tx), 'rxCount': len(rx)})
        tx, rx = ([v-min(values) for v in sorted(values)] for values in (tx, rx))
        def to_slots(values):
            slots = [x/o['gridLambda'] for x in values]
            return [_round(x) if abs(x-_round(x)) < 1e-7 else x for x in slots]
        if not _legal(to_slots(tx), o['txCount'], o) or not _legal(to_slots(rx), o['rxCount'], o):
            raise ValueError('坐标不满足格点、孔径或最小间距约束。')
    except (ValueError, TypeError, OverflowError) as exc:
        return _invalid(exc)
    return _report(tx, rx, o, {'iterations': 0, 'seed': o['seed'], 'evaluations': 1,
                               'algorithm': 'given geometry evaluation', 'globalOptimality': False},
                   workers=workers, cancelled=cancelled, progress=progress)
