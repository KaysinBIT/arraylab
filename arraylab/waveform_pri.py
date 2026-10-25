"""WaveLab v1.2 dual-period delay-residue experiment, independent of Qt.

The inverse solver receives measured residues, never the simulated target truth.
This is a delay-observation model, not a raw FMCW echo association processor.
"""
from copy import deepcopy
import math
from numbers import Real
import sys

C = 299792458
METRES_PER_NS = C / 2e9
MAX_SEARCH_M = 1e7
MAX_ENUMERATION = 10000
DEFAULTS = {'method': 'dual_period', 'period1Us': 38, 'period2Us': 43,
            'searchMaxM': 20000, 'timingResolutionNs': 2, 'targetRangeM': 8000}
SOURCES = [
    {'title': 'MathWorks · range2time / 双程传播与无模糊距离', 'url': 'https://www.mathworks.com/help/phased/ref/range2time.html'},
    {'title': 'MathWorks · Radar Range and Range Rate Ambiguities', 'url': 'https://www.mathworks.com/help/fusion/ug/track-objects-with-wrapping-azimuth-angles-and-ambiguous-range-and-range-rate-measurements.html'},
    {'title': 'MathWorks · PRF Agility Based on Target Detection', 'url': 'https://www.mathworks.com/help/radar/ug/prf-agility-based-on-target-detection.html'},
]
ASSUMPTIONS = [
    '独立的周期内时延余数模型：已正确分离Tx，并已获得每种重复周期内的时延观测。不是FMCW跨chirp原始回波、ADC或匹配滤波处理链。',
    '两次观测来自同一静止单点目标；没有移动目标补偿、多目标关联、遮挡、接收盲区或检测概率模型。',
    'T1、T2分别表示两次采用均匀重复周期的观测；不模拟单个序列内交替PRI的真实回波到达与发射时刻关联。',
    '观测为τ mod T，τ=2R/c；按timingResolutionNs四舍五入后重新取模，求解保留半个量化步长的区间不确定度。',
    '候选只由输入的时延观测、周期、误差容限与搜索上限决定；求解器不读取目标真距离。',
    '只报告给定搜索区间内的唯一性；不根据小数周期推断无限范围CRT周期，也不宣称几千米示例是车载雷达探测能力。',
    '重复周期的含义必须与时延观测一致；TDM同Tx周期与相邻物理chirp周期不可混用。此实验不修改主波形的帧预算。',
]


def _finite(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def _period_error(value):
    return not _finite(value) or not .1 <= value <= 100000


def wrap(value, period):
    """Stable nonnegative residue, including a machine-precision origin boundary."""
    remainder = ((value % period) + period) % period
    epsilon = 8 * sys.float_info.epsilon * max(1, period, abs(value))
    return 0.0 if remainder < epsilon or period - remainder < epsilon else remainder


def _output(status, reason, **extra):
    return {'status': status, 'rangeM': None, 'resolvedRangeM': None,
            'candidates': [], 'measurements': [], 'reason': reason, 'warnings': [],
            'assumptions': ASSUMPTIONS.copy(), 'sources': deepcopy(SOURCES), **extra}


def _normalize_measurements(measurements):
    if not isinstance(measurements, (list, tuple)) or len(measurements) != 2:
        raise ValueError('需要两组周期内时延观测。')
    result = []
    for index, item in enumerate(measurements, 1):
        if not isinstance(item, dict):
            raise ValueError(f'第{index}组观测格式无效。')
        period = item.get('periodUs')
        delay = item.get('delayUs', item.get('foldedDelayUs'))
        tolerance = item.get('toleranceNs')
        if _period_error(period):
            raise ValueError('重复周期须在0.1–100000 μs之间。')
        if not _finite(delay) or not 0 <= delay < period:
            raise ValueError('周期内时延须满足0 ≤ delayUs < periodUs。')
        if not _finite(tolerance) or not 0 <= tolerance < period * 1000 / 10:
            raise ValueError('时延误差容限须非负且小于对应重复周期的1/10。')
        result.append({'periodUs': float(period), 'delayUs': float(delay),
                       'foldedDelayUs': float(delay), 'toleranceNs': float(tolerance),
                       'apparentRangeM': delay * 1000 * METRES_PER_NS,
                       'unambiguousRangeM': period * 1000 * METRES_PER_NS})
    return result


def _interval_plan(measurement, maximum):
    period, delay, tolerance = measurement['periodUs'] * 1000, measurement['delayUs'] * 1000, measurement['toleranceNs']
    minimum_order = math.ceil((-tolerance - delay) / period)
    maximum_order = math.floor((maximum + tolerance - delay) / period)
    return {'period': period, 'delay': delay, 'tolerance': tolerance,
            'minimumOrder': minimum_order, 'maximumOrder': maximum_order,
            'count': max(0, maximum_order - minimum_order + 1)}


def _intervals(plan, maximum):
    result = []
    for order in range(plan['minimumOrder'], plan['maximumOrder'] + 1):
        center = plan['delay'] + order * plan['period']
        low, high = max(0, center - plan['tolerance']), min(maximum, center + plan['tolerance'])
        if low <= high:
            result.append({'low': low, 'high': high, 'center': center, 'order': order})
    return result


def _fitted_delay(first, second, tolerance1, tolerance2, low, high):
    if tolerance1 == 0:
        estimate = first
    elif tolerance2 == 0:
        estimate = second
    else:
        scale = max(tolerance1, tolerance2)
        w1, w2 = (tolerance2 / scale) ** 2, (tolerance1 / scale) ** 2
        estimate = (w1 * first + w2 * second) / (w1 + w2)
    return min(high, max(low, estimate))


def resolve(measurements, search_max_m=None):
    """Resolve two delay observations within a bounded search range.

Accepts either ``resolve([measurement, measurement], search_max_m)`` or
``resolve({'measurements': [...], 'searchMaxM': ...})``. Extra diagnostic truth
keys are ignored: they cannot select an alias candidate.
"""
    if isinstance(measurements, dict):
        search_max_m = measurements.get('searchMaxM')
        measurements = measurements.get('measurements')
    try:
        measurements = _normalize_measurements(measurements)
        if not _finite(search_max_m) or not 0 < search_max_m <= MAX_SEARCH_M:
            raise ValueError('搜索上限须大于0且不超过10000000 m。')
    except (ValueError, TypeError, OverflowError) as exc:
        return _output('blocked', str(exc))
    maximum = search_max_m / METRES_PER_NS
    plans = [_interval_plan(item, maximum) for item in measurements]
    enumeration_count = sum(item['count'] for item in plans)
    if enumeration_count > MAX_ENUMERATION:
        return _output('blocked', '候选枚举超过10000个区间；请缩小搜索范围或增大重复周期。',
                       measurements=measurements, searchMaxM=search_max_m, enumerationCount=enumeration_count)
    a, b = [_intervals(plan, maximum) for plan in plans]
    epsilon = 16 * sys.float_info.epsilon * max(1, maximum, *(plan['period'] for plan in plans))
    intersections, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        first, second = a[i], b[j]
        low, high = max(first['low'], second['low']), min(first['high'], second['high'])
        if low <= high + epsilon:
            safe_high = max(low, high)
            estimate = _fitted_delay(first['center'], second['center'], plans[0]['tolerance'], plans[1]['tolerance'], low, safe_high)
            intersections.append({'low': low, 'high': safe_high, 'estimate': estimate,
                                  'order1': first['order'], 'order2': second['order'],
                                  'residualNs': abs(first['center'] - second['center'])})
        if first['high'] < second['high']:
            i += 1
        elif second['high'] < first['high']:
            j += 1
        else:
            i += 1; j += 1
    groups = []
    for item in intersections:
        previous = groups[-1] if groups else None
        if previous is not None and item['low'] <= previous['high'] + epsilon:
            previous['high'] = max(previous['high'], item['high'])
            if item['residualNs'] < previous['residualNs']:
                for key in ('estimate', 'order1', 'order2', 'residualNs'):
                    previous[key] = item[key]
        else:
            groups.append(item.copy())
    candidates = [{'rangeM': row['estimate'] * METRES_PER_NS,
                   'minM': row['low'] * METRES_PER_NS, 'maxM': row['high'] * METRES_PER_NS,
                   'order1': row['order1'], 'order2': row['order2'], 'residualNs': row['residualNs']} for row in groups]
    warnings = []
    if abs(plans[0]['period'] - plans[1]['period']) <= epsilon:
        warnings.append('两个重复周期相同，没有新增周期多样性；唯一性只能来自有限搜索先验。')
    if any(row['minM'] <= epsilon * METRES_PER_NS or row['maxM'] >= search_max_m - epsilon * METRES_PER_NS for row in candidates):
        warnings.append('候选区间触及搜索边界；显示的是被搜索先验截断后的区间。')
    status = 'inconsistent' if not candidates else 'resolved' if len(candidates) == 1 else 'ambiguous'
    reason = (f'在0–{search_max_m:g} m搜索范围内，两组时延观测只有一个连通的一致候选区间。' if status == 'resolved' else
              f'搜索范围内仍有{len(candidates)}个分离候选区间；不选取其中任何一个作为真距离。' if status == 'ambiguous' else
              '搜索范围内没有同时满足两组时延误差容限的候选；请检查观测、容限与搜索先验。')
    unique_range = candidates[0]['rangeM'] if len(candidates) == 1 else None
    return _output(status, reason, rangeM=unique_range, resolvedRangeM=unique_range,
                   candidates=candidates, measurements=measurements, warnings=warnings,
                   searchMaxM=search_max_m, enumerationCount=enumeration_count)


def simulate(input=None):
    """Generate quantized delay residues, then pass observations to resolve."""
    if input is not None and not isinstance(input, dict):
        return _output('blocked', '跨周期实验参数须为对象。')
    options = {**DEFAULTS, **(input or {})}
    if options['method'] == 'staggered_pri':
        options['method'] = 'dual_period'
    if options['method'] not in ('off', 'dual_period'):
        return _output('blocked', '跨周期算法只支持off或dual_period。')
    if _period_error(options['period1Us']) or _period_error(options['period2Us']):
        return _output('blocked', '重复周期须在0.1–100000 μs之间。')
    minimum = min(options['period1Us'], options['period2Us']) * 1000
    resolution = options['timingResolutionNs']
    if not _finite(resolution) or not 1e-6 <= resolution < minimum / 10:
        return _output('blocked', '时延量化步长须至少为0.000001 ns，且小于较短重复周期的1/10。')
    if not _finite(options['targetRangeM']) or not 0 <= options['targetRangeM'] <= MAX_SEARCH_M:
        return _output('blocked', '实验目标距离须在0–10000000 m之间。')
    if not _finite(options['searchMaxM']) or not 0 < options['searchMaxM'] <= MAX_SEARCH_M:
        return _output('blocked', '搜索上限须大于0且不超过10000000 m。')
    delay = options['targetRangeM'] / METRES_PER_NS
    measurements = []
    for period_us in (options['period1Us'], options['period2Us']):
        period_ns = period_us * 1000
        residue = wrap(delay, period_ns)
        # JS Math.round for nonnegative residues uses half-up, unlike Python round.
        quantized = math.floor(residue / resolution + .5) * resolution
        measurements.append({'periodUs': period_us, 'delayUs': wrap(quantized, period_ns) / 1000,
                             'toleranceNs': resolution / 2})
    if options['method'] == 'off':
        return _output('off', '未开启跨周期解模糊；这里只展示两组周期内时延与表观距离。',
                       measurements=_normalize_measurements(measurements), searchMaxM=options['searchMaxM'], method='off')
    result = resolve(measurements, options['searchMaxM'])
    result['method'] = options['method']
    return result
