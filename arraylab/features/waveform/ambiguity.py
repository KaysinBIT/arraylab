"""WaveLab v1.2 real-ADC frequency-folding experiment, without Qt.

Simulation creates observations; resolve only uses the observed alias frequency,
its acquisition parameters, external Doppler prior and analog passband limits.
"""
from copy import deepcopy
import math
from numbers import Real
import sys

import numpy as np

C = 299792458
TAU = 2 * math.pi
MAX_BRANCHES = 20000
DEFAULTS = {'method': 'off', 'secondSampleRateMsps': 18, 'secondSlopeMHzUs': 10,
            'searchMaxM': 600, 'knownVelocityMps': 0, 'toleranceBins': .75}
SOURCES = [
    {'title': 'Longman & Bilik (2021) · Maximal Range Extension in LFM-CW Radars using Multi-Rate Sampling',
     'url': 'https://cris.bgu.ac.il/en/publications/maximal-range-extension-in-lfm-cw-radars-using-multi-rate-samplin/',
     'doi': '10.1109/IEEECONF53345.2021.9723111'},
    {'title': 'Analog Devices · Types of ADCs and DACs / Undersampling', 'url': 'https://www.analog.com/en/resources/technical-articles/types-of-adcs-and-dacs.html'},
    {'title': 'Analog Devices · What is an antialiasing filter?', 'url': 'https://www.analog.com/en/resources/faqs/FAQ_antialiasing_filter.html'},
]
ASSUMPTIONS = [
    '本实验针对实ADC采样造成的拍频镜像折叠；不处理跨chirp/脉冲周期的距离模糊，也不是DDM速度解模糊器。',
    '两次观测为同一个无噪声单目标，使用实际实数余弦样本、窗函数和FFT估计折叠拍频，不用真实距离选择整数模糊阶数。',
    '解算器仅使用观测频率、采样率、斜率、外部已知速度补偿、模拟IF通带先验及搜索上限；真实距离只用于生成实验信号。所有候选都必须落在两组模拟通带内。',
    '实验假定模拟IF和ADC模拟输入带宽允许所选带通欠采样。模拟滤波器已去除的回波不能靠解模糊恢复。主波形页的保守实采样规划不因此改变。',
    '外部速度先验错误会产生距离偏差、无一致解或错误一致解；本实验不独立测量目标速度。两次观测假定目标距离不变。',
    'FFT频率容差是本实验的匹配门限，并非统计置信区间；多候选会全部保留。扩大搜索区间或容差可能增加模糊解。',
    'TDM/DDM波形都可使用两组fast-time参数做本实验，但这里假定发射通道已隔离且同一目标已正确关联，不验证多Tx解码或多目标配对。',
]


def _finite(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except (TypeError, ValueError, OverflowError):
        return False


def folded(frequency, sample_rate):
    """Absolute real-ADC alias frequency in [0, fs/2]."""
    if not _finite(frequency) or not _finite(sample_rate) or sample_rate <= 0:
        raise ValueError('折叠频率需有限频率与正采样率。')
    shifted = frequency + sample_rate / 2
    if not math.isfinite(shifted):
        remainder = frequency % sample_rate
        return min(remainder, sample_rate - remainder)
    remainder = ((shifted % sample_rate) + sample_rate) % sample_rate
    return abs(remainder - sample_rate / 2)


def _output(status, reason, **extra):
    return {'status': status, 'reason': reason, 'rangeM': None, 'resolvedRangeM': None,
            'candidates': [], 'measurements': [], 'errors': [reason] if status == 'blocked' else [],
            'warnings': [], 'assumptions': ASSUMPTIONS.copy(), 'sources': deepcopy(SOURCES), **extra}


def _scaled(item, direct, scaled, multiplier):
    value = item.get(direct)
    if value is not None:
        return value
    value = item.get(scaled)
    return float(value) * multiplier if _finite(value) else None


def _normalize_measurement(item):
    if not isinstance(item, dict):
        raise ValueError('折叠拍频观测须为对象。')
    fs = _scaled(item, 'sampleRateHz', 'sampleRateMsps', 1e6)
    slope = _scaled(item, 'slopeHzPerSec', 'slopeMHzUs', 1e12)
    observation = _scaled(item, 'foldedBeatHz', 'foldedBeatMHz', 1e6)
    doppler = item.get('knownDopplerHz') if item.get('knownDopplerHz') is not None else 0
    uncertainty = item.get('uncertaintyHz') if item.get('uncertaintyHz') is not None else item.get('measurementErrorHz')
    if (not all(_finite(v) for v in (fs, slope, observation, doppler, uncertainty))
            or fs <= 0 or slope <= 0 or not 0 <= observation <= fs/2 + fs*1e-12
            or not 0 <= uncertainty < fs/2):
        raise ValueError('观测必须包含有效采样率、正斜率、[0,fs/2]折叠拍频及小于fs/2的非负频率容差。')
    bandwidth = _scaled(item, 'analogBandwidthHz', 'analogBandwidthMHz', 1e6)
    if 'analogBandwidthHz' not in item and 'analogBandwidthMHz' not in item:
        bandwidth = math.inf
    if (not isinstance(bandwidth, Real) or isinstance(bandwidth, bool)
            or math.isnan(bandwidth) or bandwidth <= 0):
        raise ValueError('模拟IF通带先验必须为正数；省略时不施加通带先验。')
    alpha = 2 * slope / C
    if not _finite(alpha) or alpha <= 0:
        raise ValueError('斜率数值超出可计算范围。')
    return {'fs': float(fs), 'slope': float(slope), 'obs': min(float(observation), fs/2),
            'doppler': float(doppler), 'uncertainty': float(uncertainty),
            'analogBandwidth': float(bandwidth), 'alpha': alpha}


def _branches(m, maximum):
    domain_min, domain_max = 0., maximum
    if math.isfinite(m['analogBandwidth']):
        lower = (-m['analogBandwidth'] - m['doppler']) / m['alpha']
        upper = (m['analogBandwidth'] - m['doppler']) / m['alpha']
        domain_min = max(domain_min, lower + 32 * sys.float_info.epsilon * max(1, abs(lower)))
        domain_max = min(domain_max, upper - 32 * sys.float_info.epsilon * max(1, abs(upper)))
    if domain_max < domain_min:
        return []
    low, high = m['alpha'] * domain_min + m['doppler'], m['alpha'] * domain_max + m['doppler']
    lower = math.ceil((min(low, high) - m['obs'] - m['uncertainty']) / m['fs']) - 1
    upper = math.floor((max(low, high) + m['obs'] + m['uncertainty']) / m['fs']) + 1
    if abs(lower) > 2**53-1 or abs(upper) > 2**53-1 or upper-lower > MAX_BRANCHES/2:
        raise ValueError('当前搜索范围产生过多整数模糊分支，请缩小搜索距离或调整采样率/斜率。')
    result = []
    for order in range(lower, upper + 1):
        for sign in (-1, 1):
            center = (order * m['fs'] + sign * m['obs'] - m['doppler']) / m['alpha']
            radius = m['uncertainty'] / m['alpha']
            minimum, maximum_range = max(domain_min, center-radius), min(domain_max, center+radius)
            if minimum <= maximum_range:
                result.append({'minRangeM': minimum, 'maxRangeM': maximum_range,
                               'center': center, 'sign': sign, 'order': order})
    return sorted(result, key=lambda row: (row['minRangeM'], row['maxRangeM']))


def _candidate(a, b, m1, m2, minimum, maximum):
    r1, r2 = max(m1['uncertainty']/m1['alpha'], 1e-10), max(m2['uncertainty']/m2['alpha'], 1e-10)
    scale = min(r1, r2); w1, w2 = (scale/r1)**2, (scale/r2)**2
    distance = min(maximum, max(minimum, (a['center']*w1+b['center']*w2)/(w1+w2)))
    residuals = [abs(folded(m['alpha']*distance + m['doppler'], m['fs'])-m['obs']) for m in (m1, m2)]
    branch = {'foldOrder1': a['order'], 'foldOrder2': b['order'], 'sign1': a['sign'], 'sign2': b['sign']}
    return {'rangeM': distance, 'minRangeM': minimum, 'maxRangeM': maximum,
            'residualHz': max(residuals), 'residualsHz': residuals, **branch, 'branches': [branch]}


def resolve(measurements, options=None):
    """Inverse real-ADC alias solver; target truth is neither needed nor read."""
    if options is not None and not isinstance(options, dict):
        return _output('blocked', '搜索参数须为对象。')
    maximum = (options or {}).get('searchMaxM', DEFAULTS['searchMaxM'])
    if not _finite(maximum) or not 0 < maximum <= 10000:
        return _output('blocked', '搜索距离上限必须在(0,10000] m内。')
    if not isinstance(measurements, (list, tuple)) or len(measurements) != 2:
        return _output('blocked', '需要两组独立的折叠拍频观测。')
    try:
        m1, m2 = map(_normalize_measurement, measurements)
        if (abs(m1['alpha']/m1['fs'] - m2['alpha']/m2['fs']) < 1e-12 * max(1, m1['alpha']/m1['fs'], m2['alpha']/m2['fs'])
                and abs(m1['doppler']/m1['fs'] - m2['doppler']/m2['fs']) < 1e-12):
            return _output('blocked', '两组归一化观测等价，没有新增解模糊信息。')
        list1, list2 = _branches(m1, maximum), _branches(m2, maximum)
        matches, begin = [], 0
        for a in list1:
            while begin < len(list2) and list2[begin]['maxRangeM'] < a['minRangeM'] - 1e-9:
                begin += 1
            j = begin
            while j < len(list2) and list2[j]['minRangeM'] <= a['maxRangeM'] + 1e-9:
                b = list2[j]; low, high = max(a['minRangeM'], b['minRangeM']), min(a['maxRangeM'], b['maxRangeM'])
                if low <= high:
                    matches.append(_candidate(a, b, m1, m2, low, high))
                j += 1
    except (ValueError, TypeError, ArithmeticError) as exc:
        return _output('blocked', str(exc) or '观测数值超出可计算范围。')
    matches.sort(key=lambda row: (row['minRangeM'], row['maxRangeM']))
    candidates = []
    for current in matches:
        previous = candidates[-1] if candidates else None
        if previous is not None and current['minRangeM'] <= previous['maxRangeM'] + 1e-9:
            previous['minRangeM'] = min(previous['minRangeM'], current['minRangeM'])
            previous['maxRangeM'] = max(previous['maxRangeM'], current['maxRangeM'])
            previous['branches'].extend(current['branches'])
            if current['residualHz'] < previous['residualHz']:
                for key in ('rangeM', 'residualHz', 'residualsHz', 'foldOrder1', 'foldOrder2', 'sign1', 'sign2'):
                    previous[key] = current[key]
        else:
            candidates.append(current)
    status = 'inconsistent' if not candidates else 'resolved' if len(candidates) == 1 else 'ambiguous'
    reason = ('在给定搜索区间和频率容差内，找到一个连通的一致候选区间。' if status == 'resolved' else
              f'找到{len(candidates)}个分离的一致候选区间，不能唯一解距离模糊。' if status == 'ambiguous' else
              '两组观测在当前容差和搜索范围内没有一致解。')
    unique = candidates[0]['rangeM'] if status == 'resolved' else None
    return _output(status, reason, candidates=candidates, rangeM=unique, resolvedRangeM=unique,
                   searchMaxM=maximum, branchCounts=[len(list1), len(list2)])


def _observe(profile, target_range, target_velocity, known_velocity, tolerance_bins):
    fs, slope = profile['sampleRateMsps'] * 1e6, profile['slopeMHzUs'] * 1e12
    center = profile['startGHz'] + profile['slopeMHzUs'] * (profile['adcStartUs'] + profile['samples']/profile['sampleRateMsps']/2)/1000
    wavelength = C / (center * 1e9)
    true_beat = 2*slope*target_range/C + 2*target_velocity/wavelength
    count, length = int(profile['samples']), int(profile['rangeFft'])
    weights = np.hanning(count) if profile['window'] == 'hann' else np.ones(count)
    samples = weights * np.cos(TAU * true_beat * np.arange(count) / fs)
    magnitude = np.abs(np.fft.rfft(samples, n=length))
    half = length//2; peak = int(np.argmax(magnitude)); offset = 0.
    if 0 < peak < half:
        left, middle, right = np.log(np.maximum(magnitude[peak-1:peak+2], 1e-30))
        denominator = left - 2*middle + right
        if abs(denominator) > 1e-20:
            offset = float(np.clip(.5*(left-right)/denominator, -.5, .5))
    bin_hz = fs / length; observed = (peak+offset)*bin_hz
    uncertainty = tolerance_bins * bin_hz
    edge = min(observed, fs/2-observed); native_bin = fs/count
    edge_limited = edge < 2*native_bin
    if edge_limited:
        uncertainty = max(uncertainty, 2*native_bin)
    uncertainty = min(uncertainty, fs*.49)
    doppler = 2*known_velocity/wavelength
    scales = np.full(len(magnitude), 2.); scales[[0, -1]] = 1.
    spectrum = 20*np.log10(np.maximum(1e-9, scales*magnitude/weights.sum()))
    return {'sampleRateMsps': profile['sampleRateMsps'], 'slopeMHzUs': profile['slopeMHzUs'],
            'foldedBeatMHz': observed/1e6, 'apparentRangeM': C*max(0, observed-doppler)/(2*slope),
            'fftBinHz': bin_hz, 'uncertaintyHz': uncertainty, 'measurementErrorHz': uncertainty,
            'knownDopplerHz': doppler, 'centerGHz': center, 'nyquistMHz': profile['sampleRateMsps']/2,
            'observedSpectrum': [{'x': k*bin_hz/1e6, 'y': float(value)} for k, value in enumerate(spectrum)],
            'trueBeatMHz': true_beat/1e6, 'edgeLimited': edge_limited, 'adcUs': count/profile['sampleRateMsps']}


def simulate(input, options=None):
    """Synthesize two bounded real-ADC observations, then invert measured aliases."""
    if options is not None and not isinstance(options, dict):
        return _output('blocked', '解模糊参数须为对象。')
    config = {key: (options or {}).get(key, default) for key, default in DEFAULTS.items()}
    if config['method'] == 'off':
        # Off performs no experiment, including no access to target truth.
        return _output('off', '距离解模糊实验已关闭。', options={k: (v if _finite(v) or isinstance(v, str) else None) for k, v in config.items()})
    if config['method'] not in ('dual_rate', 'dual_slope'):
        return _output('blocked', '不支持的距离解模糊算法。')
    if any(not _finite(config[key]) for key in DEFAULTS if key != 'method'):
        return _output('blocked', '解模糊参数必须为有限数值。')
    if not 0 < config['searchMaxM'] <= 10000 or not .05 <= config['toleranceBins'] <= 4 or abs(config['knownVelocityMps']) > 1000:
        return _output('blocked', '搜索上限须为(0,10000] m，容差须为0.05–4个FFT栅格，已知速度幅值不超过1000 m/s。', options=config)
    from arraylab.features.waveform.core import calculate
    validated = calculate(input)
    if not validated['valid']:
        return _output('blocked', '请先修正主波形参数：' + ' '.join(validated['errors']), options=config)
    p = validated['params']; second = p.copy()
    if config['method'] == 'dual_rate':
        second['sampleRateMsps'] = config['secondSampleRateMsps']
    else:
        second['slopeMHzUs'] = config['secondSlopeMHzUs']
    if any(not .1 <= q['sampleRateMsps'] <= 200 or not .01 <= q['slopeMHzUs'] <= 1000 for q in (p, second)):
        return _output('blocked', '本实验支持0.1–200 MSps与0.01–1000 MHz/μs；请收紧参数范围。', options=config)
    if abs(p['sampleRateMsps']-second['sampleRateMsps']) < 1e-10 and abs(p['slopeMHzUs']-second['slopeMHzUs']) < 1e-10:
        return _output('blocked', '两组参数相同，第二次观测没有提供独立解模糊信息。', options=config)
    if second['adcStartUs'] + second['samples']/second['sampleRateMsps'] > second['rampUs'] + 1e-9:
        return _output('blocked', '第二组ADC采样窗超出ramp；请提高第二采样率、减少采样点数或延长ramp。', options=config)
    if any(q['startGHz']+q['slopeMHzUs']*q['rampUs']/1000 > 81 or q['startGHz'] < 76 for q in (p, second)):
        return _output('blocked', '至少一组扫频超出本工具的76–81 GHz规划范围。', options=config)
    measurements = []
    for q in (p, second):
        measurement = _observe(q, p['targetRange'], p['targetVelocity'], config['knownVelocityMps'], config['toleranceBins'])
        measurement.update(analogBandwidthMHz=p['ifBandwidthMHz'], observable=abs(measurement['trueBeatMHz']) < p['ifBandwidthMHz'],
                           spectrumLabel='理想模拟IF滤波前的实采样频谱')
        measurements.append(measurement)
    warnings = []
    if any(row['edgeLimited'] for row in measurements):
        warnings.append('观测接近DC/Nyquist，正负频率主瓣可能相互影响；已扩大匹配门限，不能把插值值当成高精度测量。')
    if any(abs(row['trueBeatMHz']) > row['nyquistMHz'] for row in measurements):
        warnings.append('至少一组回波高于Nyquist；本实验要求宽模拟IF/ADC输入带宽，不能使用会滤除该回波的抗混叠滤波设置。')
    if any(not row['observable'] for row in measurements):
        return _output('blocked', '至少一组真实拍频超出配置的模拟IF带宽；该回波不可观测，不能报告解模糊结果。',
                       options=config, measurements=measurements, warnings=warnings)
    keys = ('sampleRateMsps', 'slopeMHzUs', 'foldedBeatMHz', 'knownDopplerHz', 'uncertaintyHz', 'analogBandwidthMHz')
    observed = [{key: row[key] for key in keys} for row in measurements]
    solved = resolve(observed, {'searchMaxM': config['searchMaxM']})
    return {**solved, 'options': config, 'measurements': measurements, 'warnings': warnings + solved['warnings']}
