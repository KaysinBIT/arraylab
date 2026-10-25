"""Pure numerical FMCW/TDM/DDM model ported from the frozen WaveLab v1.2.

Units and JSON field names follow the source model. No hardware control,
DDM blind decoding, antenna-gain model, GUI, or JavaScript runtime is used.
"""
from copy import deepcopy
import math
from numbers import Real

import numpy as np

C = 299792458.0
BASE = {'startGHz': 77,
 'slopeMHzUs': 12,
 'rampUs': 34,
 'idleUs': 4,
 'adcStartUs': 3,
 'samples': 512,
 'sampleRateMsps': 20,
 'loops': 128,
 'txCount': 3,
 'rxCount': 4,
 'frameMs': 50,
 'ifBandwidthMHz': 10,
 'rangeFft': 1024,
 'dopplerFft': 128,
 'window': 'hann',
 'mode': 'tdm',
 'ddmSlots': 4,
 'targetRange': 60,
 'targetVelocity': 5}

PRESETS = {'lrr': {'name': '远距巡航',
         'description': '单发射通道、短周期：优先保障距离覆盖与无模糊速度。',
         'params': {'startGHz': 76.5,
                    'slopeMHzUs': 8,
                    'rampUs': 18,
                    'idleUs': 3,
                    'adcStartUs': 2,
                    'samples': 384,
                    'sampleRateMsps': 24,
                    'loops': 256,
                    'txCount': 1,
                    'rxCount': 4,
                    'frameMs': 50,
                    'ifBandwidthMHz': 12,
                    'rangeFft': 512,
                    'dopplerFft': 256,
                    'window': 'hann',
                    'mode': 'tdm',
                    'ddmSlots': 4,
                    'targetRange': 150,
                    'targetVelocity': 35}},
 'mrr': {'name': '中距感知',
         'description': '3 Tx × 4 Rx TDM：展示多天线通道组合，以及轮流发射带来的无模糊速度收窄。',
         'params': {'startGHz': 77,
                    'slopeMHzUs': 12,
                    'rampUs': 34,
                    'idleUs': 4,
                    'adcStartUs': 3,
                    'samples': 512,
                    'sampleRateMsps': 20,
                    'loops': 128,
                    'txCount': 3,
                    'rxCount': 4,
                    'frameMs': 50,
                    'ifBandwidthMHz': 10,
                    'rangeFft': 1024,
                    'dopplerFft': 128,
                    'window': 'hann',
                    'mode': 'tdm',
                    'ddmSlots': 4,
                    'targetRange': 60,
                    'targetVelocity': 5}},
 'srr': {'name': '近距泊车',
         'description': '宽带与三发射 TDM：提升名义距离分辨，接受测速范围收窄。',
         'params': {'startGHz': 77,
                    'slopeMHzUs': 75,
                    'rampUs': 48,
                    'idleUs': 6,
                    'adcStartUs': 3,
                    'samples': 768,
                    'sampleRateMsps': 20,
                    'loops': 64,
                    'txCount': 3,
                    'rxCount': 4,
                    'frameMs': 50,
                    'ifBandwidthMHz': 10,
                    'rangeFft': 1024,
                    'dopplerFft': 128,
                    'window': 'hann',
                    'mode': 'tdm',
                    'ddmSlots': 4,
                    'targetRange': 8,
                    'targetVelocity': -2.5}}}

SOURCES = [{'title': 'TI · Programming Chirp Parameters in TI Radar Devices (SWRA553)',
  'url': 'https://www.ti.com/lit/pdf/SWRA553'},
 {'title': 'TI Radar Academy · Terminology',
  'url': 'https://dev.ti.com/tirex/explore/content/radar_academy_2_10_00_2/_build_radar_academy_2_10_00_2/source/introduction/terminology.html'},
 {'title': 'TI · DDMA MIMO design guide (§2.4.7–2.4.8)', 'url': 'https://www.ti.com/lit/ug/tiduf01/tiduf01.pdf'},
 {'title': 'MathWorks · Automotive 4D Imaging MIMO Radar (DDMA)',
  'url': 'https://www.mathworks.com/help/radar/ug/simulate-an-automotive-4d-imaging-mimo-radar.html'}]

ASSUMPTIONS = ['线性上升 FMCW，匀速单点目标；采用停跳近似，不模拟距离迁移、相位噪声、杂波、多径、天线或射频失真。',
 'TDM 每轮依次发射全部 Tx，loops 是每个 Tx 的慢时间样本数。DDM 每个 chirp 同时发射全部 Tx，loops 是同时发射的总 chirp 数。',
 'DDM 采用等间隔 K 槽相位编码：第 k 路 Tx 的逐 chirp 相位步进为 2πk/K；显示保守单槽无模糊速度，未实现空槽假设检验或多目标盲解码。',
 '实际采样采用保守实数 ADC 规划：可用 IF = min(模拟 IF 带宽, 0.45 × 采样率)。频谱使用解析复信号表示正频率分量。',
 '距离分辨率按实际 ADC 观测带宽 c/(2B) 给出，为矩形窗名义值；Hann 窗主瓣更宽。补零只细化 FFT 栅格。',
 '距离上限是 IF/采样约束，预留无模糊速度边缘的多普勒频移；不是雷达方程决定的探测距离。',
 '波长按 ADC 观测窗中点频率计算。正速度定义为使拍频增大的方向，拍频 f_b = 2SR/c + 2v/λ。',
 '原始数据吞吐按每个实数 ADC 样本 16 bit 估算，不含协议、填充、缓存和处理结果。',
 'dutyPercent 为帧内任一 Tx 正在扫频的时间占比；perTxDutyPercent 为单 Tx 占空比，totalTxDutyPercent 为各 Tx 占空比之和，DDM 时该总和可能大于 100%。',
 '76–81 GHz 是本工具预设的频段规划范围，不代表器件支持范围或法规合规结论。']

INTEGER_BOUNDS = {'samples': (8, 4096), 'loops': (4, 2048), 'txCount': (1, 12),
                  'rxCount': (1, 16), 'rangeFft': (8, 4096), 'dopplerFft': (4, 4096),
                  'ddmSlots': (1, 16)}
METRIC_NAMES = ('centerGHz lambdaMm bandwidthMHz observedBandwidthMHz adcUs chirpUs txIntervalUs '
                'burstMs rangeResolutionM rangeBinM maxRangeM maxVelocityMps velocityResolutionMps '
                'velocityBinMps dutyPercent perTxDutyPercent totalTxDutyPercent dataRateMBps '
                'frameRateHz frameSamples targetBeatMHz targetDopplerHz rangeBiasM ifLimitMHz '
                'maxRangeStaticM endGHz chirpsPerFrame coherentMs rawMaxVelocityMps slotWidthMps chirpCount').split()


def _finite(value):
    try:
        return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _clean(value):
    """Invalid reports still serialize with allow_nan=False."""
    if isinstance(value, dict): return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_clean(v) for v in value]
    if isinstance(value, (bool, str)) or value is None: return value
    if isinstance(value, (int, np.integer)):
        return int(value) if _finite(value) else None
    if isinstance(value, Real):
        try: return float(value) if math.isfinite(float(value)) else None
        except (OverflowError, ValueError): return None
    return None


def _format(value, digits=3):
    return f'{value:.{digits}f}' if _finite(value) else '—'


def _power2(value):
    return isinstance(value, int) and value > 0 and value & (value - 1) == 0


def calculate(params=None, acquisition=None):
    """Return validated timing/physics metrics without synthesizing an FFT."""
    p = deepcopy(BASE)
    errors, warnings, checks = [], [], []

    def invalid(key, label, messages):
        return {'valid': False, 'errors': list(messages), 'warnings': warnings,
                'checks': [{'key': key, 'label': label, 'status': 'fail', 'detail': ' '.join(messages)}],
                'metrics': dict.fromkeys(METRIC_NAMES), 'assumptions': list(ASSUMPTIONS), 'params': _clean(p)}

    if params is not None and not isinstance(params, dict):
        return invalid('numbers', '数值有效性', ['波形参数必须是对象。'])
    if params and set(params) - set(BASE):
        return invalid('fields', '参数字段', ['波形参数含未知字段。'])
    p.update(params or {})
    for key in BASE:
        if key in ('window', 'mode'): continue
        value = p[key]
        try:
            if value is None or isinstance(value, (bool, np.bool_)) or not isinstance(value, (Real, str)) or value == '':
                raise ValueError
            number = float(value)
            if not math.isfinite(number): raise ValueError
            p[key] = int(number) if key in INTEGER_BOUNDS and number.is_integer() else number
        except (ValueError, TypeError, OverflowError):
            errors.append(key + ' 必须是有限数值。')
    if errors: return invalid('numbers', '数值有效性', errors)
    for key in ('startGHz', 'slopeMHzUs', 'rampUs', 'sampleRateMsps', 'frameMs', 'ifBandwidthMHz'):
        if p[key] <= 0: errors.append(key + ' 必须大于 0。')
    for key in ('idleUs', 'adcStartUs', 'targetRange'):
        if p[key] < 0: errors.append(key + ' 不得小于 0。')
    for key, (low, high) in INTEGER_BOUNDS.items():
        if type(p[key]) is not int or not low <= p[key] <= high:
            errors.append(f'{key} 必须为 {low}–{high} 之间的整数。')
    if p['window'] not in ('hann', 'rect'): errors.append('窗口仅支持 hann 或 rect。')
    if p['mode'] not in ('tdm', 'ddm'): errors.append('复用模式仅支持 tdm 或 ddm。')
    if p['mode'] == 'ddm' and p['ddmSlots'] < p['txCount']: errors.append('DDM 槽数不得小于同时发射的 Tx 数。')
    if not _power2(p['rangeFft']) or not _power2(p['dopplerFft']): errors.append('FFT 长度必须是 2 的整数次幂。')
    if p['rangeFft'] < p['samples'] or p['dopplerFft'] < p['loops']: errors.append('FFT 长度不得小于对应采样点数或每 Tx 的 chirp 数。')
    if errors: return invalid('parameters', '参数范围', errors)
    acquisition = {} if acquisition is None else acquisition
    if not isinstance(acquisition, dict): return invalid('acquisition', '采集时序', ['采集时序上下文必须是对象。'])
    supplied = acquisition.get('slowTimeIntervalUs')
    nominal = (p['rampUs'] + p['idleUs']) * (p['txCount'] if p['mode'] == 'tdm' else 1)
    if 'slowTimeIntervalUs' in acquisition and (not _finite(supplied) or supplied <= 0 or supplied < nominal - 1e-9):
        return invalid('acquisition', '采集时序', ['同波形同 Tx 观测间隔必须为有限正数，且不得短于该波形的单独 MIMO 重访间隔。'])
    try:
        adc = p['samples'] / p['sampleRateMsps']; chirp = p['rampUs'] + p['idleUs']
        interval = nominal if supplied is None else float(supplied)
        center = p['startGHz'] + p['slopeMHzUs'] * (p['adcStartUs'] + adc / 2) / 1000
        wavelength = C / (center * 1e9); slope = p['slopeMHzUs'] * 1e12
        doppler = 2 * p['targetVelocity'] / wavelength
        if_limit = min(p['ifBandwidthMHz'], .45 * p['sampleRateMsps'])
        chirps = p['loops'] * (p['txCount'] if p['mode'] == 'tdm' else 1)
        samples = p['samples'] * chirps * p['rxCount']
        raw_velocity = wavelength / (4 * interval * 1e-6)
        velocity = raw_velocity / (p['ddmSlots'] if p['mode'] == 'ddm' else 1)
        max_doppler = 2 * velocity / wavelength
        m = {'centerGHz': center, 'lambdaMm': wavelength * 1000,
             'bandwidthMHz': p['slopeMHzUs'] * p['rampUs'], 'observedBandwidthMHz': p['slopeMHzUs'] * adc,
             'adcUs': adc, 'chirpUs': chirp, 'txIntervalUs': interval, 'nominalTxIntervalUs': nominal,
             'burstMs': chirps * chirp / 1000, 'coherentMs': p['loops'] * interval / 1000,
             'observationSpanMs': ((p['loops'] - 1) * interval + nominal) / 1000,
             'rangeResolutionM': C / (2 * slope * adc * 1e-6),
             'rangeBinM': C * p['sampleRateMsps'] * 1e6 / (2 * slope * p['rangeFft']),
             'maxRangeM': C * max(0, if_limit * 1e6 - max_doppler) / (2 * slope),
             'maxRangeStaticM': C * if_limit * 1e6 / (2 * slope),
             'maxVelocityMps': velocity, 'rawMaxVelocityMps': raw_velocity, 'slotWidthMps': 2 * velocity,
             'chirpCount': chirps, 'velocityResolutionMps': wavelength / (2 * p['loops'] * interval * 1e-6),
             'velocityBinMps': wavelength / (2 * p['dopplerFft'] * interval * 1e-6),
             'dutyPercent': p['rampUs'] * chirps / (p['frameMs'] * 1000) * 100,
             'dataRateMBps': samples * 2 / (p['frameMs'] / 1000) / 1e6,
             'frameRateHz': 1000 / p['frameMs'], 'frameSamples': samples, 'chirpsPerFrame': chirps,
             'targetBeatMHz': (2 * slope * p['targetRange'] / C + doppler) / 1e6,
             'targetDopplerHz': doppler, 'rangeBiasM': C * doppler / (2 * slope), 'ifLimitMHz': if_limit,
             'endGHz': p['startGHz'] + p['slopeMHzUs'] * p['rampUs'] / 1000}
        m['perTxDutyPercent'] = m['dutyPercent'] / (p['txCount'] if p['mode'] == 'tdm' else 1)
        m['totalTxDutyPercent'] = m['dutyPercent'] * (p['txCount'] if p['mode'] == 'ddm' else 1)
        if not all(_finite(v) for v in m.values()): raise ArithmeticError
        if not math.isfinite(doppler * (interval * 1e-6)): raise ArithmeticError
    except (ArithmeticError, ValueError):
        return invalid('overflow', '数值范围', ['参数数量级过大，无法得到有限计算结果。'])
    m['ddmPhasesDeg'] = [360 * k / p['ddmSlots'] for k in range(p['txCount'])] if p['mode'] == 'ddm' else []

    def check(key, label, status, detail):
        checks.append({'key': key, 'label': label, 'status': status, 'detail': detail})
        if status == 'fail': errors.append(detail)
        elif status == 'warn': warnings.append(detail)

    adc_end = p['adcStartUs'] + adc
    check('adc', 'ADC 采样窗', 'pass' if adc_end <= p['rampUs'] + 1e-9 else 'fail',
          f"采样终点 {_format(adc_end)} μs / ramp {_format(p['rampUs'])} μs" + ('；采样窗超出 ramp。' if adc_end > p['rampUs'] + 1e-9 else '；已留出的器件稳定时间需按芯片确认。'))
    check('frame', '帧时序', 'pass' if m['burstMs'] <= p['frameMs'] + 1e-9 else 'fail',
          f"chirp burst {_format(m['burstMs'])} ms / 帧周期 {_format(p['frameMs'])} ms" + ('；帧内无法容纳全部 chirp。' if m['burstMs'] > p['frameMs'] + 1e-9 else '。'))
    if supplied is not None:
        check('acquisition', '同波形观测时序', 'pass' if m['observationSpanMs'] <= p['frameMs'] + 1e-9 else 'fail',
              f"同波形同 Tx 观测间隔 {_format(interval)} μs；单波形间隔 {_format(nominal)} μs；观测跨度 {_format(m['observationSpanMs'])} ms / 帧 {_format(p['frameMs'])} ms。")
    rf_ok = p['startGHz'] >= 76 and m['endGHz'] <= 81
    check('rf', '预设频段规划', 'pass' if rf_ok else 'warn', f"{_format(p['startGHz'])}–{_format(m['endGHz'])} GHz；" + ('位于预设 76–81 GHz 范围，仍需核对器件及法规。' if rf_ok else '超出预设 76–81 GHz 规划范围。'))
    beat_ok = 0 <= m['targetBeatMHz'] < if_limit
    check('if', '目标拍频', 'pass' if beat_ok else 'warn', f"目标拍频 {_format(m['targetBeatMHz'])} MHz / 保守 IF 上限 {_format(if_limit)} MHz" + ('。' if beat_ok else '；目标超出当前正频率 IF 规划范围。'))
    velocity_ok = abs(p['targetVelocity']) < velocity
    check('velocity', '目标速度', 'pass' if velocity_ok else 'warn', f"目标 {_format(p['targetVelocity'],2)} m/s / " + ('保守单槽' if p['mode'] == 'ddm' else '') + f"无模糊区间 [−{_format(velocity,2)}, +{_format(velocity,2)}) m/s" + ('。' if velocity_ok else '；目标越过 DDM 分配子带，未实现解模糊。' if p['mode'] == 'ddm' else '；慢时间频谱会发生速度混叠。'))
    if p['mode'] == 'tdm' and p['txCount'] > 1:
        check('tdm', 'TDM 速度约束', 'warn', f"{p['txCount']} Tx 轮流发射：同一 Tx 的观测间隔为 {_format(interval)} μs，" + (f"无模糊速度为单 Tx 的 1/{p['txCount']}。" if supplied is None else '速度指标已按外部 A/B 调度的同波形重访间隔计算。'))
    if p['mode'] == 'ddm':
        check('ddm', 'DDM 编码边界', 'warn', f"{p['txCount']} Tx / {p['ddmSlots']} 槽；原始慢时间速度范围 ±{_format(raw_velocity,2)} m/s，保守单槽 ±{_format(velocity,2)} m/s。空槽不等于已完成解模糊。")
        if p['loops'] % p['ddmSlots'] or p['dopplerFft'] % p['ddmSlots']:
            check('ddmGrid', 'DDM 编码周期', 'warn', 'chirp 数和 Doppler FFT 长度建议均为槽数的整数倍；当前编码偏移未必精确落在 FFT 栅格。')
        if p['loops'] < p['ddmSlots']:
            check('ddmObservation', 'DDM 观测长度', 'warn', '观测不足一个完整 DDM 编码周期；Tx 频谱可能重叠，叠加幅度可能超过单 Tx 的 0 dB 参考，不能据此验证通道分离。')
        if p['ddmSlots'] == p['txCount']:
            check('ddmEmpty', 'DDM 空槽', 'warn', '全部 DDM 槽均被占用，没有空槽可供后续假设检验；本模型仍不执行目标解码。')
    if p['window'] == 'hann':
        check('window', '窗函数解释', 'warn', 'Hann 窗抑制旁瓣并展宽主瓣；指标卡中的 c/(2B) 与 λ/(2NT) 是矩形窗名义分辨率。')
    drift = abs(p['targetVelocity']) * m['coherentMs'] / 1000
    if drift > m['rangeResolutionM'] / 2:
        check('migration', '停跳近似', 'warn', f"目标在相干时间内移动约 {_format(drift)} m；本演示忽略距离迁移，真实频谱可能展宽。")
    assumptions = list(ASSUMPTIONS)
    if supplied is not None:
        assumptions.append('慢时间指标与频谱按提供的均匀同波形同 Tx 观测间隔计算；物理 chirp 时间与采样吞吐保持原波形参数。')
    return {'valid': not errors, 'errors': errors, 'warnings': warnings, 'checks': checks,
            'metrics': m, 'assumptions': assumptions, 'params': p}


def _tone_spectrum(count, fft_length, cycles, window):
    tones = np.atleast_1d(cycles).astype(float)
    # Sampling aliases integer cycles exactly. Reduce before constructing phases
    # so even a very large finite frequency does not overflow n*frequency.
    tones = np.where(np.abs(tones) > 1e6, np.remainder(tones, 1.), tones)
    n = np.arange(count, dtype=float)
    weights = .5 - .5 * np.cos(2 * np.pi * n / (count - 1)) if window == 'hann' else np.ones(count)
    samples = np.sum(np.exp(2j * np.pi * tones[:, None] * n), axis=0) * weights
    return 20 * np.log10(np.maximum(1e-8, abs(np.fft.fft(samples, n=fft_length)) / weights.sum()))


def _pooled(profile, maximum):
    size = min(len(profile), maximum); result = []
    for i in range(size):
        start, end = i * len(profile) // size, (i + 1) * len(profile) // size
        result.append({'x': (profile[start]['x'] + profile[end - 1]['x']) / 2,
                       'y': max(row['y'] for row in profile[start:end])})
    return result


def simulate(params=None, acquisition=None):
    """Bounded ideal single-target FFTs; heatmap pooling never changes peak reads."""
    result = calculate(params, acquisition)
    if not result['valid']:
        return {'valid': False, 'errors': result['errors'], 'rangeProfile': [], 'dopplerProfile': [],
                'rangeDoppler': {'ranges': [], 'velocities': [], 'values': [], 'width': 0, 'height': 0},
                'detected': None, 'notes': ['参数无效，未运行频谱演示。']}
    p, m = result['params'], result['metrics']
    r_spectrum = _tone_spectrum(p['samples'], p['rangeFft'], m['targetBeatMHz'] / p['sampleRateMsps'], p['window'])
    cycles = m['targetDopplerHz'] * m['txIntervalUs'] * 1e-6
    base_spectrum = _tone_spectrum(p['loops'], p['dopplerFft'], cycles, p['window'])
    d_spectrum = (_tone_spectrum(p['loops'], p['dopplerFft'], [cycles + k / p['ddmSlots'] for k in range(p['txCount'])], p['window'])
                  if p['mode'] == 'ddm' else base_spectrum)
    rp = [{'x': k * m['rangeBinM'], 'y': float(y)} for k, y in enumerate(r_spectrum[:p['rangeFft'] // 2])]
    dp = [{'x': (k - p['dopplerFft'] / 2) * m['velocityBinMps'], 'y': float(y)} for k, y in enumerate(np.fft.fftshift(d_spectrum))]
    bp = [{'x': (k - p['dopplerFft'] / 2) * m['velocityBinMps'], 'y': float(y)} for k, y in enumerate(np.fft.fftshift(base_spectrum))]
    r_peak, d_peak, base_peak = (max(profile, key=lambda row: row['y']) for profile in (rp, dp, bp))
    range_valid = 0 <= m['targetBeatMHz'] < m['ifLimitMHz']
    range_note = ('目标拍频位于规划的正频率 IF 范围内。' if range_valid else
                  '目标拍频超出规划正频率 IF 范围，不报告距离峰值；所示解析复信号频谱不包含实 ADC 的镜像折叠或模拟 IF 滤波器响应。')
    pooled_r, pooled_d = _pooled(rp, 128), _pooled(dp, 64)
    notes = [
        '由解析 IF 信号 exp[j2π(f_b n/f_s + f_D mT_Tx)] 生成样本、加窗并进行 FFT；二维可分离结果等价于该单目标模型的二维 FFT。',
        '无噪声、单点、恒定速度、无距离迁移模型；谱幅度按窗函数相干增益归一化，单位 dB。未模拟信噪比或检测概率。',
        '距离谱保留多普勒造成的距离偏移；显示的峰值距离为未经距离–多普勒耦合补偿的表观距离。',
        '热图采用区间最大值汇聚到最多 128×64 像素以保留峰值；峰值读数取完整 FFT 栅格，热图像素并非原始 FFT 分辨单元。',
        '速度越界时展示慢时间频谱混叠。距离谱仅显示解析复信号的正频率半轴，不是实 ADC 余弦信号的镜像折叠演示，且未包含模拟 IF 滤波器响应。',
        '目标拍频位于规划正频率 IF 范围内，允许读出距离 FFT 栅格峰值。' if range_valid else
        '目标拍频超出规划正频率 IF 范围，距离及耦合补偿后的距离读数无效；正半轴可能只留下旁瓣泄漏，不将其报告为目标峰值。']
    if p['mode'] == 'ddm': notes.extend([
        'DDM 频谱由各 Tx 的相位编码样本相加后计算 FFT；各 Tx 单位幅度、初相位为零，不包含阵列流形、功率预算或通道误差。',
        'DDM 目标速度读数取已知未编码参考投影，绝非对编码图的盲解码；空槽仅显示为未分配子带，未实现空槽判别。',
        '同距离多目标可能发生 DDM 子带碰撞；单目标演示不能验证碰撞抑制、解模糊或检测性能。'])
    return {'valid': True, 'rangeProfile': rp, 'dopplerProfile': dp, 'baseDopplerProfile': bp,
            'rangeDoppler': {'ranges': [r['x'] for r in pooled_r], 'velocities': [d['x'] for d in pooled_d],
                             'values': [max(-100., r['y'] + d['y']) for d in pooled_d for r in pooled_r],
                             'width': len(pooled_r), 'height': len(pooled_d)},
            'detected': {'rangeM': r_peak['x'] if range_valid else None, 'rangeValid': range_valid,
                         'rangeValidityNote': range_note, 'velocityMps': base_peak['x'], 'encodedVelocityMps': d_peak['x'],
                         'correctedRangeM': r_peak['x'] - m['rangeBiasM'] if range_valid else None,
                         'apparentRangeM': p['targetRange'] + m['rangeBiasM'],
                         'estimationLabel': '已知目标参考投影（非 DDM 盲解码）' if p['mode'] == 'ddm' else 'FFT 栅格峰值'},
            'notes': notes}
