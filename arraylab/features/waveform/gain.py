"""Physical-channel antenna gain budget ported from the WaveLab 1.4 snapshot.

The caller validates the complete A/B frame before supplying profile inputs.
No transmit-count bonus or virtual-position gain is inferred by this model.
"""
from copy import deepcopy
import math
from numbers import Real

from . import core

K = 1.380649e-23
FOUR_PI = 4 * math.pi
DEFAULTS = {
    'rcsDbsm': 0, 'rangeM': 100, 'txPowerDbm': 10, 'powerBasis': 'per_tx',
    'noiseFigureDb': 12, 'systemLossDb': 6, 'requiredSnrDb': 13, 'marginDb': 3,
    'temperatureK': 290, 'combining': 'coherent_rx', 'gainMode': 'equal',
    'fixedGainDbi': 12, 'apertureEfficiency': 0.6, 'radialVelocityMps': 0,
}
ASSUMPTIONS = [
    'Gt、Gr 是目标方向上每个物理 RF 通道所连接天线或子阵的 realized gain，单位 dBi；不是去重虚拟阵列位置数或归一化阵列因子。',
    '使用单站雷达能量方程和一个已分离 Tx 的匹配处理分支；不因 TDM/DDM 的 Tx 数量额外增加发射相干收益，也不假定 DDM 盲解码成功。',
    'per_tx 为每路发射天线端口的峰值传导功率。total_active 为同一时刻所有活动 Tx 的总传导功率：TDM 一路活动，DDM 按 Tx 数均分。功率不是 EIRP。',
    'coherent_rx 采用空间白噪声、信号相位已校准条件下的理想 Nrx 合并收益；如果输入的 Gr 已含整个接收阵列增益，应选择 single_rx，避免重复计数。',
    '能量积累时间取 ADC 采样时长 × 每 Tx 实际 chirp 数；idle、A/B 切换间隔、帧末空闲以及 FFT 补零不增加信号能量。',
    '快时间和慢时间窗效率分别为 (Σw)²/(NΣw²)，Hann 采用与本工具 FFT 相同的对称窗；两项效率均作用于处理后 SNR。',
    '噪声项为 k × 参考温度 T × 噪声因子 F。参考温度不应是已经包含接收机噪声系数的等效系统温度；系统损耗不应重复包含已计入 realized gain 的损耗。',
    '所需 SNR 与设计裕量由用户指定，未从检测概率、虚警率、CFAR、目标起伏或杂波分布推导。采用单目标、无距离迁移且相干积累理想成立的预算。',
    'A/B 分别计算，选择较高的增益乘积要求；不把两组波形能量默认相干相加。增加天线增益不能恢复已超出波形 IF/采样范围的回波。',
    'Ae=Gλ²/(4π) 是每通道有效口径。再除以用户给定孔径效率得到等效物理面积；仅是面积预算，不是实际天线几何、波束形状、互耦或宽带实现设计。',
]
SOURCES = [
    {'title': 'TI · mmWave Production Testing Overview / SNR radar equation', 'url': 'https://www.ti.com/lit/an/spracx7/spracx7.pdf'},
    {'title': 'MathWorks · radareqsnr', 'url': 'https://www.mathworks.com/help/radar/ref/radareqsnr.html'},
    {'title': 'MathWorks · Equivalent Noise Bandwidth', 'url': 'https://www.mathworks.com/help/signal/ref/enbw.html'},
    {'title': 'MathWorks · gain2aperture', 'url': 'https://www.mathworks.com/help/phased/ref/gain2aperture.html'},
]
NUMERIC_BOUNDS = {
    'rcsDbsm': (-100, 100), 'rangeM': (0.1, 100000), 'txPowerDbm': (-100, 100),
    'noiseFigureDb': (0, 60), 'systemLossDb': (0, 100), 'requiredSnrDb': (-40, 80),
    'marginDb': (0, 60), 'temperatureK': (1, 5000), 'fixedGainDbi': (-40, 100),
    'apertureEfficiency': (0.001, 1), 'radialVelocityMps': (-1000, 1000),
}
CHOICES = {'powerBasis': ('per_tx', 'total_active'),
           'combining': ('coherent_rx', 'single_rx'),
           'gainMode': ('equal', 'solve_tx', 'solve_rx')}


def _finite(value):
    try:
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _db(value):
    return 10 * math.log10(value)


def _format(value, digits=2):
    return f'{value:.{digits}f}' if _finite(value) else '—'


def _snapshot(options):
    return {key: (value if _finite(value) else value[:80] if isinstance(value, str) else None)
            for key in DEFAULTS for value in (options.get(key),)}


def _empty(options, reason=None):
    return {'valid': False, 'status': 'invalid', 'requiredTxGainDbi': None,
            'requiredRxGainDbi': None, 'requiredGainSumDb': None, 'limitingProfile': None,
            'profiles': [], 'checks': [], 'warnings': [], 'errors': [reason] if reason else [],
            'assumptions': list(ASSUMPTIONS), 'sources': deepcopy(SOURCES), 'options': _snapshot(options)}


def validate_options(options):
    """Validate a complete option dictionary; raise ValueError on invalid input."""
    if not isinstance(options, dict):
        raise ValueError('增益预算设置须为对象。')
    for key, (low, high) in NUMERIC_BOUNDS.items():
        if not _finite(options.get(key)) or not low <= options[key] <= high:
            raise ValueError(f'{key} 须为 {low}–{high} 范围内的有限数值。')
    for key, allowed in CHOICES.items():
        if options.get(key) not in allowed:
            raise ValueError('不支持的 ' + key + ' 选项。')


def window_efficiency(count, window):
    """Finite-length symmetric-window coherent SNR efficiency."""
    if not _finite(count) or int(count) != count or not 1 <= count <= 16384 or window not in ('rect', 'hann'):
        raise ValueError('窗效率需要有效样本数和 rect / hann 窗。')
    count = int(count)
    if window == 'rect':
        return 1.0
    if count < 3:
        raise ValueError('对称 Hann 窗至少需要 3 个样本。')
    total = squares = 0.0
    for n in range(count):
        weight = 0.5 - 0.5 * math.cos(2 * math.pi * n / (count - 1))
        total += weight
        squares += weight * weight
    return total * total / (count * squares)


def aperture(gain_dbi, wavelength_m, efficiency):
    effective = 10 ** (gain_dbi / 10) * wavelength_m * wavelength_m / FOUR_PI
    physical = effective / efficiency
    return {'effectiveAreaM2': effective, 'physicalAreaM2': physical,
            'equivalentSquareSideMm': math.sqrt(physical) * 1000,
            'equivalentCircularDiameterMm': math.sqrt(4 * physical / math.pi) * 1000}


def layout_warnings(result, geometry):
    """Compare equivalent area with same-kind spacing, without designing sizes.

    Geometry coordinates are wavelength-normalized ``[x, z]`` pairs. A stored
    reference frequency fixes their physical positions across A/B profiles.
    Both inputs should already have passed their domain validators.
    """
    if not result.get('valid'):
        return []
    reference = geometry.get('referenceFrequencyGHz')
    if reference is not None and (not _finite(reference) or reference <= 0):
        raise ValueError('布局参考频率须为有限正数。')
    distances = {}
    for kind in ('tx', 'rx'):
        points = geometry[kind]
        if (not isinstance(points, list) or not points or len(points) > 32
                or any(not isinstance(point, (list, tuple)) or len(point) != 2
                       or any(not _finite(value) for value in point) for point in points)):
            raise ValueError('布局须为有限 [x, z] 坐标序列。')
        distances[kind] = min((math.hypot(a[0] - b[0], a[1] - b[1])
                               for index, a in enumerate(points) for b in points[index + 1:]), default=None)
    warnings = []
    for profile in result['profiles']:
        wavelength_mm = 299.792458 / reference if reference is not None else profile['lambdaMm']
        for kind, distance in distances.items():
            side = profile[kind + 'EquivalentSquareSideMm']
            if distance is not None and side > distance * wavelength_mm:
                warnings.append(f'波形 {profile["name"]}：{kind.upper()} 等效方形边长 {_format(side)} mm '
                                f'大于最近同类通道间距 {_format(distance * wavelength_mm)} mm，需核对布局容纳能力。')
    return list(dict.fromkeys(warnings))


def calculate(profile_inputs, input_options=None):
    """Budget one or two already scheduled A/B profiles, preserving source keys.

Each profile is ``{name: 'A'|'B', params: {...}, acquisition: {...}}``.
``valid`` means finite mathematical budget; inspect ``status`` to distinguish
``designed`` from a target outside the waveform's observable IF range.
"""
    if input_options is not None and not isinstance(input_options, dict):
        return _empty(DEFAULTS, '增益预算设置须为对象。')
    options = {**DEFAULTS, **(input_options or {})}
    out = _empty(options)
    try:
        validate_options(options)
    except ValueError as exc:
        return _empty(options, str(exc))
    if not isinstance(profile_inputs, list) or not 1 <= len(profile_inputs) <= 2:
        return _empty(options, '请选择一组或两组波形进行预算。')
    names = set()
    for item in profile_inputs:
        if (not isinstance(item, dict) or item.get('name') not in ('A', 'B')
                or item['name'] in names or not isinstance(item.get('params'), dict)):
            return _empty(options, '波形须为唯一命名 A / B 的参数对象。')
        names.add(item['name'])

    def add(key, label, status, detail):
        out['checks'].append({'key': key, 'label': label, 'status': status, 'detail': detail})
        if status == 'fail': out['errors'].append(detail)
        if status == 'warn': out['warnings'].append(detail)

    invalid_waveform = False
    for item in profile_inputs:
        result = core.calculate({**item['params'], 'targetRange': options['rangeM'],
                                 'targetVelocity': options['radialVelocityMps']}, item.get('acquisition', {}))
        params, metrics, name = result['params'], result['metrics'], item['name']
        if not result['valid']:
            invalid_waveform = True
            out['profiles'].append({'name': name, 'valid': False, 'observable': False, 'errors': list(result['errors'])})
            add(name + '.timing', name + ' · 波形有效性', 'fail', name + ' 波形无效：' + ' '.join(result['errors']))
            continue
        wavelength = metrics['lambdaMm'] / 1000
        active_tx = params['txCount'] if params['mode'] == 'ddm' else 1
        power_dbm = options['txPowerDbm'] - (_db(active_tx) if options['powerBasis'] == 'total_active' else 0)
        rx_factor = params['rxCount'] if options['combining'] == 'coherent_rx' else 1
        rx_gain = _db(rx_factor)
        fast_efficiency = window_efficiency(params['samples'], params['window'])
        slow_efficiency = window_efficiency(params['loops'], params['window'])
        efficiency = fast_efficiency * slow_efficiency
        window_loss = -_db(efficiency)
        energy_ms = metrics['adcUs'] * params['loops'] / 1000
        zero_gain_snr = (power_dbm - 30 + 20 * math.log10(wavelength) + options['rcsDbsm']
                         + _db(energy_ms / 1000) - window_loss + rx_gain
                         - 30 * math.log10(FOUR_PI) - 40 * math.log10(options['rangeM'])
                         - _db(K) - _db(options['temperatureK']) - options['noiseFigureDb'] - options['systemLossDb'])
        required_snr = options['requiredSnrDb'] + options['marginDb']
        requirement = required_snr - zero_gain_snr
        observable = 0 <= metrics['targetBeatMHz'] < metrics['ifLimitMHz']
        profile = {
            'name': name, 'valid': True, 'observable': observable, 'mode': params['mode'],
            'txCount': params['txCount'], 'rxCount': params['rxCount'],
            'requiredGainSumDb': requirement, 'zeroGainSnrDb': zero_gain_snr,
            'powerPerTxDbm': power_dbm, 'powerPerTxW': 10 ** ((power_dbm - 30) / 10),
            'activeTxCount': active_tx, 'rxCombiningFactor': rx_factor, 'rxCombiningGainDb': rx_gain,
            'fastWindowEfficiency': fast_efficiency, 'slowWindowEfficiency': slow_efficiency,
            'windowLossDb': window_loss, 'energyTimeMs': energy_ms, 'effectiveTimeMs': energy_ms * efficiency,
            'lambdaMm': metrics['lambdaMm'], 'adcUs': metrics['adcUs'], 'chirpsPerTx': params['loops'],
            'nominalCoherentMs': metrics['coherentMs'], 'observationSpanMs': metrics['observationSpanMs'],
            'requiredSnrWithMarginDb': required_snr, 'targetBeatMHz': metrics['targetBeatMHz'],
            'ifLimitMHz': metrics['ifLimitMHz'], 'maxVelocityMps': metrics['maxVelocityMps'],
            'rcsSquareM': 10 ** (options['rcsDbsm'] / 10),
        }
        out['profiles'].append(profile)
        add(name + '.timing', name + ' · 波形有效性', 'pass',
            f'{name} 使用 {params["samples"]} 点 ADC × 每 Tx {params["loops"]} 个 chirp，能量积累时间 {_format(energy_ms, 4)} ms。')
        add(name + '.if', name + ' · 目标可观测性', 'pass' if observable else 'fail',
            f'{name} 目标拍频 {_format(metrics["targetBeatMHz"], 4)} MHz / 可用 IF {_format(metrics["ifLimitMHz"], 3)} MHz。'
            + ('位于规划通带内。' if observable else '超出规划正频率 IF/采样范围，增加天线增益不能恢复该回波。'))
        if abs(options['radialVelocityMps']) >= metrics['maxVelocityMps']:
            add(name + '.velocity', name + ' · 速度模糊', 'warn',
                '目标速度超出 ' + name + ' 的' + (' DDM 单子带' if params['mode'] == 'ddm' else '')
                + '无模糊范围；能量预算仍可计算，但不能宣称速度可无模糊测量。')
        for check in result['checks']:
            if check['status'] == 'warn' and check['key'] in ('migration', 'rf', 'ddmGrid', 'ddmObservation'):
                add(name + '.' + check['key'], name + ' · ' + check['label'], 'warn', check['detail'])
    if invalid_waveform:
        return out
    limiting = max(out['profiles'], key=lambda profile: profile['requiredGainSumDb'])
    requirement = limiting['requiredGainSumDb']
    if options['gainMode'] == 'equal':
        tx_gain = rx_gain = requirement / 2
    elif options['gainMode'] == 'solve_tx':
        rx_gain = options['fixedGainDbi']
        tx_gain = requirement - rx_gain
    else:
        tx_gain = options['fixedGainDbi']
        rx_gain = requirement - tx_gain
    for profile in out['profiles']:
        try:
            tx_aperture = aperture(tx_gain, profile['lambdaMm'] / 1000, options['apertureEfficiency'])
            rx_aperture = aperture(rx_gain, profile['lambdaMm'] / 1000, options['apertureEfficiency'])
        except (ArithmeticError, ValueError):
            return _empty(options, '增益或口径计算超出有限数值范围。')
        delivered = profile['zeroGainSnrDb'] + tx_gain + rx_gain
        profile.update({
            'requiredTxGainDbi': tx_gain, 'requiredRxGainDbi': rx_gain,
            'deliveredSnrDb': delivered, 'snrMarginDb': delivered - options['requiredSnrDb'],
            'budgetSlackDb': delivered - profile['requiredSnrWithMarginDb'],
            'eirpPerTxDbm': profile['powerPerTxDbm'] + tx_gain,
            'txAperture': tx_aperture, 'rxAperture': rx_aperture,
            'txEffectiveAreaM2': tx_aperture['effectiveAreaM2'], 'rxEffectiveAreaM2': rx_aperture['effectiveAreaM2'],
            'txPhysicalAreaM2': tx_aperture['physicalAreaM2'], 'rxPhysicalAreaM2': rx_aperture['physicalAreaM2'],
            'txEquivalentSquareSideMm': tx_aperture['equivalentSquareSideMm'],
            'rxEquivalentSquareSideMm': rx_aperture['equivalentSquareSideMm'],
        })
        if any(isinstance(value, Real) and not isinstance(value, bool) and not _finite(value) for value in profile.values()):
            return _empty(options, '增益或口径计算超出有限数值范围。')
    out.update({'valid': True, 'status': 'designed' if all(p['observable'] for p in out['profiles']) else 'waveform_limited',
                'requiredTxGainDbi': tx_gain, 'requiredRxGainDbi': rx_gain, 'requiredGainSumDb': requirement,
                'limitingProfile': limiting['name'], 'rcsSquareM': 10 ** (options['rcsDbsm'] / 10),
                'requiredSnrWithMarginDb': options['requiredSnrDb'] + options['marginDb']})
    add('gainReference', '增益口径', 'pass', 'Gt / Gr 为每物理通道目标方向增益；'
        + ('理想 Rx 相干合并收益单独计入。' if options['combining'] == 'coherent_rx' else '按一个接收通道预算。')
        + ' 未额外计入 Tx 数量收益。')
    if len(out['profiles']) > 1:
        add('profileBudget', 'A/B 分别预算', 'pass', '共用增益要求由 ' + limiting['name'] + ' 波形决定，两组能量未直接相加。')
    if max(tx_gain, rx_gain) > 35:
        add('highGain', '高增益实现', 'warn', '所需每通道增益较高；等效面积仅供预算，应另行核对波束覆盖、阵面尺寸、效率和带宽。')
    return out
