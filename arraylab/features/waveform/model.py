"""ArrayLab orchestration of the frozen WaveLab v1.2 waveform capabilities."""
from copy import deepcopy
import math
from numbers import Real

from arraylab import __version__
import arraylab.features.waveform.core as waveform_core; import arraylab.features.waveform.ab as waveform_ab; import arraylab.features.waveform.ambiguity as waveform_ambiguity; import arraylab.features.waveform.pri as waveform_pri
from arraylab.framework.parallel import check_cancelled, ordered_map

SOURCE_VERSION = '1.2'
SOURCE_ARCHIVE_SHA256 = '16cd6c2e1ad15f7ecb4ba0c15af957711d2427b972b0dee1fd2e07bc08d21ad4'
CONFIG_KEYS = ('params', 'profileB', 'abOptions', 'activeProfile', 'rangeOptions', 'priOptions')
VIEWS = ('overview', 'phase', 'spectra', 'rd', 'metrics', 'range_recovery', 'pri_recovery', 'comparisons', 'theory')


def _default_profile_b(a):
    ramp = a['rampUs'] + 8
    slope = max(.1, min(a['slopeMHzUs'] * 1.5, max(.1, (81-a['startGHz'])*1000/ramp*.98)))
    return {**a, 'slopeMHzUs': slope, 'rampUs': ramp, 'idleUs': a['idleUs']+2}


def default_waveform_configuration():
    a = deepcopy(waveform_core.BASE)
    b = _default_profile_b(a)
    return {'params': a, 'profileB': b, 'abOptions': deepcopy(waveform_ab.DEFAULTS),
            'activeProfile': 'A',
            'rangeOptions': {'method': 'off', 'secondSampleRateMsps': 5, 'secondSlopeMHzUs': 9.6,
                             'searchMaxM': 120, 'knownVelocityMps': 0, 'toleranceBins': .75},
            'priOptions': {**deepcopy(waveform_pri.DEFAULTS), 'method': 'off'}}


def _merge(values, defaults, label):
    if type(values) is not dict or any(type(k) is not str for k in values):
        raise ValueError(label + '须为参数对象。')
    unknown = set(values) - set(defaults)
    if unknown:
        raise ValueError(label + '包含未知字段：' + ', '.join(sorted(unknown)))
    result = deepcopy(defaults)
    for key, value in values.items():
        default = defaults[key]
        if type(default) is bool:
            if type(value) is not bool: raise ValueError(label + '.' + key + '须为布尔值。')
        elif isinstance(default, str):
            if type(value) is not str or len(value) > 128: raise ValueError(label + '.' + key + '文本无效。')
        else:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(label + '.' + key + '须为有限数值。')
            try: finite = math.isfinite(value)
            except (ValueError, OverflowError): finite = False
            if not finite: raise ValueError(label + '.' + key + '须为有限数值。')
            value = int(value) if isinstance(default, int) and float(value).is_integer() else float(value)
        result[key] = value
    return result


def _profile_inputs(p, label):
    for key in ('startGHz', 'slopeMHzUs', 'rampUs', 'sampleRateMsps', 'frameMs', 'ifBandwidthMHz'):
        if p[key] <= 0: raise ValueError(label + '.' + key + '须大于0。')
    for key in ('idleUs', 'adcStartUs', 'targetRange'):
        if p[key] < 0: raise ValueError(label + '.' + key + '不得小于0。')
    limits = {'samples': (8, 4096), 'loops': (4, 2048), 'txCount': (1, 12), 'rxCount': (1, 16),
              'rangeFft': (8, 4096), 'dopplerFft': (4, 4096), 'ddmSlots': (1, 16)}
    for key, (low, high) in limits.items():
        if type(p[key]) is not int or not low <= p[key] <= high:
            raise ValueError(f'{label}.{key}须为{low}–{high}的整数。')
    for key, count in (('rangeFft', 'samples'), ('dopplerFft', 'loops')):
        if p[key] & (p[key]-1) or p[key] < p[count]:
            raise ValueError(label + '.' + key + '须为不小于采样数的2次幂。')
    if p['window'] not in ('hann', 'rect') or p['mode'] not in ('tdm', 'ddm'):
        raise ValueError(label + '窗函数或复用模式无效。')


def normalize_waveform_configuration(configuration=None):
    """Validate editable input types without calculating spectra or recovery."""
    if configuration is None: return default_waveform_configuration()
    if type(configuration) is not dict or set(configuration) - set(CONFIG_KEYS):
        raise ValueError('波形配置字段无效。')
    defaults = default_waveform_configuration()
    result = {}
    for name in ('params', 'abOptions', 'rangeOptions', 'priOptions'):
        result[name] = _merge(configuration.get(name, {}), defaults[name], name)
    _profile_inputs(result['params'], 'A')
    result['profileB'] = _merge(configuration.get('profileB', {}), _default_profile_b(result['params']), 'profileB')
    for key in waveform_ab.SHARED_KEYS:
        result['profileB'][key] = result['params'][key]
    if result['abOptions']['enabled']:
        _profile_inputs(result['profileB'], 'B')
    active = configuration.get('activeProfile', 'A')
    if active not in ('A', 'B'): raise ValueError('当前分析组须为A或B。')
    result['activeProfile'] = active if result['abOptions']['enabled'] else 'A'
    if result['abOptions']['layout'] not in ('blocks', 'alternating'):
        raise ValueError('A/B编排须为分段或整轮交替。')
    if result['rangeOptions']['method'] not in ('off', 'dual_rate', 'dual_slope'):
        raise ValueError('ADC解模糊算法无效。')
    if result['priOptions']['method'] == 'staggered_pri':
        result['priOptions']['method'] = 'dual_period'
    if result['priOptions']['method'] not in ('off', 'dual_period'):
        raise ValueError('跨周期解模糊算法无效。')
    return result


def calculate_waveform(configuration=None, *, workers=None, cancelled=None, progress=None):
    config = normalize_waveform_configuration(configuration)
    def tell(text):
        check_cancelled(cancelled)
        if progress: progress(text)
    tell('核对波形与A/B采集时序')
    analysis = waveform_ab.calculate(config['params'], config['profileB'], config['abOptions'])
    if not analysis['valid']:
        raise ValueError('；'.join(analysis['errors']) or '波形参数不满足采集条件。')
    active = ['A', 'B'] if config['abOptions']['enabled'] else ['A']

    def profile_result(name):
        check_cancelled(cancelled)
        profile = analysis['profiles'][name]
        spectrum = waveform_core.simulate(profile['params'], profile['acquisition'])
        if not spectrum['valid']:
            raise ValueError(name + '：' + '；'.join(spectrum['errors']))
        check_cancelled(cancelled)
        recovery = waveform_ambiguity.simulate(profile['params'], config['rangeOptions'])
        check_cancelled(cancelled)
        return spectrum, recovery

    tell('生成所选波形的理想频谱与ADC解模糊候选')
    values, execution = ordered_map(profile_result, active, workers=workers,
                                    work_units=sum(config['params' if name == 'A' else 'profileB']['rangeFft'] *
                                                   config['params' if name == 'A' else 'profileB']['dopplerFft'] for name in active),
                                    cancelled=cancelled)
    tell('计算独立跨周期时延实验')
    pri = waveform_pri.simulate(config['priOptions'])
    tell('波形分析完成')
    return {'format': 'arraylab.waveform.result', 'schema_version': 1, 'app_version': __version__,
            'configuration': config, 'analysis': analysis,
            'spectra': {name: row[0] for name, row in zip(active, values)},
            'range_recovery': {name: row[1] for name, row in zip(active, values)},
            'pri_recovery': pri, 'execution': execution,
            'source': {'name': 'WaveLab', 'version': SOURCE_VERSION, 'archive_sha256': SOURCE_ARCHIVE_SHA256},
            'notices': [
                '由已完成的WaveLab v1.2快照独立移植，运行不依赖原项目、浏览器或Node.js。',
                '理想单点目标与有限采样模型；IF/采样距离上限不是雷达方程探测距离，FFT补零不提高物理分辨率。',
                'A/B独立处理，不直接合并带宽；DDM展示编码频谱，未实现多目标盲解码。',
                '两类距离解模糊为独立实验，第二次采集的额外时间不自动加入主波形帧预算。',
                '跨周期实验的目标距离独立于主波形目标；恢复候选仅在指定搜索范围与容差内有效。',
                '本轮不包含稀疏阵列反向设计、天线增益预算或硬件配置。']}
