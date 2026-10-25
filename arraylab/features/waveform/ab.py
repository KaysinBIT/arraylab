"""Independent A/B FMCW profiles with explicit physical frame scheduling."""
from copy import deepcopy
import arraylab.features.waveform.core as engine

DEFAULTS = {'enabled': False, 'layout': 'blocks', 'switchGapUs': 5}

SHARED_KEYS = ['mode', 'txCount', 'rxCount', 'frameMs', 'ddmSlots', 'targetRange', 'targetVelocity', 'window']

INDEPENDENT_KEYS = ['startGHz',
 'slopeMHzUs',
 'rampUs',
 'idleUs',
 'adcStartUs',
 'samples',
 'sampleRateMsps',
 'loops',
 'ifBandwidthMHz',
 'rangeFft',
 'dopplerFft']

SOURCES = [{'title': 'TI · Programming Chirp Parameters in TI Radar Devices, §6',
  'url': 'https://www.ti.com/lit/pdf/SWRA553'}]

ASSUMPTIONS = ['A/B 表示两组独立的上升 FMCW profile，不是固定上下扫频三角波。模式、Tx/Rx 数、帧周期、DDM 子带数、窗函数及目标参数在两组之间共享。',
 '分段模式先发完全部 A，再留出切换间隔，随后发完全部 B；每组内部的同波形同 Tx 观测间隔不变。',
 '整轮交替模式以完整 A MIMO 轮次、切换间隔、完整 B MIMO 轮次、切换间隔构成一对；两组轮数须相同，最后一组 B 后不再计切换间隔。',
 '交替模式按同波形同 Tx 的实际重访间隔计算慢时间频谱与速度指标；A/B 的距离带宽和频谱独立处理，不把不同 profile 的带宽直接相加宣称更高分辨率。',
 'DDM 的相位编码按每组 profile 自身的 chirp 序号分别递增；假设各 Tx 已正确映射，未实现 DDM 盲解码。',
 '切换间隔是用户给定的时序预算，不代表任何芯片的 PLL、校准或 profile 切换规格；需按器件要求核对。',
 '帧内 burst 不包含最后 B 到下一帧 A 的切换；这次切换必须由帧末空闲时间覆盖。',
 '单目标模型使用两组共同的距离与速度作为停跳场景，未模拟 A/B 两组之间的距离迁移、跨组相干融合或多目标关联。']
METRIC_NAMES = ('burstMs frameMs frameRateHz frameIdleMs chirpCount frameSamples dataRateMBps '
                'txRampDutyPercent perTxDutyPercent totalTxDutyPercent slowTimeIntervalAUs '
                'slowTimeIntervalBUs pairIntervalUs switchCount switchTimeMs').split()


def _profile(result, acquisition):
    return {'params': result['params'], 'result': result, 'acquisition': acquisition}


def _event(name, p, round_index, tx, start):
    return {'profile': name, 'txIndex': None if p['mode'] == 'ddm' else tx,
            'startUs': start, 'rampUs': p['rampUs'], 'idleUs': p['idleUs'],
            'endUs': start + p['rampUs'] + p['idleUs'], 'chirpIndex': round_index, 'roundIndex': round_index,
            'physicalChirpIndex': round_index * (p['txCount'] if p['mode'] == 'tdm' else 1) + tx}


def _build_preview(a, b, options, rounds=2):
    count = a['txCount'] if a['mode'] == 'tdm' else 1
    ta, tb, gap = a['rampUs'] + a['idleUs'], b['rampUs'] + b['idleUs'], options['switchGapUs']
    da, db = a['loops'] * count * ta, b['loops'] * count * tb
    pair, alternating = count * (ta + tb) + 2 * gap, options['layout'] == 'alternating'

    def start_for(name, round_index, tx):
        if alternating: return round_index * pair + (tx * ta if name == 'A' else count * ta + gap + tx * tb)
        return round_index * count * ta + tx * ta if name == 'A' else da + gap + round_index * count * tb + tx * tb

    groups = []
    for name, p, period in [('A', a, ta), ('B', b, tb)]:
        events = [_event(name, p, r, tx, start_for(name, r, tx))
                  for r in range(min(p['loops'], rounds)) for tx in range(count)][:6]
        start, end = start_for(name, 0, 0), start_for(name, p['loops'] - 1, count - 1) + period
        groups.append({'profile': name, 'events': events, 'startUs': start, 'endUs': end,
                       'spanMs': (end - start) / 1000, 'chirpCount': p['loops'] * count,
                       'truncated': len(events) < p['loops'] * count})
    events = []
    if alternating:
        for r in range(min(a['loops'], 12)):
            for name, p in [('A', a), ('B', b)]:
                for tx in range(count):
                    if len(events) < 12: events.append(_event(name, p, r, tx, start_for(name, r, tx)))
            if len(events) >= 12: break
    else:
        for name, p in [('A', a), ('B', b)]:
            for r in range(min(p['loops'], 12)):
                for tx in range(count):
                    if len(events) < 12: events.append(_event(name, p, r, tx, start_for(name, r, tx)))
                if len(events) >= 12: break
    return {'previewEvents': events, 'previewGroups': groups,
            'profileSpans': [{k: row[k] for k in ('profile', 'startUs', 'endUs', 'spanMs', 'chirpCount')} for row in groups],
            'timelineBlocks': [] if alternating else [
                {'profile': 'A', 'startUs': 0, 'endUs': da, 'chirpCount': a['loops'] * count},
                {'profile': 'B', 'startUs': da + gap, 'endUs': da + gap + db, 'chirpCount': b['loops'] * count}]}


def calculate(profile_a=None, profile_b=None, options=None):
    settings = {**DEFAULTS, **(options if isinstance(options, dict) else {})}
    out = {'valid': False, 'options': settings, 'profiles': {'A': None, 'B': None},
           'metrics': dict.fromkeys(METRIC_NAMES), 'errors': [], 'warnings': [], 'checks': [],
           'assumptions': list(ASSUMPTIONS), 'sources': deepcopy(SOURCES),
           'previewEvents': [], 'previewGroups': [], 'timelineBlocks': [], 'profileSpans': []}

    def add(key, label, status, detail):
        out['checks'].append({'key': key, 'label': label, 'status': status, 'detail': detail})
        if status == 'fail': out['errors'].append(detail)
        if status == 'warn': out['warnings'].append(detail)

    if options is not None and not isinstance(options, dict):
        add('options', 'A/B模式', 'fail', 'A/B编排参数必须是对象。'); return out
    if options and set(options) - set(DEFAULTS):
        add('options', 'A/B模式', 'fail', 'A/B编排含未知字段。'); return engine._clean(out)
    a0 = engine.calculate(profile_a); out['profiles']['A'] = _profile(a0, {})
    if type(settings['enabled']) is not bool:
        add('enabled', 'A/B 模式', 'fail', 'A/B enabled 必须为布尔值。'); return engine._clean(out)
    if not settings['enabled']:
        out.update(valid=a0['valid'], errors=list(a0['errors']), warnings=list(a0['warnings']), checks=list(a0['checks']))
        if a0['valid']:
            p, m = a0['params'], a0['metrics']
            out['metrics'] = {'burstMs': m['burstMs'], 'frameMs': p['frameMs'], 'frameRateHz': m['frameRateHz'],
                              'frameIdleMs': p['frameMs'] - m['burstMs'], 'chirpCount': m['chirpCount'],
                              'frameSamples': m['frameSamples'], 'dataRateMBps': m['dataRateMBps'],
                              'txRampDutyPercent': m['dutyPercent'], 'perTxDutyPercent': m['perTxDutyPercent'],
                              'totalTxDutyPercent': m['totalTxDutyPercent'], 'slowTimeIntervalAUs': m['txIntervalUs'],
                              'slowTimeIntervalBUs': None, 'pairIntervalUs': None, 'switchCount': 0, 'switchTimeMs': 0}
            count = p['txCount'] if p['mode'] == 'tdm' else 1
            events = [_event('A', p, r, tx, (r * count + tx) * m['chirpUs'])
                      for r in range(min(p['loops'], 2)) for tx in range(count)][:12]
            out['previewEvents'] = events
            out['previewGroups'] = [{'profile': 'A', 'events': events[:6], 'startUs': 0,
                                     'endUs': m['burstMs'] * 1000, 'spanMs': m['burstMs'],
                                     'chirpCount': m['chirpCount'], 'truncated': len(events) < m['chirpCount']}]
            out['timelineBlocks'] = [{'profile': 'A', 'startUs': 0, 'endUs': m['burstMs'] * 1000, 'chirpCount': m['chirpCount']}]
            out['profileSpans'] = deepcopy(out['timelineBlocks'])
        return engine._clean(out)
    if settings['layout'] not in ('blocks', 'alternating'):
        add('layout', 'A/B 调度', 'fail', 'A/B 调度仅支持 blocks 或 alternating。')
    if not engine._finite(settings['switchGapUs']) or not 0 <= settings['switchGapUs'] <= 1e6:
        add('gap', '切换间隔', 'fail', '切换间隔须为 0–1000000 μs 之间的有限数值。')
    b0 = engine.calculate(profile_b); out['profiles']['B'] = _profile(b0, {})
    for name, result in [('A', a0), ('B', b0)]:
        for check in result['checks']:
            add(name + '.' + check['key'], name + ' · ' + check['label'], check['status'], name + '：' + check['detail'])
        if not result['valid'] and result['errors'] and not any(c['status'] == 'fail' for c in result['checks']):
            add(name + '.invalid', name + ' 参数', 'fail', ' '.join(result['errors']))
    a, b = a0['params'], b0['params']
    for key in SHARED_KEYS:
        if a.get(key) != b.get(key): add('shared.' + key, '共享参数', 'fail', f'A/B 的 {key} 必须一致。')
    if settings['layout'] == 'alternating' and a['loops'] != b['loops']:
        add('loops', '交替轮数', 'fail', '整轮交替模式要求 A/B 每 Tx 的 chirp 数相同，才能使用均匀慢时间 FFT。')
    if out['errors']: return engine._clean(out)
    out['checks'], out['warnings'], out['errors'] = [], [], []
    count = a['txCount'] if a['mode'] == 'tdm' else 1
    ta, tb, gap = a['rampUs'] + a['idleUs'], b['rampUs'] + b['idleUs'], settings['switchGapUs']
    alternating = settings['layout'] == 'alternating'; pair = count * (ta + tb) + 2 * gap
    acq_a = {'slowTimeIntervalUs': pair} if alternating else {}
    acq_b = {'slowTimeIntervalUs': pair} if alternating else {}
    ra, rb = engine.calculate(a, acq_a), engine.calculate(b, acq_b)
    out['profiles'] = {'A': _profile(ra, acq_a), 'B': _profile(rb, acq_b)}
    for name, result in [('A', ra), ('B', rb)]:
        for check in result['checks']:
            add(name + '.' + check['key'], name + ' · ' + check['label'], check['status'], name + '：' + check['detail'])
    switches = 2 * a['loops'] - 1 if alternating else 1
    chirps = count * (a['loops'] + b['loops'])
    samples = count * a['rxCount'] * (a['loops'] * a['samples'] + b['loops'] * b['samples'])
    ramp = count * (a['loops'] * a['rampUs'] + b['loops'] * b['rampUs'])
    burst = count * (a['loops'] * ta + b['loops'] * tb) + switches * gap
    duty = ramp / (a['frameMs'] * 1000) * 100
    out['metrics'] = {'burstMs': burst / 1000, 'frameMs': a['frameMs'], 'frameRateHz': 1000 / a['frameMs'],
                      'frameIdleMs': a['frameMs'] - burst / 1000, 'chirpCount': chirps, 'frameSamples': samples,
                      'dataRateMBps': samples * 2 / (a['frameMs'] / 1000) / 1e6, 'txRampDutyPercent': duty,
                      'perTxDutyPercent': duty / (a['txCount'] if a['mode'] == 'tdm' else 1),
                      'totalTxDutyPercent': duty * (a['txCount'] if a['mode'] == 'ddm' else 1),
                      'slowTimeIntervalAUs': ra['metrics']['txIntervalUs'], 'slowTimeIntervalBUs': rb['metrics']['txIntervalUs'],
                      'pairIntervalUs': pair if alternating else None, 'switchCount': switches, 'switchTimeMs': switches * gap / 1000}
    if any(v is not None and not engine._finite(v) for v in out['metrics'].values()):
        out['metrics'] = dict.fromkeys(METRIC_NAMES); add('overflow', 'A/B 数值范围', 'fail', 'A/B 时序计算超出有限数值范围。'); return engine._clean(out)
    add('combinedFrame', 'A/B 总帧预算', 'pass' if burst <= a['frameMs'] * 1000 + 1e-6 else 'fail',
        f"A/B 总采集 {out['metrics']['burstMs']:.3f} ms（含 {switches} 次切换）/ 帧周期 {a['frameMs']:.3f} ms。")
    remaining = a['frameMs'] * 1000 - burst
    add('frameReturnGap', '跨帧 B→A 切换', 'pass' if remaining >= gap - 1e-6 else 'fail',
        f'帧末剩余 {remaining:.3f} μs，需覆盖下一帧 B→A 的 {gap:.3f} μs 切换间隔。')
    add('switchHardware', 'Profile 切换', 'warn', f'已计入每次 {gap:.3f} μs 切换间隔；该值须按具体芯片的切换与稳定时间验证。')
    if alternating:
        add('sameProfileInterval', 'A/B 慢时间间隔', 'pass', f'每组同波形同 Tx 重访间隔均为 {pair:.3f} μs；两组分别按自身中心频率计算速度指标。')
    else:
        add('profileBlocks', 'A/B 分段采集', 'pass', '两组在各自连续 burst 内独立执行慢时间 FFT；A/B 间隔不被错误计入组内速度分辨率。')
    out.update(_build_preview(a, b, settings)); out['valid'] = not out['errors']
    return engine._clean(out)


def sequence_preview(a=None, b=None, options=None, preview=None):
    result = calculate(a, b, options)
    keys = ('previewEvents', 'previewGroups', 'timelineBlocks', 'profileSpans')
    if not result['valid']:
        return {'valid': False, 'errors': result['errors'], **{key: [] for key in keys}}
    if not result['options']['enabled']:
        return {'valid': True, **{key: result[key] for key in keys}}
    value = (preview or {}).get('rounds', 2)
    rounds = max(1, min(2, value)) if type(value) is int else 2
    return {'valid': True, **_build_preview(result['profiles']['A']['params'], result['profiles']['B']['params'], result['options'], rounds)}
