"""Versioned waveform inputs, immutable manual records and bounded exports."""
from copy import deepcopy
import csv
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from arraylab.features.waveform.model import CONFIG_KEYS, LEGACY_CONFIG_KEYS, VIEWS, normalize_waveform_configuration

MAX_BYTES = 64 * 1024 * 1024


def _plain(value, depth=0):
    if depth > 40: raise ValueError('波形文件嵌套过深。')
    if value is None or type(value) in (str, bool, int): return value
    if type(value) is float:
        if not math.isfinite(value): raise ValueError('波形文件含非有限数值。')
        return value
    if isinstance(value, (list, tuple)): return [_plain(v, depth + 1) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: _plain(v, depth + 1) for k, v in value.items()}
    raise ValueError('波形文件包含不支持的数据类型。')


def _canonical(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError('JSON字段重复：' + key)
        result[key] = value
    return result


def _read(path):
    with Path(path).open('rb') as stream: content = stream.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES: raise ValueError('波形文件超过64 MiB。')
    try:
        value = json.loads(content.decode('utf-8-sig'), object_pairs_hook=_pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(ValueError('JSON数值必须有限。')))
        return _plain(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('无法读取计算记录JSON。') from exc


def _write(path, document):
    data = (json.dumps(_plain(document), ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if len(data) > MAX_BYTES: raise ValueError('波形文件超过64 MiB。')
    path = Path(path); temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.' + path.name + '.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name); stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def save_waveform_configuration(path, configuration):
    _write(path, {'format': 'arraylab.waveform', 'schema_version': 2,
                  'configuration': normalize_waveform_configuration(configuration)})


def load_waveform_configuration(path):
    document = _read(path)
    if type(document) is not dict: raise ValueError('波形方案须为JSON对象。')
    if document.get('app') == 'WaveLab':
        if type(document.get('schemaVersion')) is not int or document['schemaVersion'] not in (1, 2, 3):
            raise ValueError('只支持WaveLab v1/v2/v3方案。')
        if type(document.get('params')) is not dict: raise ValueError('WaveLab方案缺少params。')
        # The radar workbench owns these coordinates. Import never changes the
        # separate ArrayLab MIMO task and never starts a sparse search.
        selected = {k: document[k] for k in CONFIG_KEYS if k in document}
        if document['schemaVersion'] == 1:
            selected.update(abOptions={'enabled': False}, activeProfile='A')
        return normalize_waveform_configuration(selected)
    if (set(document) != {'format', 'schema_version', 'configuration'} or document.get('format') != 'arraylab.waveform'
            or type(document.get('schema_version')) is not int or document['schema_version'] not in (1, 2)):
        raise ValueError('不支持的波形参数文件类型或版本。')
    return normalize_waveform_configuration(document['configuration'])


def _series(rows, count, label):
    if type(rows) is not list or len(rows) != count: raise ValueError(label + '采样数量不匹配。')
    for row in rows:
        if type(row) is not dict or set(row) != {'x', 'y'} or any(type(row[k]) not in (int, float) for k in ('x', 'y')):
            raise ValueError(label + '须为有限x/y采样序列。')
    if any(rows[i]['x'] >= rows[i+1]['x'] for i in range(len(rows)-1)):
        raise ValueError(label + '坐标须严格递增。')


def validate_waveform_report(report):
    import arraylab.features.waveform.core as waveform_core; import arraylab.features.waveform.ab as waveform_ab
    result = _plain(report)
    expected = {'format', 'schema_version', 'app_version', 'configuration', 'analysis', 'spectra',
                'range_recovery', 'pri_recovery', 'execution', 'source', 'notices'}
    version = result.get('schema_version') if type(result) is dict else None
    if version == 2:
        expected |= {'antenna', 'gain_budget', 'sparse_design'}
    if type(result) is not dict or set(result) != expected or result['format'] != 'arraylab.waveform.result' or type(version) is not int or version not in (1, 2):
        raise ValueError('波形结果格式无效。')
    config = normalize_waveform_configuration(result['configuration'])
    comparison = {k: config[k] for k in LEGACY_CONFIG_KEYS} if version == 1 else config
    if comparison != result['configuration']: raise ValueError('波形结果配置不完整或不一致。')
    analysis = result['analysis']
    if type(analysis) is not dict or analysis.get('valid') is not True or analysis.get('options') != config['abOptions']:
        raise ValueError('波形结果缺少有效时序分析。')
    if type(analysis.get('metrics')) is not dict or set(analysis['metrics']) != set(waveform_ab.METRIC_NAMES):
        raise ValueError('波形结果缺少完整帧时序指标。')
    if any(value is not None and type(value) not in (int, float) for value in analysis['metrics'].values()):
        raise ValueError('帧时序指标须为有限数值或null。')
    active = ('A', 'B') if config['abOptions']['enabled'] else ('A',)
    if type(result['spectra']) is not dict or set(result['spectra']) != set(active) or type(result['range_recovery']) is not dict or set(result['range_recovery']) != set(active):
        raise ValueError('波形结果A/B组不匹配。')
    profiles = analysis.get('profiles')
    if type(profiles) is not dict or set(profiles) != {'A', 'B'}: raise ValueError('波形结果缺少两组定义。')
    if not config['abOptions']['enabled'] and profiles['B'] is not None:
        raise ValueError('未启用B组的记录不应含B组计算结果。')
    for name in active:
        p = config['params' if name == 'A' else 'profileB']
        entry = profiles[name]
        if type(entry) is not dict or type(entry.get('result')) is not dict or entry.get('params') != p or entry['result'].get('params') != p or entry['result'].get('valid') is not True:
            raise ValueError('波形结果与输入快照不一致。')
        metrics = entry['result'].get('metrics')
        scalar_names = set(waveform_core.METRIC_NAMES) | {'nominalTxIntervalUs', 'observationSpanMs'}
        if type(metrics) is not dict or set(metrics) != scalar_names | {'ddmPhasesDeg'} or any(type(metrics[key]) not in (int, float) for key in scalar_names):
            raise ValueError('波形结果缺少完整有限指标。')
        phases = [360 * k / p['ddmSlots'] for k in range(p['txCount'])] if p['mode'] == 'ddm' else []
        if type(metrics['ddmPhasesDeg']) is not list or metrics['ddmPhasesDeg'] != phases:
            raise ValueError('DDM逐通道相位步进与参数不一致。')
        spectrum = result['spectra'][name]
        if type(spectrum) is not dict or spectrum.get('valid') is not True: raise ValueError('缺少有效频谱。')
        _series(spectrum.get('rangeProfile'), int(p['rangeFft']) // 2, name + '距离谱')
        _series(spectrum.get('dopplerProfile'), int(p['dopplerFft']), name + '多普勒谱')
        _series(spectrum.get('baseDopplerProfile'), int(p['dopplerFft']), name + '基准速度谱')
        grid = spectrum.get('rangeDoppler', {})
        if (type(grid) is not dict or type(grid.get('ranges')) is not list or type(grid.get('velocities')) is not list
                or grid.get('width') != len(grid['ranges']) or grid.get('height') != len(grid['velocities'])
                or type(grid.get('values')) is not list):
            raise ValueError('距离–速度网格尺寸无效。')
        if grid['width'] != min(int(p['rangeFft']) // 2, 128) or grid['height'] != min(int(p['dopplerFft']), 64):
            raise ValueError('距离–速度网格与FFT采样不一致。')
        for values in (grid['ranges'], grid['velocities']):
            if any(type(v) not in (int, float) for v in values) or any(values[i] >= values[i+1] for i in range(len(values)-1)):
                raise ValueError('距离–速度坐标须为递增有限数值。')
        if len(grid['values']) != grid['width'] * grid['height']:
            raise ValueError('距离–速度网格数值数量不匹配。')
        if any(type(v) not in (float, int) for v in grid['values']): raise ValueError('距离–速度网格必须为有限数值。')
        recovery = result['range_recovery'][name]
        if type(recovery) is not dict or recovery.get('status') not in ('off', 'blocked', 'resolved', 'ambiguous', 'inconsistent'):
            raise ValueError('ADC解模糊状态无效。')
    if type(result['pri_recovery']) is not dict or result['pri_recovery'].get('status') not in ('off', 'blocked', 'resolved', 'ambiguous', 'inconsistent'):
        raise ValueError('跨周期解模糊状态无效。')
    if type(result['notices']) is not list or any(type(v) is not str for v in result['notices']): raise ValueError('波形说明无效。')
    if version == 2:
        _validate_radar_results(result, config, active)
    return result


def _validate_radar_results(report, config, active):
    antenna = report['antenna']
    if type(antenna) is not dict or set(antenna) != set(active):
        raise ValueError('雷达天线结果的A/B组不匹配。')
    for item in [*antenna.values(), report['gain_budget']]:
        if type(item) is not dict or type(item.get('valid')) is not bool:
            raise ValueError('雷达天线或增益预算状态无效。')
    from .settings import normalize_geometry
    for item in antenna.values():
        if not item['valid']:
            if type(item.get('errors')) is not list or not item['errors']:
                raise ValueError('无效天线结果缺少原因。')
            continue
        geometry = normalize_geometry({'tx': item.get('tx'), 'rx': item.get('rx')})
        count = len(geometry['tx']) * len(geometry['rx'])
        virtual = item.get('virtual')
        if (item.get('channelCount') != count or type(item.get('uniqueCount')) is not int or
                type(virtual) is not list or not virtual or len(virtual) != item['uniqueCount']):
            raise ValueError('天线通道统计与坐标不一致。')
        if any(type(v) is not dict or set(v) != {'x', 'z', 'multiplicity'} or
               type(v['x']) not in (int, float) or type(v['z']) not in (int, float) or
               type(v['multiplicity']) is not int or v['multiplicity'] < 1 for v in virtual):
            raise ValueError('虚拟阵列位置或重数无效。')
        if sum(v['multiplicity'] for v in virtual) != count:
            raise ValueError('虚拟通道重数不匹配。')
        for key in ('azimuth', 'elevation'): _series(item.get(key), 721, key)
        for key in ('beamwidthAzDeg', 'beamwidthElDeg'):
            if key not in item or item[key] is not None and (type(item[key]) not in (int, float) or item[key] <= 0):
                raise ValueError('天线波束宽度格式无效。')
    gain = report['gain_budget']
    if gain.get('scope') != config['antennaDesign']['gainScope']:
        raise ValueError('增益预算参考范围与配置不一致。')
    wanted = list(active) if gain['scope'] == 'all' else [config['activeProfile']]
    inputs = gain.get('waveformInputs')
    if type(inputs) is not list or [p.get('name') for p in inputs if type(p) is dict] != wanted:
        raise ValueError('增益预算缺少参考波形。')
    for item in inputs:
        reference = report['analysis']['profiles'][item['name']]
        if item.get('params') != reference['params'] or item.get('acquisition') != reference['acquisition']:
            raise ValueError('增益预算与波形快照不一致。')
    if gain.get('options') != config['antennaDesign']['gainOptions']:
        raise ValueError('增益预算与输入设置不一致。')
    if gain['valid']:
        if gain.get('status') not in ('designed', 'waveform_limited') or gain.get('limitingProfile') not in wanted:
            raise ValueError('增益预算状态或限制组无效。')
        for key in ('requiredTxGainDbi', 'requiredRxGainDbi', 'requiredGainSumDb', 'requiredSnrWithMarginDb'):
            if type(gain.get(key)) not in (int, float): raise ValueError('增益预算缺少有效增益或SNR。')
        profiles = gain.get('profiles')
        if type(profiles) is not list or [p.get('name') for p in profiles if type(p) is dict] != wanted:
            raise ValueError('增益预算结果组不一致。')
        for profile in profiles:
            if profile.get('valid') is not True or type(profile.get('observable')) is not bool:
                raise ValueError('增益预算可观测状态无效。')
            for key in ('requiredTxGainDbi', 'requiredRxGainDbi', 'energyTimeMs', 'windowLossDb', 'rxCombiningGainDb', 'deliveredSnrDb'):
                if type(profile.get(key)) not in (int, float): raise ValueError('增益预算逐组数值无效。')
    sparse = report['sparse_design']
    if sparse is None:
        return
    if (type(sparse) is not dict or type(sparse.get('valid')) is not bool or
            sparse.get('action') not in ('sparse', 'evaluate') or
            type(sparse.get('referenceFrequencyGHz')) not in (int, float) or sparse['referenceFrequencyGHz'] <= 0):
        raise ValueError('稀疏设计记录状态无效。')
    if sparse['valid']:
        options = sparse.get('options')
        if type(options) is not dict or any(options.get(k) != v for k, v in config['antennaDesign']['sparseOptions'].items()):
            raise ValueError('稀疏设计结果与输入设置不一致。')
        geometry = normalize_geometry(sparse.get('geometry'))
        if geometry is None or len(geometry['tx']) != options['txCount'] or len(geometry['rx']) != options['rxCount']:
            raise ValueError('稀疏设计坐标数量与通道不一致。')
        if type(sparse.get('metrics')) is not dict or type(sparse.get('checks')) is not list:
            raise ValueError('稀疏设计记录缺少指标或检查结果。')
        if sparse.get('status') not in ('met', 'unmet'):
            raise ValueError('稀疏设计达标状态无效。')
        metrics = sparse['metrics']
        for key in ('channelCount', 'uniqueCount', 'physicalApertureLambda', 'virtualApertureLambda', 'sparseFillRatio', 'worstSidelobeDb'):
            if type(metrics.get(key)) not in (int, float): raise ValueError('稀疏设计缺少有效指标：' + key)
        for key in ('worstHpbwDeg', 'hpbwDeg'):
            if key not in metrics or metrics[key] is not None and type(metrics[key]) not in (int, float):
                raise ValueError('稀疏设计波束宽度无效。')
        if metrics['channelCount'] != options['txCount'] * options['rxCount']:
            raise ValueError('稀疏设计通道数不一致。')
        if type(metrics['uniqueCount']) is not int or not 1 <= metrics['uniqueCount'] <= metrics['channelCount']:
            raise ValueError('稀疏设计独立位置统计无效。')
        if not 0 <= metrics['sparseFillRatio'] <= 1:
            raise ValueError('虚拟格点占用率无效。')
        for key in ('pattern', 'twoTargetPattern'):
            rows = sparse.get(key)
            if type(rows) is not list: raise ValueError('稀疏设计缺少图形数据。')
            _series(rows, len(rows), key)
        if len(sparse['pattern']) != 1801:
            raise ValueError('稀疏物理方向图采样数量不匹配。')
        scans = sparse.get('scanChecks')
        if type(scans) is not list or (not scans and (sparse.get('superresolution') or {}).get('valid') is not False):
            raise ValueError('稀疏设计缺少扫描验证。')
        for row in scans:
            if type(row) is not dict or type(row.get('centerDeg')) not in (int, float) or type(row.get('resolved')) is not bool:
                raise ValueError('稀疏扫描结果无效。')
        sr = sparse.get('superresolution')
        if options['criterion'] in ('music', 'mvdr', 'esprit', 'dml'):
            if type(sr) is not dict or sr.get('method') != options['criterion'] or type(sr.get('valid')) is not bool:
                raise ValueError('高分辨验证方法或状态无效。')
            if type(sr.get('validation')) is not list or type(sr.get('warnings')) is not list:
                raise ValueError('高分辨验证缺少逐角结果。')
            if sr['valid'] and (type(sr.get('successRate')) not in (int, float) or not 0 <= sr['successRate'] <= 1 or not sr['validation']):
                raise ValueError('高分辨经验通过率无效。')
            for key in ('rmseDeg', 'worstErrorDeg', 'effectiveToleranceDeg'):
                if sr.get(key) is not None and type(sr[key]) not in (int, float):
                    raise ValueError('高分辨汇总数值无效。')
            for case in sr['validation']:
                if type(case) is not dict or type(case.get('centerDeg')) not in (int, float):
                    raise ValueError('高分辨逐角记录无效。')
                for key in ('successRate', 'rmseDeg', 'worstErrorDeg'):
                    if case.get(key) is not None and type(case[key]) not in (int, float):
                        raise ValueError('高分辨逐角统计数值无效。')
                if case.get('successRate') is not None and not 0 <= case['successRate'] <= 1:
                    raise ValueError('高分辨逐角通过率超界。')
            rep = sr.get('representative')
            if rep is not None:
                if (type(rep) is not dict or type(rep.get('subset', {})) is not dict or
                        rep.get('diagnostics') is not None and type(rep['diagnostics']) is not dict):
                    raise ValueError('高分辨代表试验格式无效。')
                for key in ('truth', 'truthAngles', 'estimates', 'errors'):
                    if key in rep and (type(rep[key]) is not list or any(type(x) not in (int, float) for x in rep[key])):
                        raise ValueError('高分辨角度结果格式无效。')
                for group, keys in ((rep.get('subset') or {}, ('aperture', 'processedAperture')),
                                    (rep.get('diagnostics') or {}, ('residualPower', 'conditionNumber'))):
                    for key in keys:
                        if group.get(key) is not None and type(group[key]) not in (int, float):
                            raise ValueError('高分辨算法诊断数值无效。')
        elif sr is not None:
            raise ValueError('常规波束判据不应含高分辨算法结果。')
        for row in sparse['checks']:
            if type(row) is not dict or any(type(row.get(k)) is not str for k in ('label', 'status', 'detail')):
                raise ValueError('稀疏检查项格式无效。')


def _record_payload(report, view, comparisons, profile):
    saved = validate_waveform_report(report)
    profile = profile or saved['configuration']['activeProfile']
    if view not in VIEWS or profile not in saved['spectra']: raise ValueError('波形记录视图或分析组无效。')
    if comparisons is None: comparisons = []
    if type(comparisons) is not list or len(comparisons) > 8: raise ValueError('波形对比最多8项。')
    items = []
    for row in comparisons:
        if type(row) is not dict or set(row) != {'name', 'report'} or type(row['name']) is not str or not 1 <= len(row['name']) <= 100:
            raise ValueError('波形对比名称或字段无效。')
        items.append({'name': row['name'], 'report': validate_waveform_report(row['report'])})
    return {'report': saved, 'view': view, 'profile': profile, 'comparisons': items}


def save_waveform_record(path, report, view='overview', comparisons=None, profile=None):
    payload = _record_payload(report, view, comparisons, profile)
    _write(path, {'format': 'arraylab.waveform.record', 'schema_version': 1,
                  **payload, 'sha256': hashlib.sha256(_canonical(payload)).hexdigest()})


def load_waveform_record(path):
    d = _read(path)
    if type(d) is not dict or set(d) != {'format', 'schema_version', 'report', 'view', 'profile', 'comparisons', 'sha256'} or d['format'] != 'arraylab.waveform.record' or type(d['schema_version']) is not int or d['schema_version'] != 1:
        raise ValueError('波形记录类型或版本无效。')
    payload = {k: d[k] for k in ('report', 'view', 'profile', 'comparisons')}
    digest = d['sha256']
    if type(digest) is not str or not hmac.compare_digest(digest, hashlib.sha256(_canonical(payload)).hexdigest()):
        raise ValueError('波形记录摘要校验失败。')
    return _record_payload(d['report'], d['view'], d['comparisons'], d['profile'])


def is_waveform_record(path):
    d = _read(path)
    return type(d) is dict and d.get('format') == 'arraylab.waveform.record'


def _csv(path, columns, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(columns); writer.writerows(rows)


def export_waveform_directory(path, report, plot='all'):
    if plot not in ('none', 'all', 'overview', 'phase', 'spectra', 'rd', 'range_recovery', 'pri_recovery', 'antenna', 'sparse_design', 'gain_budget'):
        raise ValueError('波形导出图形类型无效。')
    saved = validate_waveform_report(report)
    target = Path(path).expanduser().absolute()
    if target.exists() or target.is_symlink(): raise FileExistsError('输出目录已存在，未覆盖。')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / ('.arraylab-waveform-' + uuid.uuid4().hex); staging.mkdir()
    try:
        _write(staging/'summary.json', saved)
        save_waveform_configuration(staging/'parameters.json', saved['configuration'])
        save_waveform_record(staging/'waveform-record.json', saved)
        for name, spectrum in saved['spectra'].items():
            _csv(staging/f'{name}-range.csv', ['apparent_range_m', 'normalized_amplitude_db'], ([p['x'], p['y']] for p in spectrum['rangeProfile']))
            _csv(staging/f'{name}-doppler.csv', ['velocity_mps', 'velocity_kmh', 'normalized_amplitude_db'], ([p['x'], p['x']*3.6, p['y']] for p in spectrum['dopplerProfile']))
            g = spectrum['rangeDoppler']
            _csv(staging/f'{name}-range-doppler.csv', ['range_m', 'velocity_mps', 'normalized_amplitude_db'],
                 ([x, y, g['values'][j*g['width']+i]] for j, y in enumerate(g['velocities']) for i, x in enumerate(g['ranges'])))
            _write(staging/f'{name}-range-recovery.json', saved['range_recovery'][name])
        _write(staging/'pri-recovery.json', saved['pri_recovery'])
        for key in ('antenna', 'sparse_design', 'gain_budget'):
            if key in saved and saved[key] is not None:
                _write(staging/(key.replace('_', '-') + '.json'), saved[key])
        if plot != 'none':
            from arraylab.features.waveform.plotting import render_waveform_plots
            render_waveform_plots(staging, saved, mode=plot)
        if target.exists() or target.is_symlink(): raise FileExistsError('输出目录已被创建，未覆盖。')
        staging.rename(target)
    finally:
        if staging.exists():
            for item in staging.iterdir(): item.unlink()
            staging.rmdir()
    return target
