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

from .waveform import CONFIG_KEYS, VIEWS, normalize_waveform_configuration

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
    _write(path, {'format': 'arraylab.waveform', 'schema_version': 1,
                  'configuration': normalize_waveform_configuration(configuration)})


def load_waveform_configuration(path):
    document = _read(path)
    if type(document) is not dict: raise ValueError('波形方案须为JSON对象。')
    if document.get('app') == 'WaveLab':
        if type(document.get('schemaVersion')) is not int or document['schemaVersion'] not in (1, 2):
            raise ValueError('只支持WaveLab v1/v2方案，未读取其他版本。')
        if type(document.get('params')) is not dict: raise ValueError('WaveLab方案缺少params。')
        # Only the user-selected waveform domains are imported. Antenna geometry
        # is not relabeled or silently applied to ArrayLab's separate MIMO task.
        selected = {k: document[k] for k in CONFIG_KEYS if k in document}
        if document['schemaVersion'] == 1:
            selected.update(abOptions={'enabled': False}, activeProfile='A')
        return normalize_waveform_configuration(selected)
    if (set(document) != {'format', 'schema_version', 'configuration'} or document.get('format') != 'arraylab.waveform'
            or type(document.get('schema_version')) is not int or document['schema_version'] != 1):
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
    from . import waveform_core, waveform_ab
    result = _plain(report)
    expected = {'format', 'schema_version', 'app_version', 'configuration', 'analysis', 'spectra',
                'range_recovery', 'pri_recovery', 'execution', 'source', 'notices'}
    if type(result) is not dict or set(result) != expected or result['format'] != 'arraylab.waveform.result' or type(result['schema_version']) is not int or result['schema_version'] != 1:
        raise ValueError('波形结果格式无效。')
    config = normalize_waveform_configuration(result['configuration'])
    if config != result['configuration']: raise ValueError('波形结果配置不完整或不一致。')
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
    return result


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
    if plot not in ('none', 'all', 'overview', 'phase', 'spectra', 'rd', 'range_recovery', 'pri_recovery'):
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
        if plot != 'none':
            from .waveform_plot import render_waveform_plots
            render_waveform_plots(staging, saved, mode=plot)
        if target.exists() or target.is_symlink(): raise FileExistsError('输出目录已被创建，未覆盖。')
        staging.rename(target)
    finally:
        if staging.exists():
            for item in staging.iterdir(): item.unlink()
            staging.rmdir()
    return target
