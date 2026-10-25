"""Versioned MIMO files and transactional exports; no GUI dependencies."""
from dataclasses import asdict
import csv
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import uuid

import numpy as np

MAX_FILE_BYTES = 64 * 1024 * 1024
CONFIG_FORMAT = 'arraylab.mimo'
RECORD_FORMAT = 'arraylab.mimo.record'
VIEWS = frozenset(('layout', 'cut', 'map', 'channels', 'unique'))


def _plain(value, depth=0):
    """Reject non-finite or non-JSON values before a destination is touched."""
    if depth > 40:
        raise ValueError('MIMO文件嵌套过深。')
    if isinstance(value, np.ndarray):
        return _plain(value.tolist(), depth + 1)
    if isinstance(value, np.generic):
        return _plain(value.item(), depth + 1)
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('MIMO文件包含非有限数值。')
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key: _plain(item, depth + 1) for key, item in value.items()}
    raise ValueError('MIMO文件包含不支持的数据类型。')


def _canonical(value):
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def _encoded(value):
    data = (json.dumps(_plain(value), ensure_ascii=False, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')
    if len(data) > MAX_FILE_BYTES:
        raise ValueError('MIMO文件超过64 MiB上限。')
    return data


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('MIMO文件存在重复字段：' + key)
        result[key] = value
    return result


def _read(path):
    path = Path(path)
    with path.open('rb') as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError('MIMO文件超过64 MiB上限。')
    try:
        value = json.loads(data.decode('utf-8-sig'), object_pairs_hook=_unique_object,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError('非有限JSON数值：' + value)))
        return _plain(value)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('无法读取MIMO JSON文件。') from exc


def _keys(value, expected, label):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(label + '字段不完整或包含未知字段。')


def _version(document, format_name):
    if document.get('format') != format_name or type(document.get('schema_version')) is not int or document['schema_version'] != 1:
        raise ValueError('不支持的MIMO文件类型或版本。')


def _atomic_write(path, data):
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.' + target.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_mimo_configuration(path, parameters):
    parameters.validate()
    document = {'format': CONFIG_FORMAT, 'schema_version': 1,
                'parameters': asdict(parameters)}
    _atomic_write(path, _encoded(document))


def load_mimo_configuration(path):
    from .mimo import mimo_parameters_from_dict
    document = _read(path)
    _keys(document, ('format', 'schema_version', 'parameters'), 'MIMO参数')
    _version(document, CONFIG_FORMAT)
    return mimo_parameters_from_dict(document['parameters'])


def _same(actual, expected, label):
    """Geometry recomputation only; saved fields and metrics are never replaced."""
    if isinstance(expected, np.ndarray): expected = expected.tolist()
    if isinstance(expected, np.generic): expected = expected.item()
    if type(expected) is dict:
        _keys(actual, expected, label)
        for key in expected: _same(actual[key], expected[key], label + '.' + key)
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not list or len(actual) != len(expected):
            raise ValueError(label + '形状不一致。')
        for a, e in zip(actual, expected): _same(a, e, label)
    elif type(expected) in (int, float):
        if type(actual) not in (int, float) or not math.isfinite(actual):
            raise ValueError(label + '须为有限数值。')
        if type(expected) is int and (type(actual) is not int or actual != expected):
            raise ValueError(label + '整数值不一致。')
        if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(label + '与参数不一致。')
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(label + '与参数不一致。')


def _numbers(value, label, *, nullable=False):
    if type(value) is not list:
        raise ValueError(label + '须为数组。')
    for item in value:
        if nullable and item is None: continue
        if type(item) not in (int, float) or not math.isfinite(item):
            raise ValueError(label + '须为有限数值数组。')
    return np.asarray(value, dtype=float)


def validate_mimo_report(report):
    """Validate saved structure and physical geometry without evaluating a field."""
    from .mimo import mimo_geometry, mimo_parameters_from_dict
    report = _plain(report)
    _keys(report, ('format', 'schema_version', 'app_version', 'parameters', 'counts',
                   'geometry', 'patterns', 'metrics', 'model', 'aggregation', 'notices', 'execution'), 'MIMO结果')
    _version(report, 'arraylab.mimo.result')
    if type(report['app_version']) is not str or not report['app_version']:
        raise ValueError('MIMO结果缺少程序版本。')
    parameters = mimo_parameters_from_dict(report['parameters'])
    _keys(report['parameters'], asdict(parameters), 'MIMO结果参数')
    expected = mimo_geometry(parameters)
    for key in ('counts', 'geometry', 'aggregation'):
        _same(report[key], expected[key], 'MIMO.' + key)
    _validate_patterns(report['patterns'], parameters)
    if type(report['metrics']) is not dict or type(report['metrics'].get('cut')) is not dict:
        raise ValueError('MIMO结果缺少波束指标。')
    for key in ('gain_dbi', 'directivity_dbi'):
        if key not in report['metrics'] or report['metrics'][key] is not None:
            raise ValueError('MIMO虚拟响应不能记录为物理增益或辐射方向性。')
    widths = report['metrics'].get('orthogonal_beamwidths_deg')
    if type(widths) is not list or len(widths) != 2:
        raise ValueError('MIMO正交波束宽度形状不正确。')
    for width in widths:
        if width is not None and (type(width) not in (int, float) or not 0 <= width <= 360):
            raise ValueError('MIMO正交波束宽度无效。')
    if type(report['notices']) is not list or any(type(item) is not str for item in report['notices']):
        raise ValueError('MIMO模型说明无效。')
    if type(report['execution']) is not dict or type(report['model']) not in (str, dict):
        raise ValueError('MIMO执行/模型信息无效。')
    return report


def _validate_patterns(patterns, parameters):
    _keys(patterns, ('cut', 'map'), 'MIMO方向图')
    cut = patterns['cut']
    _keys(cut, ('angles_deg', 'tx_db', 'rx_db', 'virtual_db', 'phi_deg'), 'MIMO切面')
    angles = _numbers(cut['angles_deg'], '切面角度')
    from .mimo import mimo_cut_angles
    expected_angles, _ = mimo_cut_angles(parameters)
    if angles.shape != expected_angles.shape or not np.allclose(angles, expected_angles, atol=1e-9, rtol=0):
        raise ValueError('MIMO切面角度范围或采样顺序无效。')
    _same(cut['phi_deg'], parameters.cut_phi_deg, 'MIMO切面方位角')
    for name in ('tx_db', 'rx_db', 'virtual_db'):
        values = _numbers(cut[name], name)
        if len(values) != len(angles) or np.any(values < -300.00001) or np.any(values > 1e-8):
            raise ValueError('MIMO切面幅度形状或范围无效。')
    mapping = patterns['map']
    _keys(mapping, ('axis_u', 'axis_v', 'virtual_db'), 'MIMO二维方向图')
    axis = np.linspace(-1, 1, parameters.map_size)
    for key in ('axis_u', 'axis_v'):
        values = _numbers(mapping[key], key)
        if values.shape != axis.shape or not np.allclose(values, axis, rtol=0, atol=1e-12):
            raise ValueError('MIMO二维图坐标与参数不一致。')
    values = mapping['virtual_db']
    if type(values) is not list or len(values) != len(axis):
        raise ValueError('MIMO二维图行数不一致。')
    for y, row in zip(axis, values):
        if type(row) is not list or len(row) != len(axis):
            raise ValueError('MIMO二维图列数不一致。')
        for x, value in zip(axis, row):
            visible = x*x + y*y <= 1 + 1e-14
            if not visible:
                if value is not None: raise ValueError('MIMO二维图可见域外须为null。')
            elif type(value) not in (int, float) or not math.isfinite(value) or not -300.00001 <= value <= 1e-8:
                raise ValueError('MIMO二维图可见域内须为有限幅度。')


def save_mimo_record(path, report, view='layout'):
    if type(view) is not str or view not in VIEWS:
        raise ValueError('未知MIMO记录视图。')
    payload = {'report': validate_mimo_report(report), 'view': view}
    document = {'format': RECORD_FORMAT, 'schema_version': 1,
                **payload, 'sha256': hashlib.sha256(_canonical(payload)).hexdigest()}
    _atomic_write(path, _encoded(document))


def load_mimo_record(path):
    document = _read(path)
    _keys(document, ('format', 'schema_version', 'report', 'view', 'sha256'), 'MIMO记录')
    _version(document, RECORD_FORMAT)
    if type(document['view']) is not str or document['view'] not in VIEWS:
        raise ValueError('未知MIMO记录视图。')
    payload = {'report': document['report'], 'view': document['view']}
    digest = document['sha256']
    if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None or not hmac.compare_digest(digest, hashlib.sha256(_canonical(payload)).hexdigest()):
        raise ValueError('MIMO记录摘要校验失败。')
    return {'report': validate_mimo_report(document['report']), 'view': document['view']}


def is_mimo_record(path):
    """Inspect the bounded parsed document, never a filename or text substring."""
    document = _read(path)
    return type(document) is dict and document.get('format') == RECORD_FORMAT


def _csv(path, columns, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream); writer.writerow(columns); writer.writerows(rows)


def _write_exports(directory, report, plot):
    (directory / 'summary.json').write_bytes(_encoded(report))
    from .mimo import mimo_parameters_from_dict
    save_mimo_configuration(directory / 'parameters.json', mimo_parameters_from_dict(report['parameters']))
    save_mimo_record(directory / 'mimo-record.json', report)
    geometry = report['geometry']
    for side in ('tx', 'rx'):
        data = geometry[side]
        _csv(directory / f'{side}.csv', ('element', 'x_lambda', 'y_lambda', 'weight'),
             ((index, *xy, weight) for index, (xy, weight) in enumerate(zip(data['positions_lambda'], data['weights']), 1)))
    for name, key, columns in (
            ('pairs.csv', 'pairs', ('channel', 'tx', 'rx', 'x_lambda', 'y_lambda', 'weight', 'unique_index')),
            ('unique.csv', 'virtual_unique', ('index', 'x_lambda', 'y_lambda', 'multiplicity', 'weight'))):
        _csv(directory / name, columns, ([item[column] for column in columns] for item in geometry[key]))
    cut = report['patterns']['cut']
    _csv(directory / 'cut.csv', ('angle_deg', 'tx_db', 'rx_db', 'virtual_db'),
         zip(cut['angles_deg'], cut['tx_db'], cut['rx_db'], cut['virtual_db']))
    mapping = report['patterns']['map']
    _csv(directory / 'map.csv', ('u', 'v', 'virtual_db'),
         ((u, v, value) for v, row in zip(mapping['axis_v'], mapping['virtual_db'])
          for u, value in zip(mapping['axis_u'], row)))
    if plot != 'none':
        from .mimo_plot import render_mimo_plots
        render_mimo_plots(directory, report, mode=plot)


def export_mimo_directory(path, report, plot='all'):
    if plot not in ('none', 'layout', 'cut', 'map', 'all'):
        raise ValueError('未知MIMO导出图形模式。')
    target = Path(path).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError('输出目录已存在，未覆盖：' + str(target))
    report = validate_mimo_report(report)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name('.' + target.name + '.mimo-' + uuid.uuid4().hex)
    staging.mkdir(exist_ok=False)
    try:
        _write_exports(staging, report, plot)
        if target.exists() or target.is_symlink():
            raise FileExistsError('输出目录已存在，未覆盖：' + str(target))
        staging.rename(target)
    finally:
        if staging.exists():
            if staging.resolve().parent != target.parent.resolve() or not staging.name.startswith('.' + target.name + '.mimo-'):
                raise RuntimeError('导出临时目录不在预期位置。')
            shutil.rmtree(staging)
    return target
