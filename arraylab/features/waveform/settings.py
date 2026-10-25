"""Versioned radar antenna inputs; loading settings never starts an optimizer."""
from copy import deepcopy
import math


def default_antenna_design():
    from .sparse_design import DEFAULTS as sparse_defaults
    from .gain import DEFAULTS as gain_defaults
    return {'version': 1, 'section': 'layout', 'sparseOptions': deepcopy(sparse_defaults),
            'gainOptions': deepcopy(gain_defaults), 'gainScope': 'active'}


def _options(value, defaults, label):
    if type(value) is not dict or set(value) - set(defaults):
        raise ValueError(label + '字段无效。')
    result = deepcopy(defaults)
    for key, item in value.items():
        sample = defaults[key]
        if isinstance(sample, str):
            if type(item) is not str or len(item) > 128:
                raise ValueError(label + '.' + key + '文本无效。')
        elif type(item) not in (int, float) or not math.isfinite(item):
            raise ValueError(label + '.' + key + '须为有限数值。')
        elif type(sample) is int and float(item).is_integer():
            item = int(item)
        result[key] = item
    return result


def normalize_antenna_design(value=None):
    defaults = default_antenna_design()
    if value is None:
        return defaults
    if type(value) is not dict or set(value) - set(defaults):
        raise ValueError('天线设计工作台字段无效。')
    version = value.get('version', 1)
    if type(version) is not int or version != 1:
        raise ValueError('天线设计工作台版本无效。')
    result = {**defaults, **deepcopy(value)}
    if result['section'] not in ('layout', 'sparse', 'gain') or result['gainScope'] not in ('active', 'all'):
        raise ValueError('天线工作区或参考波形无效。')
    for key in ('sparseOptions', 'gainOptions'):
        result[key] = _options(value.get(key, {}), defaults[key], key)
    choices = {'criterion': ('two_target', 'hpbw', 'music', 'mvdr', 'esprit', 'dml'),
               'axis': ('azimuth', 'elevation'), 'smoothing': ('none', 'spatial')}
    for key, options in choices.items():
        if result['sparseOptions'][key] not in options:
            raise ValueError('稀疏综合选项无效：' + key)
    for key, options in {'gainMode': ('equal', 'solve_rx', 'solve_tx'),
                         'powerBasis': ('per_tx', 'total_active'),
                         'combining': ('coherent_rx', 'single_rx')}.items():
        if result['gainOptions'][key] not in options:
            raise ValueError('增益预算选项无效：' + key)
    return result


def normalize_geometry(value):
    if value is None:
        return None
    allowed = {'tx', 'rx', 'steerAzDeg', 'steerElDeg', 'referenceFrequencyGHz'}
    if type(value) is not dict or set(value) - allowed or not {'tx', 'rx'} <= set(value):
        raise ValueError('天线坐标须包含Tx和Rx列表。')
    result = deepcopy(value)
    for key, limit in (('tx', 12), ('rx', 16)):
        rows = result[key]
        if type(rows) is not list or not 1 <= len(rows) <= limit:
            raise ValueError(f'{key}须有1–{limit}行坐标。')
        for row in rows:
            if (type(row) is not list or len(row) != 2 or
                    any(type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 100 for x in row)):
                raise ValueError('每行坐标须为±100λ以内的有限[x,z]。')
    for key in ('steerAzDeg', 'steerElDeg'):
        number = result.setdefault(key, 0)
        if type(number) not in (int, float) or not math.isfinite(number) or abs(number) > 80:
            raise ValueError('天线指向须在±80°以内。')
    if 'referenceFrequencyGHz' in result:
        number = result['referenceFrequencyGHz']
        if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
            raise ValueError('天线参考频率须为正有限值。')
    return result
