"""WaveLab 1.4 radar-coordinate geometry and ideal channel-weighted cuts.

Coordinates use the source tool's horizontal x / vertical z convention, not
ArrayLab's theta/phi convention. No electromagnetic or RF gain is implied.
"""
from copy import deepcopy
import math

from arraylab.framework.parallel import check_cancelled

RAD = math.pi / 180
STEP = .25
FLOOR_DB = -80


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _count(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f'{name}须为1–{maximum}的整数。')
    return value


def geometry_preset(key, tx_count, rx_count):
    m, n = _count(tx_count, 'Tx数量', 16), _count(rx_count, 'Rx数量', 32)
    if key not in ('ula', 'planar', 'compact'):
        raise ValueError('未知阵列布局类型。')
    return {'tx': [[0, i*.5] if key == 'planar' else
                   [i*(n*.5 if key == 'ula' else .5), 0] for i in range(m)],
            'rx': [[i*.5, 0] for i in range(n)]}


def read_geometry(points, name, maximum):
    if not isinstance(points, (list, tuple)):
        raise ValueError(name+'坐标须为数组。')
    _count(len(points), name+'数量', maximum)
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 or
           any(not _finite(v) or abs(v) > 100 for v in p) for p in points):
        raise ValueError(name+'坐标须为±100波长内的有限[x,z]坐标对。')
    return [list(p) for p in points]


def effective_geometry(geometry, frequency_ghz):
    """Keep physical geometry fixed if it carries a reference frequency."""
    result = deepcopy(geometry)
    reference = geometry.get('referenceFrequencyGHz')
    if reference is not None and (not _finite(reference) or reference <= 0):
        raise ValueError('天线参考频率无效。')
    if reference is not None and (not _finite(frequency_ghz) or frequency_ghz <= 0):
        raise ValueError('当前分析频率无效。')
    ratio = 1 if reference is None else frequency_ghz/reference
    for key in ('tx', 'rx'):
        result[key] = [[v*ratio for v in p] for p in geometry[key]]
    return result


def applied_geometry(report, reference_frequency_ghz):
    """Create the source tool's explicit applied-coordinate snapshot."""
    if not report.get('valid') or not report.get('geometry'):
        raise ValueError('没有可应用的有效综合候选。')
    if not _finite(reference_frequency_ghz) or reference_frequency_ghz <= 0:
        raise ValueError('天线参考频率无效。')
    return {**deepcopy(report['geometry']), 'referenceFrequencyGHz': reference_frequency_ghz,
            'steerAzDeg': 0, 'steerElDeg': 0}


def direction(azimuth, elevation):
    return math.sin(azimuth*RAD)*math.cos(elevation*RAD), math.sin(elevation*RAD)


def _factor(virtual, total, reference, azimuth, elevation):
    ux, uz = direction(azimuth, elevation)
    dx, dz = ux-reference[0], uz-reference[1]
    real = imaginary = 0.
    for p in virtual:
        phase = math.tau*(p['x']*dx+p['z']*dz)
        real += p['multiplicity']*math.cos(phase)
        imaginary += p['multiplicity']*math.sin(phase)
    amplitude = min(1, math.hypot(real, imaginary)/total)
    return max(FLOOR_DB, 20*math.log10(max(1e-12, amplitude)))


def _half_power_width(virtual, total, reference, az, el, is_azimuth, aperture, cancelled):
    result = {'value': None, 'limited': False}
    if aperture <= 1e-9:
        return result
    threshold = 10**(-3/20)
    if 2*max(p['multiplicity'] for p in virtual)/total-1 > threshold:
        return result
    mean_x = sum(p['x']*p['multiplicity'] for p in virtual)/total
    mean_z = sum(p['z']*p['multiplicity'] for p in virtual)/total
    spread_x = sum(abs(p['x']-mean_x)*p['multiplicity'] for p in virtual)/total
    spread_z = sum(abs(p['z']-mean_z)*p['multiplicity'] for p in virtual)/total
    steering = az if is_azimuth else el
    evaluations = 0

    def amplitude(angle):
        nonlocal evaluations
        evaluations += 1
        if evaluations % 128 == 0:
            check_cancelled(cancelled)
        if evaluations > 12000:
            result['limited'] = True
            return math.nan
        return 10**(_factor(virtual, total, reference, angle if is_azimuth else az,
                            el if is_azimuth else angle)/20)

    def slope_bound(a, b):
        low, high = min(a, b), max(a, b)
        nearest = 0 if low <= 0 <= high else min(abs(low), abs(high))
        max_cos = math.cos(nearest*RAD)
        max_sin = math.sin(max(abs(low), abs(high))*RAD)
        return math.tau*RAD*(spread_x*abs(math.cos(el*RAD))*max_cos if is_azimuth else
                             spread_x*abs(math.sin(az*RAD))*max_sin+spread_z*max_cos)

    def nearest_boundary(sign):
        limit = 90-sign*steering

        def search(a, fa, b, fb):
            if result['limited']:
                return None
            width = b-a
            bound = slope_bound(steering+sign*a, steering+sign*b)
            if min(fa, fb)-bound*width/2 > threshold+1e-12:
                return None
            if width <= 1e-7:
                return (a, b) if fa > threshold and fb <= threshold else None
            mid = (a+b)/2
            fm = amplitude(steering+sign*mid)
            return search(a, fa, mid, fm) or search(mid, fm, b, fb)

        a, fa = 0., 1.
        while a < limit and not result['limited']:
            b = min(a+STEP, limit)
            fb = amplitude(steering+sign*b)
            bracket = search(a, fa, b, fb)
            if bracket:
                left, right = bracket
                while right-left > 1e-10 and not result['limited']:
                    mid = (left+right)/2
                    if amplitude(steering+sign*mid) > threshold:
                        left = mid
                    else:
                        right = mid
                return (left+right)/2
            a, fa = b, fb
        return None

    left, right = nearest_boundary(-1), nearest_boundary(1)
    if not result['limited'] and left is not None and right is not None:
        result['value'] = left+right
    return result


def calculate(config, *, cancelled=None):
    if not isinstance(config, dict):
        raise ValueError('需要天线配置对象。')
    check_cancelled(cancelled)
    tx, rx = read_geometry(config.get('tx'), 'Tx', 16), read_geometry(config.get('rx'), 'Rx', 32)
    az, el = config.get('steerAzDeg', 0), config.get('steerElDeg', 0)
    if any(not _finite(v) or abs(v) > 80 for v in (az, el)):
        raise ValueError('指向角须为-80至80度的有限数值。')
    grouped = {}
    for t in tx:
        for r in rx:
            x, z = t[0]+r[0], t[1]+r[1]
            key = math.floor(x*1e9+.5), math.floor(z*1e9+.5)
            if key in grouped:
                grouped[key]['multiplicity'] += 1
            else:
                grouped[key] = {'x': x, 'z': z, 'multiplicity': 1}
    virtual = sorted(grouped.values(), key=lambda p: (p['z'], p['x']))
    count = len(tx)*len(rx)
    aperture_x = max(p['x'] for p in virtual)-min(p['x'] for p in virtual)
    aperture_z = max(p['z'] for p in virtual)-min(p['z'] for p in virtual)
    reference = direction(az, el)
    azimuth, elevation = [], []
    for i in range(721):
        if i % 64 == 0:
            check_cancelled(cancelled)
        angle = -90+i*STEP
        azimuth.append({'x': angle, 'y': _factor(virtual, count, reference, angle, el)})
        elevation.append({'x': angle, 'y': _factor(virtual, count, reference, az, angle)})
    az_width = _half_power_width(virtual, count, reference, az, el, True, aperture_x, cancelled)
    el_width = _half_power_width(virtual, count, reference, az, el, False, aperture_z, cancelled)
    warnings = []
    if len(virtual) < count:
        warnings.append('存在重合的虚拟位置：通道数不等于独立空间采样数；图中按重数保留均匀通道加权。')
    if aperture_x <= 1e-9:
        warnings.append('没有水平孔径，无法给出独立的方位半功率波束宽度。')
    if aperture_z <= 1e-9:
        warnings.append('没有垂直孔径；一维线阵不支持独立俯仰测角。')
    if (aperture_x > 1e-9 and az_width['value'] is None) or (aperture_z > 1e-9 and el_width['value'] is None):
        warnings.append('当前角度切面内未找到完整的主瓣半功率边界，波束宽度显示为空。')
    if az_width['limited'] or el_width['limited']:
        warnings.append('半功率边界搜索达到计算预算，未给出不确定的波束宽度数值。')
    if max(aperture_x, aperture_z) > 50:
        warnings.append('方向图显示网格为 0.25°，窄峰或栅瓣可能未充分显示；半功率波束宽度采用独立自适应求解。')
    if abs(az) > 60 or abs(el) > 60:
        warnings.append('大角度扫描会改变主瓣宽度；各向同性单元模型未计真实单元方向图。')
    warnings.append('方向图为通道已正确分离并校相后的单频远场阵列因子，未包含互耦、单元方向图、安装遮挡或 DDM 解码误差。')
    return {'tx': tx, 'rx': rx, 'virtual': virtual, 'channelCount': count, 'uniqueCount': len(virtual),
            'apertureX': aperture_x, 'apertureZ': aperture_z, 'azimuth': azimuth, 'elevation': elevation,
            'beamwidthAzDeg': az_width['value'], 'beamwidthElDeg': el_width['value'], 'warnings': warnings}
