"""Display-only FMCW scene adapted from completed waveform snapshots.

No signal, antenna or ambiguity solver is called here. Target angles and
motion direction are explicit visualization assumptions, not measured inputs.
"""
from copy import deepcopy
import math

C = 299792458.0
DEFAULT_SCENE_STATE = {'time_s': 0., 'camera': None, 'target_az_deg': 0.,
                       'target_el_deg': 0., 'motion': 'stationary'}


def _finite(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(label + '须为有限数值。')
    return float(value)


def normalize_scene_state(value=None):
    """Validate scene-only state without altering the numerical configuration."""
    if value is None:return deepcopy(DEFAULT_SCENE_STATE)
    if type(value) is not dict or set(value) - set(DEFAULT_SCENE_STATE):
        raise ValueError('FMCW场景显示状态字段无效。')
    state = {**deepcopy(DEFAULT_SCENE_STATE), **deepcopy(value)}
    for key in ('time_s', 'target_az_deg', 'target_el_deg'):
        state[key] = _finite(state[key], key)
    if state['time_s'] < 0:raise ValueError('场景显示时间不得为负。')
    if abs(state['target_az_deg']) > 180 or abs(state['target_el_deg']) > 90:
        raise ValueError('显示方位须在±180°、俯仰须在±90°以内。')
    if state['motion'] not in ('stationary', 'approaching', 'receding'):
        raise ValueError('场景运动须为静止、显示接近或显示远离。')
    camera = state['camera']
    if camera is not None:
        if type(camera) is not dict or set(camera) != {'position', 'target', 'zoom'}:
            raise ValueError('场景视角字段无效。')
        for key in ('position', 'target'):
            if type(camera[key]) is not list or len(camera[key]) != 3:
                raise ValueError('场景视角坐标须包含3个数值。')
            camera[key] = [_finite(x, '场景视角坐标') for x in camera[key]]
        camera['zoom'] = _finite(camera['zoom'], '场景视角缩放')
        if camera['zoom'] <= 0:raise ValueError('场景视角缩放须大于0。')
    return state


def scene_payload(report, profile='A', time_s=0., options=None):
    """Return the shared Three.js scene contract from one saved A/B profile."""
    state = normalize_scene_state(options)
    time_s = _finite(time_s, '场景显示时间')
    if time_s < 0:raise ValueError('场景显示时间不得为负。')
    if type(report) is not dict or profile not in ('A', 'B'):
        raise ValueError('请先完成FMCW波形计算并选择已计算的A/B组。')
    item = report.get('analysis', {}).get('profiles', {}).get(profile)
    if type(item) is not dict or type(item.get('params')) is not dict:
        raise ValueError('所选FMCW组尚无计算快照。')
    params = item['params']
    distance = _finite(params.get('targetRange'), '频谱目标距离')
    velocity = _finite(params.get('targetVelocity'), '频谱目标速度')
    if distance < 0:raise ValueError('频谱目标距离不得为负。')
    frequency = _finite(item.get('result', {}).get('metrics', {}).get('centerGHz'), '分析频率')
    if frequency <= 0:raise ValueError('分析频率须大于0。')
    az, el = math.radians(state['target_az_deg']), math.radians(state['target_el_deg'])
    direction = [math.cos(el)*math.cos(az), math.cos(el)*math.sin(az), math.sin(el)]
    speed = abs(velocity) * {'stationary': 0, 'approaching': -1, 'receding': 1}[state['motion']]
    maximum = 5.
    if state['motion'] == 'approaching' and speed and distance > 0:
        maximum = min(maximum, distance/abs(speed))
    if distance == 0:speed = 0.
    time_s = min(time_s, maximum)
    geometry = report.get('configuration', {}).get('geometry') or {}
    pointing = [_finite(geometry.get(key, 0), '已保存天线指向') for key in ('steerAzDeg', 'steerElDeg')]
    antenna = report.get('antenna', {}).get(profile) or {}
    valid_array = antenna.get('valid') is True
    widths = [antenna.get('beamwidthAzDeg'), antenna.get('beamwidthElDeg')]
    beam_visible = valid_array and all(type(w) in (int, float) and math.isfinite(w) and w > 0 for w in widths)
    wavelength = C/(frequency*1e9)
    arrays = {}
    for side in ('tx', 'rx'):
        points = antenna.get(side, []) if valid_array else []
        if type(points) is not list:raise ValueError('已保存物理阵列坐标无效。')
        converted = []
        for point in points:
            if type(point) is not list or len(point) != 2:raise ValueError('已保存物理阵列坐标无效。')
            x, z = [_finite(v, '已保存物理阵列坐标') for v in point]
            converted.append([0., x*wavelength, z*wavelength])
        arrays[side+'_positions_m'] = converted
    label = f'频谱目标 R={distance:g} m，v={velocity:g} m/s'
    project = {'name': f'FMCW {profile}组显示场景',
               'scene': {'display_min_span_m': 1., 'platform': {'position_m': [0., 0., 0.], 'velocity_mps': [0., 0., 0.],
                                      'attitude_deg': [0., 0., 0.], 'trajectory': []},
                         'targets': [{'id': label, 'position_m': [distance*v for v in direction],
                                      'velocity_mps': [speed*v for v in direction], 'trajectory': []}],
                         'terrain': {'enabled': False}},
               'array': {'pointing_deg': pointing, 'scan_rate_deg_s': [0., 0.],
                         'beam_visible': beam_visible, 'pointing_axis': valid_array,
                         'beamwidth_deg': widths if beam_visible else [0., 0.],
                         'tx_count': params['txCount'], 'rx_count': params['rxCount'], **arrays}}
    motion = {'stationary': '目标静态展示', 'approaching': '按|v|显示接近', 'receding': '按|v|显示远离'}[state['motion']]
    note = ('显示角度与运动方向仅为可视化假设，不回写波形或改变频谱。原速度正值表示拍频增加；'
            '局部原点/网格不是实际平台位置或地形。A/B共用同一频谱目标；独立时延、增益预算和稀疏双目标实验不混入场景。')
    if not beam_visible:
        note += '本组未给出完整的方位/俯仰半功率宽度，已隐藏波束锥。'
        if valid_array:note += '箭头仅表示本次保存的天线指向。'
    else:note += '透明波束为已保存指向及宽度的方向示意，不是真实辐射增益等值面。'
    note += '箭头和锥体长度为显示尺寸，不代表探测距离。'
    if distance == 0 and state['motion'] != 'stationary':note += '零距离目标保持在参考原点。'
    return {'project': project, 'grid': None, 'time_s': time_s, 'max_time_s': maximum,
            'caption': f'FMCW {profile}组 · {motion} · 角度仅作示意', 'note': note}
