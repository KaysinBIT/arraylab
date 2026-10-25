"""Ideal far-field radar MIMO geometry and digital receive-beam response.

The virtual coordinate is Tx + Rx (one-way wavelength convention). A physical
channel is never discarded merely because its coordinate overlaps another.
This is not an RF-chain model, radiation-gain calculation, or DOA simulator.
"""
from dataclasses import asdict, dataclass, fields
import math
from numbers import Real

import numpy as np
from scipy.spatial import cKDTree

from arraylab import __version__
from arraylab.framework.parallel import check_cancelled, ordered_map
from arraylab.shared.plotting.metrics import sampled_cut_metrics, spherical_beamwidths


MAX_ELEMENTS_PER_SIDE = 256
MAX_CHANNELS = 16384
MAX_VIRTUAL_DIAMETER_LAMBDA = 80.0
POSITION_TOLERANCE_LAMBDA = 1e-9
FIELD_BLOCK_VALUES = 262144
DATA_FLOOR_DB = -300.0


def _number(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name}须为数值。')
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite or not low <= value <= high:
        raise ValueError(f'{name}须为{low:g}至{high:g}的有限数值。')


@dataclass(frozen=True)
class MIMOParameters:
    dimension: str = 'linear'
    frequency_ghz: float = 10.0
    coordinate_unit: str = 'lambda'
    tx_positions: tuple = ((0.0, 0.0), (2.0, 0.0))
    rx_positions: tuple = ((0.0, 0.0), (.5, 0.0), (1.0, 0.0), (1.5, 0.0))
    tx_weights: tuple = ()
    rx_weights: tuple = ()
    theta_deg: float = 0.0
    phi_deg: float = 0.0
    cut_phi_deg: float = 0.0
    step_deg: float = .1
    map_size: int = 181

    def __post_init__(self):
        for name in ('tx_positions', 'rx_positions'):
            value = getattr(self, name)
            if isinstance(value, (list, tuple)):
                object.__setattr__(self, name, tuple(tuple(row) if isinstance(row, (list, tuple)) else row for row in value))
        for name in ('tx_weights', 'rx_weights'):
            value = getattr(self, name)
            if isinstance(value, (list, tuple)):
                object.__setattr__(self, name, tuple(value))

    @property
    def wavelength_mm(self):
        return 299.792458 / self.frequency_ghz

    def positions(self, side):
        if side not in ('tx', 'rx'):
            raise ValueError('阵列侧须为tx或rx。')
        values = np.asarray(getattr(self, side + '_positions'), dtype=float)
        return values if self.coordinate_unit == 'lambda' else values / self.wavelength_mm

    def weights(self, side):
        if side not in ('tx', 'rx'):
            raise ValueError('阵列侧须为tx或rx。')
        values = getattr(self, side + '_weights')
        result = np.asarray(values if values else np.ones(len(getattr(self, side + '_positions'))), dtype=float)
        # Scale before summing, so a harmless common amplitude factor cannot overflow.
        result = result / result.max()
        return result / result.sum()

    def validate(self):
        if self.dimension not in ('linear', 'planar'):
            raise ValueError('MIMO维度须为linear或planar。')
        if self.coordinate_unit not in ('lambda', 'mm'):
            raise ValueError('MIMO坐标单位须为lambda或mm。')
        _number(self.frequency_ghz, '频率(GHz)', .001, 1000)
        _number(self.theta_deg, '目标θ(度)', -90 if self.dimension == 'linear' else 0, 90)
        _number(self.phi_deg, '目标φ(度)', -180, 180)
        _number(self.cut_phi_deg, '切面φ(度)', -180, 180)
        _number(self.step_deg, '切面步长(度)', .01, 5)
        if type(self.map_size) is not int or not 81 <= self.map_size <= 401:
            raise ValueError('二维图采样数须为81–401的整数。')
        if self.dimension == 'linear' and (self.phi_deg != 0 or self.cut_phi_deg != 0):
            raise ValueError('一维MIMO沿x轴，目标φ和切面φ须为0；θ为从+z朝+x的有符号角。')
        positions = []
        for side in ('tx', 'rx'):
            rows = getattr(self, side + '_positions')
            if not isinstance(rows, tuple) or not 1 <= len(rows) <= MAX_ELEMENTS_PER_SIDE:
                raise ValueError(f'{side.upper()}须提供1–{MAX_ELEMENTS_PER_SIDE}个二维坐标。')
            for row in rows:
                if not isinstance(row, tuple) or len(row) != 2:
                    raise ValueError(f'{side.upper()}每个阵元须提供x、y两个坐标。')
                for value in row:
                    _number(value, side.upper() + '坐标', -1e9, 1e9)
                if self.dimension == 'linear' and row[1] != 0:
                    raise ValueError('一维MIMO的全部y坐标须为0。')
            xy = self.positions(side)
            if np.max(abs(xy)) > 1000:
                raise ValueError('各侧坐标绝对值须不超过1000λ；可平移原点。')
            if len(xy) > 1 and np.min(cKDTree(xy).query(xy, k=2)[0][:, 1]) < 1e-6:
                raise ValueError(f'{side.upper()}侧阵元不得重合，最小间距须至少为0.000001λ。')
            weights = getattr(self, side + '_weights')
            if not isinstance(weights, tuple) or (weights and len(weights) != len(rows)):
                raise ValueError(f'{side.upper()}数字幅度系数数量须等于该侧阵元数，或留空使用均匀系数。')
            for value in weights:
                _number(value, side.upper() + '数字幅度系数', np.nextafter(0.0, 1.0), np.finfo(float).max)
            if weights and math.log10(max(weights)) - math.log10(min(weights)) > 6 + 1e-12:
                raise ValueError(f'{side.upper()}数字幅度动态范围须不超过120 dB。')
            positions.append(xy)
        if len(self.tx_positions) * len(self.rx_positions) > MAX_CHANNELS:
            raise ValueError(f'首版Tx×Rx通道数不得超过{MAX_CHANNELS}，不会静默删减阵元。')
        # A conservative bounding-box diagonal protects angular sampling/work size.
        spans = np.ptp(positions[0], axis=0) + np.ptp(positions[1], axis=0)
        if np.linalg.norm(spans) > MAX_VIRTUAL_DIAMETER_LAMBDA + 1e-12:
            raise ValueError(f'虚拟阵列包围盒对角线须不超过{MAX_VIRTUAL_DIAMETER_LAMBDA:g}λ。')
        return self


def mimo_parameters_from_dict(data):
    if not isinstance(data, dict) or not all(isinstance(key, str) for key in data):
        raise ValueError('MIMO参数须为对象。')
    allowed = {field.name for field in fields(MIMOParameters)}
    if set(data) - allowed:
        raise ValueError('未知MIMO参数：' + ', '.join(sorted(set(data) - allowed)))
    return MIMOParameters(**data).validate()


def _geometry_stats(xy, weights, wavelength):
    span = np.ptp(xy, axis=0)
    minimum = float(cKDTree(xy).query(xy, k=2)[0][:, 1].min()) if len(xy) > 1 else None
    return {'positions_lambda': xy.tolist(), 'weights': weights.tolist(),
            'span_lambda': span.tolist(), 'span_mm': (span * wavelength).tolist(),
            'min_spacing_lambda': minimum,
            'min_spacing_mm': minimum * wavelength if minimum is not None else None,
            'bounding_box_diagonal_lambda': float(np.linalg.norm(span)),
            'bounding_box_diagonal_mm': float(np.linalg.norm(span) * wavelength),
            'geometric_dimension': int(np.linalg.matrix_rank(xy - xy[0], tol=1e-10))}


def mimo_geometry(parameters, *, cancelled=None):
    """Reconstruct/validate geometry only; no field or optimization is run.

    Points are not rounded onto a lattice. In lexicographic x/y/channel order,
    each unassigned point becomes a representative and absorbs still-unassigned
    points within 1e-9 lambda. This is a numerical identity rule, not a grid or
    a claim about coarray degrees of freedom. Original pair coordinates survive.
    """
    p = parameters.validate()
    check_cancelled(cancelled)
    tx, rx = p.positions('tx'), p.positions('rx')
    wt, wr = p.weights('tx'), p.weights('rx')
    xy = (tx[:, None, :] + rx[None, :, :]).reshape(-1, 2)
    weights = (wt[:, None] * wr[None, :]).ravel()
    tree = cKDTree(xy)
    order = np.lexsort((np.arange(len(xy)), xy[:, 1], xy[:, 0]))
    identifiers = np.full(len(xy), -1, dtype=np.int64)
    unique = []
    for number, index in enumerate(order):
        if number % 128 == 0:
            check_cancelled(cancelled)
        if identifiers[index] >= 0:
            continue
        neighbors = np.asarray(tree.query_ball_point(xy[index], POSITION_TOLERANCE_LAMBDA), dtype=np.int64)
        neighbors = neighbors[identifiers[neighbors] < 0]
        group = len(unique)
        identifiers[neighbors] = group
        unique.append({'index': group + 1, 'x_lambda': float(xy[index, 0]),
                       'y_lambda': float(xy[index, 1]), 'multiplicity': int(len(neighbors)),
                       'weight': float(weights[neighbors].sum())})
    unique_xy = np.array([[row['x_lambda'], row['y_lambda']] for row in unique])
    unique_weights = np.array([row['weight'] for row in unique])
    pair_rows = [{'channel': index + 1, 'tx': index // len(rx) + 1, 'rx': index % len(rx) + 1,
                  'x_lambda': float(point[0]), 'y_lambda': float(point[1]),
                  'weight': float(weight), 'unique_index': int(group + 1)}
                 for index, (point, weight, group) in enumerate(zip(xy, weights, identifiers))]
    counts = {'tx_elements': len(tx), 'rx_elements': len(rx), 'tx_rx_channels': len(xy),
              'unique_virtual_positions': len(unique), 'redundant_channels': len(xy) - len(unique),
              'redundancy_ratio': (len(xy) - len(unique)) / len(xy),
              'max_multiplicity': max(row['multiplicity'] for row in unique), 'rf_chains': None}
    return {'counts': counts,
            'geometry': {'tx': _geometry_stats(tx, wt, p.wavelength_mm),
                         'rx': _geometry_stats(rx, wr, p.wavelength_mm),
                         'pairs': pair_rows, 'virtual_unique': unique,
                         'virtual': _geometry_stats(unique_xy, unique_weights, p.wavelength_mm)},
            'aggregation': {'method': 'lexicographic_representative_radius_first_assignment',
                            'tolerance_lambda': POSITION_TOLERANCE_LAMBDA,
                            'coordinates_rounded': False,
                            'field_uses_original_coordinates': True,
                            'channel_order': 'tx_major_rx_minor', 'index_base': 1,
                            'weight_rule': 'sum original separable channel amplitudes; do not drop repeated channels'}}


def _steering_uv(p):
    t, f = np.deg2rad([p.theta_deg, p.phi_deg])
    return np.array([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f)])


def mimo_cut_angles(parameters):
    """Stable cut sampling including the exact projected steering direction."""
    p = parameters
    angles = np.linspace(-90, 90, int(math.ceil(180 / p.step_deg)) + 1)
    phi = math.radians(p.cut_phi_deg)
    direction = np.array([math.cos(phi), math.sin(phi)])
    preferred = math.degrees(math.asin(float(np.clip(_steering_uv(p) @ direction, -1, 1))))
    nearest = int(np.argmin(abs(angles - preferred)))
    if abs(angles[nearest] - preferred) <= 1e-10:
        angles[nearest] = preferred
    else:
        angles = np.sort(np.r_[angles, preferred])
    return angles, preferred


def _array_field(xy, weights, delta_uv, cancelled=None):
    result = np.empty(len(delta_uv), dtype=complex)
    block = max(1, min(2048, FIELD_BLOCK_VALUES // max(1, len(xy))))
    for start in range(0, len(delta_uv), block):
        check_cancelled(cancelled)
        phase = delta_uv[start:start + block] @ xy.T
        result[start:start + block] = np.exp(2j * np.pi * phase) @ weights
    check_cancelled(cancelled)
    return result


def _responses(p, uv, cancelled=None):
    delta = uv - _steering_uv(p)
    tx = _array_field(p.positions('tx'), p.weights('tx'), delta, cancelled)
    rx = _array_field(p.positions('rx'), p.weights('rx'), delta, cancelled)
    return tx, rx, tx * rx


def mimo_field(parameters, uv, *, cancelled=None):
    """Complex digital beam response, referenced to unity at the target angle."""
    parameters.validate()
    values = np.asarray(uv, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError('方向余弦须为有限数值的N×2数组。')
    if np.any(np.sum(values ** 2, axis=1) > 1 + 1e-12):
        raise ValueError('方向余弦须位于u²+v²≤1的可见圆盘内。')
    return _responses(parameters, values, cancelled)[2]


def _db(values):
    return 20 * np.log10(np.maximum(abs(values), 10 ** (DATA_FLOOR_DB / 20)))


def calculate_mimo(parameters, *, workers=None, cancelled=None, progress=None):
    """Compute a bounded, fully JSON-serializable ideal radar MIMO snapshot."""
    p = parameters.validate()

    def tell(message):
        check_cancelled(cancelled)
        if progress:
            progress(message)

    tell('构建Tx–Rx虚拟通道及重合位置统计')
    geometry = mimo_geometry(p, cancelled=cancelled)
    angles, preferred = mimo_cut_angles(p)
    phi = math.radians(p.cut_phi_deg)
    direction = np.array([math.cos(phi), math.sin(phi)])
    cut_uv = np.sin(np.deg2rad(angles))[:, None] * direction
    axis = np.linspace(-1, 1, p.map_size)
    uu, vv = np.meshgrid(axis, axis)
    map_uv = np.column_stack((uu.ravel(), vv.ravel()))
    visible = np.sum(map_uv ** 2, axis=1) <= 1 + 1e-14
    tell('计算切面与二维虚拟波束')

    def calculate_view(kind):
        if kind == 'cut':
            values = _responses(p, cut_uv, cancelled)
            return {'angles_deg': angles.tolist(), 'phi_deg': float(p.cut_phi_deg),
                    **{name: _db(value).tolist() for name, value in zip(('tx_db', 'rx_db', 'virtual_db'), values)}}
        values = np.full(len(map_uv), np.nan)
        values[visible] = _db(_responses(p, map_uv[visible], cancelled)[2])
        matrix = values.reshape(p.map_size, p.map_size)
        return {'axis_u': axis.tolist(), 'axis_v': axis.tolist(),
                'virtual_db': [[float(value) if np.isfinite(value) else None for value in row] for row in matrix]}

    views, execution = ordered_map(calculate_view, ('cut', 'map'), workers=workers,
                                  work_units=(len(cut_uv) + int(visible.sum())) * (len(p.tx_positions) + len(p.rx_positions)),
                                  cancelled=cancelled)
    cut = sampled_cut_metrics(angles, views[0]['virtual_db'], preferred_angle=preferred)
    tell('计算目标代表峰的局部3 dB宽度')
    scalar_geometry = geometry['counts']['unique_virtual_positions'] == 1
    widths = None
    if not scalar_geometry:
        theta = abs(p.theta_deg)
        phi_peak = 180.0 if p.dimension == 'linear' and p.theta_deg < 0 else p.phi_deg
        diameter = geometry['geometry']['virtual']['bounding_box_diagonal_lambda']

        def widths_task(_):
            return spherical_beamwidths(lambda xyz: _responses(p, xyz[:, :2], cancelled)[2], theta, phi_peak,
                                        step_deg=min(.25, 5 / max(1, diameter)))

        # Reuse the same bounded BLAS contract; never nest ordered_map locks.
        result, _ = ordered_map(widths_task, [None], workers=1, cancelled=cancelled)
        widths = result[0]
    metrics = {'cut': cut,
               'representative_peak_theta_deg': None if scalar_geometry else float(p.theta_deg),
               'representative_peak_phi_deg': None if scalar_geometry else float(p.phi_deg),
               'pointing_offset_deg': None if scalar_geometry else 0.0,
               'peak_method': 'no_angular_information' if scalar_geometry else 'theoretical_coherent_steering',
               'peak_uniqueness': 'not_established',
               'orthogonal_beamwidths_deg': widths['orthogonal_beamwidths_deg'] if widths else [None, None],
               'great_circle_widths': widths,
               'directivity_dbi': None, 'gain_dbi': None,
               'gain_status': 'digital_receive_beam_not_physical_radiation_gain',
               'angular_resolution_deg': None}
    notices = [
        '近共址、远场、窄带、静态目标；理想正交且可分离的Tx波形，各向同性点阵元。',
        '虚拟坐标使用Tx+Rx的一程波长约定；Tx–Rx通道数、独特位置数及RF链数不是同一概念。',
        '幅度系数为可分离的数字接收通道合成系数，不是发射功率，也不是双侧加权模糊函数中的平方权。',
        '重合通道的幅度按重数累加；方向图使用原始Tx/Rx坐标，数值位置归并不改变方向图。',
        '正幅度且理想扫描补偿使目标位置成为解析全局代表峰；零指向偏差不是搜索测量，不证明无栅瓣或唯一主瓣。',
        '切面指标和二维图是有限采样估计；步长或网格不足可能漏掉窄波束及旁瓣，3 dB宽度不等于双目标DOA分辨率。',
        '坐标均在z=0平面，前后空间对称；线阵还存在圆锥方向歧义。未计算RF链数、通信秩、SNR增益或实测辐射增益。',
    ]
    tell('MIMO参数计算完成')
    payload = asdict(p)
    for name in ('frequency_ghz', 'theta_deg', 'phi_deg', 'cut_phi_deg', 'step_deg'):
        payload[name] = float(payload[name])
    for side in ('tx', 'rx'):
        payload[side + '_positions'] = tuple(tuple(float(value) for value in row) for row in payload[side + '_positions'])
        payload[side + '_weights'] = tuple(float(value) for value in payload[side + '_weights'])
    return {'format': 'arraylab.mimo.result', 'schema_version': 1, 'app_version': __version__,
            'parameters': payload, **geometry, 'patterns': {'cut': views[0], 'map': views[1]},
            'metrics': metrics, 'execution': execution, 'notices': notices,
            'model': {'kind': 'ideal_coherent_quasi_monostatic_radar_mimo',
                      'virtual_coordinate_definition': 'r_virtual = r_tx + r_rx; one-way wavelength convention',
                      'channel_weight_definition': 'normalized positive separable digital amplitude: w_tx * w_rx',
                      'normalization': 'unit complex response at the configured steering direction',
                      'coordinate_convention': 'x-y plane, z=0; theta from +z, phi from +x; linear theta signed toward +x',
                      'source': 'https://www.ti.com/lit/an/swra554a/swra554a.pdf',
                      'source_section': 'Principle of the MIMO Radar, page 4'}}
