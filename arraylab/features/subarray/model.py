"""Subarray configuration and geometry/field primitives shared by its services."""
from dataclasses import dataclass, fields
import math
import numpy as np
from arraylab.shared.numerics.scenarios import scenarios


@dataclass(frozen=True)
class SubarrayConfig:
    count: int = 9
    nx: int = 2
    ny: int = 2
    spacing_x: float = .5
    spacing_y: float = .5
    aperture_x: float = 6.0
    aperture_y: float = 6.0
    min_spacing: float = .5
    reference_ghz: float = 10.0
    theta: float = 0.0
    phi: float = 0.0
    frequencies_ghz: str = ''
    scan_angles: str = ''
    exclusion_u: float = .2
    exclusion_v: float = .2
    iterations: int = 200
    restarts: int = 2
    opt_grid: int = 31
    eval_grid: int = 81
    seed: int = 42
    hole_shape: str = 'none'
    hole_x: float = 1.0
    hole_y: float = 1.0

    def validate(self):
        limits = {'count': (1, 64), 'nx': (1, 8), 'ny': (1, 8),
                  'iterations': (1, 20000), 'restarts': (1, 8),
                  'opt_grid': (21, 101), 'eval_grid': (41, 301), 'seed': (0, 2**32-1)}
        for key, (lo, hi) in limits.items():
            value = getattr(self, key)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f'{key} 须为 {lo}–{hi} 的整数。')
        for field in fields(self):
            if field.name in limits or field.name in ('frequencies_ghz', 'scan_angles', 'hole_shape'):
                continue
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f'{field.name} 须为有限数值。')
        if self.count * self.nx * self.ny > 512:
            raise ValueError('首版最多 512 个实际阵元。')
        if not all(.01 <= v <= 10 for v in (self.spacing_x, self.spacing_y)):
            raise ValueError('子阵内部间距须为 0.01–10 λ0。')
        if not all(.01 <= v <= 100 for v in (self.aperture_x, self.aperture_y)):
            raise ValueError('孔径须为 0.01–100 λ0。')
        if not 0 <= self.min_spacing <= 10:
            raise ValueError('阵元最小间距须为 0–10 λ0。')
        if self.nx > 1 and self.spacing_x < self.min_spacing-1e-10 or self.ny > 1 and self.spacing_y < self.min_spacing-1e-10:
            raise ValueError('内部阵元间距已违反最小间距，移动子阵中心无法修复。')
        if self.aperture_x < (self.nx-1)*self.spacing_x or self.aperture_y < (self.ny-1)*self.spacing_y:
            raise ValueError('孔径无法容纳一个完整子阵。')
        if not .1 <= self.reference_ghz <= 1000 or not 0 <= self.theta <= 60 or not -360 <= self.phi <= 360:
            raise ValueError('参考频率范围 0.1–1000 GHz，θ 0–60°，φ −360–360°。')
        if not all(.005 <= v <= 1 for v in (self.exclusion_u, self.exclusion_v)):
            raise ValueError('主瓣排除区的 u/v 半宽须为 0.005–1。')
        if self.eval_grid <= self.opt_grid:
            raise ValueError('复核网格须比优化网格更密。')
        if self.hole_shape not in ('none', 'rectangle', 'circle') or min(self.hole_x, self.hole_y) <= 0:
            raise ValueError('禁布区须为 none、rectangle 或 circle，尺寸须为正数。')
        cases = scenarios(self)
        for case in cases:
            _sample_spec(self, case, self.opt_grid)
            _sample_spec(self, case, self.eval_grid, shifted=True)
        return self

def template_coordinates(config):
    xx, yy = np.meshgrid((np.arange(config.nx)-(config.nx-1)/2)*config.spacing_x,
                         (np.arange(config.ny)-(config.ny-1)/2)*config.spacing_y)
    return np.column_stack([xx.ravel(), yy.ravel()])

def expanded_coordinates(config, centers):
    return (np.asarray(centers, dtype=float)[:, None, :] + template_coordinates(config)[None, :, :]).reshape(-1, 2)

def center_limits(config):
    return np.array([(config.aperture_x-(config.nx-1)*config.spacing_x)/2,
                     (config.aperture_y-(config.ny-1)*config.spacing_y)/2])

def layout_valid(config, centers):
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2 or not 1 <= len(centers) <= config.count or not np.isfinite(centers).all():
        return False
    if np.any(np.abs(centers) > center_limits(config)+1e-10):
        return False
    # The fixed, co-oriented template envelopes may touch but may not overlap.
    extent = np.array([(config.nx-1)*config.spacing_x, (config.ny-1)*config.spacing_y])
    delta = np.abs(centers[:, None, :]-centers[None, :, :])
    # A 1 x N template has a line envelope: collinear overlapping segments
    # remain forbidden even though their two-dimensional area is zero.
    overlaps_axis = np.where(extent[None, None, :] > 1e-10,
                             delta < extent-1e-10, delta <= 1e-10)
    overlap = np.all(overlaps_axis, axis=2)
    np.fill_diagonal(overlap, False)
    if overlap.any():
        return False
    xy = expanded_coordinates(config, centers)
    if config.hole_shape == 'rectangle' and np.any((abs(xy[:, 0]) <= config.hole_x/2+1e-10) & (abs(xy[:, 1]) <= config.hole_y/2+1e-10)):
        return False
    if config.hole_shape == 'circle' and np.any(np.linalg.norm(xy, axis=1) <= config.hole_x/2+1e-10):
        return False
    distance2 = np.sum((xy[:, None, :]-xy[None, :, :])**2, axis=2)
    np.fill_diagonal(distance2, np.inf)
    return not np.any(distance2 < max(config.min_spacing, 1e-8)**2-1e-18)

def _scan_uv(case):
    theta, phi = np.deg2rad([case['theta'], case['phi']])
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)])

def _sample_spec(config, case, size, shifted=False):
    axis = np.linspace(-1, 1, size)
    if shifted:
        axis = (axis[:-1]+axis[1:])/2
    uu, vv = np.meshgrid(axis, axis)
    visible = uu*uu+vv*vv <= 1+1e-12
    uv = np.column_stack([uu[visible], vv[visible]])
    delta = uv-_scan_uv(case)
    outside = (abs(delta[:, 0]) > config.exclusion_u+1e-12) | (abs(delta[:, 1]) > config.exclusion_v+1e-12)
    if not outside.any():
        raise ValueError('主瓣排除区覆盖整个可见区域，无法定义副瓣目标。')
    return axis, visible, uv, outside

def normalized_field(config, centers, case, uv):
    """Exact equal-template factorization, including every internal element."""
    uv = np.asarray(uv, dtype=float)
    shape = uv.shape[:-1]
    delta = (uv.reshape(-1, 2)-_scan_uv(case))*case['frequency_ratio']
    output = np.empty(len(delta), dtype=complex)
    template = template_coordinates(config)
    for i in range(0, len(delta), 2048):
        part = delta[i:i+2048]
        output[i:i+2048] = (np.exp(2j*np.pi*(part @ np.asarray(centers).T)).mean(axis=1)
                            * np.exp(2j*np.pi*(part @ template.T)).mean(axis=1))
    return output.reshape(shape)
