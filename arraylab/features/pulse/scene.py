"""Shared physical geometry for the renderer and the echo generator.

Local axes are x forward, y left, z up (metres). Terrain is a height field;
propagation is a straight line. A DEM supplies geometry, not material properties.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

C = 299_792_458.0


def _vector(value, name="vector", size=3):
    a = np.asarray(value, dtype=float)
    if a.shape != (size,) or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} 必须包含 {size} 个有限数值")
    return a


def rotation_matrix(yaw_pitch_roll_deg):
    """Active body-to-world Rz(yaw) Ry(-pitch) Rx(roll); pitch raises the nose."""
    yaw, pitch, roll = np.deg2rad(_vector(yaw_pitch_roll_deg, "attitude_deg"))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.]]) @ np.array(
        [[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]]) @ np.array(
        [[1., 0, 0], [0, cr, -sr], [0, sr, cr]])


def pose_at(obj, t):
    """Linear trajectory interpolation; endpoint segments extrapolate linearly.

    Without a trajectory, position_m is the position at t=0. A one-point
    trajectory uses velocity_mps about its own timestamp. Optional keyframe
    attitudes interpolate through the shortest wrapped angular difference.
    """
    if not np.isfinite(t):
        raise ValueError("时间必须为有限数值")
    attitude = _vector(obj.get("attitude_deg", [0, 0, 0]), "attitude_deg")
    velocity = _vector(obj.get("velocity_mps", [0, 0, 0]), "velocity_mps")
    keys = obj.get("trajectory") or []
    if not keys:
        return _vector(obj.get("position_m", [0, 0, 0]), "position_m") + velocity * t, attitude
    times = np.array([float(k["time_s"]) for k in keys])
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("航迹时间须为有限数值，并严格递增")
    positions = np.array([_vector(k["position_m"], "trajectory position") for k in keys])
    if len(keys) == 1:
        return positions[0] + velocity * (t - times[0]), _vector(keys[0].get("attitude_deg", attitude))
    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
    fraction = (t - times[i]) / (times[i + 1] - times[i])
    position = positions[i] + fraction * (positions[i + 1] - positions[i])
    a0 = _vector(keys[i].get("attitude_deg", attitude))
    a1 = _vector(keys[i + 1].get("attitude_deg", attitude))
    return position, a0 + fraction * ((a1 - a0 + 180) % 360 - 180)


def pose_with_terrain(obj, t, grid=None):
    position, attitude = pose_at(obj, t)
    mode = obj.get("height_mode", "absolute")
    if mode not in ("absolute", "agl"):
        raise ValueError("高度模式须为同基准绝对高度或离地高度")
    if mode == "agl":
        if grid is None:
            raise ValueError("离地高度要求启用地形")
        height = sample_height(position[0], position[1], grid)
        if height is None:
            raise ValueError("离地高度位置落在已知地形之外或无数据区")
        position = position.copy()
        position[2] += height
    return position, attitude


def velocity_at(obj,t,grid=None):
    """Derivative of the same piecewise-linear/AGL path used by pose_at.

    At a waypoint the right-hand segment is used. A long finite-difference
    step must not mix the next segment into an earlier pulse's Doppler.
    """
    keys=obj.get('trajectory') or []
    velocity=_vector(obj.get('velocity_mps',[0,0,0]),'velocity_mps').copy()
    if len(keys)>=2:
        times=np.array([float(k['time_s']) for k in keys])
        if np.any(np.diff(times)<=0):raise ValueError('航迹时间须严格递增')
        i=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-2))
        velocity=(_vector(keys[i+1]['position_m'])-_vector(keys[i]['position_m']))/(times[i+1]-times[i])
    if obj.get('height_mode','absolute')=='agl':
        if grid is None:raise ValueError('离地高度要求启用地形')
        position,_=pose_at(obj,t);x,y=position[:2]
        if sample_height(x,y,grid) is None:raise ValueError('离地高度位置落在已知地形之外或无数据区')
        gx,gy=np.asarray(grid['x_m']),np.asarray(grid['y_m']);z=np.asarray(grid['z_m'])
        ix=int(np.clip(np.searchsorted(gx,x,side='left' if velocity[0]<0 else 'right')-1,0,len(gx)-2))
        iy=int(np.clip(np.searchsorted(gy,y,side='left' if velocity[1]<0 else 'right')-1,0,len(gy)-2))
        dx=gx[ix+1]-gx[ix];dy=gy[iy+1]-gy[iy];u=(x-gx[ix])/dx;v=(y-gy[iy])/dy
        dzdx=((z[iy,ix+1]-z[iy,ix])*(1-v)+(z[iy+1,ix+1]-z[iy+1,ix])*v)/dx
        dzdy=((z[iy+1,ix]-z[iy,ix])*(1-u)+(z[iy+1,ix+1]-z[iy,ix+1])*u)/dy
        velocity[2]+=dzdx*velocity[0]+dzdy*velocity[1]
    return velocity


def attitude_rate_at(obj,t):
    keys=obj.get('trajectory') or []
    if len(keys)<2:return np.zeros(3)
    times=np.array([float(k['time_s']) for k in keys])
    i=int(np.clip(np.searchsorted(times,t,side='right')-1,0,len(times)-2))
    default=obj.get('attitude_deg',[0,0,0])
    a0=_vector(keys[i].get('attitude_deg',default));a1=_vector(keys[i+1].get('attitude_deg',default))
    return ((a1-a0+180)%360-180)/(times[i+1]-times[i])


def rotation_rate_matrix(attitude_deg,rate_deg_s):
    """Analytic derivative of Rz(yaw) Ry(-pitch) Rx(roll)."""
    y,p,r=np.deg2rad(attitude_deg);dy,dp,dr=np.deg2rad(rate_deg_s)
    cy,sy,cp,sp,cr,sr=np.cos(y),np.sin(y),np.cos(p),np.sin(p),np.cos(r),np.sin(r)
    rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    ry=np.array([[cp,0,-sp],[0,1,0],[sp,0,cp]])
    rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    dz=dy*np.array([[-sy,-cy,0],[cy,-sy,0],[0,0,0]])
    dpitch=dp*np.array([[-sp,0,-cp],[0,0,0],[cp,0,-sp]])
    droll=dr*np.array([[0,0,0],[0,-sr,-cr],[0,cr,-sr]])
    return dz@ry@rx+rz@dpitch@rx+rz@ry@droll


def beam_power_gain(direction, pointing, beamwidth_deg):
    """Unit-peak Gaussian one-way power pattern; full HPBW in each angular cut.

    This is an element/subarray envelope, not an additional array-factor gain.
    The back hemisphere of the pointing direction is suppressed.
    """
    direction = _vector(direction, "direction")
    norm = np.linalg.norm(direction)
    if norm <= 0:
        raise ValueError("方向向量不能为零")
    d = direction / norm
    pointing = _vector(pointing, "pointing", 2)
    width = _vector(beamwidth_deg, "beamwidth_deg", 2)
    if np.any(width <= 0) or np.any(width > 360):
        raise ValueError("波束宽度须大于 0°，且不超过 360°")
    azp, elp = np.deg2rad(pointing)
    p = np.array([np.cos(elp) * np.cos(azp), np.cos(elp) * np.sin(azp), np.sin(elp)])
    if np.dot(d, p) < 0:
        return 0.0
    az, el = np.rad2deg([np.arctan2(d[1], d[0]), np.arcsin(np.clip(d[2], -1, 1))])
    delta = np.array([(az - pointing[0] + 180) % 360 - 180, el - pointing[1]])
    return float(np.exp(-4 * np.log(2) * np.sum((delta / width) ** 2)))


def _planar_shape(n):
    columns = int(math.ceil(math.sqrt(n)))
    return columns, int(math.ceil(n / columns))


def element_positions(array, side, fc_hz):
    if side not in ("tx", "rx"):
        raise ValueError("阵列端口须为 tx 或 rx")
    if not np.isfinite(fc_hz) or fc_hz <= 0:
        raise ValueError("载频必须为正数")
    count = int(array.get(f"{side}_count", 1))
    if count < 1 or count > 4096:
        raise ValueError("阵元数须为 1–4096")
    geometry = array.get("geometry", "ula")
    if geometry == "custom":
        values = np.asarray(array.get(f"{side}_positions_m", []), dtype=float)
        if values.shape != (count, 3) or not np.all(np.isfinite(values)):
            raise ValueError(f"自定义 {side} 坐标须为 {count} 行、3 列的米制坐标")
        return values.copy()
    if geometry not in ("ula", "upa"):
        raise ValueError("阵列结构须为线阵、面阵或自定义坐标")
    d = float(array.get("spacing_lambda", 0.5)) * C / fc_hz
    if not np.isfinite(d) or d <= 0:
        raise ValueError("以波长计的阵元间距必须为正数")
    sy = sz = d
    if side == "tx" and array.get("mode", "phased") in ("tdm", "cdm"):
        nr = int(array.get("rx_count", 1))
        if geometry == "ula":
            sy *= nr
        else:
            cols, rows = _planar_shape(nr)
            sy *= cols
            sz *= rows
    xyz = np.zeros((count, 3))
    if geometry == "ula":
        xyz[:, 1] = (np.arange(count) - (count - 1) / 2) * sy
    else:
        columns, rows = _planar_shape(count)
        xyz[:, 1] = (np.arange(count) % columns - (columns - 1) / 2) * sy
        xyz[:, 2] = (np.arange(count) // columns - (rows - 1) / 2) * sz
        xyz -= xyz.mean(axis=0)
    return xyz


def terrain_grid(project):
    terrain = project.get("scene", {}).get("terrain", {})
    if not terrain.get("enabled", False):
        return None
    kind = terrain.get("kind", "flat")
    if kind == "dem":
        x = np.asarray(terrain.get("x_m", []), dtype=float)
        y = np.asarray(terrain.get("y_m", []), dtype=float)
        z = np.asarray(terrain.get("heights_m", []), dtype=float)
        valid = np.asarray(terrain.get("valid", np.isfinite(z)), dtype=bool) & np.isfinite(z)
        metadata = dict(terrain.get("metadata", {}))
    else:
        extent = np.asarray(terrain.get("extent_m", 2000), dtype=float)
        if extent.ndim == 0:
            extent = np.repeat(extent, 2)
        if extent.shape != (2,) or np.any(~np.isfinite(extent)) or np.any(extent <= 0):
            raise ValueError("地形尺寸须为正数或 [长度,宽度]，单位为米")
        resolution = int(terrain.get("resolution", 17))
        if resolution < 2 or resolution > 2048:
            raise ValueError("地形每轴网格数须为 2–2048")
        x = np.linspace(0, extent[0], resolution)
        y = np.linspace(-extent[1] / 2, extent[1] / 2, resolution)
        xx, yy = np.meshgrid(x, y)
        base = float(terrain.get("base_height_m", 0))
        if kind == "flat":
            z = np.full_like(xx, base)
        elif kind == "slope":
            slope = float(terrain.get("slope_deg", 5))
            if not np.isfinite(slope) or abs(slope) >= 85:
                raise ValueError("地形坡度须在 -85° 与 85° 之间")
            z = base + xx * np.tan(np.deg2rad(slope))
        elif kind == "hills":
            h = float(terrain.get("hill_height_m", 100))
            z = base + h * (np.exp(-((xx - 0.57 * extent[0]) / (0.19 * extent[0])) ** 2
                                  - (yy / (0.24 * extent[1])) ** 2)
                            + 0.45 * np.exp(-((xx - 0.27 * extent[0]) / (0.13 * extent[0])) ** 2
                                             - ((yy + .23 * extent[1]) / (.16 * extent[1])) ** 2))
        else:
            raise ValueError("地形须为平地、坡地、丘陵或 DEM")
        valid = np.ones_like(z, dtype=bool)
        metadata = {"source": "parametric", "kind": kind,
                    "vertical_datum": "local project datum", "units": "m",
                    "requested_resolution": resolution, "resampling": "none"}
    if x.ndim != 1 or y.ndim != 1 or len(x) < 2 or len(y) < 2:
        raise ValueError("地形坐标轴至少需要两个采样点")
    if (not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or
            np.any(np.diff(x) <= 0) or np.any(np.diff(y) <= 0)):
        raise ValueError("地形坐标轴须为有限数值，并严格递增")
    if z.shape != (len(y), len(x)) or valid.shape != z.shape:
        raise ValueError("地形高程与有效区矩阵须匹配 [y,x] 坐标轴")
    if len(x) * len(y) > 4_194_304:
        raise ValueError("地形分析网格超过 4,194,304 个采样点；请显式裁剪或重采样")
    if not np.any(valid):
        raise ValueError("地形没有有效高程")
    metadata.update({"analysis_shape": list(z.shape), "analysis_spacing_m": [float(np.min(np.diff(x))), float(np.min(np.diff(y)))],
                     "nodata_count": int(np.size(valid) - np.count_nonzero(valid)),
                     "surface_model": "piecewise bilinear heights; straight-ray visibility"})
    return {"x_m": x, "y_m": y, "z_m": z, "valid": valid, "metadata": metadata}


def sample_height(x, y, grid):
    """Bilinear height, or None for unknown/outside-domain samples."""
    if grid is None:
        return None
    gx, gy, gz, valid = (grid[k] for k in ("x_m", "y_m", "z_m", "valid"))
    if not np.isfinite(x + y) or x < gx[0] or x > gx[-1] or y < gy[0] or y > gy[-1]:
        return None
    ix = int(np.clip(np.searchsorted(gx, x, side="right") - 1, 0, len(gx) - 2))
    iy = int(np.clip(np.searchsorted(gy, y, side="right") - 1, 0, len(gy) - 2))
    wx = (x - gx[ix]) / (gx[ix + 1] - gx[ix])
    wy = (y - gy[iy]) / (gy[iy + 1] - gy[iy])
    weights = np.array([[(1-wx)*(1-wy), wx*(1-wy)], [(1-wx)*wy, wx*wy]])
    support = weights > 1e-12
    cell_valid = valid[iy:iy+2, ix:ix+2]
    if not np.all(cell_valid[support]):
        return None
    values = np.where(cell_valid, gz[iy:iy+2, ix:ix+2], 0.0)
    return float(np.sum(values * weights))


def line_of_sight(a, b, grid):
    """Test terrain along a segment; None means some traversed terrain is unknown.

    At every crossed grid cell the bilinear height along the ray is a quadratic.
    Its interior extremum is tested analytically, so grazing occlusion is not
    missed between fixed ray samples. Surface endpoints do not self-occlude.
    This is geometric height-field visibility, not diffraction or propagation.
    """
    if grid is None:
        return True
    a, b = _vector(a, "ray origin"), _vector(b, "ray end")
    delta = b-a
    gx, gy = grid["x_m"], grid["y_m"]
    spacing = min(np.min(np.diff(grid["x_m"])), np.min(np.diff(grid["y_m"])))
    crossings = [0., 1.]
    for axis, values in ((0, gx), (1, gy)):
        if abs(delta[axis]) > 1e-15:
            fractions = (values-a[axis])/delta[axis]
            crossings.extend(fractions[(fractions > 0) & (fractions < 1)].tolist())
    crossings = np.unique(crossings)
    if len(crossings) > 100_000:
        raise ValueError("视线穿越网格超过计算预算；请裁剪地形")
    unknown = False
    tolerance = max(1e-6, spacing * 1e-7)
    for low, high in zip(crossings[:-1], crossings[1:]):
        middle = (low+high)/2
        centre = a + middle*delta
        if centre[0] < gx[0] or centre[0] > gx[-1] or centre[1] < gy[0] or centre[1] > gy[-1]:
            unknown = True
            continue
        ix = int(np.clip(np.searchsorted(gx, centre[0], side="right")-1, 0, len(gx)-2))
        iy = int(np.clip(np.searchsorted(gy, centre[1], side="right")-1, 0, len(gy)-2))
        # A vertical ray on a known grid boundary only needs its nonzero support.
        if not np.all(grid["valid"][iy:iy+2, ix:ix+2]):
            if sample_height(centre[0], centre[1], grid) is None:
                unknown = True
                continue
        z = np.where(grid["valid"][iy:iy+2, ix:ix+2], grid["z_m"][iy:iy+2, ix:ix+2], 0.)
        ray_points = a[None, :] + np.array([low, middle, high])[:, None]*delta[None, :]
        wx = (ray_points[:, 0]-gx[ix])/(gx[ix+1]-gx[ix])
        wy = (ray_points[:, 1]-gy[iy])/(gy[iy+1]-gy[iy])
        h = z[0,0]*(1-wx)*(1-wy)+z[0,1]*wx*(1-wy)+z[1,0]*(1-wx)*wy+z[1,1]*wx*wy
        g0, gm, g1 = h-ray_points[:, 2]
        qa = 2*(g0+g1-2*gm)
        qb = g1-g0-qa
        maximum = max(g0, g1)
        if qa < 0:
            stationary = -qb/(2*qa)
            if 0 < stationary < 1:
                maximum = max(maximum, qa*stationary**2+qb*stationary+g0)
        if maximum > tolerance:
            return False
    return None if unknown else True


def terrain_scatterers(project, grid=None):
    terrain = project.get("scene", {}).get("terrain", {})
    if not terrain.get("enabled", False):
        return []
    grid = terrain_grid(project) if grid is None else grid
    x, y, z, valid = (grid[k] for k in ("x_m", "y_m", "z_m", "valid"))
    cell_valid = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    count = int(cell_valid.sum())
    maximum = int(terrain.get("max_scatterers", 512))
    if maximum < 1 or count > maximum:
        raise ValueError(f"地形需要 {count} 个散射面元，超过当前预算 {maximum}；请显式降低分析分辨率或增加面元预算")
    sigma0 = 10.0 ** (float(terrain.get("sigma0_db", -20)) / 10.0)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        raise ValueError("地形后向散射系数须为有限数值")
    rng = np.random.default_rng(int(terrain.get("seed", project.get("seed", 42))))
    gains = (rng.standard_normal(count) + 1j*rng.standard_normal(count)) / np.sqrt(2)
    result = []
    for k, (iy, ix) in enumerate(np.argwhere(cell_valid)):
        a = np.array([x[ix], y[iy], z[iy, ix]])
        b = np.array([x[ix+1], y[iy], z[iy, ix+1]])
        c = np.array([x[ix+1], y[iy+1], z[iy+1, ix+1]])
        d = np.array([x[ix], y[iy+1], z[iy+1, ix]])
        area = (np.linalg.norm(np.cross(b-a, c-a)) + np.linalg.norm(np.cross(c-a, d-a))) / 2
        result.append({"id": f"terrain-{iy}-{ix}", "position_m": ((a+b+c+d)/4).tolist(),
                       "velocity_mps": [0., 0., 0.], "rcs_m2": float(sigma0*area),
                       "surface_area_m2": float(area), "complex_gain": complex(gains[k]),
                       "fluctuation": "fixed_complex", "is_terrain": True})
    return result


def import_dem(path, options):
    """Import a north-up GeoTIFF into a rectilinear local metric height field.

    Reprojection uses a local azimuthal-equidistant CRS centred on the selected
    area. Heights retain their explicitly named vertical datum: no geoid
    conversion is implied. max_size, when supplied, explicitly authorizes
    analysis-grid resampling. Original NoData remains invalid.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import array_bounds
    from rasterio.windows import Window, from_bounds, bounds as window_bounds
    from rasterio.warp import calculate_default_transform, reproject
    from pyproj import CRS, Transformer

    options = dict(options or {})
    path = Path(path)
    with rasterio.open(path) as src:
        if src.driver != "GTiff" or src.count != 1:
            raise ValueError("请导入单波段 GeoTIFF 高程文件")
        if src.crs is None:
            raise ValueError("DEM 缺少坐标参考系；请提供带受支持地理坐标的 GeoTIFF")
        epsg = src.crs.to_epsg()
        if epsg not in (4326, 3857) and not (epsg is not None and (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760)):
            raise ValueError("支持的坐标系：EPSG:4326、3857，以及 WGS84 UTM 32601–32660 / 32701–32760")
        if abs(src.transform.b) > 1e-12 or abs(src.transform.d) > 1e-12 or src.transform.a <= 0 or src.transform.e >= 0:
            raise ValueError("DEM 须为北向上的无旋转栅格")
        tags = {**src.tags(), **src.tags(1)}
        unit = options.get("height_unit") or (src.units[0] if src.units else None) or tags.get("UNITTYPE")
        unit = str(unit or "").strip().lower()
        conversion = {"m": 1., "meter": 1., "metre": 1., "meters": 1., "metres": 1.,
                      "ft": .3048, "foot": .3048, "feet": .3048}.get(unit)
        if conversion is None:
            raise ValueError("DEM 高程单位未知；请显式选择米或英尺")
        datum = options.get("vertical_datum") or tags.get("VERTICAL_DATUM") or tags.get("vertical_datum")
        if not isinstance(datum, str) or not datum.strip():
            raise ValueError("DEM 高程基准未指定；请填写基准名称，导入后保留该基准")
        win = Window(0, 0, src.width, src.height)
        if options.get("crop_bounds") is not None:
            crop = _vector(options["crop_bounds"], "crop_bounds", 4)
            if crop[0] >= crop[2] or crop[1] >= crop[3]:
                raise ValueError("裁剪范围须为源坐标系中的 [左,下,右,上]")
            proposed = from_bounds(*crop, transform=src.transform)
            left, top = math.floor(proposed.col_off), math.floor(proposed.row_off)
            right, bottom = math.ceil(proposed.col_off+proposed.width), math.ceil(proposed.row_off+proposed.height)
            try:
                win = Window(left, top, right-left, bottom-top).intersection(win)
            except rasterio.errors.WindowError as exc:
                raise ValueError("裁剪范围与 DEM 不相交") from exc
        width, height = int(win.width), int(win.height)
        if min(width, height) < 2:
            raise ValueError("所选 DEM 区域至少需要 2×2 个源像元")
        limit = int(options.get("max_size", 2048))
        if limit < 2 or limit > 2048:
            raise ValueError("目标网格每轴上限须为 2–2048")
        if max(width, height) > limit and "max_size" not in options:
            raise ValueError("DEM 超过分析上限；请裁剪，或显式设置目标网格尺寸以允许重采样")
        ratio = min(1., limit / max(width, height))
        read_width, read_height = max(2, round(width*ratio)), max(2, round(height*ratio))
        # Source window is read at the explicitly selected analysis scale. The
        # separate nearest mask prevents filling NoData holes with nearby heights.
        values = src.read(1, window=win, out_shape=(read_height, read_width), masked=True,
                          resampling=Resampling.bilinear).astype(np.float64)
        nearest_valid = src.read_masks(1, window=win, out_shape=(read_height, read_width),
                                      resampling=Resampling.nearest) > 0
        scale = float(src.scales[0]) if src.scales else 1.
        offset = float(src.offsets[0]) if src.offsets else 0.
        data = (values.filled(np.nan)*scale + offset)*conversion
        data[~nearest_valid] = np.nan
        bounds = window_bounds(win, src.transform)
        longitude, latitude = Transformer.from_crs(src.crs, 4326, always_xy=True).transform(
            (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2)
        if not np.isfinite(longitude+latitude):
            raise ValueError("无法将 DEM 中心转换到 WGS84 坐标")
        local = CRS.from_proj4(f"+proj=aeqd +lat_0={latitude:.12g} +lon_0={longitude:.12g} +datum=WGS84 +units=m +no_defs")
        read_transform = src.window_transform(win) @ src.transform.scale(width/read_width, height/read_height)
        transform, out_width, out_height = calculate_default_transform(
            src.crs, local, read_width, read_height, *bounds,
            dst_width=read_width, dst_height=read_height)
        destination = np.full((out_height, out_width), np.nan, dtype=np.float64)
        reproject(data, destination, src_transform=read_transform, src_crs=src.crs,
                  src_nodata=np.nan, dst_transform=transform, dst_crs=local,
                  dst_nodata=np.nan, resampling=Resampling.bilinear, num_threads=1)
        validity = np.zeros(destination.shape, dtype=np.uint8)
        reproject((nearest_valid & np.isfinite(data)).astype(np.uint8), validity,
                  src_transform=read_transform, src_crs=src.crs, src_nodata=0,
                  dst_transform=transform, dst_crs=local, dst_nodata=0,
                  resampling=Resampling.nearest, num_threads=1)
        valid = (validity > 0) & np.isfinite(destination)
        if not np.any(valid):
            raise ValueError("所选 DEM 在重投影后没有有效高程")
        xs = transform.c + (np.arange(out_width)+.5)*transform.a
        ys = (transform.f + (np.arange(out_height)+.5)*transform.e)[::-1]
        destination, valid = destination[::-1], valid[::-1]
        metadata = {"source": "GeoTIFF", "source_name": path.name, "source_epsg": epsg,
                    "source_shape": [src.height, src.width], "source_resolution": list(src.res),
                    "source_window": [int(win.col_off), int(win.row_off), width, height],
                    "source_height_unit": unit, "source_scale": scale, "source_offset": offset,
                    "vertical_datum": datum.strip(), "units": "m", "origin_lon_lat": [longitude, latitude],
                    "local_crs": local.to_string(), "analysis_shape": list(destination.shape),
                    "analysis_spacing_m": [float(transform.a), float(-transform.e)],
                    "resampling": "bilinear heights; nearest validity mask; explicit max_size" if ratio < 1 else "bilinear reprojection; nearest validity mask",
                    "nodata_count": int(valid.size-valid.sum()), "curvature_refraction": "local metric height field; no refraction or geoid conversion"}
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    metadata["source_sha256"] = digest.hexdigest()
    heights = destination.astype(object)
    heights[~valid] = None
    return {"enabled": True, "kind": "dem", "x_m": xs.tolist(), "y_m": ys.tolist(),
            "heights_m": heights.tolist(), "valid": valid.tolist(), "metadata": metadata,
            "sigma0_db": float(options.get("sigma0_db", -20)), "seed": int(options.get("seed", 42)),
            "max_scatterers": int(options.get("max_scatterers", 512))}
