"""Array estimators operating exclusively on received complex snapshots."""
from __future__ import annotations

import math
import numpy as np

from .scene import C, element_positions


def _unique_rows(positions, tolerance):
    # Relative coordinates avoid depending on a potentially huge world origin.
    centred = positions - positions[0]
    _, indices = np.unique(np.round(centred/tolerance), axis=0, return_index=True)
    return np.sort(indices)


def _aperture(p):
    if len(p) < 2:
        return 0.
    return float(np.max(np.linalg.norm(p[:, None, :]-p[None, :, :], axis=-1)))


def array_metrics(array, fc_hz):
    tx, rx = element_positions(array, "tx", fc_hz), element_positions(array, "rx", fc_hz)
    wavelength = C/fc_hz
    virtual = (tx[:, None, :]+rx[None, :, :]).reshape(-1, 3)
    unique = _unique_rows(virtual, wavelength*1e-8)
    positions = virtual[unique] if array.get("mode", "phased") in ("tdm", "cdm") else rx
    aperture = _aperture(positions)
    # Broadside azimuth/elevation derivatives use y/z projected aperture.
    # The diagonal diameter is appropriate for a conservative far-field check,
    # but using it as the azimuth aperture overstates a planar array's resolution.
    az_aperture=float(np.ptp(positions[:,1]))
    el_aperture=float(np.ptp(positions[:,2]))
    az_hpbw=float(np.rad2deg(.886*wavelength/az_aperture)) if az_aperture else None
    el_hpbw=float(np.rad2deg(.886*wavelength/el_aperture)) if el_aperture else None
    return {"tx_count": len(tx), "rx_count": len(rx), "physical_channels": len(tx)+len(rx),
            "tx_rx_pairs": len(tx)*len(rx), "independent_virtual_positions": len(unique),
            "used_angle_channels": len(positions), "tx_aperture_m": _aperture(tx),
            "rx_aperture_m": _aperture(rx), "virtual_aperture_m": _aperture(virtual[unique]),
            "angle_aperture_m": aperture, "far_field_distance_m": 2*aperture**2/wavelength,
            "broadside_hpbw_proxy_deg": az_hpbw,
            "azimuth_aperture_m":az_aperture,"elevation_aperture_m":el_aperture,
            "azimuth_hpbw_proxy_deg":az_hpbw,"elevation_hpbw_proxy_deg":el_hpbw,
            "tx_positions_m": tx.tolist(), "rx_positions_m": rx.tolist(),
            "virtual_positions_m": virtual[unique].tolist(),
            "note": "相控模式使用物理接收通道；MIMO 虚拟阵列要求正确分离发射通道并补偿运动相位。方位／俯仰 HPBW 代理分别取实际 y／z 投影孔径；阵列直径仅用于远场检查。代理不保证双目标分辨率。"}


def _axis(value, step, name):
    bounds = np.asarray(value, dtype=float)
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] > bounds[1]:
        raise ValueError(f"{name} 必须为 [下限,上限]")
    if bounds[0] < -90 or bounds[1] > 90:
        raise ValueError(f"{name} 必须处于前半空间 [-90°,90°]")
    requested = (bounds[1]-bounds[0])/step
    if not np.isfinite(requested) or requested > 100_000:
        raise ValueError(f"{name} 超过角度搜索网格预算")
    n = int(np.ceil(requested)) + 1
    return np.linspace(bounds[0], bounds[1], n)


def _smooth(x, positions, length, wavelength):
    centred = positions-positions.mean(axis=0)
    _, singular, axes = np.linalg.svd(centred, full_matrices=False)
    if singular[0] <= wavelength*1e-10 or (len(singular)>1 and singular[1] > singular[0]*1e-7):
        raise ValueError("空间平滑要求共线的均匀线阵")
    projection = centred @ axes[0]
    order = np.argsort(projection)
    separations = np.diff(projection[order])
    spacing = float(np.mean(separations))
    if not np.allclose(separations, spacing, rtol=1e-6, atol=wavelength*1e-9):
        raise ValueError("空间平滑要求连续、等间距的均匀线阵")
    if spacing > wavelength/2*(1+1e-7):
        raise ValueError("空间平滑要求线阵间距不超过半波长")
    n = len(positions)
    length = int(length)
    if length < 2 or length >= n:
        raise ValueError("平滑子阵长度须至少为 2，且小于独立通道数")
    ordered = x[order]
    blocks = [ordered[i:i+length] for i in range(n-length+1)]
    return np.concatenate(blocks, axis=1), positions[order[:length]], n-length+1


def estimate_angles(snapshots, positions_m, frequency_hz, settings):
    """Bartlett/MVDR/MUSIC on actual snapshots; exp(+j k position·direction).

    The returned spectra are normalized estimator outputs, not absolute gains.
    No target list or ground-truth angle is accepted by this interface.
    """
    settings = settings or {}
    x = np.asarray(snapshots, dtype=np.complex128)
    if x.ndim == 1:
        x = x[:, None]
    p = np.asarray(positions_m, dtype=float)
    if x.ndim != 2 or p.shape != (x.shape[0], 3) or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("快拍矩阵须为 [通道,快拍]，阵元坐标须为 [通道,3]")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(p)) or not np.isfinite(frequency_hz) or frequency_hz <= 0:
        raise ValueError("阵列数据须为有限数值，频率须为正数")
    method = settings.get("method", "bartlett").lower()
    if method not in ("bartlett", "mvdr", "music"):
        raise ValueError("测角方法须为 Bartlett、MVDR 或 MUSIC")
    count = int(settings.get("source_count", 1))
    if count < 1 or count > 64:
        raise ValueError("指定源数须为 1–64")
    wavelength = C/frequency_hz
    warnings = []
    input_channels, input_snapshots = x.shape
    unique = _unique_rows(p, wavelength*1e-8)
    if len(unique) != len(p):
        warnings.append("虚拟位置重合：每个位置只保留一路实测通道，不额外计入未经校准的合并增益。")
        p, x = p[unique], x[unique]
    smoothing = settings.get("smoothing", False)
    subarrays = 1
    if smoothing:
        length = (len(p)//2+1) if isinstance(smoothing, bool) else int(smoothing)
        x, p, subarrays = _smooth(x, p, length, wavelength)
        warnings.append("前向空间平滑缩短有效孔径；重叠子阵之间相关，不代表新增独立观测。")
    channels, samples = x.shape
    if method == "music" and (count >= channels or samples <= count):
        raise ValueError("MUSIC 要求指定源数小于有效通道数，且有效快拍数多于源数")
    if channels < 2:
        warnings.append("只有一个独立阵元位置，无法确定角度。")
    if samples < channels and method != "bartlett":
        warnings.append("快拍数少于协方差维数；对角加载或平滑不能替代缺失的独立观测。")
    step = float(settings.get("step_deg", 1))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("测角步长必须为正数")
    az = _axis(settings.get("scan_az_deg", [-60, 60]), step, "scan_az_deg")
    el = _axis(settings.get("scan_el_deg", [0, 0]), step, "scan_el_deg")
    if len(az)*len(el)*channels > 8_000_000:
        raise ValueError("角度搜索超过 800 万个导向系数；请缩小扫描范围或增大步长")
    azm, elm = np.meshgrid(np.deg2rad(az), np.deg2rad(el))
    directions = np.vstack([(np.cos(elm)*np.cos(azm)).ravel(),
                            (np.cos(elm)*np.sin(azm)).ravel(), np.sin(elm).ravel()])
    p = p-p[0]
    steering = np.exp(2j*np.pi/wavelength*(p @ directions))
    covariance = x @ x.conj().T / samples
    covariance = (covariance+covariance.conj().T)/2
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = float(np.trace(covariance).real/channels)
    if scale <= np.finfo(float).tiny:
        raise ValueError("快拍中没有信号功率，无法定义角度谱")
    relative_loading = float(settings.get("diagonal_loading", 1e-3))
    if not np.isfinite(relative_loading) or relative_loading < 0:
        raise ValueError("相对对角加载必须为非负有限数值")
    loading = relative_loading*scale
    loaded_condition = None
    if method == "bartlett":
        spectrum = np.real(np.sum(steering.conj()*(covariance @ steering), axis=0))/channels**2
    elif method == "mvdr":
        loaded = covariance + np.eye(channels)*loading
        loaded_condition = float(np.linalg.cond(loaded))
        if not np.isfinite(loaded_condition) or loaded_condition > 1e13:
            raise ValueError("MVDR 协方差病态；请增加对角加载或独立快拍")
        inverse_a = np.linalg.solve(loaded, steering)
        denominator = np.real(np.sum(steering.conj()*inverse_a, axis=0))
        spectrum = 1/np.maximum(denominator, np.finfo(float).tiny)
    else:
        noise_vectors = eigenvectors[:, :channels-count]
        denominator = np.sum(np.abs(noise_vectors.conj().T @ steering)**2, axis=0)
        spectrum = 1/np.maximum(denominator, np.finfo(float).eps*channels)
    effective_rank = int(np.count_nonzero(eigenvalues > eigenvalues[-1]*1e-6))
    if method == "music" and effective_rank < count:
        warnings.append("协方差有效秩低于指定源数；相干目标或观测不足使当前 MUSIC 模型失效。")
    spectrum = np.maximum(spectrum, np.finfo(float).tiny).reshape(len(el), len(az))
    spectrum_db = np.maximum(10*np.log10(spectrum/spectrum.max()), -160)
    # True grid-local maxima, without wrapping at scan boundaries.
    candidates = np.ones(spectrum.shape, dtype=bool)
    for dy, dx in ((-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)):
        shifted = np.full_like(spectrum, -np.inf)
        ys = slice(max(0, dy), min(len(el), len(el)+dy))
        xs = slice(max(0, dx), min(len(az), len(az)+dx))
        shifted[ys, xs] = spectrum[slice(max(0,-dy), min(len(el),len(el)-dy)),
                                     slice(max(0,-dx), min(len(az),len(az)-dx))]
        candidates &= spectrum >= shifted
    peaks = [];peak_indices=[]
    if channels > 1 and np.ptp(spectrum_db) > 1e-8:
        indices = np.argwhere(candidates)
        indices = sorted(indices.tolist(), key=lambda ij: spectrum[ij[0],ij[1]], reverse=True)
        for iy, ix in indices:
            if any(abs(az[ix]-v["azimuth_deg"]) <= step and abs(el[iy]-v["elevation_deg"]) <= step for v in peaks):
                continue
            peaks.append({"azimuth_deg": float(az[ix]), "elevation_deg": float(el[iy]),
                          "relative_db": float(spectrum_db[iy,ix])})
            peak_indices.append((iy,ix))
            if len(peaks) >= count:
                break
    # Requested source count is not the number of distinguishable directions.
    # Retain exact manifold aliases independently of K. A lambda-spaced ULA,
    # for example, cannot distinguish +/-30 degrees even when K is known to be 1.
    ambiguous=[];all_aliases=set()
    if peak_indices:
        for iy,ix in peak_indices:
            ref=steering[:,iy*len(az)+ix]
            coherence=np.abs(ref.conj()@steering)/channels
            same=(coherence.reshape(spectrum.shape)>=1-1e-8)&candidates
            for ay,ax in np.argwhere(same):
                if abs(az[ax]-az[ix])<=step and abs(el[ay]-el[iy])<=step:continue
                all_aliases.add((int(ay),int(ax)))
        for iy,ix in sorted(all_aliases,key=lambda ij:-spectrum[ij[0],ij[1]])[:16]:
            ambiguous.append({'azimuth_deg':float(az[ix]),'elevation_deg':float(el[iy]),
                              'relative_db':float(spectrum_db[iy,ix]),
                              'reason':'阵列导向矢量仅相差公共相位，当前通道不能区分'})
        if ambiguous:
            warnings.append('当前扫描域存在不可区分的角度导向候选；指定源数不消除空间歧义，不能把单个最高峰作为唯一角度。')
    aperture = _aperture(p)
    if channels > 1:
        distances = np.linalg.norm(p[:,None,:]-p[None,:,:], axis=-1)
        np.fill_diagonal(distances, np.inf)
        minimum_spacing = float(np.min(distances))
        if minimum_spacing > wavelength/2*(1+1e-7):
            warnings.append("最小阵元间距超过半波长，扫描域内可能存在空间混叠候选。")
    else:
        minimum_spacing = None
    singular_positions = np.linalg.svd(p-p.mean(axis=0), compute_uv=False)
    geometry_rank = int(np.count_nonzero(singular_positions > wavelength*1e-8))
    if geometry_rank < 2 and len(el)>1:
        warnings.append("线阵只测量一个方向余弦，同时估计方位和俯仰可能存在歧义。")
    if len(el)==1:
        warnings.append(f'俯仰固定在 {float(el[0]):g}° 切面，属于输入假设，不是独立俯仰测量。')
    if len(az)==1:
        warnings.append(f'方位固定在 {float(az[0]):g}° 切面，属于输入假设，不是独立方位测量。')
    return {"peaks": peaks, "azimuth_deg": az.tolist(), "elevation_deg": el.tolist(),
            "spectrum_db": spectrum_db.tolist(),
            "ambiguous_peaks":ambiguous,
            "diagnostics": {"method": method, "input_channels": input_channels,
                            "unique_positions": len(unique), "effective_channels": channels,
                            "input_snapshots": input_snapshots, "effective_snapshot_columns": samples,
                            "smoothing_subarrays": subarrays, "effective_aperture_m": aperture,
                            "far_field_distance_m": 2*aperture**2/wavelength,
                            "minimum_spacing_m": minimum_spacing,
                            "geometry_rank": geometry_rank, "covariance_eigenvalues": eigenvalues[::-1].real.tolist(),
                            "effective_rank_relative_1e-6": effective_rank,
                            "loading_power": loading, "relative_loading": relative_loading,
                            "loaded_condition_number": loaded_condition,
                            "source_count": count, "grid_step_deg_requested": step,
                            "angle_frame":"radar body frame at CPI reference",
                            "azimuth_is_fixed":len(az)==1,"elevation_is_fixed":len(el)==1,
                            "fixed_azimuth_deg":float(az[0]) if len(az)==1 else None,
                            "fixed_elevation_deg":float(el[0]) if len(el)==1 else None,
                            "ambiguous_peak_count":len(all_aliases),
                            "ambiguous_peaks_report_limit":16,
                            "model": "已校准的窄带远场快拍；归一化估计器谱"},
            "warnings": warnings}
