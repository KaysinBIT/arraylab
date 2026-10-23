"""Uniform isotropic linear-array calculations, independent of the UI.

Array axis: x. Cut: x-z, z >= 0. theta is signed from +z towards +x.
The radiation intensity is rotationally symmetric about x. Directivity uses
the full sphere, even though the plotted cut shows only -90..90 degrees.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

C0 = 299_792_458.0
DATA_FLOOR_DB = -300.0


class InputError(ValueError):
    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


@dataclass(frozen=True)
class ArrayParameters:
    elements: int = 8
    frequency_ghz: float = 3.0
    spacing: float = 0.5
    spacing_unit: str = "lambda"
    scan_deg: float = 0.0
    step_deg: float = 0.1

    @property
    def wavelength_mm(self) -> float:
        return C0 / (self.frequency_ghz * 1e9) * 1e3

    @property
    def spacing_lambda(self) -> float:
        return self.spacing if self.spacing_unit == "lambda" else self.spacing / self.wavelength_mm

    @property
    def spacing_mm(self) -> float:
        return self.spacing_lambda * self.wavelength_mm

    def validate(self) -> None:
        if isinstance(self.elements, bool) or not isinstance(self.elements, (int, np.integer)):
            raise InputError("elements", "阵元数必须是整数。")
        if not 1 <= self.elements <= 200:
            raise InputError("elements", "阵元数须为 1–200。")
        for key in ("frequency_ghz", "spacing", "scan_deg", "step_deg"):
            if not math.isfinite(getattr(self, key)):
                raise InputError(key, "输入须为有限数值。")
        if not 0.001 <= self.frequency_ghz <= 1000:
            raise InputError("frequency_ghz", "首版频率范围为 0.001–1000 GHz。")
        if self.spacing_unit not in ("lambda", "mm"):
            raise InputError("spacing_unit", "间距单位须为 λ 或 mm。")
        if self.spacing <= 0 or not 1e-6 <= self.spacing_lambda <= 10:
            raise InputError("spacing", "首版间距范围为 0.000001–10 λ；请检查数值和单位。")
        if not -90 <= self.scan_deg <= 90:
            raise InputError("scan_deg", "扫描角须在 −90° 至 90°之间。")
        if not 0.01 <= self.step_deg <= 5:
            raise InputError("step_deg", "角度步长须在 0.01° 至 5°之间。")


def normalized_field_u(u: NDArray | float, p: ArrayParameters) -> NDArray:
    """Real field for equal weights with the phase reference at the array center.

Use the analytic removable limit at grating peaks, not a small denominator
offset. The sign is retained because an even-N centered array changes phase
by pi at some periodic peaks.
    """
    q = p.spacing_lambda * (np.asarray(u, dtype=float) - np.sin(np.deg2rad(p.scan_deg)))
    if p.elements == 1:
        return np.ones_like(q)
    denominator = np.sinc(q)
    out = np.empty_like(q)
    regular = np.abs(denominator) > 1e-12
    np.divide(np.sinc(p.elements * q), denominator, out=out, where=regular)
    np.copyto(out, np.cos(np.pi * p.elements * q) / np.where(
        np.abs(np.cos(np.pi * q)) > 1e-12, np.cos(np.pi * q), 1
    ), where=~regular)
    return out


def pattern_db_at(theta_deg: NDArray | float, p: ArrayParameters) -> NDArray:
    field = normalized_field_u(np.sin(np.deg2rad(theta_deg)), p)
    return 20 * np.log10(np.maximum(np.abs(field), 10 ** (DATA_FLOOR_DB / 20)))


def full_sphere_directivity(p: ArrayParameters) -> float:
    """Exact full-sphere integral of the finite isotropic array factor.

Integral/(4*pi) = N + 2 sum_l (N-l) sinc(2*d*l) cos(2*pi*d*l*u0).
The numerator is N**2. This is NOT a 2-D pattern integral or gain estimate.
    """
    lag = np.arange(1, p.elements, dtype=float)
    u0 = np.sin(np.deg2rad(p.scan_deg))
    terms = 2 * (p.elements - lag) * np.sinc(2 * p.spacing_lambda * lag) * np.cos(
        2 * np.pi * p.spacing_lambda * lag * u0
    )
    denominator = math.fsum([float(p.elements), *terms.tolist()])
    if not math.isfinite(denominator) or denominator <= 0:
        raise ArithmeticError("总辐射功率积分异常，请检查参数。")
    return p.elements ** 2 / denominator


def equal_peak_angles(p: ArrayParameters) -> tuple[float, ...]:
    if p.elements == 1:
        return ()
    u0 = float(np.sin(np.deg2rad(p.scan_deg)))
    d = p.spacing_lambda
    lo = math.ceil(d * (-1 - u0) - 1e-10)
    hi = math.floor(d * (1 - u0) + 1e-10)
    return tuple(float(np.rad2deg(np.arcsin(np.clip(u0 + m / d, -1, 1)))) for m in range(lo, hi + 1))


def half_power_edges(p: ArrayParameters) -> tuple[float, float] | None:
    if p.elements == 1 or len(equal_peak_angles(p)) != 1:
        return None
    u0 = float(np.sin(np.deg2rad(p.scan_deg)))
    first_null = 1 / (p.elements * p.spacing_lambda)
    left, right = max(-1.0, u0 - first_null), min(1.0, u0 + first_null)

    def f(u: float) -> float:
        return float(normalized_field_u(u, p) ** 2 - 0.5)

    if f(left) > 0 or f(right) > 0:
        return None
    roots = (brentq(f, left, u0, xtol=1e-14), brentq(f, u0, right, xtol=1e-14))
    return tuple(float(np.rad2deg(np.arcsin(np.clip(u, -1, 1)))) for u in roots)


@dataclass(frozen=True)
class ArrayResult:
    parameters: ArrayParameters
    angles_deg: NDArray
    pattern_db: NDArray
    phase_deg: NDArray
    positions_mm: NDArray
    excitation_phase_deg: NDArray
    directivity_dbi: float
    peaks_deg: tuple[float, ...]
    half_power_deg: tuple[float, float] | None
    notices: tuple[str, ...]

    @property
    def beamwidth_deg(self) -> float | None:
        return None if self.half_power_deg is None else self.half_power_deg[1] - self.half_power_deg[0]


def calculate(p: ArrayParameters) -> ArrayResult:
    p.validate()
    angles = np.arange(-90.0, 90.0, p.step_deg)
    angles = np.append(angles[angles < 90.0 - 1e-10], 90.0)
    field = normalized_field_u(np.sin(np.deg2rad(angles)), p)
    pattern = 20 * np.log10(np.maximum(np.abs(field), 10 ** (DATA_FLOOR_DB / 20)))
    phase = np.where(field < 0, 180.0, 0.0)
    offsets = np.arange(p.elements) - (p.elements - 1) / 2
    excitation_phase = -360 * p.spacing_lambda * offsets * np.sin(np.deg2rad(p.scan_deg))
    excitation_phase = (excitation_phase + 180) % 360 - 180
    peaks = equal_peak_angles(p)
    edges = half_power_edges(p)
    notices = []
    if p.elements == 1:
        notices.append("单个各向同性单元：全向辐射，无唯一主瓣或半功率波束宽度。")
    elif len(peaks) > 1:
        notices.append("所示切面存在多个等高主瓣（含栅瓣/端射对称峰）：" + "、".join(f"{a:.2f}°" for a in peaks) + "；不报告唯一主瓣宽度。")
    elif edges is None:
        notices.append("主瓣半功率边界未全部落在 −90° 至 90° 范围内，波束宽度不报告。")
    if edges and p.step_deg > (edges[1] - edges[0]) / 10:
        notices.append("绘图步长较粗，建议减小到波束宽度的 1/10 以下；当前指标独立于绘图采样计算。")
    positions = offsets * p.spacing_mm
    for array in (angles, pattern, phase, positions, excitation_phase):
        array.setflags(write=False)
    return ArrayResult(
        p, angles, pattern, phase, positions, excitation_phase,
        10 * math.log10(full_sphere_directivity(p)), peaks, edges, tuple(notices),
    )
