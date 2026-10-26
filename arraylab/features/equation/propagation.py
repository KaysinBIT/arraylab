"""Coherent propagation and pinned ITU-R P.676-13 / P.838-3 attenuation.

The gas input pressure is DRY-air pressure in hPa.  Atmospheric models here
describe homogeneous terrestrial-path absorption only: no emitted noise,
refractive bending, dispersion, annual availability or spatial rain statistics.
Path coefficients never contain carrier phase, spreading or atmospheric loss.
See PROPAGATION_SOURCES.md for equation/table provenance and validation.
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

C = 299792458.0
EPSILON_0 = 8.8541878128e-12
GAS_STANDARD = "ITU-R P.676-13 (08/2022), Annex 1"
RAIN_STANDARD = "ITU-R P.838-3 (03/2005)"


@lru_cache(maxsize=2)
def _lines(species):
    source = Path(__file__).with_name("data") / f"p676_13_{species}.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    if data["standard"] != GAS_STANDARD:
        raise RuntimeError("P.676 spectroscopic data version mismatch")
    values = np.asarray(data["rows"], dtype=float)
    values.setflags(write=False)
    return values


def _scalar_or_array(value):
    value = np.asarray(value)
    return float(value) if value.ndim == 0 else value


def _frequency(frequency_hz):
    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(~np.isfinite(frequency)) or np.any(frequency < 1e9) or np.any(frequency > 1e12):
        raise ValueError("ITU attenuation is supported only from 1 to 1000 GHz")
    return frequency / 1e9


def gas_attenuation(frequency_hz, pressure_hpa=1013.25, temperature_k=288.15,
                    vapor_density_g_m3=7.5):
    """Return dry-air and water-vapour attenuation, dB/km (Annex 1 eqs. 1-9).

    pressure_hpa is dry-air pressure p, NOT the total barometric pressure p+e.
    Frequency may have arbitrary array shape; meteorological inputs are scalar.
    """
    f = _frequency(frequency_hz)
    p, temperature, rho = map(float, (pressure_hpa, temperature_k, vapor_density_g_m3))
    if not all(np.isfinite(x) for x in (p, temperature, rho)):
        raise ValueError("Meteorological inputs must be finite")
    if p <= 0 or temperature <= 0 or rho < 0:
        raise ValueError("Dry pressure and temperature must be positive; humidity cannot be negative")
    theta = 300.0 / temperature
    e = rho * temperature / 216.7
    frequency = f[..., None]
    ox = _lines("oxygen")
    f0, a1, a2, a3, a4, a5, a6 = ox.T
    width = a3 * 1e-4 * (p * theta ** (0.8 - a4) + 1.1 * e * theta)
    width = np.sqrt(width * width + 2.25e-6)
    interference = (a5 + a6 * theta) * 1e-4 * (p + e) * theta ** 0.8
    minus, plus = f0 - frequency, f0 + frequency
    shape = frequency / f0 * (
        (width - interference * minus) / (minus * minus + width * width)
        + (width - interference * plus) / (plus * plus + width * width))
    strength = a1 * 1e-7 * p * theta ** 3 * np.exp(a2 * (1 - theta))
    debye = 5.6e-4 * (p + e) * theta ** 0.8
    continuum = f * p * theta ** 2 * (
        6.14e-5 / (debye * (1 + (f / debye) ** 2))
        + 1.4e-12 * p * theta ** 1.5 / (1 + 1.9e-5 * f ** 1.5))
    dry = 0.1820 * f * (np.sum(strength * shape, axis=-1) + continuum)

    wet = _lines("water")
    f0, b1, b2, b3, b4, b5, b6 = wet.T
    width = b3 * 1e-4 * (p * theta ** b4 + b5 * e * theta ** b6)
    width = 0.535 * width + np.sqrt(0.217 * width * width + 2.1316e-12 * f0 * f0 / theta)
    minus, plus = f0 - frequency, f0 + frequency
    shape = frequency / f0 * (width / (minus * minus + width * width)
                              + width / (plus * plus + width * width))
    strength = b1 * 1e-1 * e * theta ** 3.5 * np.exp(b2 * (1 - theta))
    water = 0.1820 * f * np.sum(strength * shape, axis=-1)
    return {"dry_db_km": _scalar_or_array(dry), "water_db_km": _scalar_or_array(water)}


# P.838-3 Tables 1-4: (a_j, b_j, c_j), slope m and intercept c.
_RAIN_FITS = {
    "kh": ([[-5.33980, -0.10008, 1.13098], [-0.35351, 1.26970, .45400],
            [-.23789, .86036, .15354], [-.94158, .64552, .16817]], -.18961, .71147),
    "kv": ([[-3.80595, .56934, .81061], [-3.44965, -.22911, .51059],
            [-.39902, .73042, .11899], [.50167, 1.07319, .27195]], -.16398, .63297),
    "ah": ([[-.14318, 1.82442, -.55187], [.29591, .77564, .19822],
            [.32177, .63773, .13164], [-5.37610, -.96230, 1.47828],
            [16.1721, -3.29980, 3.43990]], .67849, -1.95537),
    "av": ([[-.07771, 2.33840, -.76284], [.56727, .95545, .54039],
            [-.20238, 1.14520, .26809], [-48.2991, .791669, .116226],
            [48.5833, .791459, .116479]], -.053739, .83433),
}


def rain_coefficients(frequency_hz, elevation_deg=0.0, polarization_tilt_deg=0.0):
    """P.838-3 eqs. 2-5; tilt is 0 H, 90 V, 45 circular polarization."""
    f = _frequency(frequency_hz)
    elevation, tilt = float(elevation_deg), float(polarization_tilt_deg)
    if not np.isfinite(elevation) or not -90 <= elevation <= 90 or not np.isfinite(tilt):
        raise ValueError("Invalid path elevation or polarization tilt")
    logf = np.log10(f)
    fitted = {}
    for key, (coefficients, slope, intercept) in _RAIN_FITS.items():
        a, b, c = np.asarray(coefficients).T
        fitted[key] = (np.sum(a * np.exp(-((logf[..., None] - b) / c) ** 2), axis=-1)
                       + slope * logf + intercept)
    kh, kv = 10 ** fitted["kh"], 10 ** fitted["kv"]
    ah, av = fitted["ah"], fitted["av"]
    factor = np.cos(np.deg2rad(elevation)) ** 2 * np.cos(2 * np.deg2rad(tilt))
    k = (kh + kv + (kh - kv) * factor) / 2
    alpha = (kh * ah + kv * av + (kh * ah - kv * av) * factor) / (2 * k)
    return {"k": _scalar_or_array(k), "alpha": _scalar_or_array(alpha)}


def specific_attenuation(frequency_hz, atmosphere):
    """Return gas/rain/total dB/km; disabled atmosphere returns exact zeros."""
    if not atmosphere.get("enabled", False):
        f = np.asarray(frequency_hz, dtype=float)
        if np.any(~np.isfinite(f)) or np.any(f <= 0):
            raise ValueError("Frequency must be finite and positive")
        zero = _scalar_or_array(np.zeros_like(f))
        return {"gas_db_km": zero, "rain_db_km": zero, "total_db_km": zero}
    _frequency(frequency_hz)
    gas = gas_attenuation(frequency_hz, atmosphere.get("pressure_hpa", 1013.25),
                          atmosphere.get("temperature_k", 288.15),
                          atmosphere.get("vapor_density_g_m3", 7.5))
    rate = float(atmosphere.get("rain_rate_mm_h", 0.0))
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("Rain rate must be finite and nonnegative")
    rain = rain_coefficients(frequency_hz, atmosphere.get("elevation_deg", 0.0),
                             atmosphere.get("polarization_tilt_deg", 0.0))
    rain_db = np.asarray(rain["k"]) * rate ** np.asarray(rain["alpha"])
    gas_db = np.asarray(gas["dry_db_km"]) + np.asarray(gas["water_db_km"])
    return {"gas_db_km": _scalar_or_array(gas_db), "rain_db_km": _scalar_or_array(rain_db),
            "total_db_km": _scalar_or_array(gas_db + rain_db)}


def path_attenuation(frequency_hz, length_m, atmosphere):
    """Amplitude transfer for the supplied FULL path length, counted once."""
    length = np.asarray(length_m, dtype=float)
    if np.any(~np.isfinite(length)) or np.any(length < 0):
        raise ValueError("Path length must be finite and nonnegative")
    gamma = np.asarray(specific_attenuation(frequency_hz, atmosphere)["total_db_km"])
    return _scalar_or_array(10 ** (-gamma * length / 1000.0 / 20.0))


def attenuation_for_path(frequency_hz, path, atmosphere):
    """Apply rain polarization/elevation to each leg, not an average angle."""
    result = np.ones_like(np.asarray(frequency_hz, dtype=float))
    for distance, direction in ((path["tx_distance_m"], path["departure"]),
                                (path["rx_distance_m"], path["arrival"])):
        leg = dict(atmosphere)
        leg["elevation_deg"] = float(np.rad2deg(np.arcsin(np.clip(direction[2], -1, 1))))
        result *= path_attenuation(frequency_hz, distance, leg)
    return _scalar_or_array(result)


def fresnel_coefficient(frequency_hz, grazing_angle_rad, epsilon_r, conductivity_s_m=0.,
                        polarization="horizontal"):
    """Electric-field reflection on a smooth dielectric half-space.

    e^(+j omega t) convention; passive epsilon = epsilon_r - j sigma/(omega eps0).
    Vertical sign follows the standard local p-polarization Fresnel basis.
    """
    frequency = np.asarray(frequency_hz, float)
    angle = float(grazing_angle_rad)
    if np.any(~np.isfinite(frequency)) or np.any(frequency <= 0):
        raise ValueError("Reflection frequency must be positive")
    if not 0 <= angle <= np.pi / 2 or epsilon_r < 1 or conductivity_s_m < 0:
        raise ValueError("Invalid half-space reflection parameters")
    if polarization not in ("horizontal", "vertical"):
        raise ValueError("Reflection polarization must be horizontal or vertical")
    epsilon = complex(epsilon_r) - 1j * conductivity_s_m / (2 * np.pi * frequency * EPSILON_0)
    sine = np.sin(angle)
    # This equivalent form avoids losing sin²(grazing) near grazing incidence,
    # especially when identical media should produce exactly zero reflection.
    root = np.sqrt((epsilon - 1) + sine * sine + 0j)
    leading = sine if polarization == "horizontal" else epsilon * sine
    numerator, denominator = leading - root, leading + root
    # Identical media have zero reflection, including the grazing 0/0 limit.
    value = np.divide(numerator, denominator, out=np.zeros_like(root), where=np.abs(denominator) > 0)
    return complex(value) if value.ndim == 0 else value


def _direction(angles):
    azimuth, elevation = np.deg2rad(np.asarray(angles, dtype=float))
    return np.array([np.cos(elevation) * np.cos(azimuth),
                     np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])


def paths_for_target(config, target):
    """Construct direct, four-ray or direct-plus-custom paths.

    Departure and arrival are ARRAY-end look vectors, pointing from the radar
    to the real or image target, not the incoming wave propagation vector.
    Reflected-path Doppler uses geometry length derivatives at the TARGET;
    its vertical component cannot be inferred from array-end look vectors.
    A scalar radial velocity defines a 3-D velocity along the direct LOS.
    """
    from .config import target_position
    propagation = config["propagation"]
    model = propagation["model"]
    if model not in ("free", "two_ray", "custom"):
        raise ValueError("Unknown propagation model")
    origin = np.array([0., 0., float(config["scene"]["radar_height_m"])])
    position = np.asarray(target_position(config, target), dtype=float)
    direct_vector = position - origin
    distance = float(np.linalg.norm(direct_vector))
    if distance <= 0:
        raise ValueError("Target cannot coincide with the radar")
    direct = direct_vector / distance
    velocity = direct * float(target["velocity_mps"])
    frequency = float(config["waveform"]["fc_hz"])
    wavelength = C / frequency

    def path(name, tx_distance, rx_distance, coefficient, departure, arrival, doppler):
        length = float(tx_distance + rx_distance)
        return {"name": name, "length_m": length, "delay_s": length / C,
                "coefficient": complex(coefficient), "departure": np.asarray(departure, float),
                "arrival": np.asarray(arrival, float), "doppler_hz": float(doppler),
                "tx_distance_m": float(tx_distance), "rx_distance_m": float(rx_distance)}

    answer = [path("direct", distance, distance, 1., direct, direct,
                   2 * float(target["velocity_mps"]) / wavelength)]
    if model == "free":
        return answer
    if model == "custom":
        for i, custom in enumerate(propagation.get("custom_paths", [])):
            extra = float(custom["extra_delay_s"]) * C
            if extra < 0:
                raise ValueError("Custom excess delay cannot be negative")
            coefficient = 10 ** (-float(custom["attenuation_db"]) / 20) * np.exp(
                1j * np.deg2rad(float(custom["phase_deg"])))
            # Equal legs define a declared phenomenological relative-path model.
            leg = distance + extra / 2
            answer.append(path(f"custom_{i+1}", leg, leg, coefficient,
                               _direction(custom["departure_deg"]), _direction(custom["arrival_deg"]),
                               answer[0]["doppler_hz"] + float(custom["doppler_hz"])))
        return answer

    if origin[2] <= 0 or position[2] <= 0:
        raise ValueError("Flat-ground two-ray geometry requires both heights above zero")
    image = position.copy()
    image[2] *= -1
    reflected_vector = image - origin
    reflected_distance = float(np.linalg.norm(reflected_vector))
    reflected = reflected_vector / reflected_distance
    grazing = float(np.arcsin(np.clip((origin[2] + position[2]) / reflected_distance, 0, 1)))
    reflection = fresnel_coefficient(frequency, grazing, propagation["epsilon_r"],
                                     propagation["conductivity_s_m"], propagation["polarization"])
    # Derivative wrt target position, whose mirror has opposite vertical motion.
    target_end = reflected.copy()
    target_end[2] *= -1
    direct_rate = float(np.dot(direct, velocity))
    reflected_rate = float(np.dot(target_end, velocity))
    one_bounce_doppler = (direct_rate + reflected_rate) / wavelength
    answer += [path("tx_ground", reflected_distance, distance, reflection, reflected, direct,
                    one_bounce_doppler),
               path("rx_ground", distance, reflected_distance, reflection, direct, reflected,
                    one_bounce_doppler),
               path("both_ground", reflected_distance, reflected_distance, reflection ** 2,
                    reflected, reflected, 2 * reflected_rate / wavelength)]
    # Expose grazing/bounce metadata so the engine can use frequency-dependent
    # Fresnel coefficients for wideband synthesis without including phase twice.
    for item, bounces in zip(answer, (0, 1, 1, 2)):
        item["ground_bounces"] = bounces
        item["grazing_angle_rad"] = grazing
    return answer


def coefficient_for_path(frequency_hz, path, propagation):
    """Wideband reflection/custom factor, still excluding carrier phase/loss."""
    bounces = path.get("ground_bounces", 0)
    if bounces:
        return fresnel_coefficient(frequency_hz, path["grazing_angle_rad"],
                                  propagation["epsilon_r"], propagation["conductivity_s_m"],
                                  propagation["polarization"]) ** bounces
    frequency = np.asarray(frequency_hz)
    result = np.full(frequency.shape, path["coefficient"], dtype=complex)
    return complex(result) if result.ndim == 0 else result
