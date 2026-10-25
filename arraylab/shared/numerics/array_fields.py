"""Complex x-y aperture fields and full-sphere isotropic power.

Both functions consume wavelength-normalized coordinate arrays and complex
weights, without importing any business model, renderer or application state.
"""
import math
import numpy as np


def field_uv(directions, coordinates, weights):
    directions = np.asarray(directions, dtype=float)
    original_shape = directions.shape[:-1]
    directions = directions.reshape(-1, 2)
    output = np.empty(len(directions), dtype=complex)
    # Translation only adds a common field phase; power and directivity are invariant.
    centered = coordinates - np.mean(coordinates, axis=0)
    for i in range(0, len(directions), 512):
        block = directions[i:i + 512]
        phase = block[:, 0, None] * centered[:, 0] + block[:, 1, None] * centered[:, 1]
        output[i:i + 512] = np.sum(weights * np.exp(2j * np.pi * phase), axis=1)
    return output.reshape(original_shape)


def integrated_power(coordinates, weights):
    distance = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    terms = (weights[:, None] * weights[None, :].conj()).real * np.sinc(2 * distance)
    power = math.fsum(terms.ravel().tolist())
    if power <= float(np.sum(abs(weights)**2)) * 1e-9:
        diameter = float(np.max(distance))
        order = max(40, math.ceil(8 * diameter + 20))
        z, quadrature = np.polynomial.legendre.leggauss(order)
        phi = np.arange(2 * order) * (2 * np.pi / (2 * order))
        radius = np.sqrt(1-z*z)
        uv = np.stack([radius[:,None]*np.cos(phi), radius[:,None]*np.sin(phi)], axis=-1)
        values = abs(field_uv(uv, coordinates, weights))**2
        power = float(np.sum(quadrature * np.mean(values, axis=1)) / 2)
    if power <= 0 or not math.isfinite(power): raise ArithmeticError("平面阵总辐射功率无法可靠求得。")
    return power
