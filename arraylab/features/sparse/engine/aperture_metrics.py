"""Bounded pairwise work for large sparse-layout figure measurements."""
import math
import numpy as np

from arraylab.shared.numerics.array_fields import integrated_power as _small_integrated_power


def maximum_distance(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.
    rows = max(1, min(len(points), (16 * 1024**2) // (32 * len(points))))
    largest = 0.
    for start in range(0, len(points), rows):
        distances = np.linalg.norm(points[start:start+rows, None, :] - points[None, :, :], axis=2)
        largest = max(largest, float(distances.max()))
    return largest


def integrated_power(points, weights):
    """The same full-sphere sinc integral, without an N-by-N resident matrix."""
    points = np.asarray(points, dtype=float)
    weights = np.asarray(weights, dtype=complex)
    if len(points) <= 512:
        return _small_integrated_power(points, weights)
    rows = max(1, min(len(points), (16 * 1024**2) // (64 * len(points))))

    def terms():
        for start in range(0, len(points), rows):
            distance = np.linalg.norm(points[start:start+rows, None, :] - points[None, :, :], axis=2)
            values = (weights[start:start+rows, None] * weights[None, :].conj()).real * np.sinc(2 * distance)
            yield from values.ravel()

    power = math.fsum(terms())
    if power <= float(np.sum(abs(weights)**2)) * 1e-9:
        # Preserve the original quadrature check, evaluating one latitude at a
        # time rather than materializing the complete sphere-by-elements mesh.
        from arraylab.features.sparse.engine.field_math import field
        order = max(40, math.ceil(8 * maximum_distance(points) + 20))
        if order * order * 32 > 128 * 1024**2:
            raise ArithmeticError('大孔径辐射积分发生严重抵消，所需积分复核超出临时内存预算。')
        z, quadrature = np.polynomial.legendre.leggauss(order)
        phi = np.arange(2 * order) * (2 * np.pi / (2 * order))
        circle = np.column_stack((np.cos(phi), np.sin(phi)))
        centered = points - np.mean(points, axis=0)
        power = math.fsum(float(q * np.mean(abs(field(centered, weights, math.sqrt(1-height*height)*circle))**2) / 2)
                          for height, q in zip(z, quadrature))
    if power <= 0 or not math.isfinite(power):
        raise ArithmeticError('平面阵总辐射功率无法可靠求得。')
    return power
