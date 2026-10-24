"""Exact sparse-array geometry using bounded temporary allocations."""
import numpy as np

from .resources import FIELD_WORKSPACE_BYTES, _memory_state, format_bytes


def coarray(integer):
    """Count all ordered differences without storing the N by N difference set.

    Lag order and equal-area rectangle ties match the original np.unique and
    increasing-rx/increasing-ry search. No FFT approximation is involved.
    """
    points = np.asarray(integer)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError('共阵需要非空的二维整数坐标。')
    if not np.issubdtype(points.dtype, np.integer):
        if not np.all(np.isfinite(points)) or not np.all(points == np.rint(points)):
            raise ValueError('共阵坐标必须为整数。')
    points = points.astype(np.int64, copy=False)
    mx = int(points[:, 0].max()) - int(points[:, 0].min())
    my = int(points[:, 1].max()) - int(points[:, 1].min())
    width, height = 2 * mx + 1, 2 * my + 1
    cells = width * height
    budget = _memory_state()['budget_bytes']
    required = 48 * cells + FIELD_WORKSPACE_BYTES
    if required > budget:
        raise ValueError(f'共阵几何预计需要 {format_bytes(required)}，超过当前内存预算 {format_bytes(budget)}。')
    histogram = np.zeros(cells, dtype=np.int64)
    block_rows = max(1, min(len(points), FIELD_WORKSPACE_BYTES // (32 * len(points))))
    for start in range(0, len(points), block_rows):
        stop = start + block_rows
        offsets = points[start:stop, 0, None] - points[None, :, 0]
        offsets = offsets.astype(np.int64, copy=False)
        offsets += mx
        offsets *= height
        dy = points[start:stop, 1, None] - points[None, :, 1]
        offsets += dy + my
        histogram += np.bincount(offsets.ravel(), minlength=cells)
    histogram = histogram.reshape(width, height)
    indices = np.argwhere(histogram > 0)
    multiplicity = histogram[indices[:, 0], indices[:, 1]]
    lags = indices - [mx, my]
    del indices
    prefix = np.zeros((width + 1, height + 1), dtype=np.int64)
    prefix[1:, 1:] = histogram == 0
    np.cumsum(prefix, axis=0, out=prefix)
    np.cumsum(prefix, axis=1, out=prefix)

    def complete(rx, ry):
        x0, y0, x1, y1 = mx-rx, my-ry, mx+rx+1, my+ry+1
        return prefix[x1, y1] - prefix[x0, y1] - prefix[x1, y0] + prefix[x0, y0] == 0

    best = (1, 0, 0)
    for rx in range(mx + 1):
        if not complete(rx, 0):
            continue
        lo, hi = 0, my + 1
        while lo + 1 < hi:
            middle = (lo + hi) // 2
            if complete(rx, middle):
                lo = middle
            else:
                hi = middle
        area = (2 * rx + 1) * (2 * lo + 1)
        if area > best[0]:
            best = (area, rx, lo)
    return {'unique_lags': len(lags), 'bounding_holes': int(cells-len(lags)),
            'central_rectangle_lags': best[0], 'central_half_widths': [best[1], best[2]],
            'lags': lags, 'multiplicity': multiplicity}
