"""Plain text pattern exchange. Imported files never imply a 3-D directivity."""

from pathlib import Path
import io
import os
import tempfile

import numpy as np


def read_pattern(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    path = Path(path)
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("文件超过首版 16 MB 上限。")
    raw = path.read_text(encoding="utf-8-sig")
    try:
        data = np.loadtxt(io.StringIO(raw.replace(",", " ")), ndmin=2)
    except ValueError as exc:
        raise ValueError("请使用无表头的数值文件：角度(°)、幅度(dB)、可选相位(°)。") from exc
    if data.ndim != 2 or data.shape[1] not in (2, 3) or not 2 <= len(data) <= 100_000:
        raise ValueError("需要 2–100000 行，每行 2 或 3 列。")
    if not np.isfinite(data).all():
        raise ValueError("文件包含空值、NaN 或无穷大。")
    if not np.all(np.diff(data[:, 0]) > 0):
        raise ValueError("角度必须严格递增，不允许重复角度。")
    if data[0, 0] < -180 or data[-1, 0] > 180:
        raise ValueError("首版文件显示支持 −180° 至 180°；本机计算使用 −90° 至 90°。")
    return data[:, 0], data[:, 1], None if data.shape[1] == 2 else data[:, 2]


def write_pattern(path: str | Path, angles, db, phase=None) -> None:
    """Save headerless numeric columns, atomically replacing only on success."""
    target = Path(path)
    data = np.column_stack((angles, db) if phase is None else (angles, db, phase))
    if len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("没有有效的方向图可保存。")
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="ascii", newline="\n", suffix=".tmp", dir=target.parent, delete=False) as stream:
            temp_path = Path(stream.name)
            np.savetxt(stream, data, fmt="%.12g")
        os.replace(temp_path, target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
