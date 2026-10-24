"""Strict CSV input for matched branch delays at the design frequency."""
import csv
import io
import math
from pathlib import Path


def read_branch_delays_csv(path, elements):
    if type(elements) is not int or not 1 <= elements <= 200:
        raise ValueError('阵元数量须为1–200的整数。')
    with Path(path).open('rb') as source:
        raw = source.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError('支路电长度CSV不能超过1 MiB。')
    reader = csv.reader(io.StringIO(raw.decode('utf-8-sig')))
    if next(reader, None) != ['section', 'delay_deg']:
        raise ValueError('支路CSV表头须为section,delay_deg；单位为设计频率处的度。')
    count = 1 if elements == 1 else 2 * (elements - 1)
    expected = [f'T{i}' for i in range(1, count + 1)]
    values = {}
    for number, row in enumerate(reader, 2):
        if not row: continue
        if len(row) != 2:
            raise ValueError(f'支路CSV第{number}行须为支路编号和电长度两列。')
        name = row[0].strip()
        if name not in expected or name in values:
            raise ValueError(f'支路CSV第{number}行含未知或重复编号：{name}。')
        try: value = float(row[1])
        except ValueError as exc:
            raise ValueError(f'支路CSV第{number}行电长度不是数值。') from exc
        if not math.isfinite(value) or not 0 <= value <= 3600:
            raise ValueError(f'支路CSV第{number}行电长度须为0–3600°的有限数值。')
        values[name] = value
    if set(values) != set(expected):
        raise ValueError(f'支路CSV必须完整包含T1至T{count}；缺省支路请明确填写0。')
    return tuple(values[name] for name in expected)
