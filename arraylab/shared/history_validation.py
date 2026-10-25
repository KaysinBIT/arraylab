"""Bounded JSON/scientific-container validation shared by history codecs.

Domain objects are converted only through an explicitly supplied adapter.
This module has no dependency on a business model or a calculation engine.
"""
import math
import numpy as np

MAX_BYTES = 64 * 1024 * 1024
MAX_POINTS = 100_000
MAX_DEPTH = 32
MAX_STRING = 1024 * 1024



def fail(where, message):
    raise ValueError(f"{where}：{message}")


def require_object(value, where, required, exact=False):
    if not isinstance(value, dict):
        fail(where, "须为JSON对象。")
    missing = set(required) - value.keys()
    if missing:
        fail(where, "缺少字段 " + ", ".join(sorted(missing)))
    if exact and set(value) != set(required):
        fail(where, "包含未知字段。")
    return value


def require_string(value, where, limit=MAX_STRING, nonempty=False):
    if not isinstance(value, str) or len(value) > limit or (nonempty and not value.strip()):
        fail(where, f"须为{'非空' if nonempty else ''}字符串，最长{limit}字。")
    return value


def require_number(value, where, nullable=False, low=None, high=None):
    if nullable and value is None:
        return value
    if type(value) not in (int, float):
        fail(where, "须为有限数值，不能是布尔值、字符串或null。")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite or (low is not None and value < low) or (high is not None and value > high):
        fail(where, "数值非有限或超出允许范围。")
    return value


def require_integer(value, where, low, high):
    if type(value) is not int or not low <= value <= high:
        fail(where, f"须为{low}–{high}范围的整数。")
    return value


def require_boolean(value, where):
    if type(value) is not bool:
        fail(where, "须为布尔值。")


def require_list(value, where, low=0, high=MAX_POINTS):
    if not isinstance(value, list) or not low <= len(value) <= high:
        fail(where, f"须为长度{low}–{high}的列表。")
    return value


def require_vector(value, where, size=None, low=0, high=MAX_POINTS, bounds=None):
    require_list(value, where, low, high)
    if size is not None and len(value) != size:
        fail(where, "数组长度不匹配。")
    for item in value:
        require_number(item, where, low=None if bounds is None else bounds[0],
                high=None if bounds is None else bounds[1])
    return value


def require_strings(value, where, high=10000):
    for item in require_list(value, where, high=high):
        require_string(item, where)


def require_angles(value, where, bounds=(-180, 180)):
    require_vector(value, where, low=2, bounds=bounds)
    if any(b <= a for a, b in zip(value, value[1:])):
        fail(where, "角度须严格递增。")
    return value


def plain_json(value, depth=0, *, adapter=None):
    """Copy only JSON data and explicitly supported scientific containers."""
    if depth > MAX_DEPTH:
        fail("记录文件", "嵌套过深。")
    if isinstance(value, np.ndarray):
        if value.size > MAX_POINTS:
            fail("记录数组", "超过允许的点数。")
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    elif adapter is not None:
        value = adapter(value)
    if isinstance(value, dict):
        if len(value) > 10000 or any(not isinstance(key, str) for key in value):
            fail("记录对象", "字段过多或字段名不是字符串。")
        return {require_string(key, "字段名", 4096): plain_json(item, depth + 1, adapter=adapter) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_POINTS:
            fail("记录数组", "超过允许的点数。")
        return [plain_json(item, depth + 1, adapter=adapter) for item in value]
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        return require_number(value, "记录数值")
    if isinstance(value, str):
        return require_string(value, "记录文本")
    fail("记录文件", "含不支持的非JSON数据类型。")


def unique_json_keys(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            fail("记录文件", "重复字段 " + key)
        values[key] = value
    return values


def reject_json_constant(value):
    fail("记录文件", "非法JSON数值 " + value)
