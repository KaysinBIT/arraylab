"""Strict linear-array result and record codecs; no application routing."""
from dataclasses import asdict, fields
import json
import numpy as np
from arraylab.features.linear.configuration import FIELDS, parameters_from_values
from arraylab.features.linear.model import ArrayResult
from arraylab.shared.history_validation import plain_json
from arraylab.shared.history_validation import MAX_BYTES, MAX_POINTS, MAX_DEPTH, MAX_STRING, fail as _fail, require_object as _object, require_string as _string, require_number as _number, require_integer as _integer, require_boolean as _boolean, require_list as _list, require_vector as _vector, require_strings as _strings, require_angles as _angles, unique_json_keys as _unique_keys, reject_json_constant as _invalid_constant

RESULT_FIELDS = frozenset(field.name for field in fields(ArrayResult))
_RESULT_FIELDS = RESULT_FIELDS


def _adapt_array_result(value):
    if isinstance(value, ArrayResult):
        value = {field.name: getattr(value, field.name) for field in fields(ArrayResult)}
        value["parameters"] = asdict(value["parameters"])
    return value


def plain_linear_record(value, depth=0):
    """Convert supported saved values, including nested ArrayResult snapshots."""
    return plain_json(value, depth, adapter=_adapt_array_result)



def validate_parameters(value, where):
    _object(value, where, FIELDS, exact=True)
    return parameters_from_values(value)


def validate_array_result(result):
    _object(result, "线阵结果", _RESULT_FIELDS, exact=True)
    parameters = validate_parameters(result["parameters"], "线阵参数")
    angles = _angles(result["angles_deg"], "线阵角度", (-90, 90))
    for name in ("pattern_db", "phase_deg"):
        _vector(result[name], name, len(angles))
    for name in ("positions_mm", "excitation_phase_deg"):
        _vector(result[name], name, parameters.elements)
    _number(result["directivity_dbi"], "方向性")
    _vector(result["peaks_deg"], "等高峰角度", bounds=(-90, 90))
    if result["half_power_deg"] is not None:
        edges = _vector(result["half_power_deg"], "半功率边界", 2, bounds=(-90, 90))
        if edges[0] >= edges[1]:
            _fail("半功率边界", "左右边界次序错误。")
    _strings(result["notices"], "线阵提示")


def array_result_snapshot(result):
    """Return the complete JSON-compatible snapshot of an ArrayResult/dict."""
    snapshot = plain_linear_record(result)
    validate_array_result(snapshot)
    return snapshot


def array_result_from_snapshot(snapshot):
    """Recreate the immutable numeric result without invoking a solver."""
    values = array_result_snapshot(snapshot)
    values["parameters"] = parameters_from_values(values["parameters"])
    for name in ("angles_deg", "pattern_db", "phase_deg", "positions_mm", "excitation_phase_deg"):
        values[name] = np.array(values[name], dtype=float)
        values[name].setflags(write=False)
    for name in ("peaks_deg", "notices", "half_power_deg"):
        if values[name] is not None:
            values[name] = tuple(values[name])
    return ArrayResult(**values)


def validate_linear_record(record):
    _object(record, "线阵记录", {"label", "angles", "db", "phase", "result"}, exact=True)
    _string(record["label"], "记录标签", 1024, nonempty=True)
    angles = _angles(record["angles"], "曲线角度")
    _vector(record["db"], "曲线幅度", len(angles))
    if record["phase"] is not None:
        _vector(record["phase"], "曲线相位", len(angles))
    if record["result"] is not None:
        result = record["result"]
        validate_array_result(result)
        for record_key, result_key in (("angles", "angles_deg"), ("db", "pattern_db"), ("phase", "phase_deg")):
            if record[record_key] != result[result_key]:
                _fail("线阵记录", "显示曲线与计算快照不一致。")


def linear_record_bytes(record, *, max_bytes=None):
    """Encode one strictly validated snapshot for the automatic archive."""
    limit = MAX_BYTES if max_bytes is None else max_bytes
    value = plain_linear_record(record)
    validate_linear_record(value)
    raw = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(raw) > limit:
        _fail("自动历史记录", "单条快照不能超过64 MiB；结果仍保留在本次会话中。")
    return raw


def linear_record_from_bytes(raw, *, max_bytes=None):
    """Read one archive entry without invoking a calculation engine."""
    limit = MAX_BYTES if max_bytes is None else max_bytes
    if not isinstance(raw, bytes) or len(raw) > limit:
        _fail("自动历史记录", "记录类型错误或单条快照超过64 MiB。")
    try:
        value = plain_linear_record(json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys,
                                  parse_constant=_invalid_constant))
        validate_linear_record(value)
    except RecursionError as exc:
        raise ValueError("自动历史记录嵌套过深。") from exc
    return value
