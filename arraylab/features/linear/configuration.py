"""Shared, versioned analysis parameters for desktop and headless use."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from arraylab.shared.io import unique_keys as _unique_keys, invalid_constant as _invalid_constant


FORMAT = "arraylab.analysis"
SCHEMA_VERSION = 1
MAX_BYTES = 1024 * 1024
FIELDS = frozenset({"elements", "frequency_ghz", "spacing", "spacing_unit", "scan_deg", "step_deg"})


def parameters_from_values(values):
    from arraylab.features.linear.model import ArrayParameters

    unknown = set(values) - FIELDS
    if unknown:
        raise ValueError("未知参数字段：" + ", ".join(sorted(unknown)))
    for key, value in values.items():
        if key == "spacing_unit":
            if not isinstance(value, str):
                raise ValueError("spacing_unit须为字符串lambda或mm。")
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key}须为数值，不能是布尔值、字符串或null。")
        elif key == "elements" and not isinstance(value, int):
            raise ValueError("elements须为JSON整数。")
    parameters = ArrayParameters(**values)
    parameters.validate()
    return parameters






def load_configuration(path):
    """Read a complete versioned document or a legacy partial parameter object."""
    with Path(path).open("rb") as source:
        raw = source.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("配置文件不能超过 1 MB。")
    try:
        document = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys,
                              parse_constant=_invalid_constant)
    except RecursionError as exc:
        raise ValueError("配置文件嵌套过深。") from exc
    if not isinstance(document, dict):
        raise ValueError("配置须为一个JSON对象。")
    if set(document) & {"format", "schema_version", "parameters"}:
        expected = {"format", "schema_version", "parameters"}
        if set(document) != expected:
            raise ValueError("方案须且仅包含format、schema_version、parameters字段。")
        if document["format"] != FORMAT:
            raise ValueError("不支持的方案格式；请选择ArrayLab分析方案JSON。")
        if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
            raise ValueError("不支持的方案版本；当前支持schema_version=1。")
        values = document["parameters"]
        if not isinstance(values, dict) or set(values) != FIELDS:
            raise ValueError("版本1方案的parameters须完整包含6项参数，且不能有未知字段。")
    else:
        # Historical --config files may omit fields; ArrayParameters supplies defaults.
        values = document
    return parameters_from_values(values)


def save_configuration(path, parameters):
    """Validate first, then atomically replace the destination on success."""
    values = asdict(parameters)
    parameters_from_values(values)
    document = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "parameters": values}
    content = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         dir=target.parent, prefix=".arraylab-config-", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
