"""Strict JSON hooks and same-directory atomic text replacement."""
from pathlib import Path


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"配置字段重复：{key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"配置文件包含非法JSON常量：{value}")


def write_text_atomic(path, text):
    # Reuse the existing same-directory atomic replace contract.
    import os
    import tempfile
    target = Path(path)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=target.parent,
                                         suffix=".tmp", prefix=".arraylab-feed-", delete=False) as stream:
            temp_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, target)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
