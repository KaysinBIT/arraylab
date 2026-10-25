"""Versioned calculation snapshots, independent of Qt and of calculation routines.

``load_history`` returns JSON-compatible dictionaries. Linear records contain
label/angles/db/phase/result; ``array_result_from_snapshot`` restores an optional
ArrayResult without recalculating it. Feed records contain label/report. Report
extension metadata is retained, while all fields used to render it are checked.
The old feed_comparison v1 envelope is accepted as feed history, selecting its
last record. Missing legacy pointing summaries are derived from saved metrics.
An exported complete feed-network.json report is accepted as one saved design;
its electrical results are validated and retained without running the solver.
"""
import json
import os
from pathlib import Path
import tempfile
from arraylab.features.linear import array_result_snapshot, array_result_from_snapshot, RESULT_FIELDS as _RESULT_FIELDS, plain_linear_record as _plain, validate_parameters as _parameters, validate_array_result as _validate_result, validate_linear_record as _validate_linear
from arraylab.features import linear as _linear_records
from arraylab.features.feed import validate_saved_pattern as _validate_pattern, validate_feed_record as _validate_feed
from arraylab.shared.history_validation import MAX_BYTES, MAX_POINTS, MAX_DEPTH, MAX_STRING, fail as _fail, require_object as _object, require_string as _string, require_number as _number, require_integer as _integer, require_boolean as _boolean, require_list as _list, require_vector as _vector, require_strings as _strings, require_angles as _angles, unique_json_keys as _unique_keys, reject_json_constant as _invalid_constant

FORMAT = "arraylab.calculation_history"
SCHEMA_VERSION = 1
MAX_RECORDS = 5
MAX_LINEAR_RECORDS = 100_000


def linear_record_bytes(record):
    """Compatibility entry; the standalone codec lives with linear records."""
    return _linear_records.linear_record_bytes(record, max_bytes=MAX_BYTES)


def linear_record_from_bytes(raw):
    """Compatibility entry; honor this legacy entrypoint's byte budget."""
    return _linear_records.linear_record_from_bytes(raw, max_bytes=MAX_BYTES)



def _validate_document(document):
    _object(document, "记录文件", {"format", "schema_version", "kind", "records", "selected"}, exact=True)
    if document["format"] != FORMAT or type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        _fail("记录文件", "不支持的格式或版本。")
    if document["kind"] not in ("linear", "feed"):
        _fail("记录文件", "kind须为linear或feed。")
    limit = MAX_LINEAR_RECORDS if document["kind"] == "linear" else MAX_RECORDS
    records = _list(document["records"], "计算记录", 1, limit)
    _integer(document["selected"], "所选记录", 0, len(records) - 1)
    validate = _validate_linear if document["kind"] == "linear" else _validate_feed
    for record in records:
        validate(record)
    return document


def load_history(path):
    """Load and validate every snapshot before returning any history data."""
    with Path(path).open("rb") as source:
        raw = source.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        _fail("记录文件", "不能超过64 MiB。")
    try:
        document = _plain(json.loads(raw.decode("utf-8-sig"), object_pairs_hook=_unique_keys,
                                     parse_constant=_invalid_constant))
        if isinstance(document, dict) and document.get("format") == "arraylab.feed_comparison":
            _object(document, "旧比较记录", {"format", "schema_version", "records"}, exact=True)
            if type(document["schema_version"]) is not int or document["schema_version"] != 1:
                _fail("旧比较记录", "不支持的版本。")
            records = _list(document["records"], "旧比较记录", 1, MAX_RECORDS)
            document = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "kind": "feed",
                        "records": records, "selected": len(records) - 1}
        elif isinstance(document, dict) and document.get("format") == "arraylab.feed_impedance_result":
            # Result export and history export use different envelopes but carry
            # the same complete report. Preserve the original report and run the
            # usual strict snapshot checks before any widget can be changed.
            document = {"format": FORMAT, "schema_version": SCHEMA_VERSION, "kind": "feed",
                        "records": [{"label": Path(path).stem[:1000] + " · 导出设计", "report": document}],
                        "selected": 0}
        elif isinstance(document, dict) and document.get("format") in ("arraylab.feed_impedance", "arraylab.analysis"):
            _fail("记录文件", "这是参数设置，不含已计算结果。请使用“打开参数”；馈电结果请打开feed-network.json。")
        return _validate_document(document)
    except RecursionError as exc:
        raise ValueError("记录文件嵌套过深。") from exc


def save_history(path, kind, records, selected):
    """Validate complete snapshots, then atomically replace the destination."""
    try:
        document = _plain({"format": FORMAT, "schema_version": SCHEMA_VERSION, "kind": kind,
                           "records": records, "selected": selected})
        _validate_document(document)
        raw = (json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except RecursionError as exc:
        raise ValueError("记录文件嵌套过深。") from exc
    if len(raw) > MAX_BYTES:
        _fail("记录文件", "不能超过64 MiB。")
    target, temporary = Path(path), None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent, prefix=".arraylab-history-",
                                         suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
