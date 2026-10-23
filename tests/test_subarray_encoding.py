"""Frozen Windows stdout may be cp936 even when callers expect UTF-8 JSON."""
import io
import json
from unittest.mock import patch

from arraylab.subarray_cli import main


def test_subarray_json_is_utf8_on_non_utf8_console(tmp_path):
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding='gbk')
    with patch('sys.stdout', console):
        assert main(['--write-example', str(tmp_path/'中文参数.json')]) == 0
    console.flush()
    assert json.loads(raw.getvalue().decode('utf-8'))['configuration'].endswith('中文参数.json')


def test_subarray_error_json_is_utf8_on_non_utf8_console():
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding='gbk')
    with patch('sys.stdout', console):
        assert main([]) == 2
    console.flush()
    assert '--output' in json.loads(raw.getvalue().decode('utf-8'))['error']


def test_invalid_cli_argument_uses_structured_utf8_error():
    raw = io.BytesIO()
    console = io.TextIOWrapper(raw, encoding='gbk')
    with patch('sys.stdout', console):
        assert main(['--未知参数']) == 2
    console.flush()
    assert '未知参数' in json.loads(raw.getvalue().decode('utf-8'))['error']
