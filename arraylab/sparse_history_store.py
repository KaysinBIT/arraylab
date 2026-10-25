"""Immutable sparse-result snapshots with lazy, transactional local history.

Only metadata is loaded for the navigation tree. Opening one record validates
and restores that record without optimization, reevaluation or metric updates.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

import numpy as np

from .sparse2d.headless import restore_bundle


APPLICATION_ID = 0x414C5348
FORMAT = 'arraylab.sparse_history'
VERSION = 1
MAX_BYTES = 256 * 1024 * 1024
MAX_TEXT = 1024 * 1024
MAX_DEPTH = 40
MAX_RESULTS = 10000
_VIEWS = {'overview', 'layout', 'pattern', 'cuts', 'coarray', 'cases', 'weights',
          'quantization', 'element', 'comparison', 'structures'}
_TABS = ('plot', 'ports', 'cases', 'weight', 'manual')
_STATE_KEYS = {'result_index', 'case_index', 'view', 'tab', 'pattern_view', 'phase_view'}
# SQLite must not seek through an overflow BLOB to reach any navigation column.
# Keep the potentially large payload physically last in the table record.
_COLUMNS = ('id', 'label', 'kind', 'created_utc', 'summary', 'view_state', 'sha256', 'payload')


class SparseHistoryError(ValueError):
    pass


def _fail(message):
    raise SparseHistoryError(message)


def _plain(value, depth=0):
    if depth > MAX_DEPTH:
        _fail('稀疏历史嵌套过深。')
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in 'biuf':
            _fail('稀疏历史数组必须为实数、整数或布尔值。')
        return _plain(value.tolist(), depth + 1)
    if isinstance(value, np.generic):
        return _plain(value.item(), depth + 1)
    if isinstance(value, dict):
        if any(type(key) is not str or len(key) > 1024 for key in value):
            _fail('稀疏历史字段名无效。')
        return {key: _plain(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    if value is None or type(value) is bool:
        return value
    if type(value) in (int, float):
        try:
            if not math.isfinite(value): _fail('稀疏历史不能包含NaN或无穷值。')
        except OverflowError:
            _fail('稀疏历史数值过大。')
        return value
    if type(value) is str and len(value) <= MAX_TEXT:
        return value
    _fail('稀疏历史含不支持的值或过长文本。')


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result: _fail('稀疏历史含重复JSON字段。')
        result[key] = value
    return result


def _decode(raw, limit=None):
    limit = MAX_BYTES if limit is None else limit
    if not isinstance(raw, (bytes, bytearray)) or not 1 <= len(raw) <= limit:
        _fail('稀疏历史内容大小超出限制。')
    try:
        data = json.loads(raw.decode('utf-8'), object_pairs_hook=_pairs,
                          parse_constant=lambda value: _fail('稀疏历史不能包含非有限数。'))
        return _plain(data)
    except SparseHistoryError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise SparseHistoryError('稀疏历史JSON损坏，原档案未覆盖。') from exc


def _number(value, label, *, nullable=False):
    if nullable and value is None: return
    if type(value) not in (int, float) or not math.isfinite(value):
        _fail(f'稀疏历史{label}须为有限数。')


def _integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value <= upper:
        _fail(f'稀疏历史{label}索引或数量无效。')


def _required(mapping, keys, label):
    if not isinstance(mapping, dict) or not set(keys) <= set(mapping):
        _fail(f'稀疏历史{label}缺少完整字段。')


def _validate_metrics(bundle, config):
    """Check saved fields needed by the UI; never recalculate their values."""
    cases = bundle['scenarios']
    for index, case in enumerate(cases):
        _integer(case['index'], index, index, '工况')
        for key in ('theta', 'phi', 'frequency_ghz', 'frequency_ratio'):
            _number(case[key], key)
        if type(case['label']) is not str: _fail('稀疏历史工况标签无效。')
    required_metrics = ('elements', 'validation_psl_db', 'optimization_psl_db',
        'amplitude_dynamic_range_db', 'hpbw_u', 'hpbw_v', 'min_distance_lambda_ref',
        'spacing_violations', 'unique_lags', 'case_count', 'worst_case_index')
    case_numbers = ('optimization_psl_db', 'validation_psl_db', 'grid_gap_db',
        'array_factor_psl_db', 'element_at_steering_db', 'total_peak_db', 'mainlobe_peak_db')
    nullable = ('hpbw_u', 'hpbw_v', 'relative_to_main_peak_psl_db', 'min_distance_lambda_frequency')
    for index, row in enumerate(bundle['results']):
        _required(row, ('metrics', 'scenario_metrics', 'phase_quantization', 'history', 'seconds', 'notes'), f'方案{index}')
        metrics = row['metrics']; _required(metrics, required_metrics, '方案指标')
        if metrics['elements'] != len(row['indices']) or type(metrics['elements']) is not int:
            _fail('稀疏历史阵元数与布局不一致。')
        _integer(metrics['case_count'], len(cases), len(cases), '工况数')
        _integer(metrics['worst_case_index'], 0, len(cases)-1, '最差工况')
        for key in ('validation_psl_db', 'optimization_psl_db', 'amplitude_dynamic_range_db'):
            _number(metrics[key], key)
        for key in ('hpbw_u', 'hpbw_v', 'min_distance_lambda_ref'):
            _number(metrics[key], key, nullable=True)
        for key in ('spacing_violations', 'unique_lags'):
            _integer(metrics[key], 0, 10**18, key)
        _number(row['seconds'], '耗时')
        if row['seconds'] < 0 or type(row['notes']) is not str or not isinstance(row['history'], list):
            _fail('稀疏历史耗时、说明或搜索记录无效。')
        for value in row['history']: _number(value, '搜索记录')
        per_case = row['scenario_metrics']
        if not isinstance(per_case, list) or len(per_case) != len(cases):
            _fail('稀疏历史缺少完整工况指标。')
        for saved, case in zip(per_case, cases):
            _required(saved, (*case, *case_numbers, *nullable, 'mainlobe_peak_uv', 'total_peak_uv'), '工况指标')
            if any(saved[key] != value for key, value in case.items()):
                _fail('稀疏历史工况与参数不一致。')
            _integer(saved['index'], case['index'], case['index'], '工况')
            for key in ('theta', 'phi', 'frequency_ghz', 'frequency_ratio'):
                _number(saved[key], key)
            for key in case_numbers: _number(saved[key], key)
            for key in nullable: _number(saved[key], key, nullable=True)
            for key in ('mainlobe_peak_uv', 'total_peak_uv'):
                uv = saved[key]
                if not isinstance(uv, list) or len(uv) != 2:
                    _fail('稀疏历史峰值坐标无效。')
                for value in uv: _number(value, key)
                if sum(value * value for value in uv) > 1 + 1e-8:
                    _fail('稀疏历史峰值超出可见方向域。')
        quant = row['phase_quantization']
        _required(quant, ('enabled', 'bits'), '量化结果')
        if (type(quant['enabled']) is not bool or quant['enabled'] != bool(config.phase_bits)
                or type(quant['bits']) is not int or quant['bits'] != config.phase_bits):
            _fail('稀疏历史量化设置与结果不一致。')
        if config.phase_bits:
            _required(quant, ('metrics', 'scenarios'), '量化结果')
            _required(quant['metrics'], ('worst_case_index', 'validation_psl_db'), '量化指标')
            _integer(quant['metrics']['worst_case_index'], 0, len(cases)-1, '量化最差工况')
            _number(quant['metrics']['validation_psl_db'], '量化副瓣', nullable=True)
            if not isinstance(quant['scenarios'], list) or len(quant['scenarios']) != len(cases):
                _fail('稀疏历史缺少量化工况。')
            for saved, case in zip(quant['scenarios'], cases):
                _required(saved, (*case, 'validation_psl_db', 'steering_loss_db'), '量化工况')
                if any(saved[key] != value for key, value in case.items()):
                    _fail('稀疏历史量化工况不一致。')
                _integer(saved['index'], case['index'], case['index'], '量化工况')
                _number(saved['validation_psl_db'], '量化副瓣', nullable=True)
                _number(saved['steering_loss_db'], '量化损失')
    if not any(row['metrics']['elements'] == config.k and row['name'] != 'L1 weighted solution'
               for row in bundle['results']):
        _fail('稀疏历史缺少目标阵元数的完整综合方案。')


def _restore_snapshot(data):
    _required(data, ('config', 'results', 'element_pattern', 'scenarios', 'lp', 'model'), '完整结果')
    if not isinstance(data['results'], list) or not 1 <= len(data['results']) <= MAX_RESULTS:
        _fail('稀疏历史方案数量无效。')
    try:
        restored, config, _ = restore_bundle(data)
        if data['scenarios'] != restored['scenarios']:
            _fail('稀疏历史工况与保存的参数不一致。')
        _validate_metrics(data, config)
        if restored.get('restoration', {}).get('metrics_require_upgrade'):
            _fail('稀疏历史缺少完整指标，不能无重算恢复。')
        # The engine loader checks coordinates against candidate geometry. Keep
        # the exact saved values rather than replacing them with rebuilt values.
        for source, target in zip(data['results'], restored['results']):
            if np.asarray(source['integer']).dtype.kind not in 'iu':
                _fail('稀疏历史整数格点不能包含浮点或布尔坐标。')
            for key in ('indices', 'integer', 'xy', 'weights'):
                target[key] = np.array(source[key], dtype=int if key in ('indices', 'integer') else float)
        restored['config'] = data['config'].copy()
        restored['scenarios'] = data['scenarios']
        return restored
    except SparseHistoryError:
        raise
    except (ValueError, TypeError, KeyError, OverflowError, IndexError, ArithmeticError) as exc:
        raise SparseHistoryError(f'稀疏历史结果校验失败：{exc}') from exc


def snapshot_bytes(bundle):
    try:
        plain = _plain(bundle)
        raw = json.dumps(plain, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
        if len(raw) > MAX_BYTES: _fail('单条稀疏历史不能超过256 MiB，当前结果未被截断。')
        _restore_snapshot(plain)
        return raw
    except RecursionError as exc:
        raise SparseHistoryError('稀疏历史嵌套过深。') from exc


def snapshot_from_bytes(raw):
    return _restore_snapshot(_decode(raw))


def _summary(bundle):
    rows, cases, config = bundle['results'], bundle['scenarios'], bundle['config']
    candidates = [i for i, row in enumerate(rows) if row['metrics']['elements'] == config['k']
                  and row['name'] != 'L1 weighted solution']
    selected = _plain(bundle.get('last_weighted_index'))
    if selected is None: selected = min(candidates, key=lambda i: rows[i]['metrics']['validation_psl_db'])
    _integer(selected, 0, len(rows)-1, '默认方案')
    return _plain({'result_count': len(rows), 'case_count': len(cases),
            'config_summary': {key: config[key] for key in ('nx', 'ny', 'k', 'reference_ghz', 'phase_bits')},
            'default_result': selected, 'default_case': rows[selected]['metrics']['worst_case_index']})


def _state_from_summary(summary, state):
    result = {'result_index': summary['default_result'], 'case_index': summary['default_case'],
              'view': 'overview', 'tab': 'plot', 'pattern_view': 'total',
              'phase_view': 'quantized' if summary['config_summary']['phase_bits'] else 'ideal'}
    if state is not None:
        if not isinstance(state, dict) or set(state) - _STATE_KEYS:
            _fail('稀疏历史视图状态含未知字段。')
        result.update(state)
    _integer(result['result_index'], 0, summary['result_count']-1, '所选方案')
    _integer(result['case_index'], 0, summary['case_count']-1, '所选工况')
    if type(result['tab']) is int and 0 <= result['tab'] < len(_TABS):
        result['tab'] = _TABS[result['tab']]
    if (not all(type(result[key]) is str for key in ('view', 'tab', 'pattern_view', 'phase_view'))
            or result['view'] not in _VIEWS or result['tab'] not in _TABS
            or result['pattern_view'] not in {'total', 'array_factor', 'element'}
            or result['phase_view'] not in {'ideal', 'quantized'}):
        _fail('稀疏历史视图状态无效。')
    if result['phase_view'] == 'quantized' and not summary['config_summary']['phase_bits']:
        _fail('该稀疏历史没有量化结果。')
    return result


def normalize_view_state(bundle, state=None):
    """Validate indices and stable view keys without computing any fields."""
    return _state_from_summary(_summary(bundle), state)


class SparseHistoryStore:
    def __init__(self, path):
        self.path = Path(path).resolve()
        try:
            if self.path.exists():
                with self._connect() as connection: self._check_schema(connection)
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(self.path, timeout=3)
                try:
                    connection.execute('BEGIN IMMEDIATE')
                    if connection.execute('SELECT COUNT(*) FROM sqlite_master').fetchone()[0]:
                        self._check_schema(connection)
                    else:
                        connection.execute(f'PRAGMA application_id={APPLICATION_ID}')
                        connection.execute(f'PRAGMA user_version={VERSION}')
                        connection.execute('CREATE TABLE history_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
                        connection.execute('CREATE TABLE sparse_results (id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT NOT NULL, kind TEXT NOT NULL, created_utc TEXT NOT NULL, summary TEXT NOT NULL, view_state TEXT NOT NULL, sha256 TEXT NOT NULL, payload BLOB NOT NULL)')
                        connection.executemany('INSERT INTO history_meta VALUES (?,?)', [('format', FORMAT), ('selected_id', '-1')])
                    connection.commit()
                finally: connection.close()
        except (sqlite3.Error, OSError) as exc:
            raise SparseHistoryError(f'无法打开稀疏历史档案，原文件未覆盖：{exc}') from exc

    @contextmanager
    def _connect(self):
        with self.path.open('rb') as stream: header = stream.read(100)
        if (len(header) < 100 or header[:16] != b'SQLite format 3\0'
                or int.from_bytes(header[68:72], 'big') != APPLICATION_ID
                or int.from_bytes(header[60:64], 'big') != VERSION):
            _fail('不是受支持的ArrayLab稀疏历史档案；原文件未覆盖。')
        # Read access also permits SQLite's recovery of a legitimate hot journal.
        connection = sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True, timeout=3)
        try: yield connection
        finally: connection.close()

    @staticmethod
    def _check_schema(connection):
        if (connection.execute('PRAGMA application_id').fetchone()[0] != APPLICATION_ID
                or connection.execute('PRAGMA user_version').fetchone()[0] != VERSION):
            _fail('稀疏历史格式或版本不受支持。')
        objects = connection.execute("SELECT type,name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 4").fetchall()
        if set(objects) != {('table', 'history_meta'), ('table', 'sparse_results')}:
            _fail('稀疏历史数据库结构异常。')
        if tuple(row[1] for row in connection.execute('PRAGMA table_info(sparse_results)')) != _COLUMNS:
            _fail('稀疏历史记录字段异常。')
        sizes = connection.execute('SELECT COUNT(*),MAX(LENGTH(key)),MAX(LENGTH(value)) FROM history_meta').fetchone()
        if sizes[0] != 2 or not isinstance(sizes[1], int) or not isinstance(sizes[2], int) or sizes[1] > 32 or sizes[2] > 128:
            _fail('稀疏历史元数据大小异常。')
        meta = dict(connection.execute('SELECT key,value FROM history_meta'))
        if (set(meta) != {'format', 'selected_id'} or meta['format'] != FORMAT
                or type(meta['selected_id']) is not str or not re.fullmatch(r'-1|[1-9]\d*', meta['selected_id'])):
            _fail('稀疏历史元数据无效。')
        return int(meta['selected_id'])

    @staticmethod
    def _metadata(connection, row_id):
        _integer(row_id, 1, 2**63-1, '记录')
        sizes = connection.execute('SELECT typeof(label),length(label),typeof(kind),length(kind),typeof(created_utc),length(created_utc),typeof(summary),length(summary),typeof(view_state),length(view_state),typeof(sha256),length(sha256),typeof(payload),length(payload) FROM sparse_results WHERE id=?', (row_id,)).fetchone()
        if sizes is None: _fail('所选稀疏历史记录不存在。')
        for offset, kind, limit in ((0, 'text', 1024), (2, 'text', 16), (4, 'text', 64),
                                    (6, 'text', 4096), (8, 'text', 4096), (10, 'text', 64), (12, 'blob', MAX_BYTES)):
            if sizes[offset] != kind or not 1 <= sizes[offset+1] <= limit:
                _fail('稀疏历史字段类型或大小异常。')
        label, kind, created, summary, state, digest = connection.execute('SELECT label,kind,created_utc,summary,view_state,sha256 FROM sparse_results WHERE id=?', (row_id,)).fetchone()
        if kind not in ('synthesis', 'weight') or not re.fullmatch('[0-9a-f]{64}', digest):
            _fail('稀疏历史类型或摘要无效。')
        try:
            if datetime.fromisoformat(created).tzinfo is None: _fail('稀疏历史时间缺少时区。')
        except (ValueError, TypeError): _fail('稀疏历史时间无效。')
        summary = _decode(summary.encode('utf-8'), 4096)
        _required(summary, ('result_count', 'case_count', 'config_summary', 'default_result', 'default_case'), '导航摘要')
        _integer(summary['result_count'], 1, MAX_RESULTS, '方案数')
        _integer(summary['case_count'], 1, 25, '工况数')
        _integer(summary['default_result'], 0, summary['result_count']-1, '默认方案')
        _integer(summary['default_case'], 0, summary['case_count']-1, '默认工况')
        _required(summary['config_summary'], ('nx', 'ny', 'k', 'reference_ghz', 'phase_bits'), '参数摘要')
        for key in ('nx', 'ny', 'k'): _integer(summary['config_summary'][key], 1, 10**100, key)
        _integer(summary['config_summary']['phase_bits'], 0, 12, '相位位数')
        _number(summary['config_summary']['reference_ghz'], '参考频率')
        state = _state_from_summary(summary, _decode(state.encode('utf-8'), 4096))
        return {'id': row_id, 'label': label, 'kind': kind, 'created_utc': created,
                'result_count': summary['result_count'], 'case_count': summary['case_count'],
                'config_summary': summary['config_summary'], 'view_state': state}, summary, digest

    def list_records(self):
        """List lightweight navigation metadata; no result payload is read."""
        try:
            with self._connect() as connection:
                connection.execute('BEGIN')
                selected = self._check_schema(connection)
                ids = [row[0] for row in connection.execute('SELECT id FROM sparse_results ORDER BY id')]
                rows = [self._metadata(connection, row_id)[0] for row_id in ids]
                if selected != -1 and selected not in ids: _fail('稀疏历史所选记录不存在。')
                return rows, selected
        except (sqlite3.Error, OSError) as exc:
            raise SparseHistoryError(f'无法读取稀疏历史列表，原文件未覆盖：{exc}') from exc

    def load_record(self, row_id):
        _integer(row_id, 1, 2**63-1, '记录')
        try:
            with self._connect() as connection:
                connection.execute('BEGIN'); self._check_schema(connection)
                metadata, summary, digest = self._metadata(connection, row_id)
                raw = connection.execute('SELECT payload FROM sparse_results WHERE id=?', (row_id,)).fetchone()[0]
                if hashlib.sha256(raw).hexdigest() != digest: _fail('稀疏历史内容摘要不匹配，原文件未覆盖。')
            bundle = snapshot_from_bytes(raw)
            if _summary(bundle) != summary: _fail('稀疏历史导航摘要与结果不一致。')
            return {**metadata, 'bundle': bundle}
        except (sqlite3.Error, OSError) as exc:
            raise SparseHistoryError(f'无法读取所选稀疏历史，原文件未覆盖：{exc}') from exc

    def append(self, bundle, label=None, *, kind='synthesis', view_state=None):
        if kind not in ('synthesis', 'weight'): _fail('稀疏历史操作类型无效。')
        if label is not None and (type(label) is not str or not 1 <= len(label) <= 1024): _fail('稀疏历史标签无效。')
        raw = snapshot_bytes(bundle)
        summary = _summary(bundle)
        state = _state_from_summary(summary, view_state)
        digest = hashlib.sha256(raw).hexdigest()
        created = datetime.now(timezone.utc).isoformat()
        try:
            with self._connect() as connection:
                connection.execute('BEGIN IMMEDIATE'); self._check_schema(connection)
                cursor = connection.execute('INSERT INTO sparse_results(label,kind,created_utc,summary,view_state,sha256,payload) VALUES (?,?,?,?,?,?,?)',
                    (label or 'pending', kind, created, json.dumps(summary, ensure_ascii=False), json.dumps(state), digest, raw))
                row_id = cursor.lastrowid
                if label is None:
                    c = summary['config_summary']
                    label = f"#{row_id} {'综合' if kind == 'synthesis' else '加权'} · {c['nx']}×{c['ny']} · K={c['k']}"
                    connection.execute('UPDATE sparse_results SET label=? WHERE id=?', (label, row_id))
                connection.execute("UPDATE history_meta SET value=? WHERE key='selected_id'", (str(row_id),))
                metadata = self._metadata(connection, row_id)[0]
                connection.commit()
                return metadata
        except (sqlite3.Error, OSError) as exc:
            raise SparseHistoryError(f'稀疏历史自动保存失败，当前结果仍可使用：{exc}') from exc

    def select(self, row_id, view_state=None):
        _integer(row_id, 1, 2**63-1, '记录')
        try:
            with self._connect() as connection:
                connection.execute('BEGIN IMMEDIATE'); self._check_schema(connection)
                metadata, summary, _ = self._metadata(connection, row_id)
                state = _state_from_summary(summary, metadata['view_state'] if view_state is None else view_state)
                connection.execute('UPDATE sparse_results SET view_state=? WHERE id=?', (json.dumps(state), row_id))
                connection.execute("UPDATE history_meta SET value=? WHERE key='selected_id'", (str(row_id),))
                connection.commit()
        except (sqlite3.Error, OSError) as exc:
            raise SparseHistoryError(f'稀疏历史所选状态未保存：{exc}') from exc

    def update_view_state(self, row_id, state):
        self.select(row_id, state)

    def remove(self, row_id, selected_id=-1):
        _integer(row_id, 1, 2**63-1, '记录')
        _integer(selected_id, -1, 2**63-1, '删除后所选记录')
        try:
            with self._connect() as connection:
                connection.execute('BEGIN IMMEDIATE'); self._check_schema(connection)
                self._metadata(connection, row_id)
                connection.execute('DELETE FROM sparse_results WHERE id=?', (row_id,))
                if not connection.execute('SELECT 1 FROM sparse_results WHERE id=?', (selected_id,)).fetchone():
                    latest = connection.execute('SELECT id FROM sparse_results ORDER BY id DESC LIMIT 1').fetchone()
                    selected_id = latest[0] if latest else -1
                connection.execute("UPDATE history_meta SET value=? WHERE key='selected_id'", (str(selected_id),))
                connection.commit()
                return selected_id
        except (sqlite3.Error, OSError) as exc:
            raise SparseHistoryError(f'删除稀疏历史失败，档案未变更：{exc}') from exc
