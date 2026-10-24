"""JSON/Python control of the integrated sparse engine, without Qt or plotting."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np
from .core import (Config, available_mask, candidates, evaluation_domains,
                  evaluate_result, run, spacing_stats)
from .elements import build_element, make_snapshot
from .scenarios import scenarios
from .weighting import Weighting, add_weighting
from .resources import ensure_resources

VERSION = 1
ALIASES = {'regular': 'Regular thinning', 'random': 'Random symmetric',
           'greedy': 'Greedy exchange', 'annealing': 'Simulated annealing',
           'l1-topk': 'L1 top-K uniform', 'l1-weighted': 'L1 weighted solution'}


def jsonable(value):
    """Convert arrays recursively; reject nonfinite values instead of emitting NaN."""
    if isinstance(value, np.ndarray): return jsonable(value.tolist())
    if isinstance(value, np.generic): return jsonable(value.item())
    if isinstance(value, dict): return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('JSON 数值必须有限。')
    return value


def dumps(value):
    return json.dumps(jsonable(value), ensure_ascii=False, allow_nan=False)


def _object(value, allowed, label, required=()):
    if not isinstance(value, dict): raise ValueError(f'{label} 必须是 JSON 对象。')
    if set(value) - set(allowed):
        raise ValueError(f'{label} 含未知字段：{sorted(set(value)-set(allowed))}')
    if set(required) - set(value):
        raise ValueError(f'{label} 缺少字段：{sorted(set(required)-set(value))}')


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result: raise ValueError(f'JSON 含重复字段：{key}')
        result[key] = value
    return result


def read_json(path):
    """Read strict UTF-8 JSON, or one JSON document from stdin when path is '-'."""
    text = sys.stdin.read() if str(path) == '-' else Path(path).read_text(encoding='utf-8-sig')
    def constant(value): raise ValueError(f'JSON 不允许 {value}')
    return jsonable(json.loads(text.lstrip('\ufeff'), object_pairs_hook=_pairs,
                               parse_constant=constant))


def _typed(values, defaults, label):
    _object(values, defaults, label)
    merged = {**defaults, **values}
    for key, value in merged.items():
        kind = type(defaults[key])
        if kind is float:
            try: good = type(value) in (int, float) and math.isfinite(value)
            except OverflowError: good = False
        else: good = type(value) is kind
        if not good: raise ValueError(f'{label}.{key} 类型错误或数值非有限。')
    return merged


def _version(document, allowed, required):
    _object(document, allowed, '文档', required)
    if type(document['schema_version']) is not int or document['schema_version'] != VERSION:
        raise ValueError('仅支持 schema_version=1。')


def normalize_request(request, base_dir=None):
    """Validate types and ranges; paths in a file are relative to that file's folder."""
    _version(request, ('schema_version', 'config', 'weighting'), ('schema_version', 'config'))
    values = _typed(request['config'], asdict(Config()), 'config')
    if values['element_file']:
        values['element_file'] = str((Path(base_dir or Path.cwd()) / values['element_file']).resolve())
    config = Config(**values).validate()
    operations = request.get('weighting', [])
    if not isinstance(operations, list): raise ValueError('weighting 必须是数组。')
    normalized = []
    for item in operations:
        _object(item, ('source', *asdict(Weighting())), 'weighting 项')
        source = item.get('source', 'annealing')
        if not (type(source) is int and source >= 0 or type(source) is str and source.strip()):
            raise ValueError('source 必须是非负的 0 基行索引、方法名称或别名。')
        settings = _typed({k: v for k, v in item.items() if k != 'source'},
                          asdict(Weighting()), 'weighting')
        Weighting(**settings).validate(config)
        normalized.append({'source': source, **settings})
    return {'schema_version': VERSION, 'config': asdict(config), 'weighting': normalized}


def validate_request(request, base_dir=None):
    """Preflight config and element coverage; no optimization or packing proof."""
    normalized = normalize_request(request, base_dir)
    config = Config(**normalized['config'])
    element = build_element(config)
    for case in scenarios(config): element.steering_response(case)
    return normalized


def source_index(bundle, source):
    if type(source) is int:
        if 0 <= source < len(bundle['results']): return source
    elif type(source) is str:
        if source == 'dense': return 0
        if source == 'last': return len(bundle['results']) - 1
        name = ALIASES.get(source, source)
        matches = [i for i, r in enumerate(bundle['results']) if r['name'] == name]
        if len(matches) == 1: return matches[0]
    raise ValueError(f'加权源不存在或名称不唯一：{source!r}；可用 inspect 查看行索引。')


def _stamp(bundle, operation, request=None):
    import scipy
    from .. import __version__
    bundle['integration'] = {'arraylab_version': __version__, 'source_engine': 'Sparse2D 1.3.0'}
    bundle['headless'] = {'schema_version': VERSION, 'operation': operation,
                          'utc': datetime.now(timezone.utc).isoformat()}
    if request is not None: bundle['headless']['request'] = request
    bundle['runtime'] = {'python': sys.version.split()[0], 'numpy': np.__version__,
                         'scipy': scipy.__version__}
    return bundle


def compute(request, base_dir=None, progress=None, *, workers=None):
    """Importable API. Return the same in-memory numerical bundle as core.run."""
    normalized = validate_request(request, base_dir)
    bundle = run(Config(**normalized['config']), progress, workers=workers)
    for item in normalized['weighting']:
        settings = Weighting(**{k: v for k, v in item.items() if k != 'source'})
        bundle = add_weighting(bundle, source_index(bundle, item['source']), settings, progress, workers=workers)
    return _stamp(bundle, 'run', normalized)


def _snapshot(config, snapshot):
    if not isinstance(snapshot, dict) or snapshot.get('schema_version') != 1:
        raise ValueError('结果缺少有效的单元方向图快照。')
    if snapshot.get('model') != config.element_model or snapshot.get('reference_ghz') != config.reference_ghz:
        raise ValueError('单元快照与 config 的模型/参考频率不一致。')
    if config.element_model != 'imported':
        if snapshot.get('parameters') != make_snapshot(config)['parameters']:
            raise ValueError('单元快照参数与 config 不一致。')
    else:
        if snapshot.get('parameters') != {'import_mode': config.element_import_mode}:
            raise ValueError('单元导入模式与快照不一致。')
        planes = snapshot.get('planes')
        if not isinstance(planes, list) or not planes: raise ValueError('导入快照缺少采样面。')
        frequencies = []; peak = 0.
        for plane in planes:
            theta = np.asarray(plane['theta_deg'], dtype=float)
            phi = np.asarray(plane['phi_deg'], dtype=float)
            amplitude = np.asarray(plane['amplitude'], dtype=float)
            if (theta.ndim != 1 or len(theta) < 2 or phi.ndim != 1 or not len(phi)
                    or amplitude.shape != (len(theta), len(phi))
                    or not all(np.all(np.isfinite(a)) for a in (theta, phi, amplitude))
                    or np.any(np.diff(theta) <= 0) or np.any(np.diff(phi) <= 0)
                    or theta[0] != 0 or not 90 <= theta[-1] <= 180
                    or phi[0] < 0 or phi[-1] >= 360
                    or np.any(amplitude < 0) or np.any(amplitude > 1 + 1e-10)):
                raise ValueError('单元快照网格或幅度非法。')
            if config.element_import_mode == 'grid':
                if len(phi) < 4 or np.max(np.diff(np.r_[phi, phi[0]+360])) > 90+1e-8:
                    raise ValueError('单元快照方位覆盖不足。')
                for pole in (0, 180):
                    if pole in theta and np.ptp(amplitude[theta == pole]) > 1e-8:
                        raise ValueError('单元快照极点幅度不一致。')
            elif len(phi) != 1 or phi[0] != 0:
                raise ValueError('轴对称快照的 phi 必须为 [0]。')
            peak = max(peak, float(amplitude.max()))
            frequencies.append(plane['frequency_ghz'])
        independent = frequencies == [None]
        if snapshot.get('frequency_independent') is not independent or abs(peak-1) > 1e-8:
            raise ValueError('单元快照归一化或频率标识错误。')
        if not independent and (any(type(f) not in (int, float) or not .1 <= f <= 1000 for f in frequencies)
                                or np.any(np.diff(frequencies) <= 0)):
            raise ValueError('单元快照频率必须严格递增。')
    element = build_element(config, snapshot)
    for case in scenarios(config): element.steering_response(case)
    return element


def restore_bundle(data):
    """Restore array types and verify geometry before using a saved computation."""
    if not isinstance(data, dict): raise ValueError('结果必须是对象。')
    bundle = copy.deepcopy(data)
    if 'headless' in bundle:
        control = bundle['headless']
        if not isinstance(control, dict) or type(control.get('schema_version')) is not int or control['schema_version'] != VERSION:
            raise ValueError('保存结果的无头接口版本不受支持。')
    original_config = bundle['config']
    config = Config(**_typed(original_config, asdict(Config()), 'config')).validate()
    ensure_resources(config,operation='restore')
    bundle['config'] = asdict(config)
    if 'element_pattern' not in bundle and not any(k.startswith('element_') for k in original_config):
        # Results predating element support were explicitly ideal point sources.
        bundle['element_pattern'] = make_snapshot(config)
        bundle['restoration'] = {'legacy_point_source_snapshot_reconstructed': True,
                                 'metrics_require_upgrade': True}
    element = _snapshot(config, bundle['element_pattern'])
    rows = bundle['results']
    if not isinstance(rows, list) or not rows: raise ValueError('结果行不能为空。')
    integer, xy, _ = candidates(config); allowed = available_mask(config)
    for row_index, row in enumerate(rows):
        indices = np.asarray(row['indices'])
        if (indices.ndim != 1 or not 1 <= len(indices) <= len(xy)
                or indices.dtype.kind not in 'iu' or len(set(indices.tolist())) != len(indices)
                or np.any(indices < 0) or np.any(indices >= len(xy))):
            raise ValueError(f'第 {row_index} 行候选索引非法。')
        indices = indices.astype(int)
        row['indices'] = indices
        for key, expected in (('integer', integer[indices]), ('xy', xy[indices])):
            array = np.asarray(row[key], dtype=float)
            if array.shape != expected.shape or not np.allclose(array, expected, rtol=0, atol=1e-10):
                raise ValueError(f'第 {row_index} 行 {key} 与候选索引不一致。')
            row[key] = expected.copy()
        weights = np.asarray(row['weights'], dtype=float)
        if (weights.shape != indices.shape or not np.all(np.isfinite(weights))
                or np.any(weights <= 0) or abs(float(weights.sum())-1) > 1e-8):
            raise ValueError(f'第 {row_index} 行幅度应为正且和为 1。')
        lookup = dict(zip(indices.tolist(), weights.tolist()))
        if any(abs(lookup.get(len(xy)-1-i, -1)-w) > 1e-9 for i, w in lookup.items()):
            raise ValueError('保存布局必须具有中心对称位置及幅度。')
        if not np.all(allowed[indices]) or spacing_stats(row['xy'], config.min_spacing)['spacing_violations']:
            raise ValueError('保存布局违反中心禁布或最小间距约束。')
        row['weights'] = weights.copy()
        if type(row.get('name')) is not str: raise ValueError('结果行缺少方法名称。')
        if 'weighting' in row:
            source = row['weighting']['source_index']
            if type(source) is not int or not 0 <= source < row_index:
                raise ValueError('加权来源索引非法。')
            if not np.array_equal(np.sort(indices), np.sort(rows[source]['indices'])):
                raise ValueError('加权行没有保留源布局的全部位置。')
    bundle['scenarios'] = scenarios(config)
    if config.phase_bits and any(row.get('phase_quantization',{}).get('bits')!=config.phase_bits for row in rows):
        bundle['restoration']={**bundle.get('restoration',{}),'metrics_require_upgrade':True}
    # Parsing and geometry checks do not certify stored metrics. evaluate recomputes them.
    return bundle, config, element


def load_bundle(path):
    path = Path(path)
    return restore_bundle(read_json(path/'summary.json' if path.is_dir() else path))[0]


def reweight(bundle, source='annealing', settings=None, progress=None, *, workers=None):
    restored, config, _ = restore_bundle(bundle)
    values = _typed({} if settings is None else settings, asdict(Weighting()), 'weighting')
    setting = Weighting(**values).validate(config)
    if restored.get('restoration', {}).get('metrics_require_upgrade'):
        restored = reevaluate(restored, progress=progress, workers=workers)
    return _stamp(add_weighting(restored, source_index(restored, source), setting, progress, workers=workers), 'weight')


def reevaluate(bundle, eval_grid=None, progress=None, phase_bits=None, *, workers=None):
    restored, config, element = restore_bundle(bundle)
    if eval_grid is not None:
        values = {**asdict(config), 'eval_grid': eval_grid}
        config = Config(**_typed(values, asdict(Config()), 'config')).validate()
        restored['config'] = asdict(config)
    if phase_bits is not None:
        config=Config(**_typed({**asdict(config),'phase_bits':phase_bits},asdict(Config()),'config')).validate()
        restored['config']=asdict(config)
    ensure_resources(config,workers=workers,operation='evaluate')
    domains = evaluation_domains(config)
    from .core import evaluate_layouts
    restored['results'],restored['execution']=evaluate_layouts(restored['results'],config,domains,element,workers=workers,
        progress=(lambda done,total:progress(f'Validate {done}/{total}')) if progress else None)
    if 'restoration' in restored:
        restored['restoration'] = {**restored['restoration'], 'metrics_require_upgrade': False}
    return _stamp(restored, 'evaluate')


def save_bundle(bundle, output, report=False):
    """Stage in the output parent, publish only a completed new directory."""
    target = Path(output).resolve()
    if target.exists(): raise FileExistsError(f'输出已存在：{target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sparse2d-stage-', dir=target.parent) as scratch:
        staged = Path(scratch)/'result'
        if report:
            from .demo import export_bundle
            export_bundle(bundle, staged)
        else:
            staged.mkdir()
            (staged/'summary.json').write_text(dumps(bundle)+'\n', encoding='utf-8')
        if target.exists(): raise FileExistsError(f'输出已存在：{target}')
        staged.rename(target)
    return target


def overview(bundle, output=None):
    return {'schema_version': VERSION, 'ok': True, 'output': str(output) if output else None,
            'config': bundle['config'], 'case_count': len(bundle['scenarios']),
            'element_model': bundle['element_pattern']['model'],
            'restoration': bundle.get('restoration'),
            'results': [{'index': i, 'method': row['name'], **row['metrics'],
                         'phase_quantization':row.get('phase_quantization',{}).get('metrics')}
                        for i, row in enumerate(bundle['results'])]}


def _error(exc):
    code = 3 if isinstance(exc, FileExistsError) else 2 if isinstance(exc, (ValueError, TypeError, KeyError, IndexError, OverflowError)) else 1
    return code, {'type': type(exc).__name__, 'message': str(exc)}


def _progress(operation, job=None):
    def emit(message):
        print(dumps({'schema_version': VERSION, 'event': 'progress', 'operation': operation,
                     'job': job, 'message': message}), file=sys.stderr, flush=True)
    return emit


def _batch(args):
    manifest = read_json(args.manifest)
    _version(manifest, ('schema_version', 'jobs'), ('schema_version', 'jobs'))
    jobs = manifest['jobs']
    if not isinstance(jobs, list) or not jobs: raise ValueError('jobs 必须是非空数组。')
    ids = set()
    for job in jobs:
        _object(job, ('id', 'request'), 'job', ('id', 'request'))
        name = job['id']
        if (type(name) is not str or not name or len(name) > 64
                or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in name)
                or name.endswith('.') or name.upper() in {'CON', 'PRN', 'AUX', 'NUL',
                    *('COM'+str(i) for i in range(1, 10)), *('LPT'+str(i) for i in range(1, 10))}):
            raise ValueError('job.id 需为 1–64 位英文字母/数字/_/-，且不能是系统保留名。')
        if name.casefold() in ids: raise ValueError(f'job.id 重复：{name}')
        ids.add(name.casefold())
        if not isinstance(job['request'], (str, dict)): raise ValueError('job.request 应为参数文件路径或对象。')
    target = Path(args.output).resolve(); target.mkdir(parents=True, exist_ok=False)
    base = Path(args.manifest).resolve().parent if args.manifest != '-' else Path.cwd()
    status = {'schema_version': VERSION, 'ok': False, 'output': str(target), 'jobs': [], 'status': 'running'}
    def checkpoint():
        temporary = target/'batch.json.tmp'
        temporary.write_text(dumps(status)+'\n', encoding='utf-8')
        temporary.replace(target/'batch.json')
    checkpoint()
    interrupted = False; failed = False
    for job in jobs:
        if interrupted or failed and args.fail_fast:
            status['jobs'].append({'id': job['id'], 'status': 'skipped'})
            checkpoint(); continue
        entry = {'id': job['id'], 'status': 'running'}
        status['jobs'].append(entry); checkpoint()
        try:
            request = job['request']; directory = base
            if isinstance(request, str):
                path = (base/request).resolve(); request = read_json(path); directory = path.parent
            bundle = compute(request, directory, _progress('run', job['id']))
            output = save_bundle(bundle, target/job['id'], args.report)
            entry.update(status='completed', exit_code=0, result=overview(bundle, output))
        except KeyboardInterrupt:
            interrupted = True; entry.update(status='cancelled', exit_code=130)
        except Exception as exc:
            failed = True; code, error = _error(exc)
            entry.update(status='failed', exit_code=code, error=error)
        checkpoint()
    status['ok'] = not (failed or interrupted)
    status['status'] = 'cancelled' if interrupted else 'completed' if status['ok'] else 'failed'
    checkpoint()
    return status, 130 if interrupted else 0 if status['ok'] else 4


class ControlParser(argparse.ArgumentParser):
    def error(self, message): raise ValueError(message)


def parser():
    p = ControlParser(description='二维稀疏阵列无头控制：JSON、批处理、加权与复核，无桌面依赖')
    commands = p.add_subparsers(dest='command', required=True, parser_class=ControlParser)
    commands.add_parser('defaults', help='输出完整默认参数 JSON；不计算')
    validate = commands.add_parser('validate', help='检查参数与单元，不执行综合')
    validate.add_argument('--config', required=True, help='参数 JSON 路径，或 - 读取 stdin')
    compute_parser = commands.add_parser('run', help='计算全部稀疏方法，可顺序追加加权')
    compute_parser.add_argument('--config', required=True)
    for command in ('weight', 'evaluate', 'inspect'):
        item = commands.add_parser(command, help={'weight': '对保存布局追加加权',
            'evaluate': '重新计算保存布局的全部指标', 'inspect': '列出保存结果（不重新计算指标）'}[command])
        item.add_argument('--input', required=True, help='summary.json 或结果目录')
        if command == 'weight':
            item.add_argument('--source', default='annealing', help='别名/方法名；行索引用 --index')
            item.add_argument('--index', type=int, help='0 基行索引；与非默认 --source 互斥')
            item.add_argument('--settings', required=True, help='Weighting 参数 JSON，或 -')
        if command == 'evaluate':
            item.add_argument('--eval-grid', type=int)
            item.add_argument('--phase-bits', type=int, help='0=ideal; 1..12-bit replay on the saved support and amplitudes')
    batch = commands.add_parser('batch', help='顺序运行清单；单项失败后继续')
    batch.add_argument('--manifest', required=True)
    batch.add_argument('--fail-fast', action='store_true')
    for item in (compute_parser, commands.choices['weight'], commands.choices['evaluate'], batch):
        item.add_argument('--workers',type=int,default=0,help='0=自动最多4路；1=单核；2..8=有界并行复核')
        item.add_argument('--output', required=True, help='新目录；run/weight/evaluate 可用 - 输出完整 JSON')
        item.add_argument('--report', action='store_true', help='额外生成图表/CSV/HTML；需要 Matplotlib')
    return p


def main(argv=None):
    # Fixed encoding for pipes and service callers; importing the API leaves host I/O alone.
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    operation = None
    try:
        args = parser().parse_args(argv); operation = args.command
        from ..parallel import set_workers
        set_workers(getattr(args,'workers',0))
        if operation == 'defaults':
            print(dumps({'schema_version': VERSION, 'config': asdict(Config()), 'weighting': []})); return 0
        if operation == 'batch':
            if args.output == '-': raise ValueError('batch 必须指定新输出目录。')
            result, code = _batch(args); print(dumps(result)); return code
        if getattr(args, 'report', False) and args.output == '-':
            raise ValueError('--report 需要新输出目录，不能与 --output - 同用。')
        if getattr(args, 'output', '-') != '-' and Path(args.output).exists():
            raise FileExistsError(f'输出已存在：{Path(args.output).resolve()}')
        progress = _progress(operation)
        if operation in ('run', 'validate'):
            request = read_json(args.config)
            base = Path(args.config).resolve().parent if args.config != '-' else Path.cwd()
            if operation == 'validate':
                normalized = validate_request(request, base)
                print(dumps({'schema_version': VERSION, 'ok': True, 'request': normalized,
                             'case_count': len(scenarios(Config(**normalized['config']))),
                             'scope': 'types, ranges, element coverage and scan response; packing feasibility checked during run'}))
                return 0
            bundle = compute(request, base, progress)
        else:
            bundle = load_bundle(args.input)
            if operation == 'inspect':
                result = overview(bundle); result['metrics_recomputed'] = False
                print(dumps(result)); return 0
            if operation == 'weight':
                if args.index is not None and args.source != 'annealing':
                    raise ValueError('--index 与 --source 不能同时指定。')
                bundle = reweight(bundle, args.index if args.index is not None else args.source,
                                  read_json(args.settings), progress)
            else: bundle = reevaluate(bundle, args.eval_grid, progress, args.phase_bits)
        if args.output == '-': print(dumps(bundle))
        else:
            output = save_bundle(bundle, args.output, args.report)
            print(dumps(overview(bundle, output)))
        return 0
    except KeyboardInterrupt:
        print(dumps({'schema_version': VERSION, 'ok': False, 'operation': operation,
                     'error': {'type': 'Cancelled', 'message': '计算已中断。'}, 'exit_code': 130}))
        return 130
    except Exception as exc:
        code, error = _error(exc)
        print(dumps({'schema_version': VERSION, 'ok': False, 'operation': operation,
                     'error': error, 'exit_code': code}))
        return code


if __name__ == '__main__': raise SystemExit(main())
