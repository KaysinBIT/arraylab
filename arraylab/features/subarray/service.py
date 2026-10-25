"""Continuous planar subarray-center optimization with explicit element templates.

Coordinates are element-center positions in reference wavelengths, not physical
feed dimensions. Equal-amplitude elements are rephased for every scan/frequency
case. Factoring identical templates is exact and retains their element pattern.
"""
from dataclasses import asdict, dataclass, fields
import csv
import json
import math
from pathlib import Path
import uuid

import numpy as np

from arraylab.framework.parallel import check_cancelled, ordered_map, requested_workers
from arraylab.shared.io import write_text_atomic
from arraylab.shared.numerics.array_fields import integrated_power
from arraylab.shared.numerics.scenarios import scenarios
from arraylab.features.subarray.model import SubarrayConfig, template_coordinates, expanded_coordinates, center_limits, layout_valid, _scan_uv, _sample_spec, normalized_field












def initial_layout(config, cancelled=None):
    bounds = center_limits(config)
    # Search regular seeds first, with enough rows/columns to accommodate holes.
    choices = [(nx, ny) for nx in range(1, config.count+1) for ny in range(1, config.count+1)
               if config.count <= nx*ny <= max(2*config.count, config.count+8)]
    choices.sort(key=lambda shape: (shape[0]*shape[1], abs(shape[0]/shape[1]-config.aperture_x/config.aperture_y)))
    for nx, ny in choices:
        check_cancelled(cancelled)
        xx, yy = np.meshgrid(np.linspace(-bounds[0], bounds[0], nx) if nx > 1 else [0.],
                             np.linspace(-bounds[1], bounds[1], ny) if ny > 1 else [0.])
        candidate = np.column_stack([xx.ravel(), yy.ravel()])
        kept = []
        for point in candidate:
            trial = np.array(kept+[point])
            if layout_valid(config, trial):
                kept.append(point)
            if len(kept) == config.count:
                return np.array(kept)
    rng = np.random.default_rng(config.seed)
    for _ in range(20):
        kept = []
        for attempt in range(1000):
            if attempt % 32 == 0:
                check_cancelled(cancelled)
            point = rng.uniform(-bounds, bounds)
            if layout_valid(config, np.array(kept+[point])):
                kept.append(point)
            if len(kept) == config.count:
                return np.array(kept)
    raise ValueError('未找到满足孔径、禁布和间距约束的初始布局；这不构成数学不可行证明。请放宽约束或提供初始中心坐标。')






def _objective_data(config, cases):
    directions = []
    amplitudes = []
    template = template_coordinates(config)
    for case in cases:
        _, _, uv, outside = _sample_spec(config, case, config.opt_grid)
        delta = (uv[outside]-_scan_uv(case))*case['frequency_ratio']
        directions.append(delta)
        amplitudes.append(abs(np.exp(2j*np.pi*(delta @ template.T)).mean(axis=1)))
    return np.concatenate(directions), np.concatenate(amplitudes)




def _db(field):
    return 20*np.log10(np.maximum(abs(field), 1e-15))


def _evaluate(config, centers, cases, cancelled=None):
    from arraylab.shared.plotting.metrics import spherical_beamwidths
    xy = expanded_coordinates(config, centers)
    reports = []
    for case in cases:
        check_cancelled(cancelled)
        axis, visible, uv, outside = _sample_spec(config, case, config.eval_grid)
        db = _db(normalized_field(config, centers, case, uv))
        sampled = float(np.max(db[outside]))
        _, _, shifted_uv, shifted_outside = _sample_spec(config, case, config.eval_grid, shifted=True)
        shifted = float(np.max(_db(normalized_field(config, centers, case, shifted_uv))[shifted_outside]))
        grid = np.empty(visible.shape, dtype=object)
        grid[:] = None
        grid[visible] = db
        ratio = case['frequency_ratio']
        phases = -2*np.pi*ratio*(xy @ _scan_uv(case))
        weights = np.exp(1j*phases)
        directivity = 10*math.log10(len(xy)**2/integrated_power(xy*ratio, weights))
        beamwidths = spherical_beamwidths(lambda unit: normalized_field(config, centers, case, unit[:, :2]),
                                         case['theta'], case['phi'])
        theta, phi = np.deg2rad([case['theta'], case['phi']])
        origin = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        tangents = [np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)]),
                    np.array([-np.sin(phi), np.cos(phi), 0.])]
        cut_angles = np.linspace(-90, 90, 721)
        angle_rad = np.deg2rad(cut_angles)
        cuts = [_db(normalized_field(config, centers, case,
                    (np.cos(angle_rad)[:, None]*origin+np.sin(angle_rad)[:, None]*tangent)[:, :2])).tolist()
                for tangent in tangents]
        metrics = {'directivity_dbi': directivity, 'gain_dbi': None,
                   'half_power_beamwidth_deg': beamwidths['cuts'][0]['half_power_beamwidth_deg'],
                   'orthogonal_beamwidths_deg': [cut['half_power_beamwidth_deg'] for cut in beamwidths['cuts']],
                   'representative_peak_theta_deg': case['theta'], 'representative_peak_phi_deg': case['phi'],
                   'peak_sidelobe_db': max(sampled, shifted), 'pointing_offset_deg': 0.,
                   'peak_is_not_claimed_unique': True}
        reports.append({**case, 'axis_uv': axis.tolist(), 'map_db': grid.tolist(),
                        'sampled_psl_db': sampled, 'shifted_psl_db': shifted,
                        'worst_psl_db': max(sampled, shifted), 'metrics': metrics,
                        'beamwidths': beamwidths, 'cut_angles_deg': cut_angles.tolist(), 'cuts_db': cuts,
                        'element_phase_deg': ((np.rad2deg(phases)+180)%360-180).tolist(),
                        'center_phase_deg': ((-360*ratio*(centers @ _scan_uv(case))+180)%360-180).tolist(),
                        'internal_phase_deg': ((-360*ratio*(template_coordinates(config) @ _scan_uv(case))+180)%360-180).tolist()})
    distances = np.linalg.norm(xy[:, None, :]-xy[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    elements = [{'subarray_id': i//(config.nx*config.ny)+1, 'element_id': i+1,
                 'local_element_id': i%(config.nx*config.ny)+1,
                 'x_lambda0': float(point[0]), 'y_lambda0': float(point[1]), 'amplitude': 1.}
                for i, point in enumerate(xy)]
    return {'centers': centers.tolist(), 'elements': elements, 'cases': reports,
            'min_element_spacing': float(np.min(distances)) if len(xy) > 1 else None,
            'worst_psl_db': max(case['worst_psl_db'] for case in reports)}


def optimize_subarrays(config, *, initial_centers=None, workers=None, progress=None, cancelled=None):
    config.validate()
    requested_workers(workers)
    check_cancelled(cancelled)
    initial = initial_layout(config, cancelled) if initial_centers is None else np.asarray(initial_centers, dtype=float)
    if initial.shape != (config.count, 2) or not layout_valid(config, initial):
        raise ValueError('初始中心坐标数量或布局不满足约束。')
    cases = scenarios(config)
    prepared, preparation_plan = ordered_map(
        lambda _: _objective_data(config, cases), [None], workers=workers,
        work_units=len(cases)*config.opt_grid**2*config.nx*config.ny, cancelled=cancelled)
    directions, envelope = prepared[0]
    bounds = center_limits(config)

    def search(restart):
        rng = np.random.default_rng(np.random.SeedSequence([config.seed, restart]))
        centers = initial.copy()
        # Do not retain samples x centers matrices for every parallel restart.
        # Bounded blocks plus one field vector keep peak memory proportional to
        # the sample count rather than sample count x center count x workers.
        total = np.empty(len(directions), dtype=complex)
        for offset in range(0, len(directions), 2048):
            check_cancelled(cancelled)
            total[offset:offset+2048] = np.exp(2j*np.pi*(directions[offset:offset+2048] @ centers.T)).sum(axis=1)
        score = float(np.max(_db(abs(total)*envelope/config.count)))
        best, best_score = centers.copy(), score
        trace = [{'iteration': 0, 'psl_db': score}]
        accepted = 0
        for iteration in range(1, config.iterations+1):
            check_cancelled(cancelled)
            fraction = iteration/config.iterations
            temperature = 1.5*(.02/1.5)**fraction
            step = (.15*(1-fraction)+.015)*np.maximum(bounds, .01)
            index = int(rng.integers(config.count))
            point = np.clip(centers[index]+rng.normal(size=2)*step, -bounds, bounds)
            trial = centers.copy()
            trial[index] = point
            if not layout_valid(config, trial):
                continue
            component = np.exp(2j*np.pi*(directions @ point))
            previous = np.exp(2j*np.pi*(directions @ centers[index]))
            trial_total = total-previous+component
            trial_score = float(np.max(_db(abs(trial_total)*envelope/config.count)))
            if trial_score <= score or rng.random() < math.exp(min(0., (score-trial_score)/temperature)):
                centers, total, score = trial, trial_total, trial_score
                accepted += 1
                if score < best_score:
                    best, best_score = centers.copy(), score
                    trace.append({'iteration': iteration, 'psl_db': score})
        return {'centers': best, 'optimization_psl_db': best_score, 'trace': trace,
                'accepted_moves': accepted, 'restart': restart}

    results, plan = ordered_map(search, range(config.restarts), workers=workers,
                                work_units=config.restarts*config.iterations*len(directions)*config.count,
                                progress=progress, cancelled=cancelled)
    selected = min(results, key=lambda result: result['optimization_psl_db'])
    evaluations, validation_plan = ordered_map(
        lambda centers: _evaluate(config, centers, cases, cancelled),
        [initial, selected['centers']], workers=workers,
        work_units=2*len(cases)*config.eval_grid**2*(config.count+config.nx*config.ny),
        cancelled=cancelled)
    initial_report, best_report = evaluations
    candidate_validation_psl = best_report['worst_psl_db']
    # Dense-grid regressions are visible and never replace the better baseline.
    validation_reverted = best_report['worst_psl_db'] > initial_report['worst_psl_db']+1e-10
    if validation_reverted:
        best_report = initial_report
    check_cancelled(cancelled)
    xy = expanded_coordinates(config, np.asarray(best_report['centers']))
    diameter = float(np.max(np.linalg.norm(xy[:, None, :]-xy[None, :, :], axis=2)))
    electrical_diameter = diameter*max(case['frequency_ratio'] for case in cases)
    recommended_step = 1/(8*electrical_diameter) if electrical_diameter else None
    resolution_warning = recommended_step is not None and 2/(config.eval_grid-1) > recommended_step
    return {'schema': 'arraylab.subarray_centers', 'version': 1, 'config': asdict(config),
            'initial': initial_report, 'best': best_report,
            'optimization': {'method': 'bounded_coordinate_annealing',
                             'objective': 'worst_sampled_psl_outside_explicit_uv_rectangle',
                             'selected_restart': selected['restart'],
                             'optimization_psl_db': selected['optimization_psl_db'],
                             'candidate_validation_psl_db': candidate_validation_psl,
                             'validation_reverted': validation_reverted,
                             'runs': [{key: value for key, value in result.items() if key != 'centers'} for result in results]},
            'execution': plan,
            'preparation_execution': preparation_plan,
            'validation_execution': validation_plan,
            'sampling': {'max_electrical_diameter_lambda': electrical_diameter,
                         'optimization_uv_step': 2/(config.opt_grid-1),
                         'validation_uv_step': 2/(config.eval_grid-1),
                         'recommended_max_uv_step': recommended_step,
                         'resolution_warning': resolution_warning,
                         'note': '建议每个 1/电气孔径尺度至少 8 个采样点；这是分辨率提示，不是连续域验证保证。'},
            'limits': ['固定数量、同模板、同朝向、等幅子阵；连续优化二维中心位置。',
                       '每个频率/扫描工况独立重设子阵公共相位及内部阵元相位；不代表同时多波束。',
                       '各向同性独立阵元，未包含互耦、损耗、失配或物理馈线尺寸；增益未计算。',
                       '孔径和禁布区约束作用于实际阵元中心；子阵模板包络不允许内部重叠。',
                       'PSL 使用明确的 u/v 主瓣矩形排除区；可能包含主瓣裙边，非自动首零点副瓣指标。',
                       '优化及错位加密复核均为有限采样，不能保证连续域最坏副瓣或全局最优。',
                       '平面各向同性阵列具有上下半空间镜像响应；目标峰不宣称唯一。',
                       '子阵数、阵元数、独立端口数和 RF 链数不等同。']}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'JSON 包含重复字段：{key}')
        result[key] = value
    return result


def load_config(path):
    path = Path(path)
    if path.stat().st_size > 1024*1024:
        raise ValueError('子阵配置文件超过 1 MiB。')
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError('配置不允许 NaN/Infinity。')))
    except RecursionError as exc:
        raise ValueError('配置 JSON 嵌套过深。') from exc
    if not isinstance(data, dict):
        raise ValueError('配置必须为 JSON 对象。')
    allowed = {field.name for field in fields(SubarrayConfig)}
    if set(data)-allowed:
        raise ValueError('未知配置字段：'+', '.join(sorted(set(data)-allowed)))
    return SubarrayConfig(**data).validate()


def save_config(path, config):
    config.validate()
    write_text_atomic(path, json.dumps(asdict(config), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


load_subarray_configuration = load_config
save_subarray_configuration = save_config


def export_subarray(directory, report, plot='none'):
    if plot not in ('none', 'layout', 'pattern', 'cuts', '3d', 'all'):
        raise ValueError('未知子阵图形类型。')
    target = Path(directory).absolute()
    if target.exists() or target.is_symlink():
        raise FileExistsError('输出目录已存在；请选择新目录。')
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / ('.arraylab-subarray-'+uuid.uuid4().hex)
    try:
        _write_export(staging, report, plot)
        staging.rename(target)
    finally:
        # This private staging contains only the flat files written below.
        # Never recursively remove an existing user directory.
        if staging.exists():
            for item in staging.iterdir():
                item.unlink()
            staging.rmdir()
    return target


def _write_export(directory, report, plot):
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (directory/'config.json').write_text(json.dumps(report['config'], ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    config = SubarrayConfig(**report['config'])
    wavelength_mm = 299.792458/config.reference_ghz
    with (directory/'case-metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['layout', 'case_index', 'frequency_ghz', 'theta_deg', 'phi_deg',
                         'sampled_psl_db', 'shifted_psl_db', 'worst_psl_db', 'directivity_dbi',
                         'meridional_hpbw_deg', 'transverse_hpbw_deg', 'gain_dbi', 'gain_status',
                         'psl_exclusion_halfwidth_u', 'psl_exclusion_halfwidth_v'])
        for name in ('initial', 'best'):
            for case in report[name]['cases']:
                writer.writerow([name, case['index'], case['frequency_ghz'], case['theta'], case['phi'],
                                 case['sampled_psl_db'], case['shifted_psl_db'], case['worst_psl_db'],
                                 case['metrics']['directivity_dbi'], *case['metrics']['orthogonal_beamwidths_deg'],
                                 '', 'missing_radiation_efficiency', config.exclusion_u, config.exclusion_v])
    for name in ('initial', 'best'):
        layout = report[name]
        with (directory/f'{name}-centers.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['subarray_id', 'x_lambda0', 'y_lambda0', 'x_mm', 'y_mm'])
            for index, point in enumerate(layout['centers'], 1):
                writer.writerow([index, *point, *(np.array(point)*wavelength_mm)])
        for case in layout['cases']:
            with (directory/f'{name}-case-{case["index"]}-elements.csv').open('w', encoding='utf-8-sig', newline='') as stream:
                writer = csv.writer(stream)
                writer.writerow(['subarray_id', 'element_id', 'local_element_id', 'x_lambda0', 'y_lambda0', 'x_mm', 'y_mm', 'amplitude', 'phase_deg'])
                for element, phase in zip(layout['elements'], case['element_phase_deg']):
                    writer.writerow([element['subarray_id'], element['element_id'], element['local_element_id'],
                                     element['x_lambda0'], element['y_lambda0'], element['x_lambda0']*wavelength_mm,
                                     element['y_lambda0']*wavelength_mm, element['amplitude'], phase])
    for case in report['best']['cases']:
        theta_deg, phi_deg = np.meshgrid(np.arange(0., 181., 3.), np.arange(-180., 181., 3.), indexing='ij')
        theta, phi = np.deg2rad(theta_deg), np.deg2rad(phi_deg)
        uv = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)], axis=-1)
        field = normalized_field(config, report['best']['centers'], case, uv)
        with (directory/f'best-case-{case["index"]}-sphere.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['theta_deg_step3', 'phi_deg_step3', 'relative_field_magnitude', 'relative_field_db',
                             'field_phase_deg', 'display_grid_not_metric_search'])
            writer.writerows(zip(theta_deg.ravel(), phi_deg.ravel(), abs(field).ravel(), _db(field).ravel(),
                                 np.rad2deg(np.angle(field)).ravel(), np.ones(field.size, dtype=int)))
    if plot != 'none':
        from arraylab.features.subarray.plotting import render_subarray_plots
        render_subarray_plots(directory, report, plot)
