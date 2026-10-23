"""Checked transfer of a sparse ideal layout into the planar electrical model."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np

from .feed import FeedSettings
from .planar import PlanarParameters, validate_planar_design, save_planar_configuration
from .sparse2d.headless import jsonable, restore_bundle
from .sparse2d.scenarios import LIGHT_MM_GHZ


def planar_from_sparse(bundle, result_index, case_index):
    source, config, _ = restore_bundle(jsonable(bundle))
    for value, size, label in [(result_index, len(source['results']), '方案'), (case_index, len(source['scenarios']), '工况')]:
        if type(value) is not int or not 0 <= value < size:
            raise ValueError(f'{label}索引须为有效的0基整数。')
    if source['element_pattern']['model'] != 'isotropic':
        raise ValueError('平面阵电气页目前仅支持各向同性点源；该方案的单元模型无法无损传递，请继续在稀疏模块分析。')
    row, case = source['results'][result_index], source['scenarios'][case_index]
    if len(row['weights']) > 200:
        raise ValueError('平面阵电气页最多200阵元；请选取不超过200阵元的方案，不会截断布局。')
    xy_mm = row['xy'] * (LIGHT_MM_GHZ / config.reference_ghz)
    phi = (case['phi'] + 180) % 360 - 180
    p = PlanarParameters(geometry='arbitrary', frequency_ghz=case['frequency_ghz'],
                         coordinate_unit='mm', coordinates=tuple(map(tuple, xy_mm)),
                         scan_theta_deg=case['theta'], scan_phi_deg=phi, cut_phi_deg=phi)
    amplitudes = row['weights'] / np.max(row['weights'])
    settings = FeedSettings(topology='corporate', phase_mode='scan', weighting='custom',
                            custom_amplitudes=tuple(amplitudes.tolist()), custom_phases_deg=(0.,) * len(amplitudes))
    validate_planar_design(p, settings)
    selected = {'source_config': source['config'], 'indices': row['indices'], 'xy_lambda_ref': row['xy'],
                'amplitudes_sum_one': row['weights'], 'case': case, 'element_pattern': source['element_pattern']}
    fingerprint = hashlib.sha256(json.dumps(jsonable(selected), sort_keys=True, separators=(',', ':'),
                                            ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
    provenance = {'format': 'arraylab.sparse_planar_handoff', 'schema_version': 1,
                  'source_method': row['name'], 'result_index': result_index, 'case_index': case_index,
                  'source_reference_ghz': config.reference_ghz, 'selected_case': case,
                  'candidate_indices': row['indices'].tolist(), 'source_sha256': fingerprint,
                  'amplitude_mapping': 'positive relative field amplitudes, normalized by maximum; not powers',
                  'initial_feed': 'new corporate network; input/junction/load 50 ohm; continuous scan compensation',
                  'source_phase_bits': config.phase_bits, 'destination_phase_bits': 0,
                  'notes': ['保持选中方案全部位置、编号顺序及相对幅度；毫米坐标按源参考频率换算。',
                            '仅传递理想几何和幅度；原量化相位码不迁移。新馈电网络须重新综合，50 Ω为可编辑初值。',
                            '稀疏PSL与HPBW不转成电气页指标；两页保留各自的定义和归一化。']}
    return p, settings, provenance


def export_planar_handoff(target, bundle, result_index, case_index):
    p, settings, provenance = planar_from_sparse(bundle, result_index, case_index)
    target = Path(target).absolute()
    if target.exists():
        raise FileExistsError('输出目录已存在，请选择新目录。')
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.arraylab-handoff-', dir=target.parent) as scratch:
        staged = Path(scratch) / 'result'; staged.mkdir()
        save_planar_configuration(staged / 'planar-config.json', p, settings)
        (staged / 'source.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        with (staged / 'coordinates-mm.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream); writer.writerow(['element', 'x', 'y'])
            for i, xy in enumerate(p.coordinates, 1): writer.writerow([i, *xy])
        staged.rename(target)
    return provenance
