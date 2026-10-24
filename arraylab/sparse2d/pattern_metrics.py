"""Figure measurements using the selected sparse layout, case and phase mode.

Imported scalar element snapshots cover only the front hemisphere. Their
backside is never invented to infer full-sphere directivity.
"""
from collections import OrderedDict
import copy
import hashlib
import json
import math
import threading
import numpy as np

from .core import Config, response_field, scan_uv, sidelobe_mask, sample_domain, db
from .elements import build_element
from .scenarios import scenarios, case_config
from .phase_quantization import excitation_phases, angular_distance
from ..pattern_metrics import spherical_beamwidths
from .aperture_metrics import integrated_power, maximum_distance
from ..parallel import ordered_map


_CACHE_LIMIT = 8
_summary_cache = OrderedDict()
_metric_sources = OrderedDict()
_cache_lock = threading.Lock()


def _fingerprint(*values):
    """Hash contents, never object ids, including mutable numerical arrays."""
    digest = hashlib.sha256()
    def add(value):
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(b'array')
            add(array.dtype.str); add(array.shape)
            digest.update(array.tobytes())
        elif isinstance(value, dict):
            digest.update(b'dict')
            for key in sorted(value): add(key); add(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(b'list')
            add(len(value))
            for item in value: add(item)
        else:
            if isinstance(value, np.generic): value = value.item()
            digest.update(json.dumps(value, sort_keys=True, ensure_ascii=False).encode('utf-8'))
            digest.update(b'\0')
    for value in values: add(value)
    return digest.digest()


def _saved_peak(result, c, case, view_mode, phase_mode, element, source_unchanged):
    """Reuse the report's dense/shifted-grid measurements on its own snapshot."""
    if not source_unchanged or view_mode == 'element': return None
    if view_mode == 'array_factor' and element.model != 'isotropic': return None
    if phase_mode == 'quantized' and c.phase_bits:
        quantization = result.get('phase_quantization', {})
        if quantization.get('bits') != c.phase_bits: return None
        rows = quantization.get('scenarios', [])
    else:
        rows = result.get('scenario_metrics', [])
    index = case['index']
    if not 0 <= index < len(rows): return None
    row = rows[index]
    if any(row.get(key) != case[key] for key in ('frequency_ghz', 'frequency_ratio', 'theta', 'phi')):
        return None
    try:
        chosen = np.asarray(row['mainlobe_peak_uv'], dtype=float)
        values = np.asarray([row['mainlobe_peak_db'], row['total_peak_db'],
                             row['relative_to_main_peak_psl_db']], dtype=float)
        if chosen.shape != (2,) or not np.isfinite(chosen).all() or not np.isfinite(values).all(): return None
        if np.sum(chosen**2) > 1 + 1e-10 or sidelobe_mask(chosen[None, :], case_config(c, case))[0]: return None
        level, maximum = 10**(values[:2] / 20)
        return chosen, float(level), float(maximum), float(values[2])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def sparse_pattern_summary(bundle, result, case_index=0, view_mode='total', phase_mode='ideal'):
    """Measure once per scientific snapshot, outside any numerical worker batch.

    Plot callers run after synthesis/validation. A single-task ordered_map shares
    the numerical lock and BLAS limit with other calculations. Cached entries
    contain only small summaries; no cache fields enter the exported bundle.
    """
    if view_mode not in ('total', 'array_factor', 'element'): raise ValueError('Unknown pattern view')
    if phase_mode not in ('ideal', 'quantized'): raise ValueError('Unknown phase mode')
    c = Config(**bundle['config'])
    case = (bundle.get('scenarios') or scenarios(c))[case_index]
    source = _fingerprint(bundle['config'], bundle.get('element_pattern'), result['xy'], result['weights'])
    key = _fingerprint(source.hex(), case, view_mode, phase_mode,
                       result.get('scenario_metrics'), result.get('phase_quantization'))
    with _cache_lock:
        # Keep at most eight result references, not their containing bundles.
        # The reference prevents id reuse. If a caller edits geometry/weights in
        # place, unchanged stored measurements must not override the new inputs.
        identity = id(result)
        previous = _metric_sources.get(identity)
        unchanged = previous is None or (previous[0] is result and previous[1] == source)
        if previous is None:
            _metric_sources[identity] = (result, source)
        _metric_sources.move_to_end(identity)
        while len(_metric_sources) > _CACHE_LIMIT: _metric_sources.popitem(last=False)
        if key in _summary_cache:
            _summary_cache.move_to_end(key)
            return copy.deepcopy(_summary_cache[key])
    rows, _ = ordered_map(lambda _: _measure_summary(bundle, result, case_index, view_mode, phase_mode, unchanged),
                          [None], workers=1)
    summary = rows[0]
    with _cache_lock:
        _summary_cache[key] = copy.deepcopy(summary)
        _summary_cache.move_to_end(key)
        while len(_summary_cache) > _CACHE_LIMIT: _summary_cache.popitem(last=False)
    return summary


def _measure_summary(bundle, result, case_index, view_mode, phase_mode, source_unchanged):
    c = Config(**bundle['config']); case = (bundle.get('scenarios') or scenarios(c))[case_index]
    cc = case_config(c, case); element = build_element(c, bundle.get('element_pattern'))
    points = np.asarray(result['xy']) * case['frequency_ratio']
    weights = np.asarray(result['weights']); center = scan_uv(cc)
    def response(uv):
        return response_field(result['xy'], weights, uv, c, case, element, view_mode, phase_mode)
    saved = _saved_peak(result, c, case, view_mode, phase_mode, element, source_unchanged)
    if saved is None:
        # Legacy/inapplicable metrics use the original full meshes, never a
        # smaller display grid. Element-only and non-isotropic AF views differ
        # from the total-field measurements saved by the synthesis engine.
        uv = np.vstack([sample_domain(cc, c.eval_grid), sample_domain(cc, c.eval_grid+2, True)])
        magnitude = abs(response(uv))
        main = ~sidelobe_mask(uv, cc)
        indices = np.flatnonzero(main) if view_mode != 'element' else np.arange(len(uv))
        level = float(np.max(magnitude[indices]))
        ties = indices[magnitude[indices] >= level - max(1e-12, level * 1e-9)]
        index = int(ties[np.argmin(np.sum((uv[ties]-center)**2, axis=1))])
        chosen = uv[index]
        maximum = float(np.max(magnitude))
        outside = ~main
        psl = float(db(np.max(magnitude[outside])/level)) if outside.any() and level > 1e-12 and view_mode != 'element' else None
    else:
        chosen, level, maximum, psl = saved
    radial = min(1., float(np.linalg.norm(chosen)))
    theta = math.degrees(math.asin(radial)); phi = math.degrees(math.atan2(chosen[1], chosen[0])) if radial > 1e-10 else case['phi']
    diameter = maximum_distance(points)
    step = max(.001, min(.125, 2 / max(1, diameter)))
    full_sphere = view_mode == 'array_factor' or element.model == 'isotropic'
    t, p = np.deg2rad([theta, phi]); direction = np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
    tangents = [np.array([np.cos(t)*np.cos(p), np.cos(t)*np.sin(p), -np.sin(t)]), np.array([-np.sin(p), np.cos(p), 0.])]
    invariant = [bool(np.all(np.ptp(points @ np.stack([direction[:2], tangent[:2]], axis=1), axis=0) == 0)) for tangent in tangents]
    if full_sphere:
        flat = [name for name, constant in zip(('meridional', 'transverse'), invariant) if constant]
        widths = spherical_beamwidths(lambda xyz: response(xyz[:, :2]), theta, phi, step, flat_cuts=flat)['orthogonal_beamwidths_deg']
    else:
        # Bracket only inside the measured front hemisphere, on two great circles.
        widths = []
        for tangent, constant in zip(tangents, invariant):
            cut_response = response
            if constant and view_mode != 'element':
                af_center = response_field(result['xy'], weights, direction[None, :2], c, case, element, 'array_factor', phase_mode)[0]
                def cut_response(uv):
                    af = af_center * np.exp(2j*np.pi*((uv-direction[:2]) @ points[0]))
                    return af if view_mode == 'array_factor' else af * element.normalized_response(uv, case)
            edges = []
            for sign in (-1, 1):
                a = np.r_[0., np.geomspace(1e-6, step, 24), np.arange(step, 180, step)]
                previous = None
                for start in range(0, len(a), 128):
                    angles = a[start:start+128]
                    rad = np.deg2rad(sign*angles); xyz = np.cos(rad)[:,None]*direction + np.sin(rad)[:,None]*tangent
                    invalid = np.flatnonzero(xyz[:, 2] < -1e-10)
                    end = int(invalid[0]) if len(invalid) else len(angles)
                    power = abs(cut_response(xyz[:end, :2]))**2
                    hit = np.flatnonzero(power <= level**2 / 2)
                    if len(hit):
                        i = int(hit[0])
                        if start+i > 0:
                            last = power[i-1] if i else previous
                            fraction = (level**2/2-last)/(power[i]-last)
                            edges.append(sign*float(a[start+i-1]+fraction*(a[start+i]-a[start+i-1])))
                        break
                    if end < len(angles): break
                    if len(power): previous = power[-1]
            widths.append(edges[1]-edges[0] if len(edges) == 2 else None)
    directivity = None
    if full_sphere and view_mode != 'element':
        phases = excitation_phases(result['xy'], case, c.phase_bits if phase_mode == 'quantized' else 0)
        w = weights * np.exp(1j * np.deg2rad(phases['command_phase_deg']))
        directivity = float(10*np.log10(maximum**2/integrated_power(points, w)))
    elif full_sphere:
        directivity = 0.
    return {'frequency_ghz': case['frequency_ghz'], 'theta_deg': theta, 'phi_deg': phi if radial > 1e-10 else None,
            'pointing_offset_deg': angular_distance(center, chosen), 'orthogonal_beamwidths_deg': widths,
            'directivity_dbi': directivity, 'gain_dbi': None, 'peak_sidelobe_db': psl,
            'mainlobe_halfwidth_uv': [1/(c.nx*cc.spacing), 1/(c.ny*cc.spacing)],
            'scope': 'front-hemisphere sampled peak in fixed mainlobe box; local great-circle widths',
            'directivity_status': 'sampled_peak_full_sphere_integral' if full_sphere else 'unavailable_back_hemisphere',
            'phase_mode': phase_mode, 'view_mode': view_mode}


def sparse_metric_lines(summary):
    def fmt(value, suffix=''):
        return 'n/a' if value is None else f'{value:.3g}{suffix}'
    a, b = summary['orthogonal_beamwidths_deg']; x, y = summary['mainlobe_halfwidth_uv']
    lines = [f"{summary['frequency_ghz']:g} GHz | {summary['view_mode']} | {summary['phase_mode']} | representative mainlobe peak: theta {fmt(summary['theta_deg'])}, phi {fmt(summary['phi_deg'])} deg",
             f"3 dB widths (meridional / transverse): {fmt(a)} / {fmt(b)} deg | target separation {fmt(summary['pointing_offset_deg'])} deg",
             f"Peak directivity D {fmt(summary['directivity_dbi'], ' dBi')} | Gain G unavailable (efficiency missing)",
             f"Front-space PSL {fmt(summary['peak_sidelobe_db'], ' dB')} / main peak | excluded u/v halfwidth {x:.3g}/{y:.3g}",
             'Sampled estimate; peaks may be nonunique; missing boundaries give no width.']
    if summary['directivity_status'] == 'unavailable_back_hemisphere':
        lines[-1] = 'D unavailable: element data define only the front hemisphere; no backside assumed.'
    return lines
