"""Figure measurements using the selected sparse layout, case and phase mode.

Imported scalar element snapshots cover only the front hemisphere. Their
backside is never invented to infer full-sphere directivity.
"""
import math
import numpy as np

from .core import Config, response_field, scan_uv, sidelobe_mask, sample_domain, db
from .elements import build_element
from .scenarios import scenarios, case_config
from .phase_quantization import excitation_phases, angular_distance
from ..pattern_metrics import spherical_beamwidths
from ..planar_pattern import integrated_power


def sparse_pattern_summary(bundle, result, case_index=0, view_mode='total', phase_mode='ideal'):
    c = Config(**bundle['config']); case = (bundle.get('scenarios') or scenarios(c))[case_index]
    cc = case_config(c, case); element = build_element(c, bundle.get('element_pattern'))
    points = np.asarray(result['xy']) * case['frequency_ratio']
    weights = np.asarray(result['weights']); center = scan_uv(cc)
    def response(uv):
        return response_field(result['xy'], weights, uv, c, case, element, view_mode, phase_mode)
    # Same two meshes, exclusion boundaries and disk rim as stored case metrics.
    uv = np.vstack([sample_domain(cc, c.eval_grid), sample_domain(cc, c.eval_grid+2, True)])
    magnitude = abs(response(uv))
    main = ~sidelobe_mask(uv, cc)
    indices = np.flatnonzero(main) if view_mode != 'element' else np.arange(len(uv))
    level = float(np.max(magnitude[indices]))
    ties = indices[magnitude[indices] >= level - max(1e-12, level * 1e-9)]
    index = int(ties[np.argmin(np.sum((uv[ties]-center)**2, axis=1))])
    chosen = uv[index]; radial = min(1., float(np.linalg.norm(chosen)))
    theta = math.degrees(math.asin(radial)); phi = math.degrees(math.atan2(chosen[1], chosen[0])) if radial > 1e-10 else case['phi']
    diameter = float(np.max(np.linalg.norm(points[:, None] - points[None, :], axis=2)))
    step = max(.001, min(.125, 2 / max(1, diameter)))
    full_sphere = view_mode == 'array_factor' or element.model == 'isotropic'
    if full_sphere:
        widths = spherical_beamwidths(lambda xyz: response(xyz[:, :2]), theta, phi, step)['orthogonal_beamwidths_deg']
    else:
        # Bracket only inside the measured front hemisphere, on two great circles.
        t, p = np.deg2rad([theta, phi]); direction = np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
        tangents = [np.array([np.cos(t)*np.cos(p), np.cos(t)*np.sin(p), -np.sin(t)]), np.array([-np.sin(p), np.cos(p), 0.])]
        widths = []
        for tangent in tangents:
            edges = []
            for sign in (-1, 1):
                a = np.r_[0., np.geomspace(1e-6, step, 24), np.arange(step, 180, step)]
                rad = np.deg2rad(sign*a); xyz = np.cos(rad)[:,None]*direction + np.sin(rad)[:,None]*tangent
                invalid = np.flatnonzero(xyz[:, 2] < -1e-10)
                end = int(invalid[0]) if len(invalid) else len(a)
                power = abs(response(xyz[:end, :2]))**2
                hit = np.flatnonzero(power <= level**2 / 2)
                if len(hit) and hit[0] > 0:
                    i = int(hit[0]); fraction = (level**2/2-power[i-1])/(power[i]-power[i-1])
                    edges.append(sign*float(a[i-1]+fraction*(a[i]-a[i-1])))
            widths.append(edges[1]-edges[0] if len(edges) == 2 else None)
    directivity = None
    if full_sphere and view_mode != 'element':
        phases = excitation_phases(result['xy'], case, c.phase_bits if phase_mode == 'quantized' else 0)
        w = weights * np.exp(1j * np.deg2rad(phases['command_phase_deg']))
        directivity = float(10*np.log10(float(np.max(magnitude))**2/integrated_power(points, w)))
    elif full_sphere:
        directivity = 0.
    outside = ~main
    psl = float(db(np.max(magnitude[outside])/level)) if outside.any() and level > 1e-12 and view_mode != 'element' else None
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
