"""Pointing summaries derived from the existing pattern analysis, not new peak searches."""
import math


def angular_separation(theta_a, phi_a, theta_b, phi_b):
    """Spherical angular distance in degrees; phi is immaterial at the pole."""
    def direction(theta, phi):
        t, p = math.radians(theta), math.radians(phi or 0)
        return (math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), math.cos(t))
    a, b = direction(theta_a, phi_a), direction(theta_b, phi_b)
    cross = (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
    return math.degrees(math.atan2(math.sqrt(math.fsum(x*x for x in cross)), math.fsum(x*y for x,y in zip(a,b))))


def linear_pointing(parameters, settings, patterns):
    target = 0.0 if settings.phase_mode == 'equalize' else parameters.scan_deg
    output = patterns['output']['metrics']; peak = output['main_lobe_angle_deg']
    peaks = list(output['peak_angles_deg'])
    return {'requested_theta_deg': parameters.scan_deg, 'target_theta_deg': target,
            'phase_mode': settings.phase_mode, 'compensation_enabled': settings.phase_mode != 'none',
            'output_cut_peak_deg': peak, 'output_cut_peak_angles_deg': peaks,
            'output_cut_offset_deg': None if peak is None else peak - target,
            'status': 'unique_cut_peak' if peak is not None else 'multiple_cut_peaks' if peaks else 'undefined_cut_peak',
            'scope': 'signed x-z front-half-space cut; not a unique three-dimensional beam',
            'notes': ['角度从+z朝+x为正；自定义附加相位也会改变输出指向。',
                      '不补偿模式仅保留目标用于比较；正侧向模式的有效目标为0°。']}


def planar_pointing(parameters, settings, patterns):
    target = 0.0 if settings.phase_mode == 'equalize' else parameters.scan_theta_deg
    phi = parameters.scan_phi_deg if target != 0 else None
    output = patterns['output']['metrics']
    theta_out, phi_out = output['representative_peak_theta_deg'], output['representative_peak_phi_deg']
    distance = None if theta_out is None else angular_separation(target, phi, theta_out, phi_out)
    return {'requested_theta_deg': parameters.scan_theta_deg, 'requested_phi_deg': parameters.scan_phi_deg,
            'target_theta_deg': target, 'target_phi_deg': phi,
            'phase_mode': settings.phase_mode, 'compensation_enabled': settings.phase_mode != 'none',
            'output_representative_theta_deg': theta_out, 'output_representative_phi_deg': phi_out,
            'representative_peak_separation_deg': distance,
            'peak_search_method': output['peak_search_method'], 'peak_is_not_claimed_unique': True,
            'scope': 'representative upper-hemisphere peak; front/back symmetry and other equal peaks may exist',
            'notes': ['θ从+z轴量起，φ从+x朝+y；θ=0时φ未定义。',
                      '角距只比较所报告代表峰与目标，不代表唯一波束的指向误差。',
                      '自定义附加相位、量化及失效会改变输出；物理尺寸未包含。']}
