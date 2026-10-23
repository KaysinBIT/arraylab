"""Uniform B-bit phase commands and fixed-layout hardware replay; no redesign."""
import numpy as np


def validate_bits(bits):
    if type(bits) is not int or not 0<=bits<=12:
        raise ValueError('移相器位数必须为 0–12 的整数；0 表示理想相位、不量化。')


def quantize_degrees(ideal_degrees,bits):
    """Nearest code, circular error; signed half-step ties away from zero."""
    validate_bits(bits)
    ideal=np.asarray(ideal_degrees,dtype=float)
    if not np.all(np.isfinite(ideal)): raise ValueError('相位必须为有限度数。')
    if bits==0:
        return ideal.copy(),np.full(ideal.shape,-1,dtype=int),np.zeros_like(ideal)
    count=2**bits; step=360./count
    signed=(ideal+180.)%360.-180.
    # Tolerance affects only roundoff at an exact half step, not a finite dead zone.
    scaled=abs(signed)/step
    rounded=np.floor(scaled+.5+8*np.finfo(float).eps*np.maximum(1.,scaled))
    codes=(np.sign(signed)*rounded).astype(np.int64)%count
    command=codes*step
    error=(command-signed+180.)%360.-180.
    return command,codes,error


def excitation_phases(xy,case,bits):
    theta,phi=np.deg2rad([case['theta'],case['phi']])
    uv=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi)])
    ideal=-360*case['frequency_ratio']*(np.asarray(xy)@uv)
    command,code,error=quantize_degrees(ideal,bits)
    return {'ideal_phase_deg':ideal,'command_phase_deg':command,'phase_code':code,'error_deg':error}


def quantized_af(xy,weights,uv,case,bits):
    from .core import field
    theta,phi=np.deg2rad([case['theta'],case['phi']])
    center=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi)])
    if bits==0: return field(np.asarray(xy)*case['frequency_ratio'],weights,uv,center)
    phases=excitation_phases(xy,case,bits)
    taper=np.asarray(weights)*np.exp(1j*np.deg2rad(phases['error_deg']))
    return field(np.asarray(xy)*case['frequency_ratio'],taper,uv,center)


def sampled_peak(amplitude,uv,center):
    """Nearest-to-command representative among numerically equal sampled maxima."""
    amplitude=np.asarray(amplitude); uv=np.asarray(uv)
    maximum=float(amplitude.max())
    ties=np.flatnonzero(amplitude>=maximum-max(1e-14,maximum*1e-12))
    index=int(ties[np.argmin(np.sum((uv[ties]-center)**2,axis=1))])
    return maximum,uv[index].tolist()


def angular_distance(a,b):
    def vector(uv):
        u,v=np.asarray(uv,dtype=float)
        return np.array([u,v,np.sqrt(max(0.,1-u*u-v*v))])
    va,vb=vector(a),vector(b)
    # atan2 is stable near zero, unlike arccos(1-roundoff).
    return float(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(va,vb)),np.dot(va,vb))))


def _choose_peak(items,center):
    maximum=max(item[0] for item in items)
    tied=[item for item in items if item[0]>=maximum-max(1e-14,maximum*1e-12)]
    chosen=min(tied,key=lambda item:sum((a-b)**2 for a,b in zip(item[1],center)))
    return maximum,chosen[1]


def evaluate_quantization(result,c,domains,element):
    """Replay only. Ideal optimization and ideal metrics remain explicitly separate."""
    from .core import db,scan_uv,sidelobe_mask
    if not c.phase_bits:
        result['phase_quantization']={'enabled':False,'bits':0}
        return
    rows=[]
    for case,cc,grids in domains:
        center=scan_uv(cc); phases=excitation_phases(result['xy'],case,c.phase_bits)
        gain=float(abs(np.sum(result['weights']*np.exp(1j*np.deg2rad(phases['error_deg'])))))
        valid=gain>1e-8
        sides=[]; global_peaks=[]; main_peaks=[]; af_sides=[]
        for grid_index,uv in enumerate(grids):
            af=quantized_af(result['xy'],result['weights'],uv,case,c.phase_bits)
            total=af if element.model=='isotropic' else af*element.normalized_response(uv,case)
            mask=sidelobe_mask(uv,cc); amplitude=abs(total)
            sides.append(float(amplitude[mask].max())); af_sides.append(float(abs(af[mask]).max()))
            if grid_index:
                global_peaks.append(sampled_peak(amplitude,uv,center))
                main_peaks.append(sampled_peak(amplitude[~mask],uv[~mask],center))
        total_peak,total_uv=_choose_peak(global_peaks,center)
        main_peak,main_uv=_choose_peak(main_peaks,center)
        raw=float(db(max(sides[1:])))
        psl=raw-float(db(gain)) if valid else None
        ideal=result['scenario_metrics'][case['index']]
        ideal_uv=ideal['mainlobe_peak_uv']
        rows.append({**case,'bits':c.phase_bits,'step_deg':360/2**c.phase_bits,
            'max_phase_error_deg':float(np.max(abs(phases['error_deg']))),
            'rms_phase_error_deg':float(np.sqrt(np.mean(phases['error_deg']**2))),
            'steering_response_defined':valid,'steering_field_linear':gain,
            'steering_gain_db':float(db(gain)),'steering_loss_db':max(0.,-float(db(gain))),
            'ideal_validation_psl_db':ideal['validation_psl_db'],'validation_psl_db':psl,
            'delta_psl_db':psl-ideal['validation_psl_db'] if psl is not None else None,
            'validation_psl_ideal_reference_db':raw,
            'optimization_psl_ideal_reference_db':float(db(sides[0])),
            'array_factor_psl_ideal_reference_db':float(db(max(af_sides[1:]))),
            'total_peak_db':float(db(total_peak)),'total_peak_uv':total_uv,
            'mainlobe_peak_db':float(db(main_peak)),'mainlobe_peak_uv':main_uv,
            'relative_to_main_peak_psl_db':float(db(max(sides[1:])/main_peak)) if main_peak>1e-12 else None,
            'ideal_mainlobe_peak_uv':ideal_uv,
            'ideal_pointing_error_deg':angular_distance(center,ideal_uv),
            'pointing_error_deg':angular_distance(center,main_uv),
            'beam_shift_from_ideal_deg':angular_distance(ideal_uv,main_uv),
            'global_pointing_error_deg':angular_distance(center,total_uv),
            'global_peak_outside_mainlobe':bool(sidelobe_mask(np.array([total_uv]),cc)[0]),
            'pointing_definition':'nearest-to-command tied sampled peak; mainlobe peak within fixed mask; angular errors are sampled estimates'})
    invalid=[row for row in rows if not row['steering_response_defined']]
    defined=[row for row in rows if row['validation_psl_db'] is not None]
    worst=max(defined,key=lambda row:row['validation_psl_db']) if defined else None
    psl=worst['validation_psl_db'] if worst is not None and not invalid else None
    metrics={'validation_psl_db':psl,
        'delta_psl_db':psl-result['metrics']['validation_psl_db'] if psl is not None else None,
        'worst_valid_psl_db':worst['validation_psl_db'] if worst else None,
        'invalid_steering_cases':len(invalid),'worst_case_index':invalid[0]['index'] if invalid else worst['index'],
        'max_steering_loss_db':max(row['steering_loss_db'] for row in rows),
        'max_phase_error_deg':max(row['max_phase_error_deg'] for row in rows),
        'max_rms_phase_error_deg':max(row['rms_phase_error_deg'] for row in rows),
        'max_pointing_error_deg':max(row['pointing_error_deg'] for row in rows),
        'max_beam_shift_from_ideal_deg':max(row['beam_shift_from_ideal_deg'] for row in rows),
        'max_global_pointing_error_deg':max(row['global_pointing_error_deg'] for row in rows)}
    result['phase_quantization']={'enabled':True,'bits':c.phase_bits,'step_deg':360/2**c.phase_bits,
        'metrics':metrics,'scenarios':rows,
        'design':'ideal phase synthesis followed by nearest-code replay on unchanged positions and amplitudes',
        'normalization':'plots and raw fields reference ideal steering field; PSL additionally relative to actual steering and mainlobe peak',
        'tie_rule':'wrap ideal to [-180,180), nearest code; exact signed half-step away from zero',
        'frequency_mode':'each frequency is reprogrammed separately; not a fixed-code broadband model'}
