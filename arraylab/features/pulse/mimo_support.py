"""Known receive scan schedule, shared by Doppler and ambiguity processors.

Positions and electronic pointing are in the instantaneous radar body frame.
The echo synthesizer rotates that frame into world coordinates. No target
position, target velocity, or simulated source label enters these helpers.
"""
from __future__ import annotations
import numpy as np

C = 299792458.0


def receive_steering(project, plan, positions, frequencies=None):
    """Return manifold ``a[p,c]``; a beamformer uses its conjugate.

    The known commanded electronic scan is evaluated at every actual receive
    pulse event. ``frequencies`` may be scalar or one carrier per event. Using
    body-frame element coordinates is deliberate: platform rotation is already
    represented in the received element phases, and the pointing command is a
    local azimuth/elevation command rather than a world-fixed look direction.
    """
    times=np.asarray(plan['times_s'],dtype=float)
    pos=np.asarray(positions,dtype=float)
    if pos.ndim!=2 or pos.shape[1]!=3 or not np.isfinite(pos).all():
        raise ValueError('接收波束坐标须为有限的 [通道,3] 米制数组')
    array=project['array']
    pointing=np.asarray(array.get('pointing_deg',[0.,0.]),float)
    rates=np.asarray(array.get('scan_rate_deg_s',[0.,0.]),float)
    if pointing.shape!=(2,) or rates.shape!=(2,):raise ValueError('电子扫描指向与速率须为两个分量')
    angles=np.deg2rad(pointing[None,:]+times[:,None]*rates[None,:])
    az,el=angles[:,0],angles[:,1]
    direction=np.column_stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    f=np.asarray(project['waveform']['fc_hz'] if frequencies is None else frequencies,float)
    if f.ndim==0:f=np.full(len(times),float(f))
    if f.shape!=(len(times),) or not np.all(np.isfinite(f)) or np.any(f<=0):
        raise ValueError('接收载频须为正数或与实际事件数一致的频率列表')
    return np.exp(2j*np.pi/C*f[:,None]*(direction@pos.T))


def body_scan_metadata(project,plan):
    times=np.asarray(plan['times_s'],float)
    array=project['array']
    begin=np.asarray(array.get('pointing_deg',[0.,0.]),float)
    end=begin+(times[-1] if len(times) else 0)*np.asarray(array.get('scan_rate_deg_s',[0.,0.]),float)
    return {'frame':'instantaneous radar body frame',
            'reference_time_s':float(times[0]) if len(times) else 0.,
            'pointing_start_deg':begin.tolist(),'pointing_end_deg':end.tolist(),
            'weights_use_actual_event_times':True}
