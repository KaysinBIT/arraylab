"""Shared observation-only stepped-frequency templates and window conventions."""
from __future__ import annotations
import numpy as np
from scipy import signal

C = 299792458.0


def stepped_weights(project, plan):
    """Unit-energy pulse weights: one frequency taper and one group taper."""
    steps=int(project['waveform']['frequency_steps'])
    count=len(plan['times_s'])
    if steps<2 or count%steps:raise ValueError('步进频需要至少两个频点及整数个完整重复组')
    groups=count//steps
    proc=project['processing']
    group_name=proc.get('doppler_window','hann')
    frequency_name=proc.get('range_window','rect')
    if group_name=='hann' and groups<3:raise ValueError('步进频 Hann 慢时间窗至少需要 3 个重复频率组')
    if frequency_name=='hann' and steps<3:raise ValueError('步进频 Hann 频率窗至少需要 3 个频点')
    gw=np.ones(groups) if group_name=='rect' else signal.get_window(group_name,groups,fftbins=False)
    fw=np.ones(steps) if frequency_name=='rect' else signal.get_window(frequency_name,steps,fftbins=False)
    taper=(gw[:,None]*fw[None,:]).ravel()
    norm=float(np.linalg.norm(taper))
    if not np.isfinite(norm) or norm<=0:raise ValueError('步进频处理窗能量为零或无效')
    return taper/norm,{
        'frequency_taper_snr_efficiency':float(fw.sum()**2/(len(fw)*np.dot(fw,fw))),
        'group_taper_snr_efficiency':float(gw.sum()**2/(len(gw)*np.dot(gw,gw))),
        'frequency_taper':frequency_name,'group_taper':group_name,
        'frequency_steps':steps,'repeat_groups':groups,
        'pulse_weights_norm_squared':float(np.dot(taper/norm,taper/norm))}


def stepped_movement(project, plan, velocity):
    """Range change relative to acquisition t=0, using the selected comparison mode."""
    times=np.asarray(plan['times_s'],float)
    steps=int(project['waveform']['frequency_steps'])
    groups=np.repeat(times[::steps],steps)
    internal=times-groups
    mode=project['processing'].get('compensation','estimated')
    if mode=='estimated':return float(velocity)*times
    if mode=='known':return float(velocity)*groups+float(project['processing'].get('known_velocity_mps',0))*internal
    if mode=='none':return float(velocity)*groups
    raise ValueError('未知步进频运动补偿方式')


def complete_receive_range(project, plan):
    """Coarse interval for a pulse whose full support is inside its own receive gate."""
    from .receiver import complete_delay_bounds
    from .receiver import receiver_filter
    opening,latest=complete_delay_bounds(project,plan,receiver_filter(project))
    return opening*C/2,latest*C/2
