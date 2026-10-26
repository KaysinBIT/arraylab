"""Complex-baseband pulsed radar simulation with explicit acquisition times.

Positive velocity is receding. Received propagation phase is -2*pi*f*path/c.
All voltage-like arrays have units sqrt(W); power arrays are W at the specified
linear processor output. This module never supplies truth labels to estimators.
"""
from __future__ import annotations

import math
import time
import numpy as np
from scipy import signal
from scipy.linalg import hadamard

from .detection import cfar_2d, peak_cells

C = 299792458.0
KB = 1.380649e-23


def _check(cancel):
    if cancel and cancel():
        raise InterruptedError("计算已取消")


def _notify(progress, fraction, message):
    if progress: progress(float(fraction),message)


def waveform_samples(waveform, times=None):
    """Analytic baseband envelope evaluated at sample times, including fractions."""
    fs = float(waveform["sample_rate_hz"])
    tau = float(waveform["pulse_width_s"])
    if times is None:
        times = np.arange(int(math.ceil(tau*fs-1e-10)),dtype=float)/fs
    times = np.asarray(times,dtype=float)
    # Subtracting absolute pulse times can leave round-off at exact support
    # boundaries. Snap only a tiny fraction (1e-8) of a sample, never a physical
    # fractional-delay offset; otherwise an integer echo loses one sample in
    # later pulses and falsely changes coherent integration gain.
    epsilon=1e-8/fs
    times=np.where(np.abs(times)<epsilon,0.,times)
    times=np.where(np.abs(times-tau)<epsilon,tau,times)
    active = (times>=0)&(times<tau)
    kind = waveform.get("kind","lfm")
    if kind=="lfm":
        values = np.exp(1j*np.pi*float(waveform["bandwidth_hz"])/tau*(times-tau/2)**2)
    elif kind=="phase":
        code = np.asarray(waveform.get("phase_code",[1,1,1,-1,-1,1,-1]),dtype=float)
        chip_coordinate=times/tau*len(code)
        nearest=np.rint(chip_coordinate)
        chip_coordinate=np.where(abs(chip_coordinate-nearest)<epsilon*len(code)/tau,nearest,chip_coordinate)
        index = np.clip(np.floor(chip_coordinate).astype(int),0,len(code)-1)
        values = code[index].astype(complex)
    elif kind in ("rect","stepped"):
        values = np.ones(times.shape,dtype=complex)
    else:
        raise ValueError("未知发射波形")
    return np.where(active,values,0j).astype(np.complex128)


from .receiver import receiver_filter


def matched_filter(project):
    s = waveform_samples(project["waveform"])
    b = receiver_filter(project)
    received = np.convolve(s,b)
    # A stepped waveform uses range_window across frequency steps. Applying
    # it here as well would silently add a second, intrapulse mismatch loss.
    name = "rect" if project['waveform']['kind']=='stepped' else project["processing"].get("range_window","rect")
    taper = np.ones(len(received)) if name=="rect" else signal.get_window(name,len(received),fftbins=False)
    energy = float(np.vdot(received,received).real)
    if energy<=0: raise ValueError("接收参考波形能量为零")
    h = np.conj(received[::-1])*taper[::-1]/energy
    gain = complex(np.sum(received*np.conj(received)*taper)/energy)
    total = np.convolve(b,h)
    fs = float(project["waveform"]["sample_rate_hz"])
    tau = float(project["waveform"]["pulse_width_s"])
    efficiency = abs(gain)**2/(fs*tau*float(np.vdot(total,total).real))
    return h,b,{"sampled_transmit_energy_s":float(np.vdot(s,s).real/fs),
        "receive_reference_energy_s":energy/fs,"matched_amplitude_gain":float(gain.real),
        "noise_filter_energy":float(np.vdot(total,total).real),
        "single_pulse_snr_efficiency":efficiency,
        "range_window_loss_db":float(-10*np.log10(max(efficiency,1e-300))),
        "receiver_filter_taps":len(b),"filter_length":len(h)}


def pulse_response(project):
    """Measured zero-Doppler compression response and sampled sidelobe metrics."""
    h,b,_=matched_filter(project)
    received=np.convolve(waveform_samples(project["waveform"]),b)
    response=np.convolve(received,h)
    factor=8
    response=signal.resample(response,len(response)*factor)
    fs=float(project["waveform"]["sample_rate_hz"])*factor
    lags=(np.arange(len(response))-(len(received)-1)*factor)/fs
    power=abs(response)**2
    peak=int(np.argmax(power));maximum=max(float(power[peak]),1e-300)
    # Nearest local minima delimit the measured main lobe. For an unmodulated
    # triangular correlation, the support boundary is the only such minimum.
    minima=signal.find_peaks(-power)[0]
    left=minima[minima<peak];right=minima[minima>peak]
    l=int(left[-1]) if len(left) else 0
    r=int(right[0]) if len(right) else len(power)-1
    outside=np.r_[power[:l],power[r+1:]]
    main=power[l:r+1]
    pslr=float(10*np.log10(max(float(outside.max()),1e-300)/maximum)) if len(outside) else None
    islr=float(10*np.log10(max(float(outside.sum()),1e-300)/max(float(main.sum()),1e-300))) if len(outside) else None
    half=maximum/2
    below_l=np.flatnonzero(power[:peak]<half)
    below_r=np.flatnonzero(power[peak+1:]<half)
    hpbw=None
    if len(below_l) and len(below_r):
        il=int(below_l[-1]);ir=int(peak+1+below_r[0])
        xleft=lags[il]+(half-power[il])/(power[il+1]-power[il])*(lags[il+1]-lags[il])
        xright=lags[ir-1]+(half-power[ir-1])/(power[ir]-power[ir-1])*(lags[ir]-lags[ir-1])
        hpbw=float((xright-xleft)*C/2)
    metrics={"pulse_compression_pslr_db":pslr,"pulse_compression_islr_db":islr,
             "pulse_compression_hpbw_m":hpbw,
             "pulse_response_measurement":"actual sampled waveform and receive/matched filter; zero Doppler; 8x grid interpolation"}
    return metrics,{"pulse_response_lag_s":lags,
                    "pulse_response_db":10*np.log10(np.maximum(power/maximum,1e-15))}


def _historical_events(plan, horizon, project):
    if not np.isfinite(horizon) or horizon<0:raise ValueError("历史回波时间范围必须为有限非负值")
    times = np.asarray(plan["times_s"],float)
    intervals = np.asarray(plan["intervals_s"],float)
    freqs = np.asarray(plan["frequencies_hz"],float)
    indices = np.asarray(plan["tx_indices"],int)
    codes = np.asarray(plan["code_rows"],float)
    mode = project["array"]["mode"]
    events = [(float(t),float(f),int(idx),codes[i],i) for i,(t,f,idx) in enumerate(zip(times,freqs,indices))]
    # Only ordinary phased operation supports ambiguous prior returns. Past
    # transmitter pulses are real contributions, never labelled during decode.
    if mode=="phased" and project["waveform"]["kind"]!="stepped":
        timing = project["timing"]
        seq = np.asarray(timing.get("pri_sequence_s",[timing["pri_s"]]),float)
        if timing["mode"]!="staggered": seq=np.array([intervals[0]])
        fs=float(project["waveform"]["sample_rate_hz"])
        ticks=np.rint(seq*fs).astype(np.int64)
        if np.any(ticks<=0):raise ValueError("实际 PRI 至少需要一个采样时钟周期")
        elapsed_ticks,step=0,0
        while elapsed_ticks < math.ceil(horizon*fs):
            step += 1
            elapsed_ticks+=int(ticks[(-step)%len(ticks)])
            past=float(times[0])-elapsed_ticks/fs
            events.insert(0,(past,float(freqs[0]),-1,np.ones(codes.shape[1]),-step))
            if step>4096:
                raise ValueError("跨 PRI 历史脉冲超过 4096 个，请缩小距离搜索域")
    return events


def _unit_from_angles(angles):
    az,el = np.deg2rad(np.asarray(angles,float))
    return np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])


def _prepare_scene(project):
    from . import scene
    grid = scene.terrain_grid(project)
    objects = [dict(t) for t in project["scene"].get("targets",[])]
    ground = scene.terrain_scatterers(project,grid)
    for g in ground:
        g=dict(g);g["_ground"]=True
        objects.append(g)
    return grid,objects,len(ground)


def synthesize_echo(project, plan, progress=None, cancel=None):
    """Receive only the physical Tx sum; no per-target/Tx raw arrays survive."""
    from . import scene
    from .receiver import capture_layout,filter_capture
    wave,timing,radar,array = (project[k] for k in ("waveform","timing","radar","array"))
    fs,fc,tau = float(wave["sample_rate_hz"]),float(wave["fc_hz"]),float(wave["pulse_width_s"])
    b=receiver_filter(project)
    layout=capture_layout(project,plan,b)
    tstart=layout['capture_start_s']
    nfast=len(layout['relative_s'])
    times = np.asarray(plan["times_s"],float)
    relative=layout['relative_s']
    nt,nr = int(array["tx_count"]),int(array["rx_count"])
    tx = np.asarray(scene.element_positions(array,"tx",fc))
    rx = np.asarray(scene.element_positions(array,"rx",fc))
    raw = np.zeros((len(times),nr,nfast),dtype=np.complex128)
    valid=layout['input_valid']
    # Receive noise is generated independently at each physical Rx before the
    # receive filter, and is disabled during Tx/recovery/gaps.
    seed = int(project.get("seed",1))
    noise_rng = np.random.default_rng(np.random.SeedSequence([seed,11]))
    scatter_rng = np.random.default_rng(np.random.SeedSequence([seed,29]))
    n0 = KB*float(radar["temperature_k"])*10**(float(radar["noise_figure_db"])/10)
    if radar.get("noise_enabled",True):
        raw.real = noise_rng.standard_normal(raw.shape)*math.sqrt(n0*fs/2)
        raw.imag = noise_rng.standard_normal(raw.shape)*math.sqrt(n0*fs/2)
        raw *= valid[:,None,:]
    grid,objects,ground_count = _prepare_scene(project)
    platform = project["scene"]["platform"]
    pose = getattr(scene,"pose_with_terrain",None)
    def object_pose(obj,t):
        return pose(obj,t,grid) if pose else scene.pose_at(obj,t)
    p0,_ = object_pose(platform,0)
    horizon = 0.0
    for obj in objects:
        pos,_ = object_pose(obj,0)
        horizon=max(horizon,2*np.linalg.norm(np.asarray(pos)-p0)/C+tau)
    horizon = max(horizon,2*float(project["processing"].get("range_search_m",[0,0])[1])/C if project["processing"].get("ambiguity_search") else 0)
    events = _historical_events(plan,horizon,project)
    fixed_gains = []
    for obj in objects:
        if "complex_gain" in obj:
            fixed_gains.append(complex(obj["complex_gain"]))
        elif obj.get("fluctuation","none")=="swerling1":
            fixed_gains.append(complex(*scatter_rng.standard_normal(2))/math.sqrt(2))
        else: fixed_gains.append(1+0j)
    pt = float(radar["peak_power_w"])
    gains = 10**((float(radar["tx_gain_dbi"])+float(radar["rx_gain_dbi"])-float(radar["loss_db"]))/10)
    blocked,unknown,visible = 0,0,0
    for event_number,(te,freq,active_tx,code,ordinal) in enumerate(events):
        _check(cancel)
        if event_number%max(1,len(events)//50)==0:
            _notify(progress,.05+.43*event_number/max(1,len(events)),f"生成真实回波 {event_number+1}/{len(events)} 脉冲")
        pp,attitude = object_pose(platform,te)
        pv=scene.velocity_at(platform,te,grid)
        rotation = scene.rotation_matrix(attitude)
        rotation_rate=scene.rotation_rate_matrix(attitude,scene.attitude_rate_at(platform,te))
        tx_world = np.asarray(pp)+tx@rotation.T
        rx_world = np.asarray(pp)+rx@rotation.T
        tx_velocity=pv+tx@rotation_rate.T
        rx_velocity=pv+rx@rotation_rate.T
        pointing = np.asarray(array.get("pointing_deg",[0,0]),float)+te*np.asarray(array.get("scan_rate_deg_s",[0,0]),float)
        usteer = _unit_from_angles(pointing)
        if array["mode"]=="phased":
            weights=np.exp(-1j*2*np.pi*freq/C*(tx@usteer))/math.sqrt(nt)
        elif array["mode"]=="tdm":
            weights=np.zeros(nt,complex);weights[active_tx]=1
        else: weights=np.asarray(code,complex)/math.sqrt(nt)
        for oi,obj in enumerate(objects):
            if oi%64==0: _check(cancel)
            position,_ = object_pose(obj,te)
            position=np.asarray(position,float)
            velocity=scene.velocity_at(obj,te,grid)
            local_direction=rotation.T@(position-np.asarray(pp))
            distance=float(np.linalg.norm(local_direction))
            if distance<=0: continue
            pattern=float(scene.beam_power_gain(local_direction,pointing,array.get("beamwidth_deg",[180,180])))
            if pattern<=1e-14: continue
            if grid is not None:
                los=scene.line_of_sight(pp,position,grid)
                if los is False: blocked+=1;continue
                if los is None: unknown+=1;continue
            rcs=float(obj.get("rcs_m2",1))
            if rcs<=0: continue
            visible+=1
            scatter=fixed_gains[oi]
            if obj.get("fluctuation")=="swerling2":
                scatter=complex(*scatter_rng.standard_normal(2))/math.sqrt(2)
            rt=np.linalg.norm(position-tx_world,axis=1)
            rr=np.linalg.norm(position-rx_world,axis=1)
            if min(rt.min(),rr.min())<=0: continue
            vt=np.sum(((position-tx_world)/rt[:,None])*(velocity-tx_velocity),axis=1)
            vr=np.sum(((position-rx_world)/rr[:,None])*(velocity-rx_velocity),axis=1)
            for it in np.flatnonzero(abs(weights)>0):
                for ir in range(nr):
                    path=rt[it]+rr[ir]
                    rate=vt[it]+vr[ir]
                    arrival=te+path/C
                    last=arrival+tau*(1+abs(rate)/C)
                    first_interval=max(0,int(np.searchsorted(times,arrival,side="right"))-1)
                    last_interval=min(len(times)-1,int(np.searchsorted(times,last,side="right"))-1)
                    if last_interval<0 or first_interval>=len(times):continue
                    received_power=pt*gains*(C/freq)**2*rcs/((4*np.pi)**3*rt[it]**2*rr[ir]**2)
                    # The normalized element/subarray envelope applies once on
                    # each propagation leg; explicit Tx geometry adds its gain.
                    amplitude=math.sqrt(received_power)*pattern*weights[it]*scatter
                    for ip in range(first_interval,last_interval+1):
                        origin=times[ip]+tstart
                        j0=max(0,int(math.floor((arrival-origin)*fs))-1)
                        j1=min(nfast,int(math.ceil((last-origin)*fs))+2)
                        if j1<=j0:continue
                        absolute=origin+np.arange(j0,j1)/fs
                        u=absolute-te-path/C
                        # First-order stop/go delay plus intra-pulse Doppler;
                        # event-to-event motion already enters path above.
                        envelope=waveform_samples(wave,u/(1+rate/C))
                        phase=np.exp(-1j*2*np.pi*freq/C*(path+rate*u))
                        if abs(freq-float(plan["frequencies_hz"][ip]))>1e-6:
                            phase*=np.exp(1j*2*np.pi*(freq-float(plan["frequencies_hz"][ip]))*(absolute-times[ip]))
                        raw[ip,ir,j0:j1]+=amplitude*envelope*phase*valid[ip,j0:j1]
    h,b,filter_info=matched_filter(project)
    raw=filter_capture(raw,b,layout).astype(np.complex128)
    valid=layout['output_valid']
    relative=relative[layout['history_samples']:]
    diagnostics={"noise_psd_w_per_hz":n0,"input_complex_noise_power_w":n0*fs,
        "history_pulses":len(events)-len(times),"terrain_scatterers":ground_count,
        "occluded_scatterer_events":blocked,"unknown_los_scatterer_events":unknown,
        "visible_scatterer_events":visible,"sample_count":int(raw.size),
        "adc_filter_history_samples":layout['history_samples'],
        "receiver_record_model":"每 PRI 发射期间滤波器状态复位；ADC 起点仅裁剪记录，保留因果 FIR 前史",
        **filter_info}
    return raw,valid,relative,diagnostics


def pulse_compress(raw, project, cancel=None):
    from .receiver import compression_origin_s
    h,_,_=matched_filter(project)
    factor=1 if project["waveform"]["kind"]=="stepped" else int(project["processing"].get("range_fft_factor",1))
    fs=float(project['waveform']['sample_rate_hz'])
    pre=int(round((float(project['timing']['receive_start_s'])-compression_origin_s(project,len(h)))*fs))
    result=np.empty((*raw.shape[:-1],(raw.shape[-1]+pre)*factor),dtype=np.complex128)
    for ip in range(len(raw)):
        _check(cancel)
        full=signal.fftconvolve(raw[ip],h[None,:],mode="full",axes=-1)
        if factor>1:
            # Frequency zero-padding of the full linear correlation refines
            # fractional lags; resample preserves amplitude, not extra energy.
            full=signal.resample(full,full.shape[-1]*factor,axis=-1)
        result[ip]=full[:,(len(h)-1-pre)*factor:(len(h)-1+raw.shape[-1])*factor]
    return result


def mti_matrix(times,order=0):
    times=np.asarray(times,float)
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("慢时间时刻必须为有限且严格递增的一维序列")
    n=len(times)
    if order in (0,1,None):return np.eye(n)
    if order==2:
        if n<2:raise ValueError("二脉冲 MTI 至少需要两个观测")
        return np.eye(n)[1:]-np.eye(n)[:-1]
    if order==3:
        if n<3:raise ValueError("三脉冲 MTI 至少需要三个观测")
        result=np.zeros((n-2,n))
        for i in range(n-2):
            a,b=times[i+1]-times[i],times[i+2]-times[i+1]
            if min(a,b)<=0:raise ValueError("脉冲时间必须递增")
            result[i,i:i+3]=np.array([b,-(a+b),a])/max(a,b)
        return result
    raise ValueError("MTI 仅支持关闭、二脉冲或三脉冲")


def slow_filter_bank(times,velocities,frequency,mti=0,window="hann",codes=None):
    """Actual-time matched bank, with MTI applied to signal AND templates.

    Norm is taken in the original independent-noise coordinates, accounting
    for MTI-induced correlation. It does not inflate gain by padding samples.
    """
    times=np.asarray(times,float);velocities=np.asarray(velocities,float)
    if times.ndim!=1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("慢时间时刻必须为有限且严格递增的一维序列")
    if velocities.ndim!=1 or not len(velocities) or not np.isfinite(velocities).all() or not np.isfinite(frequency) or frequency<=0:
        raise ValueError("速度网格及载频必须有限，载频必须为正")
    if mti not in (0,1,None,2,3):raise ValueError("MTI 仅支持关闭、二脉冲或三脉冲")
    order=0 if mti in (0,1,None) else int(mti)
    if order not in (0,2,3) or len(times)<max(1,order):
        raise ValueError("MTI 阶数或有效观测数不合法")
    # Reduce phase cycles before exp; snap only floating-point-sized errors at
    # exact integer cycles. Otherwise a mathematically exact blind speed at a
    # long CPI produces round-off which unit-noise normalization amplifies.
    cycles=(-2*float(frequency)/C)*velocities[:,None]*(times-times[0])[None,:]
    nearest=np.rint(cycles)
    tolerance=32*np.finfo(float).eps*np.maximum(1.,abs(cycles))
    cycles=np.where(abs(cycles-nearest)<=tolerance,nearest,cycles)
    e=np.exp(2j*np.pi*np.remainder(cycles,1.))
    # Preserve absolute event reference: TDM Tx sequences have different t0.
    e*=np.exp(-1j*4*np.pi*float(frequency)/C*velocities*times[0])[:,None]
    if codes is not None:
        codes=np.asarray(codes)
        if codes.shape!=times.shape or not np.isfinite(codes).all():raise ValueError("编码序列与慢时间时刻不匹配")
        e*=codes[None,:]
    # Apply the banded canceller and its adjoint directly: no N-by-N identity
    # or dense differencing matrix is needed for a long acquisition.
    if order==0:processed=e;nout=len(times)
    elif order==2:processed=e[:,1:]-e[:,:-1];nout=len(times)-1
    else:
        dt=np.diff(times);scale=np.maximum(dt[:-1],dt[1:])
        w0=dt[1:]/scale;w1=-(dt[:-1]+dt[1:])/scale;w2=dt[:-1]/scale
        processed=e[:,:-2]*w0+e[:,1:-1]*w1+e[:,2:]*w2;nout=len(times)-2
    if isinstance(window,str):
        if window=="hann" and nout<3:
            raise ValueError("Hann 慢时间窗至少需要 3 个有效观测，请增加脉冲数或明确选择矩形窗")
        taper=np.ones(nout) if window=="rect" else signal.get_window(window,nout,fftbins=False)
    else:
        taper=np.asarray(window,float)
        if taper.shape!=(nout,) or not np.isfinite(taper).all():raise ValueError("慢时间窗长度与处理后观测数不符")
    weighted=np.conj(processed)*taper[None,:]
    if order==0:functional=weighted
    else:
        functional=np.zeros_like(e)
        if order==2:
            functional[:,:-1]-=weighted;functional[:,1:]+=weighted
        else:
            functional[:,:-2]+=weighted*w0;functional[:,1:-1]+=weighted*w1;functional[:,2:]+=weighted*w2
    norms=np.linalg.norm(functional,axis=1)
    floor=64*np.finfo(float).eps*math.sqrt(len(times))*max(1.,float(np.max(abs(taper))))
    functional/=np.where(norms>floor,norms,np.inf)[:,None]
    return functional,norms


def velocity_axis(project):
    lo,hi=map(float,project["processing"]["velocity_search_mps"])
    return np.linspace(lo,hi,int(project["processing"]["doppler_bins"]))


def doppler_process(compressed,project,plan,cancel=None):
    from . import scene
    from .mimo_support import receive_steering
    times=np.asarray(plan["times_s"],float)
    fc=float(project["waveform"]["fc_hz"])
    velocities=velocity_axis(project)
    array=project["array"];mode=array["mode"]
    nr,nt=int(array["rx_count"]),int(array["tx_count"])
    rx=np.asarray(scene.element_positions(array,"rx",fc))
    tx=np.asarray(scene.element_positions(array,"tx",fc))
    mti=project["processing"].get("mti",0)
    window=project["processing"].get("doppler_window","hann")
    functionals=[]
    if mode=="phased":
        bank,norms=slow_filter_bank(times,velocities,fc,mti,window)
        cube=np.einsum("vp,pcf->vcf",bank,compressed,optimize=True)
        positions=rx
        functionals=[bank]
    elif mode=="tdm":
        cubes=[];norms=[]
        ids=np.asarray(plan["tx_indices"],int)
        for it in range(nt):
            _check(cancel)
            take=np.flatnonzero(ids==it)
            bank,norm=slow_filter_bank(times[take],velocities,fc,mti,window)
            cubes.append(np.einsum("vp,pcf->vcf",bank,compressed[take],optimize=True))
            norms.append(norm)
            expanded=np.zeros((len(velocities),len(times)),complex)
            expanded[:,take]=bank
            functionals.append(expanded)
        cube=np.concatenate(cubes,axis=1)
        positions=(tx[:,None,:]+rx[None,:,:]).reshape(nt*nr,3)
    elif mode=="cdm":
        # Joint code-Doppler filter, exactly equivalent to motion-compensated
        # Walsh decoding. Never split source Tx data in the synthesizer.
        codes=np.asarray(plan["code_rows"],float)
        cubes=[];norms=[]
        if mti:
            # MTI must operate at the same code position across blocks, not
            # cancel adjacent pulses carrying different transmit phases.
            for it in range(nt):
                combined=np.zeros((len(velocities),nr,compressed.shape[-1]),complex)
                functional=np.zeros((len(velocities),len(times)),complex)
                for il in range(nt):
                    take=np.arange(il,len(times),nt)
                    bank,_=slow_filter_bank(times[take],velocities,fc,mti,window)
                    combined+=codes[il,it]*np.einsum("vp,pcf->vcf",bank,compressed[take],optimize=True)/math.sqrt(nt)
                    functional[:,take]=codes[il,it]*bank/math.sqrt(nt)
                cubes.append(combined)
                functionals.append(functional)
        else:
            block_count=len(times)//nt
            if window=="hann" and block_count<3:
                raise ValueError("CDM Hann 窗至少需要 3 个完整编码块")
            block_window=np.ones(block_count) if window=="rect" else signal.get_window(window,block_count,fftbins=False)
            code_window=np.repeat(block_window,nt)
            for it in range(nt):
                _check(cancel)
                bank,norm=slow_filter_bank(times,velocities,fc,0,code_window,codes[:,it])
                cubes.append(np.einsum("vp,pcf->vcf",bank,compressed,optimize=True));norms.append(norm)
                functionals.append(bank)
        cube=np.concatenate(cubes,axis=1)
        positions=(tx[:,None,:]+rx[None,:,:]).reshape(nt*nr,3)
    else:raise ValueError("未知阵列收发体制")
    # Detection beam follows the known commanded scan at every receive event.
    # Keep the unsteered cube separately for DOA. Summing Tx decoded streams
    # is normalized in the original physical Rx noise coordinates, including
    # any scan-induced loss of code orthogonality.
    combined=np.zeros((len(velocities),len(times),nr),complex)
    for it,functional in enumerate(functionals):
        pos=rx if mode=="phased" else tx[it]+rx
        steering=receive_steering(project,plan,pos)
        combined+=functional[:,:,None]*np.conj(steering)[None,:,:]
    norm=np.sqrt(np.sum(abs(combined)**2,axis=(1,2)))
    combined/=np.where(norm>1e-12,norm,np.inf)[:,None,None]
    beam=np.einsum("vpc,pcf->vf",combined,compressed,optimize=True)
    return abs(beam)**2,cube,velocities,positions


def _interpolate_pulse_ranges(data,delays,origin,fs):
    """data [pulse,channel,sample], delays [pulse,range] in seconds."""
    data=np.asarray(data)
    location=(np.asarray(delays,float)-origin)*fs
    if data.ndim!=3 or location.ndim!=2 or location.shape[0]!=len(data) or data.shape[-1]<1:
        raise ValueError('距离插值需要 [脉冲,通道,采样] 与 [脉冲,距离] 数据')
    # Preserve exact endpoints. A final integer sample does not need a sample
    # beyond the array, and a one-sample input still has one valid location.
    nearest=np.rint(location)
    location=np.where(abs(location-nearest)<1e-8,nearest,location)
    valid=np.isfinite(location)&(location>=0)&(location<=data.shape[-1]-1)
    safe=np.where(valid,location,0.)
    low=np.floor(safe).astype(int)
    fraction=safe-low
    high=np.minimum(low+1,data.shape[-1]-1)
    p=np.arange(len(data))[:,None,None]
    c=np.arange(data.shape[1])[None,:,None]
    a=data[p,c,low[:,None,:]]
    b=data[p,c,high[:,None,:]]
    return (a*(1-fraction[:,None,:])+b*fraction[:,None,:])*valid[:,None,:]


def stepped_process(compressed,project,plan,progress=None,cancel=None):
    """Joint frequency/time matched search with data-derived velocity hypotheses.

    Each velocity has its own delay alignment and frequency-dependent phase
    correction. All frequency steps and repeated groups remain distinct.
    """
    from . import scene
    from .stepped_support import stepped_weights,stepped_movement,complete_receive_range
    wave=project["waveform"];proc=project["processing"]
    fs,fc=float(wave["sample_rate_hz"]),float(wave["fc_hz"])
    times=np.asarray(plan["times_s"],float)
    frequencies=np.asarray(plan["frequencies_hz"],float)
    steps=int(wave["frequency_steps"])
    if len(times)%steps:raise ValueError("步进频需要整数个完整频率组")
    groups=len(times)//steps
    origin=float(project["timing"].get("receive_start_s",0))
    coarse_lower,coarse_upper=complete_receive_range(project,plan)
    lower=max(0,float(proc["range_search_m"][0]),coarse_lower)
    upper=min(float(proc["range_search_m"][1]),coarse_upper)
    spacing=C/(2*steps*float(wave["frequency_step_hz"]))/int(proc.get("range_fft_factor",1))
    # Honour the declared search domain: do not round its upper limit outward
    # by a fraction of a bin (which could also cross the complete-receive gate).
    count=max(0,int(math.floor((upper-lower)/spacing+1e-10))+1)
    ranges=lower+np.arange(count)*spacing
    ranges=ranges[ranges<=upper+max(1.,abs(upper))*1e-13]
    if len(ranges)<2:raise ValueError("步进频距离搜索窗至少需要两个格点")
    velocities=velocity_axis(project)
    nr=compressed.shape[1]
    estimated_bytes=(len(velocities)*nr*len(ranges)*16*2+compressed.nbytes*3)
    memory_limit=float(project["resources"]["memory_mib"])*1024**2
    per_range_workspace=len(times)*(nr*16*4+16*6)
    chunk=min(128,int((memory_limit-estimated_bytes)/max(1,per_range_workspace)))
    if chunk<1:
        raise ValueError("步进频距离–速度搜索工作内存超过预算，请明确缩小搜索范围或格点数")
    cube=np.zeros((len(velocities),nr,len(ranges)),complex)
    beam=np.zeros((len(velocities),len(ranges)),complex)
    rx=np.asarray(scene.element_positions(project["array"],"rx",fc))
    directions=np.asarray([_unit_from_angles(np.asarray(project["array"].get("pointing_deg",[0,0]))+
                          t*np.asarray(project["array"].get("scan_rate_deg_s",[0,0]))) for t in times])
    steering=np.exp(1j*2*np.pi/C*frequencies[:,None]*(directions@rx.T))
    # A unit-energy outer product keeps frequency/group window losses separate.
    weights,_=stepped_weights(project,plan)
    for vi,velocity in enumerate(velocities):
        _check(cancel)
        _notify(progress,.56+.25*vi/len(velocities),f"步进频联合距离–速度搜索 {vi+1}/{len(velocities)}")
        movement=stepped_movement(project,plan,velocity)
        for start in range(0,len(ranges),chunk):
            _check(cancel)
            segment=ranges[start:start+chunk]
            path_ranges=segment[None,:]+movement[:,None]
            z=_interpolate_pulse_ranges(compressed,2*path_ranges/C,origin,fs)
            phase=np.exp(1j*4*np.pi/C*frequencies[:,None]*path_ranges)*weights[:,None]
            cube[vi,:,start:start+len(segment)]=np.einsum("pcr,pr->cr",z,phase,optimize=True)
            beam[vi,start:start+len(segment)]=np.einsum("pcr,pr,pc->r",z,phase,np.conj(steering)/math.sqrt(nr),optimize=True)
    return abs(beam)**2,cube,velocities,ranges,rx


def _ordinary_snapshots(compressed,project,plan,range_index,velocity):
    times=np.asarray(plan["times_s"],float)
    frequency=float(project["waveform"]["fc_hz"])
    mode=project["array"]["mode"]
    nt=int(project["array"]["tx_count"])
    corrected=compressed[:,:,range_index]*np.exp(1j*4*np.pi*frequency*velocity/C*times[:,None])
    if mode=="phased":return corrected.T
    if mode=="tdm":
        ids=np.asarray(plan["tx_indices"])
        sequences=[corrected[ids==it].T for it in range(nt)]
        length=min(x.shape[1] for x in sequences)
        return np.concatenate([x[:,:length] for x in sequences],axis=0)
    codes=np.asarray(plan["code_rows"],float)
    blocks=len(times)//nt
    observations=corrected[:blocks*nt].reshape(blocks,nt,-1)
    h=codes[:nt]
    decoded=np.einsum("lt,blr->btr",h,observations,optimize=True)/nt
    return decoded.reshape(blocks,-1).T


def _doppler_local_snapshots(series):
    """Disjoint sub-CPI integrations after data-based Doppler demodulation.

    Multiplying snapshots by a common rotating phase alone would leave their
    covariance unchanged. Actual integrations are required to select Doppler.
    Four disjoint blocks are used when possible; their shorter aperture and
    lower Doppler selectivity are explicitly reported to the caller.
    """
    series=np.asarray(series,complex)
    count=series.shape[1]
    blocks=min(4,max(1,count//8))
    partitions=np.array_split(np.arange(count),blocks)
    filtered=np.stack([series[:,take].sum(axis=1)/math.sqrt(len(take)) for take in partitions],axis=1)
    return filtered,{"doppler_selection":"nonoverlapping demodulated sub-CPI integrations",
        "sub_cpi_blocks":blocks,"samples_per_block":[len(take) for take in partitions],
        "doppler_selectivity_note":"子 CPI 积累选择目标多普勒；带宽宽于完整 CPI，邻近速度目标仍可进入协方差。"}


def _stepped_angles(compressed,project,plan,range_value,velocity):
    from . import arrays,scene
    from .stepped_support import stepped_weights,stepped_movement
    wave=project["waveform"];steps=int(wave["frequency_steps"])
    times=np.asarray(plan["times_s"],float)
    frequencies=np.asarray(plan["frequencies_hz"],float)
    rx=np.asarray(scene.element_positions(project["array"],"rx",float(wave["fc_hz"])))
    movement=stepped_movement(project,plan,velocity)
    path_ranges=range_value+movement
    z=_interpolate_pulse_ranges(compressed,(2*path_ranges/C)[:,None],
          float(project["timing"].get("receive_start_s",0)),float(wave["sample_rate_hz"]))[:,:,0]
    weights,diagnostics=stepped_weights(project,plan)
    z*=np.exp(1j*4*np.pi/C*frequencies*path_ranges)[:,None]*weights[:,None]
    settings=project['processing']['doa'];step=float(settings.get('step_deg',1.))
    az=arrays._axis(settings.get('scan_az_deg',[-60,60]),step,'scan_az_deg')
    el=arrays._axis(settings.get('scan_el_deg',[0,0]),step,'scan_el_deg')
    wavelength=C/float(wave['fc_hz'])
    unique=arrays._unique_rows(rx,wavelength*1e-8)
    warnings=[]
    if len(unique)<len(rx):warnings.append('重合接收位置每个仅保留一路，用于几何测角。')
    rx=rx[unique];z=z[:,unique];channels=len(rx)
    if len(az)*len(el)*channels>8_000_000:raise ValueError('步进频角度搜索超过导向系数预算；请缩小范围或增大步长')
    azm,elm=np.meshgrid(np.deg2rad(az),np.deg2rad(el))
    directions=np.vstack([(np.cos(elm)*np.cos(azm)).ravel(),(np.cos(elm)*np.sin(azm)).ravel(),np.sin(elm).ravel()])
    # Preserve complex values until AFTER frequency/group/space matching.
    # Averaging normalized per-frequency powers discards the synthetic-range
    # phase and a per-column Doppler phase cancels from covariance entirely.
    # Sum repeated groups coherently first, then evaluate each frequency's
    # actual array manifold. Do not centre rx: a reference shift creates a
    # frequency-dependent phase and would alter the selected range template.
    vectors=z.reshape(len(times)//steps,steps,channels).sum(axis=0)
    beam=np.zeros(directions.shape[1],complex)
    for start in range(0,directions.shape[1],256):
        dots=rx@directions[:,start:start+256]
        for k in range(steps):
            beam[start:start+dots.shape[1]]+=vectors[k]@np.exp(-1j*2*np.pi*frequencies[k]/C*dots)/math.sqrt(channels)
    power=(abs(beam)**2).reshape(len(el),len(az))
    maximum=float(power.max());db=10*np.log10(np.maximum(power/max(maximum,1e-300),1e-16))
    peaks=[]
    if channels>1 and maximum>1e-300 and np.ptp(db)>1e-8:
        for iy,ix in peak_cells(power,np.ones(power.shape,dtype=bool),max_count=int(settings.get('source_count',1)),min_separation=(1,1)):
            peaks.append({'azimuth_deg':float(az[ix]),'elevation_deg':float(el[iy]),'relative_db':float(db[iy,ix])})
    else:warnings.append('独立接收孔径或所选距离–速度单元能量不足，未生成角度估计。')
    aperture=arrays._aperture(rx)
    geometry_rank=int(np.linalg.matrix_rank(rx-rx.mean(axis=0),tol=wavelength*1e-8))
    if geometry_rank<2 and len(el)>1:warnings.append('线阵仅提供一个方向余弦，同时估计方位与俯仰可能存在歧义。')
    if len(az)==1:warnings.append('方位角搜索仅包含固定切面，该角度是输入假设，不是独立测量。')
    if len(el)==1:warnings.append('俯仰角搜索仅包含固定切面，该角度是输入假设，不是独立测量。')
    diagnostics.update(method='bartlett',input_channels=compressed.shape[1],effective_channels=channels,
        input_snapshots=len(times),effective_aperture_m=aperture,far_field_distance_m=2*aperture**2/(C/max(frequencies)),
        matched_peak_power_w=maximum,
        frequency_fusion='coherent frequency/time/space matched Bartlett at selected range and velocity',
        range_m=float(range_value),velocity_mps=float(velocity),compensation=project['processing'].get('compensation','estimated'),
        angle_frame='radar body frame at CPI reference',azimuth_is_fixed=len(az)==1,elevation_is_fixed=len(el)==1,
        fixed_azimuth_deg=float(az[0]) if len(az)==1 else None,fixed_elevation_deg=float(el[0]) if len(el)==1 else None,
        geometry_rank=geometry_rank,model='跨频已校准；各频点实际阵列流形，相干保留合成距离与多普勒选择')
    return {'peaks':peaks,'azimuth_deg':az.tolist(),'elevation_deg':el.tolist(),
            'spectrum_db':db.tolist(),'diagnostics':diagnostics,'warnings':warnings}


def _exact_template_score(raw,project,plan,range_value,velocity,return_noise=False):
    """A true receive-domain template, gated and receive-filtered like the data.

    Used to verify delay/Doppler candidates. It has no access to target truth.
    The scalar beam assumes the configured steering direction; angle inference
    remains separate. This score is not a calibrated detection probability.
    """
    from . import scene
    from .receiver import capture_layout,filter_capture,capture_adjoint
    from .mimo_support import receive_steering
    wave=project["waveform"];fs=float(wave["sample_rate_hz"]);fc=float(wave["fc_hz"])
    times=np.asarray(plan["times_s"],float)
    tau=float(wave["pulse_width_s"])
    if min(range_value,range_value+velocity*times[-1])<=0:
        return (0.,0.) if return_noise else 0.
    b=receiver_filter(project);layout=capture_layout(project,plan,b)
    if raw.shape[0]!=len(times) or raw.shape[-1]!=layout['sample_count']:
        raise ValueError("候选模板与真实接收记录尺寸不一致")
    # A receive crop does not reset the analog FIR. Generate precisely the
    # causal prehistory used by synthesis, then use the same capture operator.
    rel=layout['relative_s'];valid=layout['input_valid']
    horizon=(2*range_value/C+tau)/max(1e-9,1+2*min(velocity,0.)/C)
    events=_historical_events(plan,horizon,project)
    template=np.zeros((len(times),len(rel)),complex)
    for te,_,_,_,_ in events:
        path=2*(range_value+velocity*te)
        arrival=te+path/C
        ip=int(np.searchsorted(times,arrival,side="right"))-1
        last=arrival+tau*(1+abs(2*velocity)/C)
        last_ip=int(np.searchsorted(times,last,side="right"))-1
        for p in range(max(0,ip),min(len(times),last_ip+1)):
            u=times[p]+rel-te-path/C
            template[p]+=waveform_samples(wave,u/(1+2*velocity/C))*np.exp(-1j*2*np.pi*fc/C*(path+2*velocity*u))*valid[p]
    template=filter_capture(template,b,layout)
    rx=scene.element_positions(project["array"],"rx",fc)
    steering=receive_steering(project,plan,rx)
    observed=np.einsum("pc,pcf->pf",np.conj(steering)/math.sqrt(len(rx)),raw)
    energy=float(np.vdot(template,template).real)
    if energy<=1e-20:return (0.,0.) if return_noise else 0.
    score=float(abs(np.vdot(template,observed))**2/energy)
    # Adjoint of the causal receive filter gives this template's exact thermal
    # noise variance, including both transmit/recovery input and output gates.
    adjoint=capture_adjoint(template,b,layout)
    radar=project["radar"]
    n0=KB*float(radar["temperature_k"])*10**(float(radar["noise_figure_db"])/10)
    variance=float(n0*fs*np.vdot(adjoint,adjoint).real/energy)
    return (score,variance) if return_noise else score


def ambiguity_candidates(raw,compressed,project,plan,progress=None,cancel=None):
    """Bounded coarse observation search followed by actual waveform verification.

    All transmitter timing is known; only receive samples enter the score. A
    candidate list is retained rather than declaring unique recovery from PRI.
    """
    if not project["processing"].get("ambiguity_search",False):return [],{}
    if project["array"]["mode"]!="phased" or project["waveform"]["kind"]=="stepped":return [],{}
    from . import scene
    from .mimo_support import receive_steering
    from .receiver import compression_origin_s
    proc=project["processing"];wave=project["waveform"]
    rlo,rhi=map(float,proc["range_search_m"])
    fs=float(wave["sample_rate_hz"])*int(proc.get("range_fft_factor",1));fc=float(wave["fc_hz"])
    # User-specified search domain, waveform-resolution coarse spacing, explicit
    # grid in output. Never silently use the target's true distance as a window.
    if wave['kind']=='rect':resolution=C*float(wave['pulse_width_s'])/2
    elif wave['kind']=='phase':resolution=C*float(wave['pulse_width_s'])/(2*len(wave['phase_code']))
    else:resolution=C/(2*float(wave['bandwidth_hz']))
    # At least two samples per nominal compressed range scale, without claiming
    # that a finer search grid improves physical resolution.
    step=max(C/(2*fs),resolution/2)
    rg=rlo+np.arange(int(math.floor((rhi-rlo)/step))+1)*step
    if rhi-rg[-1]>max(1e-9,step*1e-10):rg=np.r_[rg,rhi]
    if len(rg)>4096:raise ValueError("距离解模糊搜索超过 4096 个粗距离格点，请明确缩小搜索域")
    velocities=velocity_axis(project)
    times=np.asarray(plan["times_s"],float)
    horizon=(2*rhi/C+float(wave['pulse_width_s']))/max(1e-9,1+2*min(float(velocities.min()),0.)/C)
    events=_historical_events(plan,horizon,project)
    emitted=np.array([e[0] for e in events])
    rx=scene.element_positions(project["array"],"rx",fc)
    steering=receive_steering(project,plan,rx)
    observed=np.einsum("pc,pcf->pf",np.conj(steering)/math.sqrt(len(rx)),compressed)
    origin=compression_origin_s(project,len(matched_filter(project)[0]))
    grid=np.zeros((len(velocities),len(rg)))
    # Bound temporary work by 32 velocities rather than Nv * all pulse events.
    # The final displayed grid is unchanged; no hypothesis is discarded.
    blocks=(len(velocities)+31)//32
    for block,vstart in enumerate(range(0,len(velocities),32)):
        _check(cancel)
        current=velocities[vstart:vstart+32]
        motion=current[:,None]*emitted[None,:]
        bank=np.exp(1j*4*np.pi*fc/C*motion)
        for ir,r in enumerate(rg):
            if ir%32==0:
                _check(cancel)
                fraction_done=(block*len(rg)+ir)/(blocks*len(rg))
                _notify(progress,.82+.08*fraction_done,f"真实时序距离解模糊：速度块 {block+1}/{blocks}，距离 {ir+1}/{len(rg)}")
            # Delay AND carrier phase use candidate velocity. A fixed delay
            # loses migrating targets before receive-domain verification.
            path_ranges=r+motion
            arrival=emitted[None,:]+2*path_ranges/C
            pi=np.searchsorted(times,arrival,side="right")-1
            okay=(pi>=0)&(pi<len(times))&(path_ranges>0);pi=np.clip(pi,0,len(times)-1)
            # For an up-chirp s(u)=exp(j*pi*S*(u-tau/2)^2), a receding
            # target has fd=-2*fc*v/c. Its ZERO-Doppler pulse-compression
            # peak lies at delay-fd/S. Candidate range is the physical delay,
            # so sample that known ambiguity ridge, not the biased lag itself.
            coupling=0.
            if wave['kind']=='lfm':
                slope=float(wave['bandwidth_hz'])/float(wave['pulse_width_s'])
                coupling=(2*fc/C*current[:,None])/slope
            loc=(arrival-times[pi]-origin+coupling)*fs
            li=np.floor(loc).astype(int);fraction=loc-li
            okay&=(loc>=0)&(loc<=observed.shape[-1]-1)
            li=np.clip(li,0,observed.shape[-1]-1);upper=np.minimum(li+1,observed.shape[-1]-1)
            values=observed[pi,li]*(1-fraction)+observed[pi,upper]*fraction
            values=np.where(okay,values,0j)
            count=okay.sum(axis=1)
            grid[vstart:vstart+len(current),ir]=abs(np.sum(bank*values,axis=1))**2/np.maximum(1,count)
    # Candidate peaks only from observations; full receive-domain templates
    # verify the strongest proposals on the declared coarse grid.
    peaks=peak_cells(grid,np.isfinite(grid)&(grid>0),max_count=16,min_separation=(2,2))
    candidates=[]
    dv=float(velocities[1]-velocities[0]) if len(velocities)>1 else 0.
    # Every possible coarse cell has the same finite 5x3 refinement stencil.
    # Bonferroni therefore counts the full possible set, not only data-selected
    # evaluations. Clipping to the declared domain only reduces that count.
    multiplicity=len(rg)*len(velocities)*15
    thermal_threshold=-math.log(float(proc["cfar"]["pfa"])/multiplicity)
    evaluated={}
    for index,(vi,ri) in enumerate(peaks):
        _check(cancel)
        _notify(progress,.90+.08*index/max(1,len(peaks)),f"接收波形模板细化 {index+1}/{len(peaks)}")
        rr=np.unique(np.clip(float(rg[ri])+step*np.array([-.5,-.25,0,.25,.5]),rlo,rhi))
        vv=np.unique(np.clip(float(velocities[vi])+dv*np.array([-.5,0,.5]),velocities[0],velocities[-1]))
        best=None
        for r in rr:
            for v in vv:
                _check(cancel)
                key=(float(r),float(v))
                if key not in evaluated:
                    evaluated[key]=_exact_template_score(raw,project,plan,*key,return_noise=True)
                score,variance=evaluated[key]
                if score<=0 or variance<=0:continue
                ratio=score/variance
                if best is None or ratio>best[0]:best=(ratio,score,variance,key)
        if best is None:continue
        ratio,score,variance,(r,v)=best
        candidates.append({"range_m":r,"velocity_mps":v,
            "score":score,"coarse_score":float(grid[vi,ri]),"verified_waveform_template":True,
            "coarse_range_m":float(rg[ri]),"coarse_velocity_mps":float(velocities[vi]),
            "thermal_noise_score_db":float(10*np.log10(max(ratio,1e-300))),
            "thermal_noise_threshold_db":float(10*np.log10(thermal_threshold)),
            "above_thermal_noise_threshold":bool(ratio>thermal_threshold),
            "thermal_noise_ratio":float(ratio),
            "range_grid_step_m":float(step),
            "refinement_range_step_m":float(step/4),"refinement_velocity_step_mps":float(dv/2),
            "refinement_hypotheses":len(rr)*len(vv),"full_grid_hypotheses_bound":multiplicity,
            "significance_assumption":"thermal noise only; full coarse-grid times 15-point refinement multiplicity; clutter not calibrated; radial constant-speed template"})
    # Different eclipsing/filter crops can have different noise variance. Rank
    # comparable normalized statistics, retaining absolute score separately.
    candidates.sort(key=lambda x:-x["thermal_noise_ratio"])
    unique=[];seen=set()
    for item in candidates:
        key=(item['range_m'],item['velocity_mps'])
        if key not in seen:unique.append(item);seen.add(key)
    candidates=unique
    peak=max((x["thermal_noise_ratio"] for x in candidates),default=1e-300)
    for item in candidates:item["relative_db"]=float(10*np.log10(max(item["thermal_noise_ratio"],1e-300)/max(peak,1e-300)))
    return candidates,{"ambiguity_ranges_m":rg,"ambiguity_velocities_mps":velocities,"ambiguity_power":grid}


def simulate(project,progress=None,cancel=None):
    from .config import normalize_project,validate_project,acquisition_plan
    from . import arrays as array_module
    started=time.perf_counter()
    project=normalize_project(project)
    validation=validate_project(project)
    if not validation["valid"]:raise ValueError("；".join(validation["errors"]))
    plan=acquisition_plan(project)
    _notify(progress,.02,"检查采集计划与复基带能量标定")
    raw,valid,relative,diagnostics=synthesize_echo(project,plan,progress,cancel)
    generated=time.perf_counter()
    _notify(progress,.50,"接收带限与匹配滤波")
    compressed=pulse_compress(raw,project,cancel)
    stepped=project["waveform"]["kind"]=="stepped"
    if stepped:
        power,cube,velocities,ranges,positions=stepped_process(compressed,project,plan,progress,cancel)
    else:
        _notify(progress,.58,"真实时间戳多普勒处理与通道解码")
        power,cube,velocities,positions=doppler_process(compressed,project,plan,cancel)
        factor=int(project["processing"].get("range_fft_factor",1))
        from .receiver import compression_origin_s
        origin=compression_origin_s(project,len(matched_filter(project)[0]))
        ranges=(origin+
                np.arange(compressed.shape[-1])/(float(project["waveform"]["sample_rate_hz"])*factor))*C/2
    _check(cancel)
    _notify(progress,.81,"CFAR 门限与检测")
    threshold,mask,noise=cfar_2d(power,project["processing"]["cfar"],cancel)
    proposed_peaks=peak_cells(power,mask,max_count=65)
    truncated=len(proposed_peaks)>64
    peaks=proposed_peaks[:64]
    detections=[];angle_results=None
    doa=project["processing"].get("doa",{})
    warning=list(validation.get("warnings",[]))
    for index,(vi,ri) in enumerate(peaks):
        _check(cancel)
        d={"range_m":float(ranges[ri]),"velocity_mps":float(velocities[vi]),
            "snr_db":float(10*np.log10(max(power[vi,ri],1e-300)/max(noise[vi,ri],1e-300))),
            "power_w":float(power[vi,ri]),"threshold_w":float(threshold[vi,ri]),
            "range_index":int(ri),"velocity_index":int(vi)}
        if project['waveform']['kind']=='lfm':
            slope=project['waveform']['bandwidth_hz']/project['waveform']['pulse_width_s']
            d.update(range_kind='apparent_zero_doppler_matched_filter',
                     lfm_ambiguity_ridge_offset_m=project['waveform']['fc_hz']*float(velocities[vi])/slope)
        d['angle_status']='disabled' if not doa.get('enabled',True) else ('single_channel' if len(positions)<=1 else ('not_processed_budget' if index>=16 else 'pending'))
        if doa.get("enabled",True) and len(positions)>1 and index<16:
            try:
                if stepped:estimate=_stepped_angles(compressed,project,plan,float(ranges[ri]),float(velocities[vi]))
                else:
                    if doa.get("method","bartlett")=="bartlett":
                        # The complete-CPI Doppler cell is already a spatial
                        # snapshot; covariance of unfiltered pulses would mix
                        # different velocities at the same range.
                        snapshots=cube[vi,:,ri][:,None]
                        temporal={"doppler_selection":"full-CPI matched Doppler cell"}
                    else:
                        series=_ordinary_snapshots(compressed,project,plan,ri,float(velocities[vi]))
                        snapshots,temporal=_doppler_local_snapshots(series)
                    estimate=array_module.estimate_angles(snapshots,positions,float(project["waveform"]["fc_hz"]),doa)
                    estimate.setdefault("diagnostics",{}).update(temporal)
                if estimate.get("peaks"):
                    diagnostic=estimate.get('diagnostics',{})
                    for axis in ('azimuth','elevation'):
                        key=axis+'_deg'
                        if diagnostic.get(axis+'_is_fixed'):
                            d[key]=None;d[axis+'_assumption_deg']=float(estimate['peaks'][0][key])
                        else:d[key]=float(estimate['peaks'][0][key])
                    all_peaks=estimate['peaks']+estimate.get('ambiguous_peaks',[])
                    unique={(v['azimuth_deg'],v['elevation_deg']):v for v in all_peaks}
                    d['angle_candidates']=list(unique.values())
                d["angle_diagnostics"]=estimate.get("diagnostics",{})
                d['angle_status']='estimated_under_model'
                if estimate.get('ambiguous_peaks'):d['angle_status']='spatially_ambiguous'
                farfield=estimate.get('diagnostics',{}).get('far_field_distance_m',validation.get('metrics',{}).get('far_field_distance_m',0))
                if d['range_m']<farfield:
                    d['azimuth_deg']=None;d['elevation_deg']=None;d['angle_status']='outside_far_field_model'
                    warning.append('部分检测距离小于实际孔径的远场参考界限；平面波角谱仅供失配分析，不报告有效测角值。')
                warning.extend(estimate.get("warnings",[]))
                if angle_results is None:angle_results=estimate
            except ValueError as exc:
                d["angle_error"]=str(exc);d['angle_status']='invalid';warning.append(str(exc))
        detections.append(d)
    candidates,extra=ambiguity_candidates(raw,compressed,project,plan,progress,cancel)
    if diagnostics["unknown_los_scatterer_events"]:
        warning.append(f"{diagnostics['unknown_los_scatterer_events']} 个散射点–脉冲事件的地形视线未知，已明确排除；不代表无遮挡。")
    if not np.any(np.isfinite(threshold)):
        warning.append("当前距离／多普勒网格不足以容纳 CFAR 训练窗，没有可检测单元。")
    if truncated:warning.append('检测摘要仅保留最强 64 个分离峰；完整 CFAR 掩码保存在数组导出中。')
    if doa.get('enabled',True) and len(peaks)>16:warning.append('测角仅处理最强 16 个检测峰，其余峰标为未处理；不将空白角度解释为零度。')
    if project['waveform']['kind']=='lfm':
        warning.append('普通 LFM 距离轴与检测距离为零多普勒匹配滤波的表观延迟距离，保留真实距离–多普勒耦合；fc·v/S 仅为模糊脊线近似，不是有限脉冲/带限下的精确峰偏移。可用真实时域模板解模糊候选复核。')
    warning.append("CFAR 名义 Pfa 假定独立指数功率训练单元；脉压、加窗、MTI 与地杂波可改变实测虚警率。")
    if project["array"]["mode"]=="cdm":
        warning.append("CDM 按完整编码块重访；折叠多普勒可能污染解码与测角，首版不保证码分速度解模糊。")
    if candidates:
        warning.append("解模糊结果是限定搜索域、网格及有限观测下的候选，不自动声明唯一解；已按真实接收波形模板复核。")
        if project["radar"].get("noise_enabled",True) and not any(x["above_thermal_noise_threshold"] for x in candidates):
            warning.append("解模糊候选均未超过热噪声全搜索域保守门限；当前没有显著候选，不能将最高峰视作恢复成功。")
    if stepped:warning.append("步进频使用相干频率–时间–空间 Bartlett 匹配，保留精细距离与速度选择；运动补偿为恒速假设。")
    if project['array']['mode'] in ('tdm','cdm') and any(t.get('fluctuation')=='swerling2' for t in project['scene']['targets']):
        warning.append('逐脉冲 Swerling II 复散射会破坏跨 Tx/码块相干性；此场景是通道分离与测角的失效实验，虚拟孔径不保证可用。')
    if stepped and any(t.get('fluctuation')=='swerling2' for t in project['scene']['targets']):
        warning.append('Swerling II 在不同频点脉冲间改变复散射系数，破坏步进频相干合成；本场景用于失配实验，不保证高分辨或补偿成功。')
    s=waveform_samples(project["waveform"])
    response_metrics,response_arrays=pulse_response(project)
    arrays={"raw":raw,"receive_valid":valid,"compressed":compressed,
        "rd_power":power,"cfar_threshold":threshold,"cfar_mask":mask,
        "cfar_noise":noise,"ranges_m":ranges,"velocities_mps":velocities,
        "waveform_real":s.real,"waveform_imag":s.imag,
        "waveform_time_s":np.arange(len(s))/float(project["waveform"]["sample_rate_hz"]),
        "pulse_times_s":np.asarray(plan["times_s"]),"pulse_frequencies_hz":np.asarray(plan["frequencies_hz"]),
        "channel_positions_m":positions,**extra,**response_arrays}
    if stepped:
        k=int(project["waveform"]["frequency_steps"])
        arrays["frequency_data"]=compressed.reshape(len(compressed)//k,k,*compressed.shape[1:])
    if angle_results is not None:
        arrays["angle_azimuth_deg"]=np.asarray(angle_results["azimuth_deg"])
        arrays["angle_elevation_deg"]=np.asarray(angle_results["elevation_deg"])
        arrays["angle_spectrum_db"]=np.asarray(angle_results["spectrum_db"])
    metrics={**validation.get("metrics",{}),**diagnostics,**response_metrics,
        "detections":len(detections),"detection_summary_limit":64,"detections_truncated":truncated,"angle_processing_limit":16,
        "valid_cfar_cells":int(np.isfinite(threshold).sum()),
        "actual_receive_samples":int(valid.sum()),"processing_grid_shape":list(power.shape),
        "range_grid_spacing_m":float(ranges[1]-ranges[0]) if len(ranges)>1 else 0,
        "velocity_grid_spacing_mps":float(velocities[1]-velocities[0]) if len(velocities)>1 else 0}
    if stepped:
        from .stepped_support import stepped_weights
        metrics.update(stepped_weights(project,plan)[1])
    summary={"metrics":metrics,"detections":detections,"warnings":list(dict.fromkeys(warning)),
        "assumptions":["复基带 sqrt(W)，正径向速度表示远离；传播相位 exp(-j2πf(Rtx+Rrx)/c)。",
          "Pt 是同一时刻所有 Tx 的总传导功率；天线增益是每个物理通道的单元／子阵增益。",
          "脉内采用一阶恒速传播，脉间更新三维几何；不求解全波、多径或折射。",
          "CFAR 输入为指向波束的单个复输出功率；SNR 栏是检测单元功率相对训练噪声估计的比值。",
          "步进频外的阵列测角使用 CPI 参考局部阵列；快速转向或明显角迁移会降低快拍一致性。"],
        "timings":{"echo_seconds":generated-started,"total_seconds":time.perf_counter()-started},
        "waveform":project["waveform"]["kind"],"acquisition":plan,
        "range_axis_definition":'表观零多普勒匹配滤波延迟距离' if project['waveform']['kind']=='lfm' else ('相干频率–时间匹配的物理距离假设' if stepped else '匹配滤波延迟距离'),
        "ambiguity_candidates":candidates,
        "ambiguity_status":("noise_free_candidates" if not project["radar"].get("noise_enabled",True)
                            else "significant_candidates" if any(x["above_thermal_noise_threshold"] for x in candidates)
                            else "no_significant_candidate") if candidates else ('no_candidate' if project['processing'].get('ambiguity_search') else 'not_run'),
        "angle_diagnostics":angle_results.get("diagnostics",{}) if angle_results else {},
        "actual_axes":{"raw":["pulse","rx","fast_time"],"rd_power":["velocity","range"],
                       "frequency_data":["repeat_group","frequency_step","rx","fast_time"] if stepped else None}}
    _notify(progress,1,"仿真完成")
    return summary,arrays
