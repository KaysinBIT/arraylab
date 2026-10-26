"""Physical-Rx sampled radar experiment and truth-independent linear receiver.

The independent thermal samples precede the explicitly sampled receiver FIR.
``raw`` in the public result is the filtered ADC record; ``receiver_input``
includes the FIR's real noise/signal history.  The combined fast-time matrix
maps those independent input coordinates to the displayed range samples.
Consequently noise powers include FIR correlation, receive gates, truncation,
windowing, fractional range interpolation, coding and spatial combination.
"""
from __future__ import annotations

from collections import OrderedDict
import math
import time

import numpy as np
from scipy import ndimage, signal as scipy_signal
from scipy.linalg import qr

from .config import (C, checked_config, complex_weights, direction, event_plan,
                     noise_density, resource_estimate)
from .propagation import (paths_for_target, attenuation_for_path,
                          coefficient_for_path)
from .receiver_spec import receiver_filter_spec, observable_path, far_field_spec


_FAST_CACHE: OrderedDict = OrderedDict()
_CACHE_BYTES = 48 * 1024**2


def _check(cancel):
    if cancel is not None and cancel():
        raise InterruptedError("计算已取消，未发布部分结果")


def _progress(callback, fraction, message):
    if callback is not None:
        callback(float(fraction), message)


def window_values(name, count):
    count = int(count)
    if count < 1:
        raise ValueError("窗口需要至少一个样本")
    if name == "rect":
        return np.ones(count)
    if name != "hann" or count < 3:
        raise ValueError("对称 Hann 窗至少需要三个有效样本或完整周期")
    return np.hanning(count)


def coding_cycle(config):
    a = config["array"]
    if a["mode"] == "ddm":
        return int(a["ddm_slots"])
    if a["mode"] in ("tdm", "cdm"):
        return int(a["tx_count"])
    return 1


def slow_window(config):
    block = int(config["waveform"]["pulses"]) // int(config["processing"]["looks"])
    cycle = coding_cycle(config)
    if block % cycle:
        raise ValueError("每个非相干块必须包含完整的 Tx / 码周期")
    return np.repeat(window_values(config["processing"]["slow_window"], block // cycle), cycle)


def receiver_taps(config):
    # FMCW UI bandwidth is the usable positive beat-frequency edge.  Pulse
    # bandwidth is the full two-sided complex-baseband receiver bandwidth.
    spec = receiver_filter_spec(config)
    if spec["filter_taps"] == 1:
        return np.ones(1)
    return scipy_signal.firwin(spec["filter_taps"], max(1e-9, spec["normalized_cutoff"]), window=("kaiser", 7.0))


def pulse_envelope(config, relative_times):
    """Finite-support pulse in the negative-time analytic I/Q convention."""
    w = config["waveform"]
    tau = float(w["pulse_width_s"])
    t = np.asarray(relative_times, dtype=float)
    eps = 1e-8 / float(w["fs_hz"])
    t = np.where(abs(t) < eps, 0., t)
    t = np.where(abs(t-tau) < eps, tau, t)
    values = np.ones(t.shape, complex)
    if w["kind"] == "lfm":
        values = np.exp(-1j*np.pi*float(w["bandwidth_hz"])/tau*(t-tau/2)**2)
    return np.where((t >= 0) & (t < tau), values, 0j)


def _fast_key(config):
    w = config["waveform"]
    return tuple(w[k] for k in ("kind", "bandwidth_hz", "ramp_s", "adc_start_s",
                 "fs_hz", "samples", "pulse_width_s", "range_fft", "range_max_m",
                 "recovery_s", "rx_bandwidth_hz", "pri_s")) + (config["processing"]["fast_window"],)


def _build_fast_operator(config):
    """Return exact finite-record linear receiver coefficients and its axes.

    H[r,n] maps independent pre-FIR samples to range outputs. ``adc_matrix``
    is deliberately not materialized: its 33-tap convolution is evaluated by
    ``adc_record``. Input times include both sides of the delay-compensated
    causal FIR. This is acquired history, not fabricated zero-noise padding.
    """
    key = _fast_key(config)
    w = config["waveform"]
    n, nfft, fs = int(w["samples"]), int(w["range_fft"]), float(w["fs_hz"])
    taps = receiver_taps(config)
    half = (len(taps)-1)//2
    start = float(w["adc_start_s"]) if w["kind"] == "fmcw" else 0.
    adc_times = start + np.arange(n)/fs
    input_times = start + (np.arange(n+2*half)-half)/fs
    if len(input_times)/fs > float(w["pri_s"])*(1+1e-12):
        raise ValueError("含 FIR 噪声历史的相邻采集记录重叠；请缩短采样窗、增加 PRI 或扩大接收带宽")
    input_gate = np.ones(len(input_times), dtype=bool)
    output_gate = np.ones(n, dtype=bool)
    if w["kind"] != "fmcw":
        opening = float(w["pulse_width_s"])+float(w["recovery_s"])
        # Repeating transmit blanking also applies to the FIR history.
        input_gate = np.remainder(input_times, float(w["pri_s"])) >= opening-1e-15
        output_gate = adc_times >= opening-1e-15
    if w["kind"] == "fmcw":
        slope = float(w["bandwidth_hz"])/float(w["ramp_s"])
        freqs = np.fft.fftfreq(nfft, 1/fs)
        use = (freqs >= 0) & (C*freqs/(2*slope) <= float(w["range_max_m"])*(1+1e-12))
        bins = np.flatnonzero(use)
        ranges = C*freqs[use]/(2*slope)
        taper = window_values(config["processing"]["fast_window"], n)
        F = np.exp(-2j*np.pi*bins[:, None]*np.arange(n)[None, :]/nfft)*taper[None, :]/taper.sum()
        reference = np.ones(n, complex)
        reference_times = adc_times
    else:
        ranges = np.linspace(0., min(float(w["range_max_m"]), C*(n-1)/(2*fs)), nfft)
        n_pulse = int(math.ceil(float(w["pulse_width_s"])*fs-1e-10))
        reference = np.convolve(pulse_envelope(config, np.arange(n_pulse)/fs), taps)
        reference_times = (np.arange(len(reference))-half)/fs
        energy = float(np.vdot(reference, reference).real)
        if energy <= 0:
            raise ValueError("接收参考波形能量为零")
        # Evaluate each delayed continuous pulse at the acquisition samples.
        # Interpolating an integer-delay complex reference does not reproduce
        # a fractional-delay LFM pulse, even when that delay is on our grid.
        # The window belongs to the transmitted pulse support, before the
        # receiver FIR: its shape must not change with the FIR tap count.
        F = np.empty((len(ranges), n), complex)
        for ri, distance in enumerate(ranges):
            relative = input_times-2*distance/C
            prototype = pulse_envelope(config, relative)
            if config["processing"]["fast_window"] == "hann":
                position = relative*fs
                near = np.rint(position)
                position = np.where(abs(position-near)<1e-8, near, position)
                taper = np.where((position >= 0) & (position <= n_pulse-1),
                                 .5-.5*np.cos(2*np.pi*position/(n_pulse-1)), 0.)
                prototype *= taper
            matched = sum(b*prototype[j:j+n] for j,b in enumerate(taps))
            F[ri] = np.conj(matched)/energy
    F *= output_gate[None, :]
    H = np.zeros((len(ranges), len(input_times)), complex)
    for j, b in enumerate(taps):
        H[:, j:j+n] += F*b
    H *= input_gate[None, :]
    result = {"matrix": H, "ranges_m": ranges, "noise_gain": np.sum(abs(H)**2, axis=1),
              "taps": taps, "input_times_s": input_times, "adc_times_s": adc_times,
              "input_gate": input_gate, "output_gate": output_gate,
              "waveform": reference, "waveform_times_s": reference_times,
              "filter_history_samples": half}
    size = sum(x.nbytes for x in result.values() if isinstance(x, np.ndarray))
    if size <= _CACHE_BYTES//2:
        while _FAST_CACHE and sum(v[1] for v in _FAST_CACHE.values())+size > _CACHE_BYTES:
            _FAST_CACHE.popitem(last=False)
        _FAST_CACHE[key] = (result, size)
    return result


def fast_operator(config):
    """Cached exact receiver; the bounded cache never stores scene data."""
    key = _fast_key(config)
    if key in _FAST_CACHE:
        _FAST_CACHE.move_to_end(key)
        return _FAST_CACHE[key][0]
    return _build_fast_operator(config)


def adc_record(receiver_input, op):
    x = np.asarray(receiver_input, complex)
    n = len(op["adc_times_s"])
    out = np.zeros(x.shape[:-1]+(n,), complex)
    for j, b in enumerate(op["taps"]):
        out += b*x[..., j:j+n]
    out *= op["output_gate"]
    return out


def swerling_samples(model, phase_model, events, rng):
    """Unit-mean RCS amplitudes; I/III are fixed over a complete acquisition."""
    model = int(model)
    count = int(events) if model in (2, 4) else 1
    if model == 0:
        power = np.ones(count)
    elif model in (1, 2):
        power = rng.gamma(1., 1., count)
    elif model in (3, 4):
        power = rng.gamma(2., .5, count)
    else:
        raise ValueError("Swerling 模型只支持 0–IV")
    phase = rng.uniform(-np.pi, np.pi, count) if phase_model == "random" else np.zeros(count)
    value = np.sqrt(power)*np.exp(1j*phase)
    return np.broadcast_to(value, (int(events),)).copy()


def spatial_transmit_weights(config, plan=None):
    plan = event_plan(config) if plan is None else plan
    a = config["array"]
    k = 2*np.pi*float(config["waveform"]["fc_hz"])/C
    steer = np.exp(1j*k*(np.asarray(a["tx_positions_m"])@direction(a["pointing_deg"])))
    return np.asarray(plan["weights"])*steer[None, :]


def _path_template(config, plan, op, path, rcs=1., cancel=None):
    """One deterministic path at all physical Rx, before any Tx separation."""
    w, a, radar = config["waveform"], config["array"], config["radar"]
    fc, fs = float(w["fc_hz"]), float(w["fs_hz"])
    times = np.asarray(plan["times_s"])
    sample_times = op["input_times_s"]
    departure, arrival = np.asarray(path["departure"]), np.asarray(path["arrival"])
    tx = np.asarray(a["tx_positions_m"], float)
    rx = np.asarray(a["rx_positions_m"], float)
    weights = spatial_transmit_weights(config, plan)
    # Far-field element-dependent path offsets retain the physical virtual
    # phase and sub-sample delay, without applying carrier phase twice.
    offsets = tx@departure
    path_lengths = float(path["length_m"])-offsets[:, None]-(rx@arrival)[None, :]
    if np.any(path_lengths <= 0):
        raise ValueError("路径长度须大于阵列远场相位模型的几何偏移")
    rt, rr = float(path["tx_distance_m"]), float(path["rx_distance_m"])
    power = float(plan["per_tx_power_w"])*10**((radar["tx_gain_dbi"]+radar["rx_gain_dbi"]-radar["loss_db"])/10)
    power *= (C/fc)**2*float(rcs)/((4*np.pi)**3*rt**2*rr**2)
    out = np.zeros((len(times), int(a["rx_count"]), len(sample_times)), complex)
    fd = float(path["doppler_hz"])
    for it in range(int(a["tx_count"])):
        _check(cancel)
        for ir in range(int(a["rx_count"])):
            delay = path_lengths[it, ir]/C
            u = sample_times-delay
            if w["kind"] == "fmcw":
                slope = float(w["bandwidth_hz"])/float(w["ramp_s"])
                active = (u >= -1e-15) & (u < float(w["ramp_s"])-1e-15)
                # rx_minus * conj(tx_minus), for an up-chirp centered at fc.
                envelope = np.exp(1j*(2*np.pi*slope*delay*sample_times-np.pi*slope*delay**2
                                     -np.pi*float(w["bandwidth_hz"])*delay)) * active
                rf = fc+slope*(u-float(w["ramp_s"])/2)
            else:
                envelope = pulse_envelope(config, u)
                active = abs(envelope) > 0
                rf = fc + (float(w["bandwidth_hz"])/float(w["pulse_width_s"])*(u-float(w["pulse_width_s"])/2) if w["kind"] == "lfm" else np.zeros_like(u))
            # The wavelength in the radar amplitude uses the specified carrier
            # (narrowband antenna/RCS model); atmospheric/Fresnel coefficients
            # may vary over the actual chirp's instantaneous RF frequency.
            # The pulse is identically zero outside its support. Evaluating
            # an extrapolated RF there can spuriously fail a valid ITU band.
            # Frequencies in the actual pulse are never clipped or replaced.
            fast = np.zeros(len(sample_times), complex)
            if np.any(active):
                coeff = coefficient_for_path(rf[active], path, config["propagation"])
                # Fresnel is supplied for exp(+j omega t); our I/Q is its
                # conjugate. User custom phase is already receiver-I/Q phase.
                if path.get("ground_bounces", 0):
                    coeff = np.conj(coeff)
                atten = attenuation_for_path(rf[active], path, config["propagation"]["atmosphere"])
                fast[active] = math.sqrt(power)*envelope[active]*coeff*atten
            fast *= np.exp(2j*np.pi*fc*delay+2j*np.pi*fd*sample_times)
            slow = weights[:, it]*np.exp(2j*np.pi*fd*times)
            out[:, ir, :] += slow[:, None]*fast[None, :]
    out *= op["input_gate"][None, None, :]
    return out


def _path_observability(config, op, path):
    observation = {"path":path["name"],
                   **observable_path(config, path["delay_s"], path["doppler_hz"])}
    # Share the same geometric contract with the independent budget. This
    # checks only supplied channel coordinates, never an unknown element size.
    observation["far_field"] = far_field_spec(config, path["tx_distance_m"], path["rx_distance_m"])
    covered = observation["far_field"]["valid"]
    observation["physical_observable"] = bool(observation["physical_observable"] and covered)
    observation["observable"] = bool(observation["observable"] and covered)
    return observation


def synthesize_raw(config, seed=None, signal=True, noise=True, clutter=True, cancel=None, progress=None):
    """Generate the physical mixture and independent pre-receiver noise."""
    p = checked_config(config)
    plan, op = event_plan(p), fast_operator(p)
    seed = int(p["experiment"]["seed"] if seed is None else seed)
    e, nr, nf = int(p["waveform"]["pulses"]), int(p["array"]["rx_count"]), len(op["input_times_s"])
    _check(cancel)
    raw = np.zeros((e, nr, nf), complex)
    observations, warnings = [], []
    if signal:
        for i, target in enumerate(p["scene"]["targets"]):
            _check(cancel)
            rng = np.random.default_rng(np.random.SeedSequence([seed, 103, i]))
            scatter = swerling_samples(target["swerling"], target["phase_model"], e, rng)
            for path in paths_for_target(p, target):
                raw += _path_template(p, plan, op, path, target["rcs_m2"], cancel)*scatter[:, None, None]
                observation = _path_observability(p, op, path)
                observation["target"] = target["name"]
                observations.append(observation)
            if target["swerling"] in (2, 4) and p["array"]["mode"] in ("tdm", "ddm", "cdm"):
                warnings.append("逐事件起伏改变 Tx/码周期内的散射系数；真实串扰保留，通道分离可能失效。")
            _progress(progress, .08+.25*(i+1)/max(1,len(p["scene"]["targets"])), "生成物理接收通道目标回波")
    clutter_meta = {"enabled":False}
    cl = p["clutter"]
    if clutter and cl["kind"] != "none":
        cl_seed = int(cl["seed"]) if cl["fixed"] else seed
        rng = np.random.default_rng(np.random.SeedSequence([cl_seed, 211, int(cl["seed"])]))
        nsc = int(cl["scatterers"])
        clutter_raw = np.zeros_like(raw)
        expected = 0.
        # A shared unit-mean gamma texture gives a compound-Gaussian K
        # background. It is fixed over this CPI; it is NOT power-normalized.
        texture = rng.gamma(float(cl["shape"]), 1/float(cl["shape"])) if cl["kind"] == "k" else 1.
        valid = op["input_gate"] & (op["input_times_s"] >= op["adc_times_s"][0]) & (op["input_times_s"] <= op["adc_times_s"][-1])
        for j in range(nsc):
            _check(cancel)
            r = rng.uniform(float(cl["range_min_m"]), float(cl["range_max_m"]))
            target = {"name":f"杂波单元 {j+1}","range_m":r,"rcs_m2":1.,"height_m":p["scene"]["radar_height_m"],
                      "az_deg":rng.uniform(-float(cl["az_span_deg"]),float(cl["az_span_deg"])),"velocity_mps":0.}
            doppler = rng.normal(float(cl["doppler_hz"]),float(cl["doppler_width_hz"]))
            template = np.zeros_like(raw)
            for source_path in paths_for_target(p, target):
                path = dict(source_path)
                path["doppler_hz"] = float(path["doppler_hz"])+doppler
                template += _path_template(p, plan, op, path, cancel=cancel)
            expected += float(np.mean(abs(template[...,valid])**2)) if np.any(valid) else 0.
            coefficient = (rng.standard_normal()+1j*rng.standard_normal())/math.sqrt(2)
            edge = 10**(float(cl["edge_db"])/20) if cl["kind"] == "edge" and r >= float(cl["edge_range_m"]) else 1.
            clutter_raw += math.sqrt(texture)*coefficient*edge*template
            if j % max(1,nsc//10) == 0:
                _progress(progress,.35+.2*(j+1)/nsc,"生成经过波形、Tx 编码与传播的统计杂波")
        requested = noise_density(p)*float(p["waveform"]["fs_hz"])*10**(float(cl["cnr_db"])/10)
        scale = math.sqrt(requested/expected) if expected > 0 else 0.
        raw += scale*clutter_raw
        if expected == 0:
            warnings.append("所有杂波路径位于采集窗外，未产生可观测杂波。")
        clutter_meta = {"enabled":True,"kind":cl["kind"],"scatterers":nsc,"fixed":bool(cl["fixed"]),
                        "requested_baseline_mean_input_power_w":requested,"template_expected_input_power_w":expected,
                        "texture_realization":float(texture),"normalization":"deterministic response ensemble energy; no realization power normalization",
                        "texture_model":"one Gamma texture shared by the acquisition and scatterers" if cl["kind"] == "k" else "none",
                        "cnr_reference":"mean valid pre-FIR physical-Rx samples; edge gain is additional"}
    if noise and p["radar"]["noise_enabled"]:
        rng = np.random.default_rng(np.random.SeedSequence([seed, 307]))
        std = math.sqrt(noise_density(p)*float(p["waveform"]["fs_hz"])/2)
        raw += (rng.standard_normal(raw.shape)+1j*rng.standard_normal(raw.shape))*std*op["input_gate"]
    _check(cancel)
    return raw, {"observability":observations,"warnings":warnings,"clutter":clutter_meta,"seed":seed}, op


def velocity_grid(config):
    w, a = config["waveform"], config["array"]
    stride = int(a["tx_count"]) if a["mode"] in ("tdm", "cdm") else 1
    frequencies = np.fft.fftshift(np.fft.fftfreq(int(w["doppler_fft"]), float(w["pri_s"])*stride))
    return frequencies*C/(2*float(w["fc_hz"])), frequencies


def slow_beam_operator(config, block_index=0):
    """Unit-noise-gain weights [velocity,event,physical Rx], no scene input."""
    w, a = config["waveform"], config["array"]
    plan = event_plan(config)
    n = int(plan["block_size"])
    take = slice(block_index*n,(block_index+1)*n)
    times = np.asarray(plan["times_s"])[take]
    _, frequencies = velocity_grid(config)
    taper = slow_window(config)
    # At the commanded look direction, geometric transmit phases cancel the
    # known steering phases. Match actual complex user weights, not an equal
    # average of decoded virtual channels.
    reference = np.asarray(plan["weights"])[take].sum(axis=1)
    rx_user = complex_weights(a["rx_weights"],1.)
    rx_phase = np.exp(-2j*np.pi*float(w["fc_hz"])/C*(np.asarray(a["rx_positions_m"])@direction(a["pointing_deg"])))
    rx_match = np.conj(rx_user*rx_phase)
    time_match = np.exp(-2j*np.pi*frequencies[:,None]*times[None,:])*np.conj(reference)[None,:]*taper[None,:]
    functional = time_match[:,:,None]*rx_match[None,None,:]
    norms = np.sqrt(np.sum(abs(functional)**2,axis=(1,2)))
    functional /= np.where(norms > 1e-20,norms,np.inf)[:,None,None]
    return functional


def _decoded_spectra(fast, config, block_index):
    """Code-matched physical-Rx spectra, retaining actual leakage."""
    w,a = config["waveform"],config["array"]
    plan=event_plan(config); n=int(plan["block_size"]);start=block_index*n
    times=np.asarray(plan["times_s"])[start:start+n]
    _,frequencies=velocity_grid(config)
    taper=slow_window(config)
    if a["mode"] == "ddm":
        code=np.exp(2j*np.pi*np.arange(start,start+n)[:,None]*np.arange(int(a["ddm_slots"]))[None,:]/int(a["ddm_slots"]))
    elif a["mode"] == "tdm":
        code=np.eye(int(a["tx_count"]))[np.arange(start,start+n)%int(a["tx_count"])]
    elif a["mode"] == "cdm":
        from scipy.linalg import hadamard
        code=hadamard(int(a["tx_count"]))[np.arange(start,start+n)%int(a["tx_count"])]
    else:
        code=np.ones((n,1),complex)
    out=[]
    for it in range(code.shape[1]):
        bank=np.exp(-2j*np.pi*frequencies[:,None]*times[None,:])*np.conj(code[:,it])[None,:]*taper[None,:]
        norm=np.sqrt(np.sum(abs(bank)**2,axis=1))
        bank/=np.where(norm>1e-20,norm,np.inf)[:,None]
        out.append(np.einsum("ve,erc->vrc",bank,fast,optimize=True))
    return np.stack(out,axis=1)


def ddm_reconstruction_residual(fast, config, cancel=None):
    """Weighted least-squares residual of actual physical-Rx fast-time data.

    For each look and candidate Doppler, the only model columns are the known
    transmit event weights times that candidate's Doppler phasor. A complex
    coefficient per Tx/Rx/range is fitted from the observations. Column-pivoted
    QR retains only numerical-rank columns, so an inactive transmitter cannot
    introduce an arbitrary orthonormal direction into the reconstructed model.
    The objective weights are the actual slow-time window (sqrt(w) premultiplies
    A and y). Residual and data energies are summed across physical Rx and
    disjoint looks *before* division; target truth never enters the function.
    """
    x=np.asarray(fast,complex)
    if x.ndim!=3 or x.shape[0]!=int(config["waveform"]["pulses"]) or x.shape[1]!=int(config["array"]["rx_count"]):
        raise ValueError("DDM 重构需要 [物理事件, 物理 Rx, 距离] 复接收数组")
    if not np.isfinite(x).all():
        raise ValueError("DDM 重构接收数组必须为有限数值")
    plan=event_plan(config);n=int(plan["block_size"])
    _,frequencies=velocity_grid(config)
    taper=slow_window(config)
    root_weight=np.sqrt(taper)
    residual=np.zeros((len(frequencies),x.shape[-1]),float)
    data_energy=np.zeros(x.shape[-1],float)
    for look in range(int(config["processing"]["looks"])):
        _check(cancel)
        take=slice(look*n,(look+1)*n)
        times=np.asarray(plan["times_s"])[take]
        known_weights=np.asarray(plan["weights"])[take]
        weighted=x[take]*root_weight[:,None,None]
        data_energy+=np.sum(abs(weighted)**2,axis=(0,1))
        y=weighted.reshape(n,-1)
        for vi,frequency in enumerate(frequencies):
            _check(cancel)
            basis=known_weights*np.exp(2j*np.pi*frequency*times)[:,None]*root_weight[:,None]
            q,r,_=qr(basis,mode="economic",pivoting=True,check_finite=False)
            diagonal=abs(np.diag(r))
            maximum=float(diagonal.max()) if len(diagonal) else 0.
            tolerance=np.finfo(float).eps*max(basis.shape)*maximum
            rank=int(np.count_nonzero(diagonal>tolerance)) if maximum>0 else 0
            if rank:
                q=q[:,:rank]
                difference=y-q@(q.conj().T@y)
            else:
                difference=y
            residual[vi]+=np.sum(abs(difference.reshape(weighted.shape))**2,axis=(0,1))
    ratio=np.divide(residual,data_energy[None,:],out=np.ones_like(residual),where=data_energy[None,:]>np.finfo(float).tiny)
    # Values outside [0,1] can arise only from final round-off of an orthogonal
    # projection. A zero-energy input is not evidence of a successful fit.
    return np.clip(ratio,0.,1.)


def ddm_recovery(slot_power, config, ranges, velocities, residual_map=None):
    """Resolve cyclic slot hypotheses from observations; retain failures.

    slot_power has [velocity,slot,range] and is summed over independent looks
    and physical Rx channels. No target list or simulator labels are read.
    ``residual_map`` is an independent weighted-QR reconstruction test; the
    production processing path always supplies it. Omitting it is useful only
    for isolated empty-slot hypothesis unit checks.
    """
    a=config["array"];k=int(a["ddm_slots"]);nt=int(a["tx_count"])
    nv=slot_power.shape[0]
    if nv%k:
        raise ValueError("DDM 多普勒网格点数须为槽数的整数倍")
    active=np.mean(slot_power[:,:nt,:],axis=1)
    empty=np.mean(slot_power[:,nt:,:],axis=1)
    floor=np.finfo(float).tiny
    gap=10*np.log10(np.maximum(active,floor)/np.maximum(empty,floor))
    valid=np.zeros_like(active,dtype=bool)
    ambiguous=np.zeros_like(valid)
    spacing=nv//k
    requirement=float(config["detection"]["ddm_gap_db"])
    for base in range(spacing):
        ids=base+spacing*np.arange(k)
        scores=gap[ids]
        order=np.argsort(scores,axis=0)
        best=order[-1];runner=order[-2]
        cols=np.arange(scores.shape[1])
        maximum=scores[best,cols]
        separation=maximum-scores[runner,cols]
        good=(maximum>=requirement)&(separation>=requirement)
        valid[ids[best],cols]=good
        ambiguous[ids,:]=(scores>=maximum[None,:]-requirement)&(maximum[None,:]>=requirement)&(~good[None,:])
    # Candidate extraction uses observed local peaks, never target locations.
    valid &= active > max(float(active.max())*1e-12,np.finfo(float).tiny)
    residual_limit=float(config["detection"].get("ddm_residual_limit",.65))
    if residual_map is not None:
        residual_map=np.asarray(residual_map,float)
        if residual_map.shape!=active.shape or not np.isfinite(residual_map).all() or np.any(residual_map<0) or np.any(residual_map>1):
            raise ValueError("DDM 重构残差须为与速度-距离网格一致的 [0,1] 有限数组")
        reconstruction_ok=residual_map<=residual_limit
        valid &= reconstruction_ok
    else:
        reconstruction_ok=np.ones_like(valid)
    local=active>=ndimage.maximum_filter(active,size=(3,3),mode="nearest")
    # Suppress floating-point nulls in this diagnostic list only. This is
    # neither a detection threshold nor a modification of the power grid.
    picks=np.argwhere(local & (active>max(float(active.max())*1e-12,np.finfo(float).tiny)))
    if len(picks):
        order=np.argsort(active[picks[:,0],picks[:,1]])[::-1][:64]
        picks=picks[order]
    candidates=[]
    for vi,ri in picks:
        status="resolved" if valid[vi,ri] else "ambiguous" if ambiguous[vi,ri] and reconstruction_ok[vi,ri] else "collision_or_low_gap"
        f=float(velocities[vi])*2*float(config["waveform"]["fc_hz"])/C
        prf=1/float(config["waveform"]["pri_s"])
        mapping=[{"tx":j,"encoded_frequency_hz":float((f+j*prf/k+prf/2)%prf-prf/2),"slot_offset":j} for j in range(nt)]
        corrected_range=float(ranges[ri])-float(config["waveform"]["fc_hz"])*float(velocities[vi])/(float(config["waveform"]["bandwidth_hz"])/float(config["waveform"]["ramp_s"]))
        candidates.append({"range_m":corrected_range,"apparent_range_m":float(ranges[ri]),"velocity_mps":float(velocities[vi]),"status":status,
                           "score":float(gap[vi,ri]),"active_power_w":float(active[vi,ri]),
                           "empty_power_w":float(empty[vi,ri]),"inferred_tx_mapping":mapping,
                           "reconstruction_residual":float(residual_map[vi,ri]) if residual_map is not None else None,
                           "reconstruction_residual_limit":residual_limit,
                           "reconstruction_passed":bool(reconstruction_ok[vi,ri]) if residual_map is not None else None})
    return valid,gap,candidates


def process_raw(receiver_input, config, cancel=None, progress=None, op=None):
    """Receive-only processing. Changing scene truth cannot change this result."""
    # Deliberately read only waveform/array/processing/detection/noise fields.
    op=fast_operator(config) if op is None else op
    plan=event_plan(config)
    x=np.asarray(receiver_input,complex)
    expected=(int(config["waveform"]["pulses"]),int(config["array"]["rx_count"]),len(op["input_times_s"]))
    if x.shape != expected:
        raise ValueError(f"接收数组形状应为 {expected}，实际为 {x.shape}")
    _check(cancel)
    fast=x@op["matrix"].T
    looks=int(config["processing"]["looks"]);n=int(plan["block_size"])
    velocities,_=velocity_grid(config)
    beams=[];decoded=[];ddm_slot=None
    for look in range(looks):
        _check(cancel)
        block=fast[look*n:(look+1)*n]
        functional=slow_beam_operator(config,look)
        beams.append(np.einsum("ver,erc->vc",functional,block,optimize=True))
        separated=_decoded_spectra(block,config,look)
        decoded.append(separated[:,:int(config["array"]["tx_count"]),:,:])
        if config["array"]["mode"]=="ddm":
            power=np.sum(abs(separated)**2,axis=2)
            ddm_slot=power if ddm_slot is None else ddm_slot+power
        _progress(progress,.65+.3*(look+1)/looks,"按实际码周期与空间权重进行相干/非相干处理")
    beam=np.stack(beams)
    power=np.sum(abs(beam)**2,axis=0)
    fast_noise=noise_density(config)*float(config["waveform"]["fs_hz"])*op["noise_gain"]
    noise_power=np.broadcast_to(looks*fast_noise[None,:],power.shape).copy()
    # A null transmit look yields a zero linear receiver, not unit noise gain.
    functional_norm=np.sum(abs(slow_beam_operator(config,0))**2,axis=(1,2))
    noise_power*=functional_norm[:,None]
    arrays={"raw":adc_record(x,op),"receiver_input":x,"fast_complex":fast,
            "beam_complex":beam,"decoded_complex":np.stack(decoded),"rd_power":power,"noise_power":noise_power,
            "ranges_m":op["ranges_m"].copy(),"velocities_mps":velocities,
            "waveform_real":op["waveform"].real.copy(),"waveform_imag":op["waveform"].imag.copy(),
            "waveform_time_s":op["waveform_times_s"].copy(),"adc_time_s":op["adc_times_s"].copy(),
            "receiver_input_time_s":op["input_times_s"].copy(),"fast_noise_gain":op["noise_gain"].copy()}
    candidates=[]
    if ddm_slot is not None:
        reconstruction=ddm_reconstruction_residual(fast,config,cancel)
        valid,gap,candidates=ddm_recovery(ddm_slot,config,op["ranges_m"],velocities,reconstruction)
        arrays.update(ddm_valid_mask=valid,ddm_gap_db=gap,ddm_slot_power=ddm_slot,
                      ddm_reconstruction_residual=reconstruction)
    _check(cancel)
    return arrays,candidates


def simulate(config, seed=None, signal=True, noise=True, clutter=True, cancel=None, progress=None):
    start=time.perf_counter()
    p=checked_config(config)
    # Must validate before allocating raw data, templates, or receiver matrices.
    resources=resource_estimate(p)
    if resources["estimated_mib"]>float(p["experiment"]["memory_mib"]):
        raise ValueError("预计接收处理内存超过工程预算")
    slow_window(p)
    if p["array"]["mode"]=="ddm" and int(p["waveform"]["doppler_fft"])%int(p["array"]["ddm_slots"]):
        raise ValueError("DDM 多普勒网格点数须为槽数的整数倍")
    raw,meta,op=synthesize_raw(p,seed,signal,noise,clutter,cancel,progress)
    arrays,candidates=process_raw(raw,p,cancel,progress,op)
    warnings=list(meta["warnings"])
    if any(not x["observable"] for x in meta["observability"]):
        warnings.append("至少一条目标路径超出完整回波/IF采集/处理网格或已知阵列坐标的远场条件；采样窗内截断和频率混叠保留，未重建更早 PRI 的历史目标回波，不能视为完整参考通过。")
    if any(not x["far_field"]["valid"] for x in meta["observability"]):
        warnings.append("已知 Tx/Rx 通道坐标不满足对应路径的 2D²/λ 远场距离；当前仍为线性化远场演示，结果不可作为已验证物理覆盖。该检查不验证未提供的真实天线单元口径。")
    if p["waveform"]["kind"]!="fmcw":
        count=float(p["waveform"]["pulse_width_s"])*float(p["waveform"]["fs_hz"])
        if abs(count-round(count))>1e-8:
            warnings.append("脉宽不是整数个采样间隔；实际样本能量和分数时延效应保留，不能将连续时间 τ 公式误差全部归为软件计算误差。")
    if p["array"]["mode"]=="ddm":
        warnings.append("DDM 使用接收谱空槽检验及已知编码/候选多普勒的加权最小二乘 QR 重构残差；码碰撞、杂波、高残差或并列假设可导致失败/歧义。线性功率图保留全部假设，检测后再应用解码有效掩码。")
    if p["array"]["mode"]=="cdm":
        warnings.append("CDM 显示完整码块的保守无模糊速度范围；超界速度和逐事件起伏造成的实际串扰不被人为消除。")
    max_movement=max((abs(float(t["velocity_mps"]))*float(p["waveform"]["pri_s"])*int(p["waveform"]["pulses"]) for t in p["scene"]["targets"]),default=0.)
    resolution=C/(2*(float(p["waveform"]["bandwidth_hz"])*float(p["waveform"]["samples"])/float(p["waveform"]["fs_hz"])/float(p["waveform"]["ramp_s"]))) if p["waveform"]["kind"]=="fmcw" else C*(float(p["waveform"]["pulse_width_s"]) if p["waveform"]["kind"]=="rect" else 1/float(p["waveform"]["bandwidth_hz"]))/2
    if max_movement>.25*resolution:
        warnings.append("CPI 内运动超过四分之一距离分辨单元；本首版固定包络时延并保留路径多普勒，未模拟距离徙动。")
    a=p["array"]
    summary={"warnings":list(dict.fromkeys(warnings)),"timing":{"elapsed_s":time.perf_counter()-start,
             "physical_events":int(p["waveform"]["pulses"]),"block_events":int(event_plan(p)["block_size"]),
             "looks":int(p["processing"]["looks"]),"pri_s":float(p["waveform"]["pri_s"])},
             "channels":{"mode":a["mode"],"physical_tx":int(a["tx_count"]),"physical_rx":int(a["rx_count"]),
             "virtual_pairs":int(a["tx_count"])*int(a["rx_count"]),"raw_axes":["event","physical_rx","adc_sample"],
             "decoded_axes":["look","velocity","decoded_tx","physical_rx","range"]},
             "ddm_candidates":candidates,"observability":meta["observability"],"clutter":meta["clutter"],
             "seed":meta["seed"],"resources":resources,
             "noise":{"density_w_hz":noise_density(p),"independent_input_sample_power_w":noise_density(p)*p["waveform"]["fs_hz"],
                      "filter_taps":len(op["taps"]),"filter_history_samples":op["filter_history_samples"],
                      "reported_power":"exact thermal expectation of ungated linear beam, summed over independent looks",
                      "enabled_in_realization":bool(noise and p["radar"]["noise_enabled"])},
             "model":{"velocity_sign":"positive receding -> positive Doppler; FMCW positive delay -> positive beat",
                      "complex_convention":"negative-time analytic I/Q exp(-j omega t); FMCW rx * conj(tx); fc is chirp center; custom phase is receiver-I/Q phase",
                      "propagation":"coherent path sum before physical-Rx processing; carrier phase applied once",
                      "motion":"fixed envelope delay over CPI plus path Doppler; no range migration",
                      "amplitude":"carrier-wavelength narrowband antenna/RCS amplitude; frequency-dependent atmosphere/reflection",
                      "pulse_match":"continuous fractional-delay pulse sampled directly; Hann on pulse support before receiver FIR; exact linear noise coefficients",
                      "array_geometry":"linearized far-field element delays; known channel-coordinate extents checked against 2D^2/lambda on each path leg; antenna-element aperture unspecified",
                      "ddm_reconstruction":"pivoted QR of sqrt(slow_window) times known Tx event weights and candidate Doppler; residual/data energy summed over physical Rx and looks",
                      "noncoherent":"sum of squared magnitudes of disjoint complete-code coherent blocks"}}
    _progress(progress,1.,"真实采样链计算完成")
    return summary,arrays
