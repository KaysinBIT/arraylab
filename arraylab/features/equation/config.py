"""Versioned configuration, units, timing and bounded resource contracts."""
from __future__ import annotations
import copy
import hashlib
import json
import math
import numpy as np
from scipy.linalg import hadamard
from .receiver_spec import receiver_layout

C = 299792458.0
K = 1.380649e-23
T0 = 290.0


def default_config(kind="fmcw", frequency_ghz=77):
    pulse = kind != "fmcw"
    return {
        "schema_version": 1, "name": f"{frequency_ghz:g} GHz {kind.upper()} 基准",
        "waveform": {"kind": kind, "fc_hz": frequency_ghz*1e9,
            "bandwidth_hz": 5e6 if pulse else 160e6, "ramp_s": 64e-6,
            "adc_start_s": 4e-6, "fs_hz": 20e6 if pulse else 5e6,
            "samples": 512 if pulse else 256, "pulse_width_s": 1e-6,
            "pri_s": 40e-6 if pulse else 80e-6, "pulses": 32,
            "range_fft": 1024 if pulse else 256, "doppler_fft": 32,
            "range_max_m": 600 if pulse else 140, "recovery_s": .1e-6,
            "rx_bandwidth_hz": 8e6 if pulse else 2.3e6},
        "radar": {"power_w": 1.0 if pulse else .01, "power_basis": "per_tx",
            "tx_gain_dbi": 12., "rx_gain_dbi": 12., "loss_db": 3.,
            "noise_mode": "nf", "antenna_temperature_k": 290.,
            "noise_figure_db": 10., "system_temperature_k": 2900., "noise_enabled": True},
        "array": {"mode": "simo", "tx_count": 1, "rx_count": 4,
            "tx_positions_m": [], "rx_positions_m": [], "pointing_deg": [0.,0.],
            "tx_weights": [], "rx_weights": [], "ddm_slots": 4},
        "scene": {"radar_height_m": 1., "targets": [{"name":"目标 1", "range_m":300. if pulse else 50.,
            "az_deg":0.,"height_m":1.,"velocity_mps":0.,"rcs_m2":1.,"swerling":0,"phase_model":"fixed"}]},
        "propagation": {"model":"free","polarization":"horizontal","epsilon_r":4.,"conductivity_s_m":.01,
            "custom_paths":[],"atmosphere":{"enabled":False,"pressure_hpa":1013.25,"temperature_k":288.15,
                "vapor_density_g_m3":7.5,"rain_rate_mm_h":0.,"polarization_tilt_deg":0.}},
        "clutter": {"kind":"none","cnr_db":0.,"scatterers":32,"range_min_m":200. if pulse else 10.,
            "range_max_m":550. if pulse else 130.,"az_span_deg":20.,"doppler_hz":0.,"doppler_width_hz":50.,
            "edge_range_m":350. if pulse else 80.,"edge_db":12.,"shape":.7,"fixed":False,"seed":314},
        "processing": {"fast_window":"rect","slow_window":"rect","looks":1},
        "detection": {"method":"ca","pfa":.001,"training":[4,2],"guard":[1,1],"rank_fraction":.75,
            "fixed_snr_db":13.,"threshold_mode":"analytic","calibration_factor":None,"calibration_hash":None,"ddm_gap_db":3.,"ddm_residual_limit":.65},
        "requirements": {"range_m":300. if pulse else 100.,"snr_db":13.,"margin_db":3.,"pd":.9,"pfa":.001},
        "experiment": {"seed":42,"trials":256,"confidence":.95,"cut_range_m":300. if pulse else 50.,
            "cut_velocity_mps":0.,"distance_min_m":200. if pulse else 10.,"distance_max_m":500. if pulse else 130.,
            "distance_count":9,"roc_points":9,"memory_mib":1024}
    }


def _merge(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict): _merge(base[key], value)
        else: base[key] = copy.deepcopy(value)
    return base


def normalize_config(config):
    if not isinstance(config,dict): raise ValueError("工程必须是 JSON 对象")
    for key in ("waveform","radar","array","scene","propagation","clutter","processing","detection","requirements","experiment"):
        if key in config and not isinstance(config[key],dict):raise ValueError(f"{key} 必须是 JSON 对象")
    kind=config.get("waveform",{}).get("kind","fmcw")
    p=_merge(default_config(kind),config)
    frequency=float(p["waveform"]["fc_hz"])
    if not math.isfinite(frequency) or frequency<=0:raise ValueError("载频必须是有限正数")
    a=p["array"]; lam=C/frequency
    for prefix in ("tx","rx"):
        count=int(a[prefix+"_count"])
        if count<1 or count>64: continue
        if not a.get(prefix+"_positions_m"):
            step=lam/2*(int(a["rx_count"]) if prefix=="tx" else 1)
            a[prefix+"_positions_m"]=[[0.,float((i-(count-1)/2)*step),0.] for i in range(count)]
        if not a.get(prefix+"_weights"): a[prefix+"_weights"]=[[1.,0.] for _ in range(count)]
    return p


def config_hash(config):
    def canonical(value):
        if isinstance(value,dict):return {key:canonical(v) for key,v in value.items()}
        if isinstance(value,list):return [canonical(v) for v in value]
        if isinstance(value,float) and math.isfinite(value) and value.is_integer():return int(value)
        return value
    return hashlib.sha256(json.dumps(canonical(config),sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def noise_density(p):
    r=p["radar"]
    ts=r["system_temperature_k"] if r["noise_mode"]=="tsys" else r["antenna_temperature_k"]+(10**(r["noise_figure_db"]/10)-1)*T0
    return K*ts


def direction(angles):
    az,el=np.deg2rad(angles)
    return np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])


def target_position(p,target):
    r=float(target["range_m"]); dz=float(target["height_m"])-p["scene"]["radar_height_m"]
    horizontal=math.sqrt(max(0.,r*r-dz*dz)); az=math.radians(target["az_deg"])
    return np.array([horizontal*math.cos(az),horizontal*math.sin(az),float(target["height_m"])])


def complex_weights(values, norm=1.):
    a=np.asarray(values,float)
    # Weight scale carries no physical meaning. Scale before squaring so that
    # finite, nonzero weights remain equivalent even near float64 extremes.
    scale=float(np.max(abs(a)))
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("复权重须为有限非零数组")
    a=a/scale; w=a[:,0]+1j*a[:,1]
    return w*math.sqrt(norm/float(np.vdot(w,w).real))


def event_plan(p):
    w=p["waveform"];a=p["array"];n=int(w["pulses"]);nt=int(a["tx_count"])
    weights=np.ones((n,nt),complex); mode=a["mode"]
    if mode=="tdm": weights=np.eye(nt,dtype=complex)[np.arange(n)%nt]
    elif mode=="ddm": weights=np.exp(2j*np.pi*np.arange(n)[:,None]*np.arange(nt)[None,:]/int(a["ddm_slots"]))
    elif mode=="cdm": weights=hadamard(nt)[np.arange(n)%nt,:].astype(complex)
    weights*=complex_weights(a["tx_weights"],nt)[None,:]
    active=1 if mode in ("simo","tdm") else nt
    if p["radar"]["power_basis"]=="total":
        row_energy=np.sum(abs(weights)**2,axis=1)
        if np.any(row_energy<=0):raise ValueError("总功率模式不能有全部静默的发射时隙")
        weights*=np.sqrt(active/row_energy)[:,None]
    per=float(p["radar"]["power_w"])/(active if p["radar"]["power_basis"]=="total" else 1)
    return {"times_s":np.arange(n)*float(w["pri_s"]),"weights":weights,"per_tx_power_w":per,
            "block_size":n//int(p["processing"]["looks"]),"active_tx":active}


def resource_estimate(p):
    w=p["waveform"];a=p["array"];e=int(w["pulses"]);ns=int(w["samples"]);rx=int(a["rx_count"])
    nr=int(w["range_fft"]);nv=int(w["doppler_fft"]);nt=int(a["tx_count"])
    looks=int(p["processing"]["looks"]);block=e//looks
    ni=receiver_layout(p)["input_samples"]
    slots=int(a["ddm_slots"]) if a["mode"]=="ddm" else nt if a["mode"] in ("tdm","cdm") else 1
    # Conservative sum of simultaneously retained arrays and expression
    # workspaces. In particular the slow receiver is O(Nv*events*Rx), not
    # O(Nv*range). No large arrays are allocated to perform this estimate.
    raw=e*rx*ns*16
    parts={
        "input_and_synthesis_bytes":8*e*rx*ni*16,
        "event_schedule_bytes":6*e*nt*16+e*32,
        "fast_operator_workspace_bytes":4*nr*(ns+ni)*16+16*(ns+ni),
        "fast_outputs_bytes":4*e*rx*nr*16+raw,
        "slow_operator_workspace_bytes":4*nv*block*16+3*nv*block*rx*16,
        "decoded_and_beam_bytes":3*looks*nv*slots*rx*nr*16+3*looks*nv*nr*16,
        "power_maps_bytes":4*nv*slots*nr*8+8*nv*nr*8,
        # Existing unrelated fast operators may remain in the bounded cache.
        "bounded_cache_bytes":48*1024**2,
        "overhead_bytes":16*1024**2,
    }
    if a["mode"]=="ddm":
        parts["ddm_qr_workspace_bytes"]=(4*block*nt*16+3*nt*nt*16+
            4*block*rx*nr*16+2*nt*rx*nr*16+6*nv*nr*8)
    estimate=sum(parts.values())
    return {"raw_bytes":raw,"estimated_bytes":estimate,"estimated_mib":estimate/1024**2,
            "limit_mib":p["experiment"]["memory_mib"],"components":parts,
            "method":"conservative retained arrays plus expression workspaces and bounded cache"}


def validate_config(p):
    errors=[]
    try:
        if p.get("schema_version")!=1: errors.append("不支持的工程版本；当前为 schema_version=1")
        w=p["waveform"];a=p["array"];r=p["radar"];s=p["scene"];d=p["detection"];e=p["experiment"];q=p["processing"];pr=p["propagation"];cl=p["clutter"]
        def finite(value,name,lo=None,hi=None,integer=False):
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value): errors.append(name+" 必须为有限数值");return
            if lo is not None and value<lo or hi is not None and value>hi: errors.append(f"{name} 超出范围 [{lo}, {hi}]")
            if integer and int(value)!=value: errors.append(name+" 必须是整数")
        def choice(value,options,name):
            if value not in options: errors.append(name+" 选项无效")
        choice(w["kind"],["fmcw","rect","lfm"],"波形");choice(a["mode"],["simo","phased","tdm","ddm","cdm"],"通道体制")
        for key in ("fc_hz","bandwidth_hz","ramp_s","fs_hz","pulse_width_s","pri_s","range_max_m","rx_bandwidth_hz"): finite(w[key],key,1e-12,1e12)
        finite(w["fc_hz"],"载频 Hz",1e9,1000e9)
        for key in ("samples","pulses","range_fft","doppler_fft"): finite(w[key],key,2,1048576,True)
        for key in ("adc_start_s","recovery_s"):finite(w[key],key,0,10)
        finite(a["tx_count"],"Tx 数",1,16,True);finite(a["rx_count"],"Rx 数",1,32,True);finite(a["ddm_slots"],"DDM 槽数",2,32,True)
        finite(q["looks"],"非相干块数",1,w["pulses"],True)
        for key in ("fast_window","slow_window"):choice(q[key],["rect","hann"],key)
        if q["fast_window"]=="hann" and w["samples"]<3:errors.append("Hann 窗至少需要 3 个样本")
        if w["range_fft"]<w["samples"]:errors.append("距离 FFT 点数不能小于采样点数")
        if w["pulses"]%q["looks"]:errors.append("总事件数必须能被非相干块数整除")
        block=w["pulses"]//q["looks"]
        cycle=a["tx_count"] if a["mode"] in ("tdm","cdm") else a["ddm_slots"] if a["mode"]=="ddm" else 1
        if block%cycle:errors.append("每个处理块须包含完整的 TDM / 编码周期")
        if w["doppler_fft"]<block//(a["tx_count"] if a["mode"]=="tdm" else 1):errors.append("多普勒 FFT 点数小于每块有效慢时间采样数")
        if q["slow_window"]=="hann" and block//cycle<3:errors.append("慢时间 Hann 窗至少需要 3 个完整时序周期")
        if a["mode"]=="ddm" and w["doppler_fft"]%a["ddm_slots"]:errors.append("DDM 多普勒 FFT 点数须能被槽数整除")
        if a["mode"]=="simo" and a["tx_count"]!=1:errors.append("SIMO 需要 1 路 Tx")
        if a["mode"]=="ddm" and (w["kind"]!="fmcw" or a["ddm_slots"]<=a["tx_count"]):errors.append("DDM 仅用于 FMCW，且至少保留一个空子带")
        if a["mode"]=="cdm" and (w["kind"]=="fmcw" or a["tx_count"] not in (2,4)):errors.append("Walsh-CDM 用于脉冲，支持 2/4 Tx")
        for prefix in ("tx","rx"):
            count=int(a[prefix+"_count"])
            pos=np.asarray(a[prefix+"_positions_m"],float);weights=np.asarray(a[prefix+"_weights"],float)
            if pos.shape!=(count,3) or not np.isfinite(pos).all():errors.append(prefix+" 坐标须为 [通道数,3] 有限米制数组")
            if weights.shape!=(count,2) or not np.isfinite(weights).all() or not np.any(weights):errors.append(prefix+" 复权重须为非零的 [通道数,2] 数组")
            if prefix=="tx" and a["mode"]=="tdm" and r["power_basis"]=="total" and weights.shape==(count,2) and np.any(~np.any(weights!=0,axis=1)):errors.append("TDM 总功率模式每个发射时隙均须有非零 Tx 权重")
        if np.asarray(a["pointing_deg"]).shape!=(2,) or not np.isfinite(a["pointing_deg"]).all():errors.append("扫描方向须为有限的 [方位,俯仰] 度数组")
        if w["kind"]=="fmcw":
            if w["adc_start_s"]+w["samples"]/w["fs_hz"]>w["ramp_s"]*(1+1e-12):errors.append("ADC 窗口超出调频 ramp")
            if w["ramp_s"]>w["pri_s"]:errors.append("ramp 不得超过重复周期")
        else:
            if w["pulse_width_s"]+w["recovery_s"]>=w["pri_s"]:errors.append("脉宽与恢复时间之和须小于 PRI")
            if w["samples"]/w["fs_hz"]>w["pri_s"]*(1+1e-12):errors.append("接收窗口不能跨越下一 PRI")
            if w["pulse_width_s"]*w["fs_hz"]<2:errors.append("一个脉冲至少需要 2 个采样点")
            if abs(w["pulse_width_s"]*w["fs_hz"]-round(w["pulse_width_s"]*w["fs_hz"]))>1e-7:errors.append("本版脉冲数值验证要求脉宽对应整数采样点，请调整脉宽或采样率")
            if q["fast_window"]=="hann" and round(w["pulse_width_s"]*w["fs_hz"])<3:errors.append("脉冲 Hann 匹配窗至少需要 3 个脉内样本")
            if w["kind"]=="lfm" and w["bandwidth_hz"]>=w["fs_hz"]:errors.append("LFM 带宽须小于复基带采样率")
        choice(r["power_basis"],["per_tx","total"],"功率口径");choice(r["noise_mode"],["nf","tsys"],"噪声模式")
        finite(r["power_w"],"功率 W",1e-15,1e9);finite(r["loss_db"],"损耗 dB",0,200)
        for key in ("tx_gain_dbi","rx_gain_dbi"):finite(r[key],key,-50,100)
        finite(r["noise_figure_db"],"NF dB",0,80);finite(r["antenna_temperature_k"],"天线噪声温度",.001,1e7);finite(r["system_temperature_k"],"系统噪声温度",.001,1e9)
        finite(s["radar_height_m"],"雷达高度",.001,1e6)
        if not isinstance(s["targets"],list) or len(s["targets"])>64:errors.append("目标表最多支持 64 个目标")
        for i,t in enumerate(s["targets"]):
            for key in ("range_m","rcs_m2","height_m"):finite(t[key],f"目标{i+1} {key}",1e-12,1e9)
            for key in ("az_deg","velocity_mps"):finite(t[key],f"目标{i+1} {key}",-1e6,1e6)
            if t["range_m"]<=abs(t["height_m"]-s["radar_height_m"]):errors.append(f"目标{i+1} 斜距须大于高差")
            choice(t["swerling"],list(range(5)),"Swerling 模型");choice(t["phase_model"],["fixed","random"],"散射相位")
        choice(pr["model"],["free","two_ray","custom"],"传播模型");choice(pr["polarization"],["horizontal","vertical"],"反射极化")
        finite(pr["epsilon_r"],"介电常数",1,1e5);finite(pr["conductivity_s_m"],"电导率",0,1e8)
        if len(pr["custom_paths"])>32:errors.append("自定义路径最多 32 条")
        for path in pr["custom_paths"]:
            finite(path["extra_delay_s"],"额外时延",0,1);finite(path["attenuation_db"],"路径衰减",0,300)
            for key in ("phase_deg","doppler_hz"):finite(path[key],key,-1e10,1e10)
            for key in ("departure_deg","arrival_deg"):
                if np.asarray(path[key]).shape!=(2,) or not np.isfinite(path[key]).all():errors.append(key+" 须为 [方位,俯仰]")
        atm=pr["atmosphere"]
        rf_half=w["bandwidth_hz"]/2 if w["kind"] in ("fmcw","lfm") else 0.
        rf_low,rf_high=w["fc_hz"]-rf_half,w["fc_hz"]+rf_half
        if rf_low<=0:errors.append("实际调频频带的下边缘 fc−B/2 必须大于 0 Hz")
        if atm["enabled"] and (rf_low<1e9 or rf_high>1e12):errors.append("启用大气模型时，完整实际射频频带须在 ITU 支持的 1–1000 GHz 内")
        finite(atm["temperature_k"],"气温 K",150,350);finite(atm["pressure_hpa"],"干空气气压 hPa",1,1200)
        finite(atm["vapor_density_g_m3"],"水汽密度",0,100);finite(atm["rain_rate_mm_h"],"雨强",0,500);finite(atm["polarization_tilt_deg"],"极化倾角",-180,180)
        choice(cl["kind"],["none","gaussian","edge","k"],"杂波模型")
        finite(cl["scatterers"],"散射单元数",1,4096,True);finite(cl["shape"],"K 纹理形状",.01,1000)
        for key in ("range_min_m","range_max_m","edge_range_m"):finite(cl[key],key,.001,1e7)
        if cl["range_max_m"]<=cl["range_min_m"]:errors.append("杂波范围上限须大于下限")
        for key in ("cnr_db","edge_db"):finite(cl[key],key,-100,100)
        finite(cl["doppler_hz"],"杂波中心频率",-1e8,1e8);finite(cl["doppler_width_hz"],"杂波谱宽",0,1e8);finite(cl["az_span_deg"],"杂波方位半宽",0,180)
        choice(d["method"],["ca","os","fixed"],"检测器");choice(d["threshold_mode"],["analytic","calibrated"],"门限模式")
        finite(d["pfa"],"名义 Pfa",1e-12,.5);finite(d["rank_fraction"],"OS 排序分位",.01,1);finite(d["fixed_snr_db"],"固定门限 dB",-30,100);finite(d["ddm_gap_db"],"DDM 空槽差值",0,60)
        finite(d["ddm_residual_limit"],"DDM 归一化重构残差上限",0,1)
        for key in ("training","guard"):
            if len(d[key])!=2:errors.append(key+" 须为两个整数")
            else:
                for x in d[key]:finite(x,key,0,128,True)
        if sum(d["training"])<1:errors.append("CFAR 至少有一个训练维度")
        for key in ("pd","pfa"):finite(p["requirements"][key],"要求 "+key,1e-12,1-1e-12)
        finite(p["requirements"]["range_m"],"预算距离",.001,1e9)
        for key in ("snr_db","margin_db"):finite(p["requirements"][key],key,-50,100)
        for key in ("trials","distance_count","roc_points","memory_mib"):
            finite(e[key],key,1,100000000,True)
        if e["memory_mib"]>8192:errors.append("内存预算上限为 8192 MiB")
        if e["distance_count"]>201 or e["roc_points"]>101:errors.append("距离点最多 201，ROC 点最多 101")
        finite(e["confidence"],"置信水平",.5,.9999)
        for key in ("cut_range_m","distance_min_m","distance_max_m"):finite(e[key],key,.001,1e9)
        finite(e["cut_velocity_mps"],"CUT 速度",-1e6,1e6)
        finite(cl["seed"],"杂波随机种子",0,2**32-1,True)
        for key in ("seed",):finite(e[key],key,0,2**32-1,True)
        if e["distance_min_m"]<=0 or e["distance_max_m"]<=e["distance_min_m"]:errors.append("距离扫描须满足 0 < 下限 < 上限")
        if not errors:
            if receiver_layout(p)["input_record_duration_s"]>w["pri_s"]*(1+1e-12):
                errors.append("含 FIR 噪声历史的相邻采集记录重叠；请缩短采样窗、增加 PRI 或扩大接收带宽")
            mem=resource_estimate(p)
            if mem["estimated_mib"]>e["memory_mib"]:errors.append(f"预计内存 {mem['estimated_mib']:.1f} MiB 超过预算 {e['memory_mib']} MiB")
    except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError) as exc:
        errors.append("配置结构或数值无效："+str(exc))
    return errors


def checked_config(config):
    p=normalize_config(config); errors=validate_config(p)
    if errors:raise ValueError("；".join(errors))
    return p
