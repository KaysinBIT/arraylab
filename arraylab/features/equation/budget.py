"""Independent energy-equation reference. Never imports the waveform simulator."""
from __future__ import annotations
import math
import numpy as np
from .config import C,K,normalize_config,validate_config,noise_density,event_plan,complex_weights,resource_estimate,config_hash
from .receiver_spec import receiver_layout, observable_path, far_field_spec


def db(x):return 10*math.log10(max(float(x),1e-300))


def window_efficiency(n,kind):
    n=int(n)
    if kind!="hann":return 1.
    if n==1:return 1.
    if n==2:return 0.
    if n==3:return 1/3
    return 2*(n-1)/(3*n)


def required_event_count(base_snr_per_event,threshold,cycle,looks,window):
    unit=cycle*looks
    minimum=max(math.ceil(2/unit),3 if window=="hann" else 1)
    def achieved(groups):return base_snr_per_event*(groups*unit)*window_efficiency(groups,window)
    lo=minimum;hi=max(minimum,math.ceil(threshold/(base_snr_per_event*unit)))
    while achieved(hi)<threshold:hi*=2
    while lo<hi:
        mid=(lo+hi)//2
        if achieved(mid)>=threshold:hi=mid
        else:lo=mid+1
    return lo*unit


def observability(p,range_m,velocity_mps=0.):
    w=p["waveform"]; lam=C/w["fc_hz"]; tau=2*range_m/C
    cycle=p["array"]["tx_count"] if p["array"]["mode"] in ("tdm","cdm") else 1
    vmax=lam/(4*w["pri_s"]*cycle)
    full_vmax=lam/(4*w["pri_s"])
    reasons=[]; layout=receiver_layout(p)
    actual=observable_path(p,tau,2*velocity_mps/lam)
    actual["far_field"]=far_field_spec(p,range_m,range_m)
    if not actual["far_field"]["valid"]:
        actual["physical_observable"]=False
        actual["observable"]=False
        reasons.append("需求距离不满足已知通道坐标的远场条件；该预算仍仅为远场能量参考")
    tail=layout["filter_history_samples"]/w["fs_hz"]
    if w["kind"]=="fmcw":
        slope=w["bandwidth_hz"]/w["ramp_s"];fbeat=slope*tau+2*velocity_mps/lam
        limit=min(w["fs_hz"]/2,w["rx_bandwidth_hz"])
        if not actual["within_if"]:reasons.append("拍频超出正频率 IF/采样通带")
        if not actual["complete_echo"]:reasons.append("ADC 窗口未完整落在同一 chirp 回波内")
        range_limit=C*limit/(2*slope)
        full_limit=C*w["adc_start_s"]/2
        filter_limit=max(0.,C*(w["adc_start_s"]-tail)/2)
        filter_blind=max(0.,C*(layout["input_last_s"]-w["ramp_s"])/2)
        blind=0.
    else:
        fbeat=None;range_limit=C*w["samples"]/w["fs_hz"]/2
        blind=C*(w["pulse_width_s"]+w["recovery_s"])/2
        full_limit=C*(min(w["pri_s"],w["samples"]/w["fs_hz"])-w["pulse_width_s"])/2
        filter_limit=max(0.,full_limit-C*tail/2)
        filter_blind=blind+C*tail/2
        if range_m<blind:reasons.append("回波与发射/恢复盲区重叠")
        if range_m>full_limit:reasons.append("完整回波超出接收窗或下一 PRI")
    if not actual["filter_settled"]:reasons.append("FIR 滤波历史或回波尾部未完整落在有效接收窗内")
    if not actual["unambiguous_doppler"]:reasons.append("速度超过该体制原始慢时间无模糊范围 [-vmax, vmax)")
    if not actual["within_processed_grid"]:reasons.append("当前输出/校核网格不覆盖该距离或速度坐标；此项不是物理 IF 或无模糊上限")
    return {**actual,"reasons":reasons,"beat_frequency_hz":fbeat,
        "if_sampling_limit_m":range_limit,"full_echo_limit_m":full_limit,"blind_range_m":blind,
        "filter_complete_echo_limit_m":filter_limit,"filter_blind_range_m":filter_blind,
        "unambiguous_range_m":C*w["pri_s"]/2,"max_velocity_mps":vmax,"physical_max_velocity_mps":full_vmax,
        "ddm_conservative_velocity_mps":full_vmax/p["array"]["ddm_slots"] if p["array"]["mode"]=="ddm" else None}


def calculate(config):
    try:p=normalize_config(config)
    except (ValueError,TypeError,KeyError,ZeroDivisionError,OverflowError) as exc:return {"valid":False,"errors":[str(exc)],"warnings":[]}
    errors=validate_config(p)
    if errors:return {"valid":False,"errors":errors,"warnings":[]}
    w=p["waveform"];a=p["array"];r=p["radar"];q=p["processing"];req=p["requirements"]
    plan=event_plan(p); lam=C/w["fc_hz"];n0=noise_density(p); looks=q["looks"];block=plan["block_size"]
    tunit=w["samples"]/w["fs_hz"] if w["kind"]=="fmcw" else w["pulse_width_s"]
    cycle=a["tx_count"] if a["mode"] in ("tdm","cdm") else a["ddm_slots"] if a["mode"]=="ddm" else 1
    fast_n=w["samples"] if w["kind"]=="fmcw" else max(2,round(w["pulse_width_s"]*w["fs_hz"]))
    ef=window_efficiency(fast_n,q["fast_window"]);es=window_efficiency(block//cycle,q["slow_window"])
    rxw=complex_weights(a["rx_weights"]); rxgain=float(abs(rxw.sum())**2)
    txw=complex_weights(a["tx_weights"],a["tx_count"])
    txgain=float(abs(txw.sum())**2) if a["mode"]=="phased" else (a["tx_count"] if a["mode"] in ("ddm","cdm") else 1.)
    if txgain*rxgain<=1e-24:
        return {"valid":False,"config_valid":True,"errors":["扫描指向处的复权重形成零陷；保持当前权重时不存在有限功率或增益反解。"],"warnings":[],"inverse":None}
    pr_per_rcs=plan["per_tx_power_w"]*10**((r["tx_gain_dbi"]+r["rx_gain_dbi"]-r["loss_db"])/10)*lam**2/(4*math.pi)**3
    rcs=p["scene"]["targets"][0]["rcs_m2"] if p["scene"]["targets"] else 1.
    distance=req["range_m"];pr=pr_per_rcs*rcs/distance**4
    energy_snr=pr*tunit*w["pulses"]*txgain*rxgain*ef*es/n0
    perlook_snr=energy_snr/looks
    threshold=10**((req["snr_db"]+req["margin_db"])/10)
    rmax=distance*(energy_snr/threshold)**.25
    factor=threshold/max(energy_snr,1e-300)
    inverse_events=required_event_count(pr*tunit*txgain*rxgain*ef/n0,threshold,cycle,looks,q["slow_window"])
    txon=w["ramp_s"] if w["kind"]=="fmcw" else w["pulse_width_s"]
    average_power=plan["per_tx_power_w"]*plan["active_tx"]*txon/w["pri_s"]
    received_dbm=db(pr)+30
    obs=observability(p,distance)
    warnings=["预算为指向中心、稳定点目标和理想已分离通道的能量基准；场景检测性能由数值实验给出。",
              "NF 为接收天线参考面上的总接收链噪声系数，已计入其中的前端被动损耗不再计入额外损耗。"]
    if looks>1:warnings.append("非相干积累的总能量 SNR 为各 look 能量参数之和；输出功率比 SNR 为信号功率和/噪声功率和，当前等强 look 下低 10log10(looks) dB。该口径差不是处理损耗，也不是等价检测 SNR；四次根距离仅为总能量参考。")
    if p["propagation"]["model"]!="free" or p["propagation"]["atmosphere"]["enabled"]:warnings.append("此处自由空间预算不包含场景多径和大气；请运行场景实验/距离统计。")
    if p["clutter"]["kind"]!="none":warnings.append("杂波不进入自由空间热噪声预算；SINR 由实际联合背景输出测量。")
    if any(t["swerling"] for t in p["scene"]["targets"]):warnings.append("目标起伏可能破坏相干积累与编码正交性，理想处理增益不作为达标结论。")
    if w["kind"]!="fmcw" and w["rx_bandwidth_hz"]<(w["bandwidth_hz"] if w["kind"]=="lfm" else 2/w["pulse_width_s"]):warnings.append("接收限带可能截断脉冲频谱；离散滤波损失以信号链结果为准。")
    resolution=C/(2*(w["bandwidth_hz"]/w["ramp_s"]*tunit)) if w["kind"]=="fmcw" else C/(2*w["bandwidth_hz"]) if w["kind"]=="lfm" else C*w["pulse_width_s"]/2
    curve_ranges=np.geomspace(max(.1,min(distance/10,rmax/10)),max(distance*2,rmax*1.5),121)
    curve_snr=db(energy_snr)+40*np.log10(distance/curve_ranges)
    ledger=[{"name":"每 Tx 端口功率","db":db(plan["per_tx_power_w"])+30,"unit":"dBm"},
        {"name":"Tx 天线增益","db":r["tx_gain_dbi"],"unit":"dBi"},{"name":"Rx 天线增益","db":r["rx_gain_dbi"],"unit":"dBi"},
        {"name":"波长平方 / (4π)³","db":db(lam**2/(4*math.pi)**3),"unit":"dB"},{"name":"目标 RCS","db":db(rcs),"unit":"dBsm"},
        {"name":"距离四次方","db":-40*math.log10(distance),"unit":"dB"},{"name":"额外损耗","db":-r["loss_db"],"unit":"dB"},
        {"name":"有效观测时间","db":db(tunit*w["pulses"]),"unit":"dB·s"},{"name":"发射/编码能量因子","db":db(txgain),"unit":"dB"},
        {"name":"接收合并","db":db(rxgain),"unit":"dB"},{"name":"快/慢窗效率","db":db(ef*es),"unit":"dB"},
        {"name":"噪声谱密度","db":-(db(n0)+30),"unit":"−dBm/Hz"}]
    return {"valid":True,"errors":[],"warnings":warnings,"config_hash":config_hash(p),"kind":"budget",
        "received_power_w":pr,"received_power_dbm":received_dbm,"noise_density_w_hz":n0,"noise_density_dbm_hz":db(n0)+30,
        "system_temperature_k":n0/K,"input_noise_w":n0*w["fs_hz"],"input_snr_db":db(pr/(n0*w["fs_hz"])),
        "output_snr_db":db(energy_snr),"energy_snr_db":db(energy_snr),
        "summed_power_snr_db":db(perlook_snr),"per_look_snr_db":db(perlook_snr),"snr_margin_db":db(energy_snr)-req["snr_db"]-req["margin_db"],
        "max_range_m":rmax,"range_m":distance,"threshold_snr_db":req["snr_db"]+req["margin_db"],"rcs_m2":rcs,
        "per_tx_power_w":plan["per_tx_power_w"],"total_active_power_w":plan["per_tx_power_w"]*plan["active_tx"],"average_power_w":average_power,
        "per_tx_peak_power_w":(plan["per_tx_power_w"]*np.max(abs(plan["weights"])**2,axis=0)).tolist(),
        "event_total_power_w":(plan["per_tx_power_w"]*np.sum(abs(plan["weights"])**2,axis=1)).tolist(),
        "effective_time_s":tunit*w["pulses"],"per_tx_events":w["pulses"]//a["tx_count"] if a["mode"]=="tdm" else w["pulses"],
        "fast_window_efficiency":ef,"slow_window_efficiency":es,"rx_combining_gain_db":db(rxgain),"tx_energy_gain_db":db(txgain),
        "range_resolution_m":resolution,"velocity_resolution_mps":lam/(2*block*w["pri_s"]),"observability":obs,
        "inverse":{"required_power_w":r["power_w"]*factor,"required_tx_gain_dbi":r["tx_gain_dbi"]+db(factor),
            "required_rx_gain_dbi":r["rx_gain_dbi"]+db(factor),"required_equal_gain_dbi":(r["tx_gain_dbi"]+r["rx_gain_dbi"]+db(factor))/2,
            "minimum_rcs_m2":rcs*factor,"required_events":inverse_events},
        "ledger":ledger,"curve":{"range_m":curve_ranges.tolist(),"snr_db":curve_snr.tolist()},"resources":resource_estimate(p),
        "assumptions":{"power_reference":"每物理 Tx 天线端口传导功率","noise_reference":"接收天线端口输入等效噪声",
            "propagation":"自由空间","target":"稳定点目标，扫描指向中心","noncoherent_reference":looks>1}}
