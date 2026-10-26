"""Independent H0/H1 experiments. Truth is used only for fixed evaluation gates."""
from __future__ import annotations
import copy
import math
import time
import numpy as np
from .config import checked_config,config_hash,C
from .budget import calculate,observability
from .detection import run_detector,detector_statistic,calibration_fingerprint,binomial_interval,analytic_assumptions


def _check(cancel):
    if cancel and cancel():raise InterruptedError("计算已取消")


def _seed(seed,purpose,index,branch=0):
    return int(np.random.SeedSequence([int(seed),purpose,index,branch]).generate_state(1)[0])


def _cell(arrays,distance,velocity):
    if not arrays["ranges_m"][0]<=distance<=arrays["ranges_m"][-1] or not arrays["velocities_mps"][0]<=velocity<=arrays["velocities_mps"][-1]:
        raise ValueError("指定 CUT 超出实际距离/速度处理网格，不能静默截到边界")
    return (int(np.argmin(abs(arrays["velocities_mps"]-velocity))),int(np.argmin(abs(arrays["ranges_m"]-distance))))


def apparent_range(p,distance,velocity):
    w=p["waveform"]
    return distance+w["fc_hz"]*velocity/(w["bandwidth_hz"]/w["ramp_s"]) if w["kind"]=="fmcw" else distance


def _simulate(p,seed,signal=True,noise=True,clutter=True,cancel=None):
    from .engine import simulate
    summary,arrays=simulate(p,seed=seed,signal=signal,noise=noise,clutter=clutter,cancel=cancel)
    arrays["noise_power"]=np.broadcast_to(arrays["noise_power"],arrays["rd_power"].shape)
    return summary,arrays


def _attach_detector(p,arrays,cancel=None):
    d=run_detector(p,arrays,cancel)
    arrays["cfar_threshold"]=d["threshold"];arrays["cfar_mask"]=d["mask"]
    return d


def simulation(config,progress=None,cancel=None):
    p=checked_config(config);_check(cancel)
    if progress:progress(.05,"生成物理接收通道与处理算子")
    summary,arrays=_simulate(p,p["experiment"]["seed"],cancel=cancel)
    d=_attach_detector(p,arrays,cancel)
    summary.update(kind="simulation",detections=d["detections"],threshold_factor=d["factor"],threshold_provenance=d["provenance"])
    summary["warnings"]=summary.get("warnings",[])+d["warnings"]
    if progress:progress(1.,"仿真完成")
    return summary,arrays


def snr_experiment(config,progress=None,cancel=None):
    p=checked_config(config);n=int(p["experiment"]["trials"]);seed=p["experiment"]["seed"]
    if not p["scene"]["targets"]:raise ValueError("SNR 校核至少需要一个目标")
    if not p["radar"]["noise_enabled"]:raise ValueError("SNR 校核要求启用热噪声")
    target=p["scene"]["targets"][0]
    desired=copy.deepcopy(p);desired["scene"]["targets"]=[copy.deepcopy(target)]
    interferers=copy.deepcopy(p);interferers["scene"]["targets"]=copy.deepcopy(p["scene"]["targets"][1:])
    sig_summary,sig=_simulate(desired,_seed(seed,110,0),noise=False,clutter=False,cancel=cancel)
    iv,ir=_cell(sig,p["experiment"]["cut_range_m"],p["experiment"]["cut_velocity_mps"])
    sp=float(sig["rd_power"][iv,ir]);expected=float(sig["noise_power"][iv,ir])
    if expected<=0 or not np.isfinite(expected):raise ValueError("选定校核单元不在有效处理区")
    samples=[];background=[];last=sig
    for i in range(n):
        _check(cancel)
        _,a=_simulate(p,_seed(seed,111,i),signal=False,clutter=False,cancel=cancel)
        samples.append(float(a["rd_power"][iv,ir]));last=a
        if p["clutter"]["kind"]!="none" or interferers["scene"]["targets"]:
            _,b=_simulate(interferers,_seed(seed,112,i),signal=bool(interferers["scene"]["targets"]),clutter=True,cancel=cancel)
            background.append(float(b["rd_power"][iv,ir]))
        if progress:progress((i+1)/n,f"完整处理链噪声试验 {i+1}/{n}")
    measured=float(np.mean(samples));theory=10*math.log10(max(sp/expected,1e-300));snr=10*math.log10(max(sp/measured,1e-300))
    bcfg=copy.deepcopy(p);bcfg["requirements"]["range_m"]=target["range_m"]
    budget=calculate(bcfg);uncertainty=1.96*float(np.std(samples,ddof=1))/math.sqrt(n)/measured*10/math.log(10) if n>1 else None
    _,arrays=_simulate(p,_seed(seed,113,0),cancel=cancel);d=_attach_detector(p,arrays,cancel)
    arrays["noise_trial_power"]=np.array(samples);arrays["background_trial_power"]=np.array(background)
    simple=(len(p["scene"]["targets"])==1 and target["swerling"]==0 and p["propagation"]["model"]=="free"
            and not p["propagation"]["atmosphere"]["enabled"] and p["clutter"]["kind"]=="none" and p["processing"]["looks"]==1)
    complete=bool(sig_summary.get("observability")) and all(x["observable"] for x in sig_summary["observability"])
    noise_pass=n>=2048 and abs(snr-theory)<=.4
    energy_pass=bool(simple and complete and budget["valid"] and abs(budget["output_snr_db"]-theory)<=.4)
    result={"kind":"snr","theoretical_snr_db":theory,"measured_snr_db":snr,"error_db":snr-theory,
        "energy_reference_snr_db":budget.get("output_snr_db"),"discrete_loss_db":budget["output_snr_db"]-theory if budget["valid"] else None,
        "signal_power_w":sp,"noise_power_w":measured,"expected_noise_power_w":expected,"noise_trials":n,
        "approximate_noise_uncertainty_db":uncertainty,"sinr_db":10*math.log10(max(sp/float(np.mean(background)),1e-300)) if background else snr,
        "cut":{"range_m":float(sig["ranges_m"][ir]),"velocity_mps":float(sig["velocities_mps"][iv])},"detections":d["detections"],
        "validated":noise_pass and energy_pass,"noise_calibration_passed":noise_pass,"energy_equation_passed":energy_pass,
        "energy_reference_applicable":simple,"complete_path_observation":complete,"observability":sig_summary.get("observability",[]),
        "warnings":["数值 SNR 在预先指定的单元比较无噪声输出与独立噪声输出，不使用全图最大峰。",
            "离散理论值使用真实线性系数的噪声标定；另列独立能量方程及离散失配损失。",
            "信号分子只取首个目标及其相干多径的固定实现；SINR 分母联合处理其余目标、杂波和热噪声。该校核不等同于目标检测概率。"]+d["warnings"]}
    return result,arrays


def calibrate(config,progress=None,cancel=None):
    p=checked_config(config);e=p["experiment"];n=int(e["trials"]);scores=[];last=None
    if not p["radar"]["noise_enabled"]:raise ValueError("H0 校准要求启用热噪声")
    for i in range(n):
        _check(cancel)
        _,a=_simulate(p,_seed(e["seed"],210,i),signal=False,cancel=cancel)
        iv,ir=_cell(a,e["cut_range_m"],e["cut_velocity_mps"])
        stat,_,valid,_=detector_statistic(a["rd_power"],a["noise_power"],p["detection"],cancel)
        if not valid[iv,ir] or not np.isfinite(stat[iv,ir]):raise ValueError("校准 CUT 位于 CFAR 边界或无效单元，请调整 CUT/训练窗")
        decoded=bool(np.asarray(a.get("ddm_valid_mask",np.ones_like(valid)))[iv,ir])
        scores.append(float(stat[iv,ir]) if decoded else 0.);last=a
        if progress:progress((i+1)/n,f"独立 H0 校准 {i+1}/{n}")
    pfa=p["detection"]["pfa"]
    factor=float(np.quantile(scores,1-pfa,method="higher"))
    ready=n*pfa>=20 and factor>0
    warnings=["校准与验收使用不同随机流；本结果尚未经独立验收。", "校准只覆盖当前配置和指定 CUT，杂波边缘需分别选择各区域校准/验证。"]
    if not ready:warnings.append(f"尾部样本不足或 DDM 可辨识样本不足；建议至少 {math.ceil(20/pfa):,} 次校准，当前门限不能应用为有效校准。")
    result={"kind":"calibration","factor":factor,"calibration_hash":calibration_fingerprint(p) if ready else None,
        "trials":n,"pfa":pfa,"tail_events":int(np.count_nonzero(np.asarray(scores)>factor)),"ready":ready,"verified":False,
        "expected_tail_events":n*pfa,"recommended_trials":math.ceil(20/pfa),"warnings":warnings}
    last["calibration_scores"]=np.asarray(scores)
    return result,last


def _truth_gates(p):
    w=p["waveform"]
    resolution=C/(2*w["bandwidth_hz"]*w["samples"]/w["fs_hz"]*1/w["ramp_s"]) if w["kind"]=="fmcw" else C/(2*w["bandwidth_hz"]) if w["kind"]=="lfm" else C*w["pulse_width_s"]/2
    dr=max(resolution,C/(2*w["fs_hz"]));dv=C/w["fc_hz"]/(2*(w["pulses"]//p["processing"]["looks"])*w["pri_s"])
    direct=[(t["range_m"],t["velocity_mps"]) for t in p["scene"]["targets"]]
    ghosts=[]
    if p["propagation"]["model"]!="free":
        from .propagation import paths_for_target
        for t in p["scene"]["targets"]:
            for path in paths_for_target(p,t)[1:]:
                rg=path["length_m"]/2;vv=path["doppler_hz"]*C/p["waveform"]["fc_hz"]/2
                if not any(abs(rg-r)<=dr and abs(vv-v)<=dv for r,v in direct):ghosts.append((rg,vv))
    return dr,dv,direct,ghosts


def statistics(config,progress=None,cancel=None):
    p=checked_config(config);e=p["experiment"];n=int(e["trials"]);seed=e["seed"]
    if not p["radar"]["noise_enabled"]:raise ValueError("检测统计要求启用热噪声")
    if not p["scene"]["targets"]:raise ValueError("检测统计至少需要一个目标")
    target=p["scene"]["targets"][0];dr,dv,direct,ghosts=_truth_gates(p)
    hits=fas=associations=ghost_hits=scene_fas=0;records=[];last=None;warnings=[]
    for i in range(n):
        _check(cancel)
        _,h0=_simulate(p,_seed(seed,310,i,0),signal=False,cancel=cancel);d0=_attach_detector(p,h0,cancel)
        iv,ir=_cell(h0,e["cut_range_m"],e["cut_velocity_mps"])
        if not d0["valid"][iv,ir]:raise ValueError("虚警 CUT 位于 CFAR 边界或无效单元，请调整 CUT/训练窗")
        fa=bool(d0["mask"][iv,ir]);fas+=fa;scene_fas+=bool(np.any(d0["mask"]))
        h1_summary,h1=_simulate(p,_seed(seed,310,i,1),cancel=cancel);d1=_attach_detector(p,h1,cancel)
        vt,rt=_cell(h1,apparent_range(p,target["range_m"],target["velocity_mps"]),target["velocity_mps"])
        if not d1["valid"][vt,rt]:raise ValueError("目标 CUT 位于 CFAR 边界或无效单元，请调整处理区域")
        hit=bool(d1["mask"][vt,rt]);hits+=hit
        assoc=any(abs(x["range_m"]-target["range_m"])<=dr and abs(x["velocity_mps"]-target["velocity_mps"])<=dv for x in d1["detections"])
        ghost=any(abs(x["range_m"]-rg)<=dr and abs(x["velocity_mps"]-vv)<=dv for x in d1["detections"] for rg,vv in ghosts)
        associations+=assoc;ghost_hits+=ghost;records.append([i,int(hit),int(fa),int(assoc),int(ghost)])
        last=h1;warnings=d1["warnings"]
        if progress:progress((i+1)/n,f"独立 H0/H1 检测试验 {i+1}/{n}")
    pd=binomial_interval(hits,n,e["confidence"]);pfa=binomial_interval(fas,n,e["confidence"])
    assoc=binomial_interval(associations,n,e["confidence"])
    obs=observability(p,target["range_m"],target["velocity_mps"])
    calibrated=p["detection"]["threshold_mode"]=="calibrated" or p["detection"]["method"]=="fixed" or not analytic_assumptions(p)
    complete=bool(h1_summary.get("observability")) and all(x["observable"] for x in h1_summary["observability"])
    motion=max(abs(t["velocity_mps"])*p["waveform"]["pri_s"]*p["waveform"]["pulses"] for t in p["scene"]["targets"])<=.25*dr
    valid=bool(calibrated and obs["observable"] and complete and motion and pd["lower"]>=p["requirements"]["pd"] and pfa["upper"]<=p["requirements"]["pfa"])
    warnings=warnings+["每次独立试验只使用一个预定 CUT 测 Pfa；整图虚警事件另列，相关像素不算独立试验。",
        "鬼影率按预先固定的可分辨多径位置窗口评估；不包括未分辨多径与所有门外噪声峰。"]
    if n*p["requirements"]["pfa"]<100:warnings.append(f"当前虚警尾部采样有限；建议约 {math.ceil(100/p['requirements']['pfa']):,} 次独立试验。零虚警不等于 Pfa=0。")
    result={"kind":"statistics","pd":pd,"pfa":pfa,"association":assoc,"ghost":binomial_interval(ghost_hits,n,e["confidence"]),
        "scene_false_alarm":binomial_interval(scene_fas,n,e["confidence"]),"confidence":e["confidence"],"validated":valid,
        "observability":obs,"nominal_pfa":p["detection"]["pfa"],"recommended_trials":math.ceil(100/p["requirements"]["pfa"]),
        "calibrated":calibrated,"complete_path_observation":complete,"motion_within_model":motion,"path_observability":h1_summary.get("observability",[]),"cut":{"range_m":e["cut_range_m"],"velocity_mps":e["cut_velocity_mps"]},
        "association_gate":{"range_m":dr,"velocity_mps":dv},"ghost_gates":ghosts,"warnings":warnings,
        "detections":d1["detections"],"clutter_ensemble":"fixed_background" if p["clutter"]["fixed"] else "independent_background"}
    last["trial_results"]=np.asarray(records,dtype=np.int64)
    return result,last


def distance_scan(config,progress=None,cancel=None):
    p=checked_config(config);e=p["experiment"]
    if not p["scene"]["targets"]:raise ValueError("距离扫描至少需要一个目标")
    points=[];last={};grid=np.linspace(e["distance_min_m"],e["distance_max_m"],int(e["distance_count"]))
    for j,r in enumerate(grid):
        _check(cancel);cfg=copy.deepcopy(p);cfg["scene"]["targets"][0]["range_m"]=float(r);cfg["experiment"]["cut_range_m"]=apparent_range(cfg,float(r),cfg["scene"]["targets"][0]["velocity_mps"])
        cfg["experiment"]["seed"]=_seed(e["seed"],410,j)
        obs=observability(cfg,float(r),cfg["scene"]["targets"][0]["velocity_mps"])
        if not obs["observable"]:
            points.append({"range_m":float(r),"status":"unobservable","pd":None,"pfa":None,"reasons":obs["reasons"]})
            continue
        try:
            if cfg["detection"]["threshold_mode"]=="calibrated":
                c,_=calibrate(cfg,cancel=cancel)
                if not c["ready"]:
                    points.append({"range_m":float(r),"status":"insufficient","pd":None,"pfa":None,"reasons":c["warnings"]});continue
                cfg["detection"].update(calibration_factor=c["factor"],calibration_hash=c["calibration_hash"])
            result,last=statistics(cfg,progress=(lambda f,m:progress((j+f)/len(grid),f"距离 {r:.2f} m · {m}")) if progress else None,cancel=cancel)
            failed=result["pd"]["upper"]<p["requirements"]["pd"] or result["pfa"]["lower"]>p["requirements"]["pfa"]
            status="passed" if result["validated"] else "failed" if failed else "insufficient"
            points.append({"range_m":float(r),"status":status,"pd":result["pd"],"pfa":result["pfa"],"association":result["association"]})
        except ValueError as exc:points.append({"range_m":float(r),"status":"unavailable","pd":None,"pfa":None,"reasons":[str(exc)]})
    intervals=[];start=None
    for i,point in enumerate(points):
        if point["status"]=="passed" and start is None:start=i
        if start is not None and (point["status"]!="passed" or i==len(points)-1):
            end=i if point["status"]=="passed" else i-1
            intervals.append({"start_m":points[start]["range_m"],"end_m":points[end]["range_m"],"sample_count":end-start+1});start=None
    return {"kind":"distance","points":points,"intervals":intervals,"grid_step_m":float(grid[1]-grid[0]) if len(grid)>1 else None,
        "warnings":["区间表示相邻采样点通过，不保证未采样距离连续达标。", "自定义路径保持额外时延和相对复增益；平地双径按每个距离重新计算几何。"]},last


def roc(config,progress=None,cancel=None):
    p=checked_config(config);e=p["experiment"];n=int(e["trials"]);count=int(e["roc_points"])
    if not p["radar"]["noise_enabled"]:raise ValueError("ROC 检测统计要求启用热噪声")
    if not p["scene"]["targets"]:raise ValueError("ROC 至少需要一个目标")
    target=p["scene"]["targets"][0];h0scores=[];h1scores=[];last={};base=None
    for i in range(n):
        _check(cancel)
        for branch,out in ((0,h0scores),(1,h1scores)):
            _,a=_simulate(p,_seed(e["seed"],510,i,branch),signal=bool(branch),cancel=cancel)
            d=run_detector(p,a,cancel);base=d["factor"]
            cell=_cell(a,apparent_range(p,target["range_m"],target["velocity_mps"]),target["velocity_mps"]) if branch else _cell(a,e["cut_range_m"],e["cut_velocity_mps"])
            if not d["valid"][cell]:raise ValueError("ROC 的 CUT 位于无效/边界单元")
            permitted=bool(np.asarray(a.get("ddm_valid_mask",np.ones_like(d["valid"])))[cell])
            out.append(float(d["statistic"][cell]) if permitted else 0.);last=a
        if progress:progress((i+1)/n,f"ROC 独立 H0/H1 {i+1}/{n}")
    factors=base*np.logspace(-1,1,count)
    points=[{"threshold_db":10*math.log10(float(f)),"pd":binomial_interval(np.count_nonzero(np.asarray(h1scores)>f),n,e["confidence"]),
             "pfa":binomial_interval(np.count_nonzero(np.asarray(h0scores)>f),n,e["confidence"])} for f in factors]
    last["h0_scores"]=np.array(h0scores);last["h1_scores"]=np.array(h1scores)
    return {"kind":"roc","points":points,"warnings":["同一组独立试验评估多个预设门限；曲线点之间相关，各置信区间为逐点区间。", "ROC 的数据用于评估，不自动选择并宣称门限已通过独立验收。"]},last


RUNNERS={"simulation":simulation,"snr":snr_experiment,"calibration":calibrate,"statistics":statistics,"distance":distance_scan,"roc":roc}


def execute(kind,config,progress=None,cancel=None):
    p=checked_config(config);started=time.perf_counter()
    if kind not in RUNNERS:raise ValueError("未知任务类型")
    result,arrays=RUNNERS[kind](p,progress,cancel)
    result.update(config_hash=config_hash(p),schema_version=1,algorithm_version="1.0.0",seed=p["experiment"]["seed"],
        elapsed_s=time.perf_counter()-started,standards={"gas":"ITU-R P.676-13","rain":"ITU-R P.838-3"},config=p)
    return result,arrays
