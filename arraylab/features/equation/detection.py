"""Data-only detectors, explicit threshold assumptions and exact intervals."""
from __future__ import annotations
import copy
import math
from functools import lru_cache
import numpy as np
from scipy import ndimage, signal
from scipy.optimize import brentq
from scipy.stats import beta
from .config import config_hash


@lru_cache(maxsize=256)
def os_factor(n,k,pfa):
    terms=np.arange(n-k+1,n+1,dtype=float)
    def f(alpha):return float(np.log(terms/(terms+alpha)).sum()-math.log(pfa))
    high=1.
    while f(high)>0:high*=2
    return float(brentq(f,0,high,xtol=1e-12))


def binomial_interval(count,trials,confidence=.95):
    if not all(isinstance(x,(int,float,np.integer,np.floating)) and not isinstance(x,(bool,np.bool_)) and np.isfinite(x) for x in (count,trials,confidence)):
        raise ValueError("二项试验参数必须是有限数值")
    if int(count)!=count or int(trials)!=trials or not 0<confidence<1:raise ValueError("计数须为整数，置信水平须介于 0 与 1")
    k=int(count);n=int(trials)
    if n<1 or not 0<=k<=n:raise ValueError("无效二项试验计数")
    alpha=1-confidence
    return {"estimate":k/n,"lower":0. if k==0 else float(beta.ppf(alpha/2,k,n-k+1)),
        "upper":1. if k==n else float(beta.ppf(1-alpha/2,k+1,n-k)),"count":k,"trials":n,"confidence":confidence,
        "zero_event_upper":float(-math.expm1(math.log(1-confidence)/n)) if k==0 else None}


def calibration_fingerprint(p):
    # H0 calibration depends on receiver/background, CUT and detector, not target truth or validation RNG.
    x={k:copy.deepcopy(p[k]) for k in ("waveform","radar","array","propagation","clutter","processing","detection")}
    for k in ("calibration_factor","calibration_hash"):x["detection"].pop(k,None)
    x["detection"].pop("threshold_mode",None)
    x["radar"]["noise_enabled"]=True
    x["radar_height_m"]=p["scene"]["radar_height_m"]
    x["cut"]=[p["experiment"]["cut_range_m"],p["experiment"]["cut_velocity_mps"]]
    x["calibration_model_version"]="1.0.0"
    return config_hash(x)


def analytic_assumptions(p):
    why=[];w=p["waveform"];q=p["processing"]
    if q["looks"]!=1:why.append("非相干功率和不服从单次指数分布")
    if p["clutter"]["kind"]!="none":why.append("杂波背景需按场景校准")
    if p["array"]["mode"]=="ddm":why.append("DDM 空子带选择改变最终检测概率")
    if q["fast_window"]!="rect" or q["slow_window"]!="rect":why.append("加窗引入训练单元相关")
    expected=w["pulses"]//q["looks"]//(p["array"]["tx_count"] if p["array"]["mode"]=="tdm" else 1)
    if w["range_fft"]!=w["samples"] or w["doppler_fft"]!=expected:why.append("补零/重采样单元相关")
    if w["kind"]!="fmcw":why.append("脉压邻近单元相关")
    if w["rx_bandwidth_hz"]<w["fs_hz"]/2:why.append("接收限带改变有限记录相关性")
    return why


def detector_statistic(power,noise_power,settings,cancel=None):
    power=np.asarray(power,float);noise_power=np.broadcast_to(np.asarray(noise_power,float),power.shape)
    if power.ndim!=2:raise ValueError("检测输入必须是距离-速度二维功率")
    finite=np.isfinite(power)&(power>=0)&np.isfinite(noise_power)&(noise_power>0)
    method=settings["method"]
    if method=="fixed":
        stat=np.divide(power,noise_power,out=np.full_like(power,np.nan),where=finite)
        return stat,noise_power,finite,1
    tr,tv=map(int,settings["training"]);gr,gv=map(int,settings["guard"])
    hr,hv=tr+gr,tv+gv; shape=(2*hv+1,2*hr+1)
    footprint=np.ones(shape,bool);footprint[hv-gv:hv+gv+1,hr-gr:hr+gr+1]=False
    n=int(footprint.sum());base=np.full_like(power,np.nan);valid=np.zeros_like(finite)
    if n==0 or power.shape[0]<shape[0] or power.shape[1]<shape[1]:return base,base,valid,n
    # Locally normalize known heteroscedastic processing noise before CFAR.
    normalized=np.divide(power,noise_power,out=np.zeros_like(power),where=finite)
    train_valid=signal.convolve2d(finite.astype(float),footprint.astype(float),mode="same")>=n-.5
    valid[hv:power.shape[0]-hv if hv else None,hr:power.shape[1]-hr if hr else None]=True
    valid &= finite & train_valid
    if method=="ca":base=signal.convolve2d(normalized,footprint/n,mode="same")
    elif method=="os":
        order=max(1,min(n,math.ceil(n*settings["rank_fraction"])))
        view=np.lib.stride_tricks.sliding_window_view(normalized,shape)
        for row in range(view.shape[0]):
            if cancel and cancel():raise InterruptedError("已取消")
            data=view[row].reshape(view.shape[1],-1)[:,footprint.ravel()]
            base[row+hv,hr:power.shape[1]-hr if hr else None]=np.partition(data,order-1,axis=1)[:,order-1]
    else:raise ValueError("未知检测器")
    base=np.where(valid,base*noise_power,np.nan)
    stat=np.divide(power,base,out=np.full_like(power,np.nan),where=valid&(base>0))
    return stat,base,valid,n


def run_detector(p,arrays,cancel=None,allow_uncalibrated=True):
    d=p["detection"];stat,base,valid,n=detector_statistic(arrays["rd_power"],arrays["noise_power"],d,cancel)
    why=analytic_assumptions(p);warnings=[]
    if d["threshold_mode"]=="calibrated":
        if d["calibration_hash"]!=calibration_fingerprint(p):raise ValueError("校准已过期或不属于当前配置/CUT，请重新执行 H0 校准")
        factor=d["calibration_factor"]
        if factor is None or not math.isfinite(factor) or factor<=0:raise ValueError("缺少有效校准门限因子")
        provenance="independent_h0_calibration"
    elif d["method"]=="fixed":factor=10**(d["fixed_snr_db"]/10);provenance="manual_fixed_threshold"
    else:
        factor=n*math.expm1(-math.log(d["pfa"])/n) if d["method"]=="ca" and n else os_factor(n,max(1,math.ceil(n*d["rank_fraction"])),d["pfa"]) if n else 1.
        provenance="nominal_exponential_reference"
        if why:warnings.append("名义 CFAR 门限未按当前场景校准："+"；".join(why))
        if why and not allow_uncalibrated:raise ValueError("当前场景需要独立 H0 校准")
    rawmask=valid&(stat>factor)
    mask=rawmask & np.asarray(arrays.get("ddm_valid_mask",np.ones_like(rawmask)),bool)
    power=np.asarray(arrays["rd_power"])
    peaks=mask&(power==ndimage.maximum_filter(np.where(np.isfinite(power),power,-np.inf),size=(3,3),mode="constant",cval=-np.inf))
    points=np.argwhere(peaks);points=sorted(points.tolist(),key=lambda v:(-power[tuple(v)],v[0],v[1]))[:256]
    detections=[{"range_m":float(arrays["ranges_m"][ir]),"velocity_mps":float(arrays["velocities_mps"][iv]),
        "power_dbw":10*math.log10(max(float(power[iv,ir]),1e-300)),"cell_snr_db":10*math.log10(max(float(power[iv,ir]/arrays["noise_power"][iv,ir]),1e-300)),
        "range_index":ir,"velocity_index":iv} for iv,ir in points]
    if p["waveform"]["kind"]=="fmcw":
        slope=p["waveform"]["bandwidth_hz"]/p["waveform"]["ramp_s"]
        for item in detections:
            item["apparent_range_m"]=item["range_m"]
            item["range_m"]-=p["waveform"]["fc_hz"]*item["velocity_mps"]/slope
    return {"threshold":base*factor,"mask":mask,"raw_mask":rawmask,"statistic":stat,"valid":valid,"factor":float(factor),
        "provenance":provenance,"warnings":warnings,"analytic_assumptions":why,"detections":detections}
