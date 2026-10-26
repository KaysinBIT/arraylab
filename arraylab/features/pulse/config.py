"""Versioned configuration, physical constraints, and actual acquisition timing."""
from __future__ import annotations
import copy
import math
import hashlib
import json
import numpy as np

C = 299792458.0
K_B = 1.380649e-23


def default_project():
    return {
        'app': 'PulseLab', 'schema_version': 1, 'name': 'X 波段多目标基准', 'seed': 20261026,
        'waveform': {'kind': 'lfm', 'fc_hz': 10e9, 'pulse_width_s': 2e-6,
                     'bandwidth_hz': 4e6, 'sample_rate_hz': 12e6,
                     'phase_code': [1,1,1,1,1,-1,-1,1,1,-1,1,-1,1],
                     'frequency_steps': 16, 'frequency_step_hz': 1e6},
        'timing': {'mode': 'uniform', 'pri_s': 80e-6, 'pri_sequence_s': [80e-6, 92e-6],
                   'pulses': 64, 'groups': 8, 'recovery_s': .5e-6,
                   'receive_start_s': 0., 'receive_end_s': 60e-6},
        'radar': {'peak_power_w': 200., 'tx_gain_dbi': 25., 'rx_gain_dbi': 15.,
                  'noise_figure_db': 5., 'temperature_k': 290., 'loss_db': 4.,
                  'noise_enabled': True, 'receiver_bandwidth_hz': 5e6},
        'array': {'mode': 'phased', 'geometry': 'ula', 'tx_count': 1, 'rx_count': 8,
                  'tx_positions_m': [], 'rx_positions_m': [], 'spacing_lambda': .5,
                  'pointing_deg': [0., 0.], 'beamwidth_deg': [60., 40.],
                  'scan_rate_deg_s': [0., 0.]},
        'scene': {'platform': {'position_m': [0.,0.,100.], 'velocity_mps': [0.,0.,0.],
                               'attitude_deg': [0.,0.,0.], 'trajectory': []},
                  'targets': [
                      {'id': 'T1', 'position_m': [3000.,100.,200.], 'velocity_mps': [12.,0.,0.],
                       'rcs_m2': 3., 'fluctuation': 'none', 'trajectory': []},
                      {'id': 'T2', 'position_m': [4500.,800.,300.], 'velocity_mps': [-18.,0.,0.],
                       'rcs_m2': 5., 'fluctuation': 'none', 'trajectory': []}],
                  'terrain': {'enabled': False, 'kind': 'hills', 'extent_m': 6000.,
                              'resolution': 17, 'base_height_m': 0., 'slope_deg': 2.,
                              'hill_height_m': 120., 'sigma0_db': -35.,
                              'max_scatterers': 512, 'seed': 83}},
        'processing': {'range_window': 'rect', 'doppler_window': 'hann', 'mti': 0,
                       'doppler_bins': 128, 'range_fft_factor': 1,
                       'velocity_search_mps': [-80.,80.], 'range_search_m': [500.,12000.],
                       'ambiguity_search': False, 'compensation': 'estimated',
                       'known_velocity_mps': 0.,
                       'cfar': {'method': 'ca', 'pfa': 1e-3, 'training': [8,4],
                                'guard': [3,2], 'rank_fraction': .75},
                       'doa': {'enabled': True, 'method': 'bartlett', 'source_count': 1,
                               'diagonal_loading': .01, 'smoothing': 0,
                               'scan_az_deg': [-60.,60.], 'scan_el_deg': [0.,0.], 'step_deg': 1.}},
        'requirements': {'range_m': 5000., 'range_resolution_m': 50.,
                          'velocity_resolution_m': 5., 'max_velocity_mps': 30.,
                          'angle_resolution_deg': 15., 'pd': .9, 'pfa': 1e-3, 'rcs_m2': 3.},
        'design': {'bandwidth_hz': [2e6,4e6,6e6], 'pulse_width_s': [2e-6,4e-6],
                    'pri_s': [80e-6,100e-6], 'pulses': [64,128],
                    'frequency_steps':[8,16,32], 'frequency_step_hz':[.5e6,1e6,2e6], 'groups':[4,8,16],
                    'peak_power_w': [100.,200.,500.], 'rx_count': [4,8],
                    'max_candidates': 96, 'verify_top': 3, 'verification_trials': 16},
        'statistics': {'trials': 32, 'target_index': 0, 'cut_range_m': 3000.,
                        'cut_velocity_mps': 12., 'seed': 87123},
        'resources': {'memory_mib': 1024}
    }


def normalize_project(project):
    if not isinstance(project, dict):
        raise ValueError('工程必须为 JSON 对象')
    if project.get('app', 'PulseLab') != 'PulseLab':
        raise ValueError('此文件不是 PulseLab 工程；WaveLab 配置不兼容')
    if project.get('schema_version', 1) != 1:
        raise ValueError('不支持的工程版本')
    def merge(base, supplied):
        for key, value in supplied.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                merge(base[key], value)
            else:
                base[key] = copy.deepcopy(value)
        return base
    return merge(default_project(), project)


def project_hash(project):
    return hashlib.sha256(json.dumps(project, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def acquisition_plan(project):
    p = normalize_project(project)
    w,t,a = p['waveform'],p['timing'],p['array']
    n = int(w['frequency_steps']*t['groups']) if w['kind']=='stepped' else int(t['pulses'])
    if t['mode']=='uniform':
        intervals=np.full(n,float(t['pri_s']))
    elif t['mode']=='staggered':
        intervals=np.resize(np.asarray(t['pri_sequence_s'],float),n)
    else:
        seq=np.asarray(t['pri_sequence_s'],float)
        intervals=seq[np.minimum(np.arange(n)*len(seq)//n,len(seq)-1)]
    # Timing is quantized to the sample clock; preserve actual times in every export.
    fs=float(w['sample_rate_hz'])
    ticks=np.rint(intervals*fs).astype(np.int64)
    intervals=ticks/fs
    times=np.r_[np.int64(0),np.cumsum(ticks[:-1],dtype=np.int64)]/fs
    ntx=int(a['tx_count'])
    rows=np.ones((n,ntx),float)
    tx=np.full(n,-1,int)
    if a['mode']=='tdm':
        tx=np.arange(n)%ntx
        rows[:]=0
        rows[np.arange(n),tx]=1
    elif a['mode']=='cdm':
        from scipy.linalg import hadamard
        rows=hadamard(ntx)[np.arange(n)%ntx,:].astype(float)
    freqs=np.full(n,w['fc_hz'],float)
    if w['kind']=='stepped':
        freqs+=(np.arange(n)%int(w['frequency_steps'])-(w['frequency_steps']-1)/2)*w['frequency_step_hz']
    return {'times_s':times.tolist(),'intervals_s':intervals.tolist(),
            'frequencies_hz':freqs.tolist(),'tx_indices':tx.tolist(),'code_rows':rows.tolist(),
            'pulse_count':n,'duration_s':float(ticks.sum()/fs),'interval_ticks':ticks.tolist(),
            'sample_rate_hz':fs,'axes':['pulse','rx','fasttime'],
            'stepped_axes':['repeat_group','frequency_step','rx','fasttime'] if w['kind']=='stepped' else None,
            'timing_quantization_s':1/fs}


def metrics(project):
    p=normalize_project(project); w,t,r,a=p['waveform'],p['timing'],p['radar'],p['array']
    plan=acquisition_plan(p); n=plan['pulse_count']; pri=np.asarray(plan['intervals_s'])
    lam=C/w['fc_hz']; tau=w['pulse_width_s']
    duration=plan['duration_s']; ngroup=a['tx_count'] if a['mode'] in ('tdm','cdm') else 1
    if w['kind']=='stepped':ngroup=w['frequency_steps']
    revisit=float(np.mean(pri))*ngroup
    end=min(t['receive_end_s'],float(pri.max()))
    samples=max(0,int(math.ceil((end-t['receive_start_s'])*w['sample_rate_hz'])))
    from .receiver import receiver_filter
    from .receiver import complete_delay_bounds
    taps=receiver_filter(p)
    opening,latest=complete_delay_bounds(p,plan,taps)
    filter_tail=(len(taps)-1)/w['sample_rate_hz']
    pulse_samples=int(math.ceil(tau*w['sample_rate_hz']-1e-10))
    matched_length=pulse_samples+len(taps)-1
    raw=n*a['rx_count']*samples*16
    fftbins=p['processing']['doppler_bins']
    # Includes raw, compressed, temporaries, multichannel Doppler cube and exports.
    factor=p['processing']['range_fft_factor'] if w['kind']!='stepped' else 1
    correlation_prefix=min(matched_length-1,int(math.floor(t['receive_start_s']*w['sample_rate_hz']+1e-10))) if w['kind']!='stepped' else 0
    range_count=(samples+correlation_prefix)*factor
    if w['kind']=='stepped':
        lower=max(0,p['processing']['range_search_m'][0],opening*C/2)
        upper=min(p['processing']['range_search_m'][1],latest*C/2)
        spacing=C/(2*w['frequency_steps']*w['frequency_step_hz']*p['processing']['range_fft_factor'])
        range_count=max(1,int(math.ceil((upper-lower)/spacing))+1)
    angle=p['processing']['doa']; angle_bytes=0
    if angle['enabled']:
        angle_count=math.prod(int(math.ceil((angle[k][1]-angle[k][0])/angle['step_deg']))+1 for k in ('scan_az_deg','scan_el_deg'))
        channels=a['rx_count']*(a['tx_count'] if a['mode']!='phased' else 1)
        angle_bytes=angle_count*channels*16*4+channels**2*16*6+angle_count*32
    pulse_workspace=(2*matched_length+samples)*16*a['rx_count']*factor*4+(2*matched_length-1)*8*16*5
    history_workspace=n*a['rx_count']*(len(taps)-1)*16*4
    memory=raw*(3+4*factor)+fftbins*a['rx_count']*a['tx_count']*range_count*16*3+angle_bytes+pulse_workspace+history_workspace+64*2**20
    R=p['requirements']['range_m']; sigma=p['requirements']['rcs_m2']
    gt=10**(r['tx_gain_dbi']/10);gr=10**(r['rx_gain_dbi']/10)
    n0=K_B*r['temperature_k']*10**(r['noise_figure_db']/10)
    pr=r['peak_power_w']*gt*gr*lam**2*sigma/((4*math.pi)**3*R**4*10**(r['loss_db']/10))
    snr=pr*tau/n0
    # Total port power: TDM allocates the active pulse entirely to one Tx,
    # CDM shares total power between simultaneous ports. Ideal bound only.
    equivalent=n*(a['tx_count'] if a['mode']=='phased' else 1)
    frequency_energy_ratio=float(np.mean((w['fc_hz']/np.asarray(plan['frequencies_hz']))**2))
    ideal=snr*equivalent*a['rx_count']*frequency_energy_ratio
    observed_span=float(plan['times_s'][-1]-plan['times_s'][0])
    velocity_time=duration if t['mode']=='uniform' else observed_span
    if t['mode']!='uniform' and a['mode']=='tdm':
        ts=np.asarray(plan['times_s']);txid=np.asarray(plan['tx_indices'])
        velocity_time=min(float(np.ptp(ts[txid==i])) for i in range(a['tx_count']))
    if w['kind']=='rect': resolution=C*tau/2
    elif w['kind']=='phase':resolution=C*tau/(2*len(w['phase_code']))
    elif w['kind']=='stepped':resolution=C/(2*w['frequency_steps']*w['frequency_step_hz'])
    else:resolution=C/(2*w['bandwidth_hz'])
    return {'wavelength_m':lam,'pulse_width_actual_s':tau,'pulse_count':n,
            'duration_s':duration,'cpi_s':duration,'sample_count':n*a['rx_count']*samples,'fasttime_sample_count':samples,
            'peak_power_w':r['peak_power_w'],'average_power_w':r['peak_power_w']*n*tau/duration,
            'duty_cycle':n*tau/duration,'blind_range_m':C*(tau+t['recovery_s'])/2,
            'receive_range_max_m':max(0,C*latest/2),'full_receive_range_min_m':C*opening/2,
            'receiver_filter_tail_s':filter_tail,'observed_time_span_s':observed_span,
            'frequency_energy_ratio':frequency_energy_ratio,
            'unambiguous_range_m':C*float(pri.min())/2,
            'unambiguous_velocity_mps':lam/(4*revisit) if t['mode']=='uniform' else None,
            'velocity_resolution_mps':lam/(2*velocity_time) if velocity_time>0 else None, 'range_resolution_m':resolution,
            'velocity_resolution_note':'均匀序列为矩形慢时间孔径基准；非均匀序列仅为实际同通道时间跨度的倒数尺度，实际主瓣/旁瓣受时序、窗与 MTI 影响',
            'range_sample_spacing_m':C/(2*w['sample_rate_hz']),
            'single_pulse_snr_db':10*math.log10(max(snr,1e-300)),
            'ideal_integrated_snr_db':10*math.log10(max(ideal,1e-300)),
            'noise_psd_w_hz':n0,'noise_sample_power_w':n0*w['sample_rate_hz'],
            'raw_bytes':raw,'estimated_memory_mib':memory/2**20,'data_rate_bytes_s':raw/duration,
            'tx_count':a['tx_count'],'rx_count':a['rx_count'],'channel_pairs':a['tx_count']*a['rx_count'],
            'frequency_span_hz':(w['frequency_steps']-1)*w['frequency_step_hz'] if w['kind']=='stepped' else 0,
            'frequency_range_period_m':C/(2*w['frequency_step_hz']) if w['kind']=='stepped' else None,
            'snr_budget_note':'单阵元增益与总传导峰值功率；步进频按各频点 λ² 累计理想能量上界，未计方向、窗、遮挡、MTI、检测及起伏损失'}


def validate_project(project):
    errors=[];warnings=[];m={}
    try:
        p=normalize_project(project)
        # Reject non-finite scalars anywhere, including imported terrain/project JSON.
        def finite(o):
            if isinstance(o,dict):return all(finite(v) for v in o.values())
            if isinstance(o,list):return all(finite(v) for v in o)
            return not isinstance(o,(int,float)) or math.isfinite(o)
        if not finite(p):raise ValueError('参数包含 NaN 或无穷大')
        w,t,r,a,pr=p['waveform'],p['timing'],p['radar'],p['array'],p['processing']
        def choice(v,vals,label):
            if v not in vals:errors.append(label+' 不支持此选项')
        def bound(v,lo,hi,label,integer=False):
            if not isinstance(v,(int,float)) or isinstance(v,bool) or not lo<=v<=hi or (integer and v!=int(v)):
                errors.append(f'{label} 必须在 {lo:g}–{hi:g}'+(' 之间且为整数' if integer else ' 之间'))
        choice(w['kind'],['rect','lfm','phase','stepped'],'波形')
        choice(t['mode'],['uniform','staggered','multi_prf'],'PRI 编排')
        choice(a['mode'],['phased','tdm','cdm'],'阵列体制')
        choice(a['geometry'],['ula','upa','custom'],'阵列结构')
        choice(pr['range_window'],['rect','hann','hamming'],'脉压窗')
        choice(pr['doppler_window'],['rect','hann'],'慢时间窗')
        choice(pr['compensation'],['estimated','known','none'],'补偿方式')
        choice(pr['mti'],[0,2,3],'MTI')
        choice(pr['cfar']['method'],['ca','os'],'CFAR')
        choice(pr['doa']['method'],['bartlett','mvdr','music'],'测角')
        for key,lo,hi in [('fc_hz',1e6,300e9),('pulse_width_s',1e-9,1.),('sample_rate_hz',1e3,2e9),('bandwidth_hz',1.,1e9),('frequency_step_hz',1.,1e9)]:bound(w[key],lo,hi,key)
        bound(w['frequency_steps'],2,256,'频点数',True)
        bound(t['pulses'],4,4096,'脉冲数',True);bound(t['groups'],2,1024,'重复组数',True)
        bound(t['pri_s'],1e-8,10.,'PRI');bound(t['recovery_s'],0.,1.,'接收恢复时间')
        bound(t['receive_start_s'],0.,10.,'接收起点');bound(t['receive_end_s'],1e-9,10.,'接收终点')
        for key,lo,hi in [('peak_power_w',1e-12,1e9),('noise_figure_db',0,80),('temperature_k',1,5000),('loss_db',0,150),('tx_gain_dbi',-50,80),('rx_gain_dbi',-50,80),('receiver_bandwidth_hz',1,2e9)]:bound(r[key],lo,hi,key)
        bound(a['tx_count'],1,16,'Tx 数',True);bound(a['rx_count'],1,64,'Rx 数',True)
        bound(a['spacing_lambda'],.01,100,'阵元间距')
        bound(pr['doppler_bins'],8,2048,'多普勒网格数',True)
        bound(pr['range_fft_factor'],1,8,'距离显示补零倍数',True)
        bound(pr['cfar']['pfa'],1e-10,.25,'名义 Pfa')
        bound(pr['cfar']['rank_fraction'],.05,1.,'OS 分位')
        bound(pr['doa']['source_count'],1,64,'MUSIC 源数',True)
        bound(pr['doa']['step_deg'],.1,10,'角度步长')
        bound(pr['doa']['diagonal_loading'],0,10,'对角加载')
        bound(pr['doa']['smoothing'],0,64,'平滑子阵长度',True)
        if pr['doa']['smoothing']==1:errors.append('空间平滑长度为 0（关闭）或至少 2')
        bound(p['resources']['memory_mib'],256,4096,'内存预算 MiB')
        for value,label in [(p['seed'],'工程种子'),(p['statistics']['seed'],'统计种子'),(p['scene']['terrain']['seed'],'地形种子')]:bound(value,0,2**32-1,label,True)
        bound(p['statistics']['trials'],2,100000,'统计试验数',True)
        bound(p['design']['max_candidates'],1,20000,'搜索候选上限',True)
        bound(p['design']['verify_top'],1,20,'验证方案数',True)
        bound(p['design']['verification_trials'],2,10000,'独立验证次数',True)
        for key,lo,hi in [('range_m',.1,1e7),('range_resolution_m',.001,1e6),('velocity_resolution_m',.0001,1e6),('max_velocity_mps',.0001,1e6),('angle_resolution_deg',.01,180),('pd',.001,.9999999),('pfa',1e-10,.25),('rcs_m2',1e-12,1e12)]:bound(p['requirements'][key],lo,hi,key)
        if not isinstance(t['pri_sequence_s'],list) or not t['pri_sequence_s'] or any(not isinstance(x,(int,float)) or x<=0 for x in t['pri_sequence_s']):errors.append('PRI 序列必须为正数列表')
        code=w['phase_code']
        if not isinstance(code,list) or not 1<=len(code)<=1024 or any(x not in (-1,1) for x in code):errors.append('二相码必须由 +1 / -1 构成，长度 1–1024')
        for key in ('velocity_search_mps','range_search_m'):
            val=pr[key]
            if not isinstance(val,list) or len(val)!=2 or val[0]>=val[1]:errors.append(key+' 必须为递增的两个边界')
        for key in ('scan_az_deg','scan_el_deg'):
            val=pr['doa'][key]
            if not isinstance(val,list) or len(val)!=2 or not -90<=val[0]<=val[1]<=90:errors.append(key+' 必须为 [-90,90] 内的递增边界')
        for key in ('guard','training'):
            val=pr['cfar'][key]
            if not isinstance(val,list) or len(val)!=2 or any(int(x)!=x or not (0 if key=='guard' else 1)<=x<=128 for x in val):errors.append('CFAR '+key+' 必须为两项非负整数，训练宽度至少 1')
        if errors:return {'valid':False,'errors':errors,'warnings':warnings,'metrics':m}
        plan=acquisition_plan(p); interval=np.asarray(plan['intervals_s']);tau=w['pulse_width_s'];fs=w['sample_rate_hz']
        if interval.min()<=tau+t['recovery_s']:errors.append('发射脉宽与恢复时间必须小于每个实际 PRI')
        if t['receive_end_s']<=t['receive_start_s']:errors.append('接收终点必须晚于起点')
        if t['receive_end_s']>interval.max()+.5/fs:errors.append('接收终点不能超过最大实际 PRI')
        if fs*tau<4:errors.append('每脉冲至少需要 4 个实际采样点')
        if w['kind']=='phase' and fs*tau/len(code)<2:errors.append('每码片至少需要 2 个采样点；请提高采样率或脉宽')
        if w['kind']=='lfm' and w['bandwidth_hz']>=.9*fs:errors.append('LFM 带宽必须小于复采样率的 90%')
        if r['receiver_bandwidth_hz']>fs:errors.append('复基带接收带宽不能超过采样率')
        if w['kind']=='lfm' and r['receiver_bandwidth_hz']<w['bandwidth_hz']:warnings.append('接收带宽小于 LFM 带宽，信号将被滤波，分辨率与能量受损')
        if a['mode']=='tdm' and t['pulses']%a['tx_count']:errors.append('TDM 脉冲数必须包含完整 Tx 轮次')
        if a['mode']=='cdm':
            if a['tx_count'] not in (2,4):errors.append('码分模式支持 2Tx 或 4Tx Walsh 编码')
            if t['pulses']%a['tx_count']:errors.append('CDM 需要完整 Walsh 编码块')
            if t['mode']!='uniform':errors.append('首版 CDM 仅支持均匀 PRI')
        if w['kind']=='stepped':
            if a['mode']!='phased' or t['mode']!='uniform':errors.append('步进频仅支持相控发射与均匀 PRI，不与 TDM/CDM/交错混用')
            if pr['mti']!=0:errors.append('步进频先在频率/时间模板中处理运动，请关闭常规 MTI')
            if pr['doa']['enabled'] and pr['doa']['method']!='bartlett':errors.append('步进频测角首版使用逐频率 Bartlett 融合')
            if min(plan['frequencies_hz'])<=0:errors.append('步进频下边界频率必须为正')
        if a['mode']!='phased' and pr['mti']!=0:errors.append('TDM/CDM 首版不将常规 MTI 跨 Tx/编码位置应用；请关闭 MTI')
        effective=t['groups'] if w['kind']=='stepped' else plan['pulse_count']//(a['tx_count'] if a['mode'] in ('tdm','cdm') else 1)
        effective-=max(0,pr['mti']-1)
        if effective<2:errors.append('每个独立慢时间序列至少需要两个有效观测')
        if pr['doppler_window']=='hann' and effective<3:errors.append('Hann 慢时间窗至少需要三个有效观测／完整编码块／重复组')
        if w['kind']=='stepped' and pr['range_window']=='hann' and w['frequency_steps']<3:errors.append('Hann 频点窗至少需要三个频点')
        if a['geometry']=='custom':
            for side in ('tx','rx'):
                v=np.asarray(a[side+'_positions_m'],float)
                if v.shape!=(a[side+'_count'],3) or not np.isfinite(v).all():errors.append(side.upper()+' 自定义坐标需为通道数 × 3 的米制矩阵')
        for name,obj in [('雷达平台',p['scene']['platform'])]+[(str(q.get('id','目标')),q) for q in p['scene']['targets']]:
            for field in ('position_m','velocity_mps'):
                if np.asarray(obj.get(field,[])).shape!=(3,):errors.append(name+' '+field+' 需要三个坐标')
            traj=obj.get('trajectory',[])
            if traj:
                times=[k['time_s'] for k in traj]
                if len(traj)<2 or any(b<=aa for aa,b in zip(times,times[1:])) or any(len(k['position_m'])!=3 for k in traj):errors.append(name+' 航迹至少两个点，时间必须严格递增')
        for target in p['scene']['targets']:
            bound(target['rcs_m2'],1e-12,1e12,'目标 RCS')
            choice(target['fluctuation'],['none','swerling1','swerling2'],'目标起伏')
        if len(p['scene']['targets'])>128:errors.append('点目标最多 128 个')
        terr=p['scene']['terrain']
        if terr['enabled']:
            bound(terr['resolution'],3,2048,'地形分析网格边长',True)
            bound(terr['max_scatterers'],1,100000,'地形散射面元预算',True)
            warnings.append('地形采用固定相干统计散射系数；DEM 几何不自动提供真实地物反射率')
            if not errors:
                from .scene import terrain_grid,terrain_scatterers
                terrain_scatterers(p,terrain_grid(p))
        if not errors:
            m=metrics(p)
            mem=p['resources']['memory_mib']
            try:
                import psutil
                mem=min(mem,psutil.virtual_memory().available*.4/2**20)
            except ImportError:pass
            m['allowed_memory_mib']=mem
            if m['estimated_memory_mib']>mem:
                errors.append(f"预计工作内存 {m['estimated_memory_mib']:.0f} MiB 超过本次预算 {mem:.0f} MiB；请显式缩小采集规模")
        # Explicitly bound all retained moving paths over the acquisition to allowed receive gates.
        if not errors and (a['mode']!='phased' or w['kind']=='stepped'):
            from .scene import pose_with_terrain,terrain_grid,terrain_scatterers,element_positions,rotation_matrix
            from .receiver import complete_delay_bounds
            from .receiver import receiver_filter
            times=np.asarray(plan['times_s'])
            gate_grid=terrain_grid(p)
            scat=list(p['scene']['targets'])
            if terr['enabled']:scat+=terrain_scatterers(p,gate_grid)
            lower,upper=complete_delay_bounds(p,plan,receiver_filter(p))
            tx_local=element_positions(a,'tx',w['fc_hz']);rx_local=element_positions(a,'rx',w['fc_hz'])
            platform_poses=[pose_with_terrain(p['scene']['platform'],float(ti),gate_grid) for ti in times]
            for target in scat:
                for event,ti in enumerate(times):
                    rp,att=platform_poses[event];tp,_=pose_with_terrain(target,float(ti),gate_grid)
                    rot=rotation_matrix(att)
                    active=tx_local[[plan['tx_indices'][event]]] if a['mode']=='tdm' else tx_local
                    rt=np.linalg.norm(tp-(rp+active@rot.T),axis=1);rr=np.linalg.norm(tp-(rp+rx_local@rot.T),axis=1)
                    minimum=(float(rt.min())+float(rr.min()))/C;maximum=(float(rt.max())+float(rr.max()))/C
                    if minimum<lower-1e-8/fs or maximum>upper+1e-8/fs:
                        errors.append('当前体制要求所有保留 Tx–Rx 回波在对应接收窗内完整接收，含发射/恢复盲区、ADC 起点和接收 FIR 尾部；请调整窗口或场景')
                        break
                if errors:break
        if errors:return {'valid':False,'errors':errors,'warnings':warnings,'metrics':m}
        m=metrics(p)
        from .arrays import array_metrics
        m.update(array_metrics(a,w['fc_hz']))
        if a['spacing_lambda']>.5:warnings.append('阵元间距超过半波长，扫描视场内可能出现栅瓣')
        if t['mode']!='uniform':warnings.append('非均匀时序的模糊响应由实际事件表决定，不提供单一均匀 PRF 无模糊速度保证')
        if pr['ambiguity_search']:warnings.append('距离–速度解模糊仅限指定搜索域，保留多解；有限网格不是全局唯一性证明')
        if pr['doa']['method']=='music':warnings.append('MUSIC 使用用户源数；同距同速相干目标可能秩亏，平滑须满足 ULA 条件')
        if a['mode']=='cdm':warnings.append('CDM 有效速度范围按编码块重访定义，超界速度可产生码间串扰')
        mem=p['resources']['memory_mib']
        try:
            import psutil
            mem=min(mem,psutil.virtual_memory().available*.4/2**20)
        except ImportError:pass
        m['allowed_memory_mib']=mem
        if m['estimated_memory_mib']>mem:errors.append(f"预计工作内存 {m['estimated_memory_mib']:.0f} MiB 超过本次预算 {mem:.0f} MiB；请显式缩小采集规模")
        if pr['cfar']['guard'][1]+pr['cfar']['training'][1]>=pr['doppler_bins']/2:errors.append('多普勒网格不足以容纳 CFAR 训练与保护窗口')
        warnings.append('理想分辨率与预算上界不等于检测概率或硬件性能；实际结果包含窗、遮挡和杂波影响')
    except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError) as exc:
        errors.append('参数结构或数值无效：'+str(exc))
    return {'valid':not errors,'errors':errors,'warnings':list(dict.fromkeys(warnings)),'metrics':m}


def presets():
    base=default_project();result={}
    def add(key,name,description,fn):
        q=copy.deepcopy(base);q['name']=name;fn(q);result[key]={'project':q,'name':name,'description':description}
    add('baseline','LFM 多目标基准','同一回波链上的距离、速度、角度与 CFAR',lambda p:None)
    def rect(p):p['waveform'].update(kind='rect',pulse_width_s=1e-6);p['processing']['range_window']='rect'
    add('rect','常规短脉冲','常规脉冲的脉宽、盲区与距离分辨率',rect)
    def phase(p):p['waveform'].update(kind='phase',pulse_width_s=6.5e-6);p['timing']['recovery_s']=1e-6
    add('barker','Barker-13 脉压','脉内二相码、匹配滤波和旁瓣',phase)
    def stagger(p):p['timing']['mode']='staggered';p['processing'].update(ambiguity_search=True,velocity_search_mps=[-150,150],range_search_m=[500,18000])
    add('staggered','交错 PRI 与模糊搜索','真实事件时刻的非均匀多普勒与有界候选',stagger)
    def sf(p):
        p['waveform'].update(kind='stepped',frequency_steps=16,frequency_step_hz=1e6)
        p['timing'].update(groups=8,pulses=128)
        p['processing'].update(velocity_search_mps=[-5,5],doppler_bins=65,range_fft_factor=4,range_search_m=[2500,3500])
        p['scene']['targets']=p['scene']['targets'][:1]
        p['scene']['targets'][0]['velocity_mps']=[3.,0.,0.]
    add('stepped','步进频运动补偿','频点 × 重复组，估计速度后合成高分辨距离',sf)
    def sf_alias(p):
        sf(p);p['scene']['targets'][0]['velocity_mps']=[12.,0.,0.]
        p['processing'].update(velocity_search_mps=[-30,30],doppler_bins=129)
    add('stepped-alias','步进频速度歧义','超出同频组重访速度范围的失效与候选实验',sf_alias)
    def multi(p):
        p['timing'].update(mode='multi_prf',pri_sequence_s=[80e-6,100e-6])
        p['processing']['ambiguity_search']=True
    add('multi-prf','多 PRF 分组','两个独立均匀脉冲段的真实时间表与联合候选',multi)
    def tdm(p):p['array'].update(mode='tdm',tx_count=2,rx_count=4);p['processing']['velocity_search_mps']=[-40,40]
    add('tdm','2Tx4Rx TDM-MIMO','时分发射、运动相位补偿与虚拟阵列',tdm)
    def cdm(p):tdm(p);p['array']['mode']='cdm'
    add('cdm','2Tx4Rx Walsh CDM','复合回波、速度补偿、实际 Hadamard 解码',cdm)
    def terrain(p):
        p['scene']['terrain'].update(enabled=True,resolution=9,max_scatterers=128,extent_m=4000,hill_height_m=60)
        p['scene']['platform'].update(position_m=[100,0,500],velocity_mps=[40,0,0])
        p['scene']['targets'][0].update(position_m=[3000,300,400],velocity_mps=[0,0,0])
        p['scene']['targets']=p['scene']['targets'][:1]
        p['timing']['pulses']=32;p['processing']['doppler_bins']=64
    add('terrain','运动平台与丘陵杂波','地形遮挡、面元散射和平台运动产生的地杂波',terrain)
    return result
