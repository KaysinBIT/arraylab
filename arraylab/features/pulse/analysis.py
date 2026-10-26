"""Independent-trial performance evaluation and bounded engineering search."""
from __future__ import annotations
import copy
import itertools
import math
import time
import numpy as np
from scipy.stats import beta
from scipy.optimize import linear_sum_assignment
from .config import normalize_project, validate_project, metrics, project_hash, C


def binomial_interval(successes, trials, confidence=.95):
    """Exact equal-tailed Clopper-Pearson interval, including zero/all counts."""
    if trials<1 or not 0<=successes<=trials:raise ValueError('无效二项计数')
    alpha=1-confidence
    lo=0. if successes==0 else float(beta.ppf(alpha/2,successes,trials-successes+1))
    hi=1. if successes==trials else float(beta.ppf(1-alpha/2,successes+1,trials-successes))
    return [lo,hi]


def _check(cancel):
    if cancel and cancel():raise InterruptedError('用户取消')


def _cell(arrays,range_m,velocity_mps):
    r=np.asarray(arrays['ranges_m']);v=np.asarray(arrays['velocities_mps'])
    if not (np.isfinite(range_m) and np.isfinite(velocity_mps) and r.min()<=range_m<=r.max() and v.min()<=velocity_mps<=v.max()):
        raise ValueError('统计参考单元超出实际距离/速度搜索网格；不能将其夹取到边界后报告 Pd/Pfa')
    return int(np.argmin(abs(v-velocity_mps))),int(np.argmin(abs(r-range_m)))


def truth_range_velocity(project,grid=None):
    """Evaluation-only geometry; never passed into an estimator."""
    from .scene import pose_with_terrain,velocity_at
    platform=project['scene']['platform'];rp,_=pose_with_terrain(platform,0,grid)
    rv=velocity_at(platform,0,grid);truth=[]
    for target in project['scene']['targets']:
        tp,_=pose_with_terrain(target,0,grid);delta=tp-rp;distance=float(np.linalg.norm(delta))
        if distance<=0:raise ValueError('目标与平台重合，雷达方程和径向速度无定义')
        velocity=float(np.dot(velocity_at(target,0,grid)-rv,delta/distance))
        truth.append((distance,velocity))
    return truth


def associate_detections(truth,detections,range_gate,velocity_gate):
    """Fixed-gate one-to-one association. A detection cannot count twice."""
    if not truth:return set()
    count=len(truth);nd=len(detections)
    # Each target has access to an unmatched dummy; real matches cost <2.
    cost=np.full((count,nd+count),3.,float)
    for i,(distance,velocity) in enumerate(truth):
        for j,d in enumerate(detections):
            er=abs(d['range_m']-distance)/range_gate;ev=abs(d['velocity_mps']-velocity)/velocity_gate
            cost[i,j]=er**2+ev**2 if er<=1 and ev<=1 else 1e6
    rows,cols=linear_sum_assignment(cost)
    return {int(i) for i,j in zip(rows,cols) if j<nd and cost[i,j]<=2+1e-12}


def design_reference_scene(project,requirements,velocity):
    """Place an explicit benchmark at the actual initial beam direction."""
    from .scene import terrain_grid,pose_with_terrain,velocity_at,rotation_matrix
    grid=terrain_grid(project);platform=project['scene']['platform']
    rp,att=pose_with_terrain(platform,0,grid);rv=velocity_at(platform,0,grid)
    az,el=np.deg2rad(project['array']['pointing_deg'])
    direction=rotation_matrix(att)@np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    project['scene']['targets']=[{'id':'设计验证目标','position_m':(rp+requirements['range_m']*direction).tolist(),
        'velocity_mps':(rv+velocity*direction).tolist(),'height_mode':'absolute',
        'rcs_m2':requirements['rcs_m2'],'fluctuation':'none','trajectory':[]}]
    return project


def statistics(project,progress=None,cancel=None):
    from .engine import simulate
    p=normalize_project(project); cfg=p['statistics'];n=int(cfg['trials'])
    if not p['radar']['noise_enabled']:raise ValueError('检测统计要求启用接收热噪声')
    if not p['scene']['targets']:raise ValueError('有目标统计需要至少一个目标')
    idx=int(cfg['target_index'])
    if not 0<=idx<len(p['scene']['targets']):raise ValueError('统计目标索引超出范围')
    # Evaluation truth is used only after the estimator has returned.
    from .scene import terrain_grid
    grid=terrain_grid(p)
    truth=truth_range_velocity(p,grid);dist,velocity=truth[idx]
    budget=metrics(p)
    rr=max(budget['range_resolution_m'],C/(2*p['waveform']['sample_rate_hz']))
    vv=2*budget['velocity_resolution_mps']
    pd_count=0;fa_count=0;scene_count=0;records=[]
    seeds=np.random.SeedSequence(int(cfg['seed'])).spawn(2*n)
    started=time.perf_counter();ref=None
    for i in range(n):
        _check(cancel)
        h0=copy.deepcopy(p);h0['scene']['targets']=[];h0['seed']=int(seeds[2*i].generate_state(1)[0])
        h0['processing']['doa']['enabled']=False;h0['processing']['ambiguity_search']=False
        s0,a0=simulate(h0,cancel=cancel)
        vi,ri=_cell(a0,cfg['cut_range_m'],cfg['cut_velocity_mps'])
        threshold=np.asarray(a0['cfar_threshold'])
        if not np.isfinite(threshold[vi,ri]):raise ValueError('固定虚警 CUT 落在无效或 CFAR 边界单元；请修改 CUT')
        fa=bool(np.asarray(a0['cfar_mask'])[vi,ri]);fa_count+=fa
        _check(cancel)
        h1=copy.deepcopy(p);h1['seed']=int(seeds[2*i+1].generate_state(1)[0])
        h1['processing']['doa']['enabled']=False;h1['processing']['ambiguity_search']=False
        s1,a1=simulate(h1,cancel=cancel)
        vt,rt=_cell(a1,dist,velocity)
        if not np.isfinite(np.asarray(a1['cfar_threshold'])[vt,rt]):raise ValueError('目标固定 CUT 落在 CFAR 无效区，不能报告 Pd')
        hit=bool(np.asarray(a1['cfar_mask'])[vt,rt]);pd_count+=hit
        # Scene association gate is fixed before every trial, not tuned to success.
        matched=idx in associate_detections(truth,s1.get('detections',[]),rr,vv)
        scene_count+=matched
        records.append([i,int(hit),int(fa),int(matched)])
        ref=a1
        if progress:progress((i+1)/n,f'独立 H0/H1 试验 {i+1}/{n}')
    result={'kind':'statistics','trials':n,
        'pd':{'estimate':pd_count/n,'successes':pd_count,'trials':n,'ci95':binomial_interval(pd_count,n)},
        'pfa':{'estimate':fa_count/n,'successes':fa_count,'trials':n,'ci95':binomial_interval(fa_count,n),
               'zero_count_upper95':float(1-.05**(1/n)) if fa_count==0 else None},
        'nominal_pfa':p['processing']['cfar']['pfa'],
        'scene_detection_rate':{'estimate':scene_count/n,'successes':scene_count,'trials':n,'ci95':binomial_interval(scene_count,n)},
        'conditions':{'noise_cut':{'range_m':cfg['cut_range_m'],'velocity_mps':cfg['cut_velocity_mps']},
                      'target_cut':{'range_m':dist,'velocity_mps':velocity},'target_index':idx,'seed':cfg['seed'],
                      'clutter_retained_in_h0':p['scene']['terrain']['enabled'],'association_range_gate_m':rr,
                      'association_velocity_gate_mps':vv,'association_method':'fixed-gate global one-to-one assignment',
                      'pd_definition':'指定真值参考 CUT 的越阈频率，包含其他目标/杂波贡献，不是已证明的目标身份识别率'},
        'warnings':['每次独立运行仅使用一个固定 CUT 统计虚警，不将相关图像像素当独立试验',
                    '固定 CUT 的 Pd 可受同单元其他目标和杂波影响；场景关联率另用固定门限的一对一分配',
                    '有限试验及此场景不证明所有环境性能；极小 Pfa 需要足够试验或专门统计分析'],
        'timings':{'total_s':time.perf_counter()-started}}
    arrays={'trial_results':np.asarray(records,dtype=np.int64)}
    if ref is not None:
        arrays.update({k:v for k,v in ref.items() if k in ('rd_power','ranges_m','velocities_mps','cfar_mask','cfar_threshold','waveform_real','waveform_imag','waveform_time_s')})
    return result,arrays


def inverse_design(project,progress=None,cancel=None):
    from .engine import simulate
    p=normalize_project(project);req=p['requirements'];cfg=p['design'];started=time.perf_counter()
    if p['array']['geometry']=='custom':raise ValueError('反向设计生成 ULA/UPA 候选；请先选择线阵或面阵')
    kind=p['waveform']['kind']
    if kind=='stepped':
        fields=[('waveform','frequency_steps'),('waveform','frequency_step_hz'),('waveform','pulse_width_s'),
                ('timing','pri_s'),('timing','groups'),('radar','peak_power_w'),('array','rx_count')]
    else:
        fields=([('waveform','bandwidth_hz')] if kind=='lfm' else [])+[
            ('waveform','pulse_width_s'),('timing','pri_s'),('timing','pulses'),('radar','peak_power_w'),('array','rx_count')]
    axes=[]
    for section,key in fields:
        vals=cfg.get(key,[])
        if not isinstance(vals,list) or not vals or len(vals)>100:raise ValueError('候选轴 '+key+' 需要 1–100 个明确值')
        axes.append(vals)
    total=math.prod(map(len,axes));cap=int(cfg['max_candidates'])
    # Evenly sample product indices when capped, rather than silently taking a
    # lexicographic prefix that misses all high-power designs.
    selected=np.unique(np.linspace(0,total-1,min(cap,total),dtype=np.int64)).tolist()
    survivors=[];reasons={};screened=0
    for flat_index in selected:
        value_indices=[0]*len(axes);remainder=int(flat_index)
        for axis_index in range(len(axes)-1,-1,-1):
            remainder,value_indices[axis_index]=divmod(remainder,len(axes[axis_index]))
        values=[axis[index] for axis,index in zip(axes,value_indices)]
        _check(cancel);q=copy.deepcopy(p)
        for (section,key),value in zip(fields,values):q[section][key]=value
        # Preserve the requested PRI pattern ratios when sweeping its base scale.
        if q['timing']['mode']!='uniform':
            ratio=q['timing']['pri_s']/p['timing']['pri_s'];q['timing']['pri_sequence_s']=[x*ratio for x in p['timing']['pri_sequence_s']]
        q['timing']['receive_end_s']=min(p['timing']['receive_end_s'],q['timing']['pri_s'])
        q['processing']['cfar']['pfa']=req['pfa']
        design_reference_scene(q,req,req['max_velocity_mps'])
        val=validate_project(q);screened+=1;fails=[]
        if not val['valid']:fails=['配置约束']
        else:
            m=val['metrics']
            if m['range_resolution_m']>req['range_resolution_m']:fails.append('距离分辨率')
            if m['velocity_resolution_mps']>req['velocity_resolution_m']:fails.append('速度分辨率')
            if m['receive_range_max_m']<req['range_m']:fails.append('接收窗')
            if m['blind_range_m']>=req['range_m']:fails.append('距离盲区')
            if m['full_receive_range_min_m']>req['range_m']:fails.append('完整接收下界')
            if m['unambiguous_velocity_mps'] is not None and m['unambiguous_velocity_mps']<req['max_velocity_mps']:fails.append('速度范围')
            if not q['processing']['velocity_search_mps'][0]<=-req['max_velocity_mps']<=req['max_velocity_mps']<=q['processing']['velocity_search_mps'][1]:fails.append('处理速度搜索域')
            # Aperture is only a conventional beamwidth proxy, never claimed as
            # a measured high-resolution estimator guarantee.
            from .scene import element_positions
            coord=element_positions(q['array'],'rx',q['waveform']['fc_hz'])
            if q['array']['mode']!='phased':
                tx=element_positions(q['array'],'tx',q['waveform']['fc_hz'])
                coord=(tx[:,None,:]+coord[None,:,:]).reshape(-1,3)
            aperture=float(np.ptp(coord[:,1]))/(C/q['waveform']['fc_hz'])
            hpbw=math.degrees(.886/aperture) if aperture else 180.
            m['angle_design_hpbw_proxy_deg']=hpbw
            if hpbw>req['angle_resolution_deg']:fails.append('常规角度孔径代理')
        if fails:
            for reason in fails:reasons[reason]=reasons.get(reason,0)+1
        else:survivors.append((q,m))
        if progress:progress(.2*screened/min(total,cap),f'解析筛选 {screened}/{min(total,cap)}')
    survivors.sort(key=lambda z:(z[1]['average_power_w'],z[1]['duration_s'],z[1]['raw_bytes']))
    chosen=survivors[:int(cfg['verify_top'])];candidates=[];lastarrays={};array_source=None
    for i,(q,m) in enumerate(chosen):
        _check(cancel)
        # Verification target is a declared benchmark at the specified design
        # requirement, not a hidden target consulted by any estimator.
        q['seed']=(int(p['seed'])+104729*(i+1))%(2**32)
        q['statistics'].update(trials=int(cfg['verification_trials']),target_index=0,
            cut_range_m=req['range_m'],cut_velocity_mps=req['max_velocity_mps'],seed=(q['seed']+99991)%(2**32))
        try:
            val=validate_project(q)
            if not val['valid']:raise ValueError('; '.join(val['errors']))
            summary,arrays=simulate(q,cancel=cancel);lastarrays=arrays
            array_source={'candidate_rank':i+1,'statistics_seed':q['statistics']['seed'],'config_hash':project_hash(q)}
            scenarios=[]
            for case,vmax in enumerate([-req['max_velocity_mps'],req['max_velocity_mps']]):
                case_project=design_reference_scene(copy.deepcopy(q),req,vmax)
                case_project['statistics'].update(cut_velocity_mps=vmax,seed=(q['statistics']['seed']+case*15485863)%(2**32))
                case_validation=validate_project(case_project)
                if not case_validation['valid']:raise ValueError('; '.join(case_validation['errors']))
                stat,_=statistics(case_project,cancel=cancel,progress=lambda f,msg:progress(.2+.8*(i+(case+f)/2)/max(1,len(chosen)),f'候选 {i+1} / 速度端点 {case+1}: '+msg) if progress else None)
                # Four confidence statements (Pd/Pfa at two endpoints):
                # Bonferroni gives at least 95% simultaneous family coverage.
                confidence=1-.05/4
                pd_ci=binomial_interval(stat['pd']['successes'],stat['trials'],confidence)
                fa_ci=binomial_interval(stat['pfa']['successes'],stat['trials'],confidence)
                scenarios.append({'velocity_mps':vmax,'pd':stat['pd'],'pfa':stat['pfa'],
                    'pd_joint_ci':pd_ci,'pfa_joint_ci':fa_ci,'seed':case_project['statistics']['seed']})
            pd_ci=[min(s['pd_joint_ci'][j] for s in scenarios) for j in (0,1)]
            fa_ci=[max(s['pfa_joint_ci'][j] for s in scenarios) for j in (0,1)]
            q['statistics']['seed']=scenarios[-1]['seed']
            verification={'pd_estimate':min(s['pd']['estimate'] for s in scenarios),'pd_ci95':pd_ci,
                           'pfa_estimate':max(s['pfa']['estimate'] for s in scenarios),'pfa_ci95':fa_ci,
                           'trials':2*int(cfg['verification_trials']),'trials_per_endpoint':int(cfg['verification_trials']),
                           'seed':q['statistics']['seed'],'scenarios':scenarios,
                           'confidence_method':'95% simultaneous Bonferroni bounds across Pd/Pfa and two velocity endpoints',
                           'signal_detections':summary.get('detections',[])}
            passed=pd_ci[0]>=req['pd'] and fa_ci[1]<=req['pfa']
            reason='解析约束及独立统计置信界通过' if passed else '解析约束通过；有限独立试验尚未证明 Pd/Pfa 要求'
            candidates.append({'rank':i+1,'project':q,'metrics':m,'passed':passed,'reason':reason,
                               'validation':val,'verification':verification})
        except InterruptedError:raise
        except ValueError as exc:
            candidates.append({'rank':i+1,'project':q,'metrics':m,'passed':False,'reason':str(exc),'validation':{'valid':False},'verification':{}})
    if progress:progress(1.,'候选搜索与独立验证完成')
    return {'kind':'inverse','candidates':candidates,'screened_count':screened,'rejected_count':screened-len(survivors),
            'analytic_survivors':len(survivors),'rejection_reasons':reasons,'search_exhausted':screened==total,
            'total_combinations':total,'requested_requirements':req,'searched_parameters':[key for section,key in fields],
            'assumptions':['有界离散搜索，未找到方案不等于不可实现',
                           '角度约束采用实际阵列 y 轴孔径的方位 HPBW 代理；真实测角另由处理实验验收',
                           '验证使用实际平台位置/姿态的初始波束中心、需求距离/RCS及正负最大速度两个端点；当前地形及平台保留',
                           '统计通过只覆盖这些已声明端点条件，不证明全部连续速度或角域；同时置信界使用 Bonferroni 校正',
                           '候选只有在 Pd 下界和 Pfa 上界同时满足要求时才标为通过'],
            'array_source':array_source,'timings':{'total_s':time.perf_counter()-started}},lastarrays
