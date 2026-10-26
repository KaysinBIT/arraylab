"""Plots and tables over saved EquationLab results; no numerical reruns."""
import math
import numpy as np
from arraylab.shared.plotting.surface import clear_figure
from arraylab.shared.plotting.metrics import draw_metric_footer


def _number(value, digits=6):
    return '—' if value is None else f'{float(value):.{digits}g}'


def _budget(result):
    return result if result.get('kind')=='budget' else result.get('budget',{})


def available_views(result, arrays):
    views=['budget']
    if 'rd_power' in arrays:views.append('rd')
    if result.get('kind')=='snr':views.append('snr')
    if result.get('kind') in ('statistics','calibration','distance','roc'):views.append('statistics')
    if result.get('detections'):views.append('detections')
    return views


def _empty(axis,text):
    axis.set_axis_off();axis.text(.5,.5,text,ha='center',va='center',wrap=True,transform=axis.transAxes)


def draw(figure,result,arrays,view='budget',selection=0):
    clear_figure(figure);figure.set_layout_engine('constrained',w_pad=.08,h_pad=.08)
    axis=figure.add_subplot(111)
    result=result or {};arrays=arrays or {};kind=result.get('kind')
    if not result:
        _empty(axis,'设置工程后，先核对系统预算，再显式运行仿真或验证。');return figure
    note='结果来自已保存输入；切换视图不重新计算。'
    if view=='budget':
        budget=_budget(result)
        if not budget.get('valid'):
            _empty(axis,'；'.join(budget.get('errors',[])) or '尚无有效预算。');return figure
        curve=budget['curve'];axis.plot(curve['range_m'],curve['snr_db'],color='#168b9c',label='独立能量方程')
        axis.axhline(budget['threshold_snr_db'],color='#bd8432',ls='--',label='SNR要求含裕量')
        axis.axvline(budget['range_m'],color='#6b60b6',ls=':',label='预算距离')
        axis.set(xlabel='距离 / m',ylabel='处理后参考 SNR / dB',title='自由空间能量参考与设计裕量')
        axis.legend(fontsize=8);axis.grid(alpha=.25)
        note=f"预算距离 {_number(budget['range_m'])} m · SNR {_number(budget['output_snr_db'])} dB · 裕量 {_number(budget['snr_margin_db'])} dB"
    elif view=='rd':
        if 'rd_power' not in arrays:
            _empty(axis,'先运行回波仿真、SNR校核或统计，获取真实处理数组。');return figure
        power=np.asarray(arrays['rd_power']);ranges=np.asarray(arrays['ranges_m']);velocity=np.asarray(arrays['velocities_mps'])
        if power.ndim!=2 or power.shape!=(velocity.size,ranges.size) or not power.size:
            raise ValueError('距离–速度数组与坐标轴不一致。')
        vs=max(1,math.ceil(power.shape[0]/128));rs=max(1,math.ceil(power.shape[1]/512))
        shown=10*np.log10(np.maximum(power[::vs,::rs],1e-300));maximum=float(np.nanmax(shown))
        mesh=axis.pcolormesh(ranges[::rs],velocity[::vs],shown,shading='auto',cmap='viridis',
                            vmin=maximum-60,vmax=maximum)
        figure.colorbar(mesh,ax=axis,label='处理功率 / dBW')
        detections=result.get('detections',[])
        if detections:
            axis.scatter([p.get('apparent_range_m',p['range_m']) for p in detections],
                         [p['velocity_mps'] for p in detections],facecolors='none',edgecolors='white',s=30,label='检测点')
        axis.set(xlabel='表观距离 / m',ylabel='速度（远离为正）/ m/s',title='真实复信号处理的距离–速度功率')
        note=f'显示步长 {vs}×{rs}；原始数组与检测网格完整保留。FMCW检测表另列耦合校正后的距离。'
    elif view=='snr':
        if kind!='snr':
            _empty(axis,'先运行“固定CUT的SNR验证”；普通仿真不自动构成方程闭合验证。');return figure
        fields=[('energy_reference_snr_db','独立能量方程'),('theoretical_snr_db','实际处理系数理论值'),('measured_snr_db','独立噪声试验')]
        values=[(name,result.get(key)) for key,name in fields if result.get(key) is not None]
        bars=axis.bar([p[0] for p in values],[p[1] for p in values],color=['#168b9c','#6b60b6','#bd8432'][:len(values)])
        axis.bar_label(bars,fmt='%.3f');axis.set(ylabel='SNR / dB',title='预先指定CUT的独立方程 / 实际处理对照')
        axis.grid(axis='y',alpha=.25)
        note=f"噪声试验 {result['noise_trials']} 次 · 实测差 {_number(result['error_db'])} dB · 噪声标定 {'通过' if result['noise_calibration_passed'] else '未证明'} · 独立方程 {'通过' if result['energy_equation_passed'] else '未闭合/不适用'}"
    elif view=='statistics':
        if kind in ('distance','roc'):
            points=[p for p in result.get('points',[]) if p.get('pd') and p.get('pfa')]
            if not points:
                _empty(axis,'没有可用统计点；查看表格中的不可观测、不可用或证据不足原因。');return figure
            x=[p['range_m'] if kind=='distance' else p['pfa']['estimate'] for p in points]
            y=np.array([p['pd']['estimate'] for p in points])
            lower=np.array([p['pd']['lower'] for p in points]);upper=np.array([p['pd']['upper'] for p in points])
            axis.errorbar(x,y,yerr=[np.maximum(0,y-lower),np.maximum(0,upper-y)],fmt='o-',capsize=3,color='#168b9c')
            axis.set(xlabel='采样距离 / m' if kind=='distance' else '实测单CUT Pfa',ylabel='实测单CUT Pd',ylim=(-.03,1.03),
                     title='离散距离条件的统计结果' if kind=='distance' else '经验ROC（逐点置信区间）');axis.grid(alpha=.25)
            note='误差线是逐点二项置信区间；只覆盖已采样条件，零虚警仍有非零上界。'
        elif kind=='statistics':
            fields=[('pd','固定CUT Pd'),('pfa','固定CUT Pfa'),('association','目标关联率'),('ghost','可分辨鬼影率')]
            values=[(name,result[key]) for key,name in fields if key in result]
            x=np.arange(len(values));y=np.array([v['estimate'] for _,v in values])
            lower=np.array([v['lower'] for _,v in values]);upper=np.array([v['upper'] for _,v in values])
            axis.errorbar(x,y,yerr=[y-lower,upper-y],fmt='o',capsize=6,color='#168b9c')
            axis.set_xticks(x,[name for name,_ in values]);axis.set(ylim=(-.03,1.03),ylabel='频率 / 概率',title='独立H0/H1统计与置信区间')
            axis.grid(axis='y',alpha=.25);note='固定CUT检出率、关联率和鬼影率具有不同定义，不能互相替代。'
        elif kind=='calibration':
            scores=np.asarray(arrays.get('calibration_scores',[]))
            if not scores.size:_empty(axis,'没有校准统计数组。');return figure
            axis.hist(scores,bins=min(40,max(5,int(math.sqrt(scores.size)))),color='#168b9c')
            axis.axvline(result['factor'],color='#bd8432',ls='--',label='候选门限因子');axis.legend(fontsize=8)
            axis.set(xlabel='预定CUT检测统计量',ylabel='独立试验计数',title='独立H0门限校准')
            note=f"{result['trials']} 次 · {'允许应用，但尚未独立验收' if result['ready'] else '尾部样本不足，不允许应用'}"
        else:
            _empty(axis,'显式运行H0校准、检测统计、ROC或距离扫描后查看。');return figure
    elif view=='detections':
        detections=result.get('detections',[])
        if detections:
            axis.scatter([p['range_m'] for p in detections],[p['velocity_mps'] for p in detections],
                         c=[p.get('cell_snr_db',0) for p in detections],cmap='viridis')
            axis.set(xlabel='估计距离 / m',ylabel='速度 / m/s',title='数据驱动检测点（可能包含虚警与旁瓣）');axis.grid(alpha=.25)
        else:_empty(axis,'当前结果没有通过门限的检测点；这不等于场景中没有目标。')
        note='FMCW表中距离按估计速度修正耦合；图中点不是场景真值列表。'
    else:raise ValueError('未知雷达方程视图：'+str(view))
    draw_metric_footer(figure,[note,'单站远场合成模型；不代表硬件或连续条件性能保证。'])
    return figure


def table(result, arrays, view='budget', selection=0):
    result=result or {};kind=result.get('kind')
    if view=='budget':
        b=_budget(result)
        if not b.get('valid'):return ['检查'],[[e] for e in b.get('errors',[])],'尚无有效预算。'
        rows=[[p['name'],_number(p['db']),p['unit']] for p in b['ledger']]
        fields=[('received_power_dbm','接收功率','dBm'),('output_snr_db','能量参考SNR','dB'),
                ('snr_margin_db','SNR裕量','dB'),('max_range_m','自由空间能量参考距离','m'),
                ('effective_time_s','全部事件有效观测时间','s'),('average_power_w','平均总发射功率','W')]
        rows += [[name,_number(b.get(key)),unit] for key,name,unit in fields]
        inverse=b.get('inverse') or {}
        for key,name,unit in [('required_power_w','所需传导功率','W'),('required_tx_gain_dbi','所需Tx增益','dBi'),
                              ('required_rx_gain_dbi','所需Rx增益','dBi'),('required_equal_gain_dbi','两侧相同增益','dBi'),
                              ('minimum_rcs_m2','最低RCS','m²'),('required_events','所需完整物理事件数','个')]:
            rows.append([name,_number(inverse.get(key)),unit])
        return ['分项 / 指标','数值','单位'],rows,'能量参考与可观测性分别判断；增大功率不能恢复窗口外回波。'
    if view in ('detections','rd'):
        return ['表观距离/m','估计距离/m','速度/m/s','功率/dBW','单元SNR/dB'],[
            [_number(p.get('apparent_range_m',p.get('range_m'))),_number(p.get('range_m')),
             _number(p.get('velocity_mps')),_number(p.get('power_dbw')),_number(p.get('cell_snr_db'))]
            for p in result.get('detections',[])],'检测来自接收数据；可能包含虚警、旁瓣或错误解码。'
    if view=='snr':
        keys=[('energy_reference_snr_db','独立能量方程SNR'),('theoretical_snr_db','实际系数理论SNR'),
              ('measured_snr_db','实测SNR'),('error_db','实测相对处理系数差'),('discrete_loss_db','能量参考与离散处理差'),
              ('sinr_db','含其余目标/杂波的SINR')]
        rows=[[name,_number(result.get(key)),'dB'] for key,name in keys] if kind=='snr' else []
        return ['指标','数值','单位'],rows,'固定预先指定CUT；没有用最大峰或真值位置筛选验收输出。'
    if kind in ('distance','roc'):
        return ['距离/m' if kind=='distance' else '门限/dB','Pd','Pd下界','Pd上界','Pfa','Pfa下界','Pfa上界','状态/原因'],[
            [_number(p.get('range_m',p.get('threshold_db'))),*[_number((p.get('pd') or {}).get(k)) for k in ('estimate','lower','upper')],
             *[_number((p.get('pfa') or {}).get(k)) for k in ('estimate','lower','upper')],
             p.get('status','')+'；'+'；'.join(p.get('reasons',[]))] for p in result.get('points',[])], '只保证所列离散点；空值保留为未定义。'
    if kind=='statistics':
        return ['口径','次数','计数','估计','下界','上界'],[[label,result[key]['trials'],result[key]['count'],
            *[_number(result[key][field]) for field in ('estimate','lower','upper')]] for key,label in
            [('pd','固定CUT Pd'),('pfa','固定CUT Pfa'),('association','关联率'),('ghost','鬼影率'),('scene_false_alarm','整图虚警事件')]],'零事件不等于零概率；置信区间和名义Pfa分别报告。'
    if kind=='calibration':
        return ['指标','值'],[['独立试验数',result['trials']],['期望尾部事件',_number(result['expected_tail_events'])],
            ['校准因子',_number(result['factor'])],['允许应用',result['ready']],['已独立验收',result['verified']],
            ['推荐至少次数',result['recommended_trials']]],'仅ready且配置/CUT指纹一致时可应用；不是独立验收通过。'
    return ['说明'],[],'先运行对应的显式验证操作。'


render_figure = draw
result_table = table
