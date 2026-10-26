"""Small view arrays and scene geometry; full result arrays stay untouched."""
import numpy as np
from .config import normalize_project
from .metadata import VIEWS


def display_result(summary,arrays):
    from arraylab.shared.simulation_io import clean_json
    out={'summary':summary,'detections':summary.get('detections',[])}
    power=arrays.get('rd_power');rr=arrays.get('ranges_m');vv=arrays.get('velocities_mps')
    if power is not None and np.asarray(power).ndim==2 and rr is not None and vv is not None:
        power=np.asarray(power);rr=np.asarray(rr);vv=np.asarray(vv)
        if power.shape!=(len(vv),len(rr)):raise ValueError('距离–速度数组与轴尺寸不一致。')
        ridx=np.linspace(0,len(rr),min(len(rr),640)+1,dtype=int)
        vidx=np.linspace(0,len(vv),min(len(vv),256)+1,dtype=int)
        down=np.maximum.reduceat(np.maximum.reduceat(power,vidx[:-1],axis=0),ridx[:-1],axis=1)
        out.update(ranges_m=rr[ridx[:-1]],velocities_mps=vv[vidx[:-1]],power_db=10*np.log10(np.maximum(down,1e-300)),
            range_profile_db=10*np.log10(np.maximum(power.max(axis=0)[ridx[:-1]],1e-300)),
            doppler_profile_db=10*np.log10(np.maximum(power.max(axis=1)[vidx[:-1]],1e-300)),
            display_note='显示采用分块峰值聚合；检测与数组导出保留完整计算网格')
        if 'cfar_threshold' in arrays:
            out['threshold_db']=10*np.log10(np.maximum(np.asarray(arrays['cfar_threshold'])[np.ix_(vidx[:-1],ridx[:-1])],1e-300))
    if 'waveform_time_s' in arrays:
        wt=np.asarray(arrays['waveform_time_s']);step=max(1,len(wt)//2048)
        out['waveform']={'time_s':wt[::step],'real':np.asarray(arrays['waveform_real'])[::step],
                         'imag':np.asarray(arrays['waveform_imag'])[::step]}
    if summary.get('angles'):out['angles']=summary['angles']
    for keys in [('angle_azimuth_deg','angle_elevation_deg','angle_spectrum_db'),
                 ('azimuth_deg','elevation_deg','spectrum_db'),('doa_azimuth_deg','doa_elevation_deg','doa_spectrum_db')]:
        if all(key in arrays for key in keys):
            out['angles']={key:arrays[name] for key,name in zip(('azimuth_deg','elevation_deg','spectrum_db'),keys)};break
    return clean_json(out,complex_style='object')


def scene_preview(configuration,time_s=0.):
    from arraylab.shared.simulation_io import clean_json
    from .scene import terrain_grid,pose_with_terrain,element_positions
    project=normalize_project(configuration);grid=terrain_grid(project)
    def pose(obj):
        position,attitude=pose_with_terrain(obj,float(time_s),grid)
        return {'id':obj.get('id','平台'),'position_m':position.tolist(),'attitude_deg':attitude.tolist()}
    platform=pose(project['scene']['platform']);targets=[pose(row) for row in project['scene']['targets']]
    if grid is not None:
        sx=max(1,len(grid['x_m'])//128);sy=max(1,len(grid['y_m'])//128)
        grid={**grid,'x_m':np.asarray(grid['x_m'])[::sx],'y_m':np.asarray(grid['y_m'])[::sy],
              'z_m':np.asarray(grid['z_m'])[::sy,::sx],'valid':np.asarray(grid['valid'])[::sy,::sx]}
    return clean_json({'terrain':grid,'platform':platform,'targets':targets,
        'array':{'tx':element_positions(project['array'],'tx',project['waveform']['fc_hz']),
                 'rx':element_positions(project['array'],'rx',project['waveform']['fc_hz'])}},complex_style='object')


def _empty(figure,text):
    ax=figure.add_subplot(111);ax.set_axis_off();ax.text(.5,.5,text,ha='center',va='center',wrap=True,transform=ax.transAxes)


def _number(value):
    if value is None:return '—'
    if type(value) is bool:return '是' if value else '否'
    return f'{value:.6g}' if isinstance(value,(int,float,np.number)) else str(value)


def _scene_context(result,selection):
    time_s=float(selection)
    if not np.isfinite(time_s):raise ValueError('场景显示时间须为有限数值。')
    source=result.get('array_source')
    if result.get('kind')=='inverse' and source:
        from .service import array_configuration
        project=array_configuration(result)
        label=f"候选 #{source['candidate_rank']} 的输入场景"
    else:
        project=result.get('configuration') or normalize_project({})
        label='输入场景'
    return project,time_s,f'{label} · t={time_s:g} s'


def scene_payload(result,time_s=0.):
    """Prepare display-only geometry for the original offline Three.js scene."""
    project,time_s,caption=_scene_context(result,time_s)
    preview=scene_preview(project,time_s)
    scene=project['scene'];items=[scene['platform'],*scene['targets']]
    maximum=max([5.,*[float(key['time_s']) for item in items for key in item.get('trajectory',[])]])
    return {'project':project,'grid':preview['terrain'],'time_s':time_s,'max_time_s':maximum,'caption':caption}


def table(result,arrays,view='metrics',selection=0):
    if view in ('rd','profiles','detections','angles'):
        headers=['检测点','距离 / m','速度 / m/s','方位 / °','俯仰 / °','单元/训练噪声 / dB']
        rows=[[i+1]+[_number(item.get(key)) for key in ('range_m','velocity_mps','azimuth_deg','elevation_deg','snr_db')]
              for i,item in enumerate(result.get('detections',[]))]
        return headers,rows,result.get('range_axis_definition','检测点来自接收数据，不等于场景真值列表。')
    if view=='ambiguity':
        rows=[[i+1]+[_number(item.get(key)) for key in ('range_m','velocity_mps','above_thermal_noise_threshold')]
              for i,item in enumerate(result.get('ambiguity_candidates',[]))]
        return ['候选','距离 / m','速度 / m/s','超过热噪声门限'],rows,'有限搜索域保留多解；状态：'+str(result.get('ambiguity_status','未运行'))
    if view=='statistics':
        rows=[]
        for key,label in [('pd','固定目标CUT Pd'),('pfa','固定无目标CUT Pfa'),('scene_detection_rate','场景关联检出率')]:
            item=result.get(key,{})
            if item:rows.append([label,_number(item.get('estimate')),item.get('successes'),item.get('trials'),
                                ' – '.join(_number(x) for x in item.get('ci95',[]))])
        return ['统计量','估计','成功计数','独立次数','95%区间'],rows,'零虚警不代表Pfa为零；独立试验与固定CUT统计。'
    if view=='candidates':
        rows=[]
        for item in result.get('candidates',[]):
            m=item['metrics'];v=item.get('verification',{})
            rows.append([item['rank'],'通过' if item['passed'] else '尚未证明',_number(m.get('average_power_w')),
                _number(m.get('duration_s',m.get('cpi_s'))),_number(v.get('pd_estimate')),_number(v.get('pfa_estimate')),item['reason']])
        return ['排序','结论','平均功率 / W','采集时长 / s','Pd估计','Pfa估计','依据'],rows,'候选需通过独立端点统计置信界；未找到不等于不可实现。'
    if view=='scene':
        project,time_s,caption=_scene_context(result,selection)
        preview=scene_preview(project,time_s)
        rows=[['平台']+[_number(v) for v in preview['platform']['position_m']]]
        rows += [[item['id']]+[_number(v) for v in item['position_m']] for item in preview['targets']]
        return ['场景输入','x前向 / m','y左向 / m','z向上 / m'],rows,caption+'。输入真值与检测输出分别显示；时间变化不重算信号，地形和平台使用同一高程基准。'
    if view=='waveform':
        t=np.asarray(arrays.get('waveform_time_s',[]));real=np.asarray(arrays.get('waveform_real',[]));imag=np.asarray(arrays.get('waveform_imag',[]))
        step=max(1,len(t)//2048)
        return ['时间 / s','实部','虚部'],[[_number(t[i]),_number(real[i]),_number(imag[i])] for i in range(0,len(t),step)],'此表为显示抽样；NPZ保留完整波形。'
    labels={
        'wavelength_m':'参考波长 / m','pulse_width_actual_s':'实际采样脉宽 / s',
        'pulse_count':'总物理脉冲数','duration_s':'采集时长 / s','range_resolution_m':'距离分辨基准 / m',
        'cpi_s':'相干处理时长CPI / s','sample_count':'总复数采样数','fasttime_sample_count':'每脉冲快时间采样数',
        'duty_cycle':'发射占空比（比例）','full_receive_range_min_m':'完整回波接收距离下界 / m',
        'receiver_filter_tail_s':'接收滤波尾部时间 / s','observed_time_span_s':'实际采样时间跨度 / s',
        'frequency_energy_ratio':'逐频能量相对基准比值','velocity_resolution_note':'速度分辨口径',
        'range_sample_spacing_m':'原始距离采样间隔 / m',
        'velocity_resolution_mps':'速度分辨基准 / m/s','blind_range_m':'完整接收盲区 / m',
        'receive_range_max_m':'完整接收距离上限 / m','unambiguous_range_m':'最小PRI距离周期 / m',
        'unambiguous_velocity_mps':'同通道无模糊速度 / m/s','peak_power_w':'总传导峰值功率 / W',
        'average_power_w':'平均发射功率 / W','single_pulse_snr_db':'需求距离单脉冲SNR基准 / dB',
        'ideal_integrated_snr_db':'理想积累SNR基准 / dB','tx_count':'物理Tx数','rx_count':'物理Rx数',
        'noise_psd_w_hz':'输入噪声功率谱密度 / W/Hz','noise_sample_power_w':'每个复样本噪声功率 / W',
        'raw_bytes':'原始复数组数据量 / byte','data_rate_bytes_s':'原始数据率 / byte/s',
        'frequency_span_hz':'步进频总频率跨度 / Hz','frequency_range_period_m':'频步距离周期 / m',
        'snr_budget_note':'SNR预算参考面与条件','physical_channels':'Tx与Rx物理通道总数',
        'tx_rx_pairs':'Tx–Rx配对通道数','used_angle_channels':'孔径评估使用的位置数',
        'tx_aperture_m':'发射阵列最大孔径 / m','rx_aperture_m':'接收阵列最大孔径 / m',
        'virtual_aperture_m':'虚拟阵列最大孔径 / m','angle_aperture_m':'测角孔径基准 / m',
        'far_field_distance_m':'远场距离基准 / m','broadside_hpbw_proxy_deg':'正侧向HPBW代理 / °',
        'azimuth_aperture_m':'方位投影孔径 / m','elevation_aperture_m':'俯仰投影孔径 / m',
        'azimuth_hpbw_proxy_deg':'方位HPBW代理 / °','elevation_hpbw_proxy_deg':'俯仰HPBW代理 / °',
        'angle_design_hpbw_proxy_deg':'反向设计角宽代理 / °','note':'阵列模型说明',
        'channel_pairs':'Tx–Rx通道对数','independent_virtual_positions':'独立虚拟位置数',
        'estimated_memory_mib':'预计工作内存 / MiB','allowed_memory_mib':'本次可用预算 / MiB',
        'actual_receive_samples':'实际接收样本数','detections':'实际检测点数',
        'noise_psd_w_per_hz':'信号链输入噪声谱密度 / W/Hz','input_complex_noise_power_w':'输入复噪声功率 / W',
        'history_pulses':'计入的历史发射脉冲数','terrain_scatterers':'地形散射面元数',
        'occluded_scatterer_events':'被遮挡的散射体–事件对数','unknown_los_scatterer_events':'视线未知的散射体–事件对数',
        'visible_scatterer_events':'可见散射体–事件对数','adc_filter_history_samples':'ADC之前保留的滤波前史样本数',
        'receiver_record_model':'接收记录与滤波前史口径','sampled_transmit_energy_s':'发射采样能量等效时间 / s',
        'receive_reference_energy_s':'接收参考波形能量等效时间 / s','matched_amplitude_gain':'匹配处理幅度增益',
        'noise_filter_energy':'噪声滤波系数平方和','single_pulse_snr_efficiency':'单脉冲SNR效率（比例）',
        'range_window_loss_db':'脉压/距离窗损耗 / dB','receiver_filter_taps':'接收FIR系数个数','filter_length':'匹配滤波器长度',
        'pulse_compression_hpbw_m':'实际脉压半功率宽度 / m','pulse_compression_pslr_db':'实际脉压峰值旁瓣 / dB',
        'pulse_compression_islr_db':'实际脉压积分旁瓣 / dB','pulse_response_measurement':'脉压响应测量口径',
        'detection_summary_limit':'检测摘要数量上限','detections_truncated':'检测摘要已截断',
        'angle_processing_limit':'最多测角检测点数','valid_cfar_cells':'有效CFAR网格单元数',
        'range_grid_spacing_m':'处理距离网格间隔 / m','velocity_grid_spacing_mps':'处理速度网格间隔 / m/s',
        'frequency_taper_snr_efficiency':'频点窗SNR效率（比例）','group_taper_snr_efficiency':'重复组窗SNR效率（比例）',
        'frequency_taper':'频点窗函数','group_taper':'重复组窗函数','frequency_steps':'实际频点数',
        'repeat_groups':'实际完整重复组数','pulse_weights_norm_squared':'处理脉冲权重平方和'}
    def shown(key,value):
        if key=='pulse_response_measurement' and value=='actual sampled waveform and receive/matched filter; zero Doppler; 8x grid interpolation':
            return '实际采样波形及接收/匹配滤波；零多普勒；8倍网格插值'
        if key in ('frequency_taper','group_taper'):return {'rect':'矩形窗','hann':'Hann窗','hamming':'Hamming窗'}.get(value,str(value))
        return _number(value)
    rows=[[labels.get(key,key),shown(key,value)] for key,value in result.get('metrics',{}).items() if not isinstance(value,(dict,list))]
    return ['指标','数值'],rows,'\n'.join(result.get('errors',[])+result.get('warnings',[]))


def draw(figure,result,arrays,view='rd',selection=0):
    """Draw stored arrays. For scene, selection is the display time in seconds."""
    from arraylab.shared.plotting.surface import clear_figure,register_surface
    from arraylab.shared.plotting.metrics import draw_metric_footer
    from arraylab.shared.plotting.layout import responsive_subplots
    clear_figure(figure);figure.set_layout_engine('constrained')
    if hasattr(figure,'_arraylab_responsive_plots'):del figure._arraylab_responsive_plots
    if result is None:_empty(figure,'设置参数后显式运行预算、仿真、反向设计或统计。');return figure
    if view not in dict(VIEWS):raise ValueError('未知脉冲雷达视图：'+str(view))
    data=display_result(result,arrays)
    note=[]
    if view=='rd' and 'power_db' in data:
        ax=figure.add_subplot(111)
        mesh=ax.pcolormesh(data['ranges_m'],data['velocities_mps'],data['power_db'],shading='auto',cmap='viridis')
        figure.colorbar(mesh,ax=ax,label='处理输出功率 / dBW')
        detections=result.get('detections',[])
        if detections:ax.scatter([x['range_m'] for x in detections],[x['velocity_mps'] for x in detections],facecolors='none',edgecolors='white',s=25,label='数据驱动检测')
        ax.set(xlabel='表观距离 / m' if result.get('waveform')=='lfm' else '距离 / m',ylabel='径向速度 / m/s',title='真实接收处理的距离–速度图')
        note=[result.get('range_axis_definition',''),data['display_note'],'检测点可能包含虚警、旁瓣和杂波，不是输入目标清单。']
    elif view=='profiles' and 'power_db' in data:
        axes=figure.subplots(1,2)
        axes[0].plot(data['ranges_m'],data['range_profile_db']);axes[0].set(xlabel='距离 / m',ylabel='dBW',title='距离峰值切线')
        axes[1].plot(data['velocities_mps'],data['doppler_profile_db']);axes[1].set(xlabel='速度 / m/s',ylabel='dBW',title='速度峰值切线')
        for ax in axes:ax.grid(alpha=.25)
        responsive_subplots(figure,axes,2)
        note=[result.get('range_axis_definition',''),'完整处理数据见NPZ；这些切线不等于目标真值。']
    elif view=='waveform' and data.get('waveform'):
        axes=figure.subplots(1,2);wave=data['waveform']
        axes[0].plot(wave['time_s'],wave['real'],label='实部');axes[0].plot(wave['time_s'],wave['imag'],label='虚部')
        axes[0].set(xlabel='时间 / s',ylabel='归一化幅度',title='实际采样基带波形');axes[0].legend()
        if 'pulse_response_lag_s' in arrays:
            axes[1].plot(arrays['pulse_response_lag_s'],arrays['pulse_response_db'])
            axes[1].set(xlabel='延迟 / s',ylabel='归一化功率 / dB',title='实际接收/匹配滤波响应')
        else:axes[1].set_axis_off()
        for ax in axes:ax.grid(alpha=.25)
        responsive_subplots(figure,axes,2)
        note=['距离补零仅细化网格，不增加带宽、能量或物理分辨率。']
    elif view=='angles' and data.get('angles'):
        angle=data['angles'];az=np.asarray(angle['azimuth_deg']);el=np.asarray(angle['elevation_deg']);spectrum=np.asarray(angle['spectrum_db'])
        ax=figure.add_subplot(111)
        if len(el)==1:ax.plot(az,spectrum[0]);ax.set(xlabel='阵列方位 / °',ylabel='相对谱 / dB')
        elif len(az)==1:ax.plot(el,spectrum[:,0]);ax.set(xlabel='阵列俯仰 / °',ylabel='相对谱 / dB')
        else:figure.colorbar(ax.pcolormesh(az,el,spectrum,shading='auto'),ax=ax,label='相对谱 / dB');ax.set(xlabel='方位 / °',ylabel='俯仰 / °')
        ax.set_title('接收数据角谱');ax.grid(alpha=.2)
        note=['固定单点角切面是输入假设，不是该轴独立测量；空间别名保留候选。']
    elif view=='scene' and result.get('configuration'):
        project,time_s,caption=_scene_context(result,selection);preview=scene_preview(project,time_s)
        ax=figure.add_subplot(111,projection='3d');grid=preview['terrain'];positions=[]
        if grid:
            x,y=np.meshgrid(grid['x_m'],grid['y_m']);z=np.asarray(grid['z_m'],float);valid=np.asarray(grid['valid'],bool)
            from matplotlib import colormaps
            finite=z[valid];low,high=(float(finite.min()),float(finite.max())) if finite.size else (0.,1.)
            colors=colormaps['terrain']((np.nan_to_num(z,nan=low)-low)/max(high-low,1e-12));colors[...,3]=valid.astype(float)
            shown=np.where(valid,z,np.nan)
            surface=ax.plot_surface(x,y,shown,facecolors=colors,rstride=1,cstride=1,linewidth=0,antialiased=False,shade=False)
            register_surface(figure,ax,surface,x,y,shown,colors)
        platform=preview['platform']['position_m'];positions.append(platform)
        ax.scatter(*platform,color='#137b9a',marker='^',s=55,label='雷达平台')
        for target in preview['targets']:
            position=target['position_m'];positions.append(position);ax.scatter(*position,color='#b57622',s=35)
            ax.text(*position,str(target['id']))
        from .scene import pose_with_terrain,terrain_grid
        full_grid=terrain_grid(project)
        for item in [project['scene']['platform'],*project['scene']['targets']]:
            keys=item.get('trajectory') or []
            if len(keys)>=2:
                trajectory=np.asarray([pose_with_terrain(item,float(key['time_s']),full_grid)[0] for key in keys])
                ax.plot(*trajectory.T,linestyle='--',alpha=.6)
        ax.set(xlabel='x前向 / m',ylabel='y左向 / m',zlabel='z向上 / m',title=caption)
        ax.view_init(28,-60);ax.legend(loc='upper left')
        note=[caption+'；输入场景不等于检测结果，时间滑动不重算已保存回波。','显示地形与分析网格分离；无效高程保持空白。']
    elif view=='statistics' and result.get('kind')=='statistics':
        ax=figure.add_subplot(111)
        keys=['pd','pfa','scene_detection_rate'];labels=['固定CUT Pd','固定CUT Pfa','场景关联率']
        for i,key in enumerate(keys):
            item=result[key];estimate=item['estimate'];low,high=item['ci95']
            ax.errorbar(i,estimate,yerr=[[estimate-low],[high-estimate]],fmt='o',capsize=5)
            ax.annotate(f'{estimate:.4g}',(i,estimate),xytext=(5,8),textcoords='offset points')
        ax.set_xticks(range(3),labels);ax.set(ylim=(-.05,1.05),ylabel='概率',title=f"{result['trials']}次独立试验与95%区间");ax.grid(alpha=.2)
        note=['零虚警仍有非零置信上界；固定CUT与场景关联率分别报告。']
    elif view=='candidates' and result.get('candidates'):
        ax=figure.add_subplot(111);candidates=result['candidates']
        ax.bar([row['rank'] for row in candidates],[row['metrics']['average_power_w'] for row in candidates],
               color=['#16835e' if row['passed'] else '#c38a3d' for row in candidates])
        ax.set(xlabel='候选排序',ylabel='平均功率 / W',title='反向设计：绿色为独立统计已证明通过');ax.grid(axis='y',alpha=.2)
        note=['橙色候选未证明满足全部要求；详细原因和配置见候选表。']
    else:
        headers,rows,explanation=table(result,arrays,view,selection)
        _empty(figure,explanation if rows else '此操作尚无该视图的数据；请选择预算、仿真、反向设计或统计对应结果。')
    if note and result.get('kind')=='inverse' and result.get('array_source'):
        note.append(f"伴随数组来自候选 #{result['array_source']['candidate_rank']}；不代表搜索原始场景的重新计算。")
    if note:draw_metric_footer(figure,note)
    return figure
