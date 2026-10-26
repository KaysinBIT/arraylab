"""Explicit, non-mutating handoffs between completed array/radar snapshots.

These bridges prepare editable inputs. They never run an array, signal or
statistics solver, apply a result to widgets, or infer antenna gain.
"""
from copy import deepcopy
import math
import numpy as np

C=299792458.0


def _panel(host,name):
    panel=getattr(host,name+'_panel',None)
    if panel is None:panel=getattr(host,'simulation_panels',{}).get(name)
    if panel is None:raise ValueError('当前工作台没有'+name+'模块。')
    return panel


def _idle(panel):
    if getattr(panel,'_busy',False) or getattr(panel,'thread',None) is not None:
        raise ValueError('相关模块仍在计算，请完成或取消后再导入。')


def _configuration(panel):
    form=getattr(panel,'parameter_widget',None)
    if form is not None and callable(getattr(form,'value',None)):
        raw=form.value()
        if not isinstance(raw,dict):raise ValueError('目标模块输入须为配置对象。')
        return deepcopy(raw)
    reader=getattr(panel,'configuration',None) or getattr(panel,'config',None)
    if reader is None:raise ValueError('目标模块缺少配置接口。')
    return deepcopy(reader())


def _phased_snapshot(host):
    from arraylab.features.phased import array_snapshot
    panel=_panel(host,'phased');_idle(panel)
    report=getattr(panel,'report',None)
    if not report:raise ValueError('请先完成相控阵分析或综合。')
    result=report.get('result') if report.get('format')=='arraylab.simulation.result' else report
    if not isinstance(result,dict) or 'analysis' not in result:raise ValueError('相控阵记录缺少已计算快照。')
    selection=getattr(panel,'selection',None)
    index=selection.currentData() if selection is not None else 0
    index=0 if index is None else int(index)
    if result.get('caseAnalyses') and not 0<=index<len(result['caseAnalyses']):raise ValueError('所选相控阵工况不存在。')
    analysis=result['caseAnalyses'][index] if result.get('caseAnalyses') else result['analysis']
    if analysis['config'].get('elementPower',0)!=0:
        raise ValueError('当前相控阵使用非各向同性单元；目标模块不能直接继承该方向图。')
    return array_snapshot(result,index),analysis


def _source_array(host,source,side):
    if source=='linear':
        snap=host.selected_linear_array_snapshot()
    elif source=='sparse':
        panel=_panel(host,'sparse');_idle(panel);snap=panel.selected_array_snapshot()
    elif source=='mimo':
        panel=_panel(host,'mimo');_idle(panel);report=getattr(panel,'report',None)
        if not report or report.get('format')!='arraylab.mimo.result':raise ValueError('请先计算MIMO阵列。')
        data=report['geometry'][side]
        snap={'positions_lambda':data['positions_lambda'],'weights':data['weights'],
              'frequency_ghz':report['parameters']['frequency_ghz'],'element_model':'isotropic',
              'origin':'MIMO已计算'+side.upper()+'物理阵列'}
    elif source=='phased':
        snap,_=_phased_snapshot(host)
        return _checked_array(snap['coordinates_m'],snap['amplitudes']),float(snap['frequency_ghz']),f"相控阵所选工况（参考{snap['reference_frequency_ghz']:g} GHz）"
    else:raise ValueError('阵列来源须为linear、sparse、mimo或phased。')
    if snap.get('element_model')!='isotropic':
        raise ValueError('源快照使用非各向同性单元；不能直接迁移其单元方向图或增益。')
    frequency=float(snap['frequency_ghz'])
    if not math.isfinite(frequency) or frequency<=0:raise ValueError('源频率无效。')
    xy=np.asarray(snap['positions_lambda'],dtype=float)
    if xy.ndim!=2 or xy.shape[1]!=2:raise ValueError('源阵列须为x-y平面坐标。')
    xyz=np.column_stack((xy,np.zeros(len(xy))))*(C/(frequency*1e9))
    return _checked_array(xyz,snap['weights']),frequency,str(snap.get('origin',source))


def _checked_array(coordinates,amplitudes):
    if np.iscomplexobj(coordinates) or np.iscomplexobj(amplitudes):raise ValueError('坐标/幅度接口不能静默丢弃未知复相位。')
    xyz=np.asarray(coordinates,dtype=float);weights=np.asarray(amplitudes,dtype=float)
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or weights.shape!=(len(xyz),)
            or not np.isfinite(xyz).all() or not np.isfinite(weights).all() or np.any(weights<0) or not np.any(weights>0)):
        raise ValueError('源阵列坐标或幅度无效。')
    return xyz.copy(),(weights/weights.max()).copy()


def _invalidate_calibration(config):
    detection=config.get('detection',{})
    had=detection.get('calibration_factor') is not None or detection.get('calibration_hash') is not None
    detection['calibration_factor']=None;detection['calibration_hash']=None
    return '原H0校准已清除，门限模式保留，检测前须按新配置重新校准。' if had else ''


def import_array(host,destination,side,source):
    """Return destination inputs, preserving its frequency and other aperture."""
    if destination not in ('pulse','equation') or side not in ('tx','rx'):
        raise ValueError('目标须为pulse/equation，导入侧须为tx/rx。')
    panel=_panel(host,destination);_idle(panel);config=_configuration(panel)
    (xyz,amplitudes),frequency,origin=_source_array(host,source,side)
    mapped=xyz[:,[2,0,1]]  # +z normal -> +x forward, +x aperture -> +y left.
    count=len(mapped)
    if destination=='pulse':
        from arraylab.features.pulse import validate_configuration,normalize_configuration,element_positions
        maximum=16 if side=='tx' else 64
        if count>maximum:raise ValueError(f'脉冲雷达{side.upper()}最多{maximum}通道，不截断源阵列。')
        if not np.allclose(amplitudes,1.,rtol=0,atol=1e-12):
            raise ValueError('脉冲雷达1.0.1仅支持等幅通道；不能静默丢弃源幅度加权。')
        config=normalize_configuration(config)
        other='rx' if side=='tx' else 'tx'
        try:preserved=element_positions(config['array'],other,config['waveform']['fc_hz'])
        except ValueError as exc:raise ValueError('目标另一侧阵列几何无效，请先修正：'+str(exc)) from exc
        config['array'][other+'_positions_m']=preserved.tolist()
        config['array'].update(geometry='custom')
        config['array'][side+'_count']=count;config['array'][side+'_positions_m']=mapped.tolist()
        config=normalize_configuration(config);validation=validate_configuration(config)
        if not validation['valid']:raise ValueError('导入阵列与当前脉冲体制不兼容：'+'；'.join(validation['errors']))
        extra='目标使用等幅通道。'
    else:
        from arraylab.features.equation import normalize_configuration
        maximum=16 if side=='tx' else 32
        if count>maximum:raise ValueError(f'雷达方程{side.upper()}最多{maximum}通道，不截断源阵列。')
        if side=='tx' and count>1 and config['array']['mode']=='simo':
            raise ValueError('SIMO只接受1路Tx；请先在目标模块明确选择相控/TDM/DDM等体制。')
        config['array'][side+'_count']=count;config['array'][side+'_positions_m']=mapped.tolist()
        config['array'][side+'_weights']=[[float(value),0.] for value in amplitudes]
        extra=_invalidate_calibration(config)
        try:config=normalize_configuration(config)
        except ValueError as exc:raise ValueError('导入阵列与当前方程体制/码周期不兼容：'+str(exc)) from exc
    note=(f'已从{origin}读取{count}个{side.upper()}位置，源分析频率为{frequency:g} GHz；'
          '旧阵列(x,y,z)按(z,x,y)映射为雷达(x前向,y左向,z向上)，坐标已转为米。'
          '仅导入几何和可表达的相对幅度；原扫描/量化馈相不复制，目标频率、另一侧阵列、功率、增益和指向保持当前设置。'
          '需在目标模块重新计算。'+extra)
    return config,note


def import_waveform(host):
    """Prepare an independent uniform FMCW group; never merge A/B energy."""
    from arraylab.features.equation import normalize_configuration
    source=_panel(host,'waveform');target=_panel(host,'equation');_idle(source);_idle(target)
    saved=getattr(source,'report',None)
    if not saved or saved.get('format')!='arraylab.waveform.result':raise ValueError('请先完成FMCW波形计算。')
    analysis=saved['analysis']
    if analysis.get('valid') is not True:raise ValueError('源波形时序未通过检查。')
    if analysis['options']['enabled'] and analysis['options']['layout']=='alternating':
        raise ValueError('A/B整轮交替不能直接当作同一均匀采集。请选择单组或分段配置后重新计算；不会合并A/B能量。')
    selected=source.result_profile.currentData() if hasattr(source,'result_profile') else saved['configuration']['activeProfile']
    group=analysis['profiles'].get(selected)
    if selected not in ('A','B') or not group:raise ValueError('没有所选A/B组的已计算快照。')
    p=group['params'];metrics=group['result']['metrics'];config=_configuration(target)
    cycle=p['txCount'] if p['mode']=='tdm' else 1
    expected_interval=(p['rampUs']+p['idleUs'])*cycle
    if not math.isclose(float(metrics['txIntervalUs']),expected_interval,rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError('源组的实际重访时序不是目标模块支持的均匀模式。')
    ramp=p['rampUs']*1e-6;slope=p['slopeMHzUs']*1e12;fc=p['startGHz']*1e9+slope*ramp/2
    config['name']=f'FMCW {selected}组独立方程验证'
    config['waveform'].update(kind='fmcw',fc_hz=fc,bandwidth_hz=slope*ramp,ramp_s=ramp,
        adc_start_s=p['adcStartUs']*1e-6,fs_hz=p['sampleRateMsps']*1e6,samples=p['samples'],
        pri_s=(p['rampUs']+p['idleUs'])*1e-6,pulses=p['loops']*cycle,
        range_fft=p['rangeFft'],doppler_fft=p['dopplerFft'],rx_bandwidth_hz=p['ifBandwidthMHz']*1e6,
        range_max_m=metrics['maxRangeStaticM'])
    config['array'].update(mode=p['mode'],tx_count=p['txCount'],rx_count=p['rxCount'],ddm_slots=p['ddmSlots'])
    geometry=(saved.get('antenna') or {}).get(selected)
    geometry_note='目标原有米制阵列保持。'
    if geometry and geometry.get('valid'):
        wavelength=float(metrics['lambdaMm'])*1e-3
        for side in ('tx','rx'):
            points=np.asarray(geometry[side],float)
            if points.shape!=(p[side+'Count'],2) or not np.isfinite(points).all():raise ValueError('源波形阵列快照尺寸无效。')
            config['array'][side+'_positions_m']=[[0.,float(x)*wavelength,float(z)*wavelength] for x,z in points]
            config['array'][side+'_weights']=[[1.,0.] for _ in points]
        geometry_note='同时采用所选组已计算的等幅阵列，源(x,z)映射为雷达(0,x,z)并转为米。'
    elif any(len(config['array'][side+'_positions_m'])!=p[side+'Count'] for side in ('tx','rx')):
        raise ValueError('源记录缺少有效天线快照且通道数与目标布局不一致；请先重新计算源天线或明确目标坐标。')
    config['processing'].update(fast_window=p['window'],slow_window=p['window'],looks=1)
    if not config['scene']['targets']:
        config['scene']['targets']=[{'name':'导入的单目标参考','range_m':p['targetRange'],'velocity_mps':p['targetVelocity'],
            'az_deg':0.,'height_m':config['scene']['radar_height_m'],'rcs_m2':1.,'swerling':0,'phase_model':'fixed'}]
        scene_note='源未定义方位/高程，新增同高正前方单目标参考。'
    else:scene_note='首个目标的方位/高程及其余场景保持目标模块输入。'
    config['scene']['targets'][0].update(range_m=p['targetRange'],velocity_mps=p['targetVelocity'])
    config['experiment'].update(cut_range_m=p['targetRange']+fc*p['targetVelocity']/slope,cut_velocity_mps=p['targetVelocity'])
    gain=(saved['configuration'].get('antennaDesign') or {}).get('gainOptions')
    gain_note='源记录未提供增益预算输入，目标功率/噪声参数保持。'
    if gain:
        if not (saved.get('gain_budget') or {}).get('valid'):raise ValueError('源增益预算输入无效，请修正并重新计算。')
        config['radar'].update(power_w=10**((gain['txPowerDbm']-30)/10),
            power_basis='total' if gain['powerBasis']=='total_active' else 'per_tx',loss_db=gain['systemLossDb'],
            noise_mode='tsys',system_temperature_k=gain['temperatureK']*10**(gain['noiseFigureDb']/10))
        config['requirements'].update(range_m=gain['rangeM'],snr_db=gain['requiredSnrDb'],margin_db=gain['marginDb'])
        config['scene']['targets'][0]['rcs_m2']=10**(gain['rcsDbsm']/10)
        gain_note='仅迁移源预算的功率、损耗、参考RCS和SNR要求；Tsys=T×F直接输入，避免重复计NF。每通道Tx/Rx增益保留目标手工值，未把反算所需增益当成已实现增益。'
        if gain['combining']=='single_rx' and p['rxCount']>1:
            gain_note+='源single_rx预算口径不直接迁移，目标模型按物理Rx处理，请重新核对增益参考面。'
    calibration_note=_invalidate_calibration(config)
    try:config=normalize_configuration(config)
    except ValueError as exc:raise ValueError('源波形不能直接进入当前方程模型：'+str(exc)) from exc
    note=(f'已准备{selected}组独立均匀实验：TDM总事件数=每Tx次数×Tx数，DDM总事件数=组内次数；不包含另一组能量、A/B切换或整帧空闲。'
          '载频按完整Ramp中心重建以保持RF扫频边界；原波形波长按ADC窗中心，窄带基准存在差别。编码慢时间窗按目标模型的完整码周期处理。'
          +geometry_note+scene_note+gain_note+calibration_note+'尚未运行方程或回波验证。')
    return config,note


def apply_phased_to_planar(host):
    """Return validated planar electrical inputs for the displayed phased case."""
    from arraylab.features.planar import PlanarParameters,validate_planar_design
    from arraylab.features.feed import FeedSettings
    snap,_=_phased_snapshot(host)
    xyz,amplitudes=_checked_array(snap['coordinates_m'],snap['amplitudes'])
    if np.any(amplitudes<=0):raise ValueError('平面馈电要求正幅度端口，不能静默删除源阵列中的零幅度阵元。')
    if len(xyz)>200:raise ValueError('平面馈电最多200阵元，不能截断所选相控阵。')
    if np.any(abs(xyz[:,2])>1e-12):raise ValueError('平面馈电需要x-y平面几何。')
    u,v=float(snap['steer_u']),float(snap['steer_v']);radius=math.hypot(u,v)
    if radius>1+1e-12:raise ValueError('命令方向余弦超出可见域。')
    theta=math.degrees(math.asin(min(1.,radius)));phi=math.degrees(math.atan2(v,u)) if radius else 0.
    frequency=float(snap['frequency_ghz']);xy=xyz[:,:2];wavelength=C/(frequency*1e9)
    phases=np.asarray(snap['phases_deg'],float)
    if phases.shape!=amplitudes.shape or not np.isfinite(phases).all():raise ValueError('源快照缺少有效实际馈相。')
    base=-360*(xy[:,0]*u+xy[:,1]*v)/wavelength
    offsets=(phases-base+180)%360-180
    parameters=PlanarParameters(geometry='arbitrary',frequency_ghz=frequency,coordinate_unit='mm',
        coordinates=tuple(tuple(float(value)*1000 for value in row) for row in xy),
        scan_theta_deg=theta,scan_phi_deg=phi,cut_phi_deg=phi)
    settings=FeedSettings(topology='corporate',weighting='custom',phase_mode='scan',
        custom_amplitudes=tuple(amplitudes),custom_phases_deg=tuple(offsets),phase_bits=0)
    validate_planar_design(parameters,settings)
    return {'parameters':parameters,'settings':settings,
        'provenance':{'source':'phased','frequency_ghz':frequency,'reference_frequency_ghz':snap['reference_frequency_ghz'],
                      'actual_phase_preserved':True},
        'note':'已准备所选相控阵工况的米制几何、幅度和实际馈相；以50Ω并联电气初值继续设计，尚未计算。源量化/斜视相位作为当前频点偏置保留，不宣称宽带硬件等效。'}
