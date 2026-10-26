"""Qt-free ArrayLab service contract for ArrayStudio 1.2.2 capabilities."""
from copy import deepcopy
import math
import numpy as np
from arraylab import __version__
from arraylab.framework.parallel import check_cancelled,ordered_map,requested_workers
from . import engine,linear_optimizer,planar_optimizer


def default_configuration():
    return {'array':engine.defaults(),'scan':{'maxAngle':60,'step':5},'linear':deepcopy(linear_optimizer.DEFAULTS),
            'planar':planar_optimizer.defaults(),'mapSize':81}


def normalize_configuration(value=None):
    value=deepcopy(value) if value is not None else {}
    if type(value) is not dict:raise ValueError('相控阵配置须为对象。')
    if value.get('format')=='arraylab.phased':
        if type(value.get('schema_version')) is not int or value['schema_version']!=1:raise ValueError('相控阵配置版本无效。')
        value=value.get('configuration')
        if type(value) is not dict:raise ValueError('相控阵配置缺少configuration。')
    if value.get('format')=='array-studio':
        version=value.get('schemaVersion')
        if type(version) is not int or version not in (1,2,3) or type(value.get('config')) is not dict:raise ValueError('只支持ArrayStudio schema1/2/3配置。')
        synthesis=value.get('synthesis');array={k:v for k,v in value['config'].items() if k in engine.defaults()}
        if type(array.get('name')) is not str or not 1<=len(array['name'].strip())<=64:raise ValueError('方案名称须为1～64字。')
        value={'array':array}
        if synthesis is not None:
            if type(synthesis) is not dict or synthesis.get('type')!='planar2d' or type(synthesis.get('options')) is not dict:raise ValueError('二维综合设置格式无效。')
            if version<3 and synthesis['options'].get('mode')=='continuous':raise ValueError('连续位置与幅度模式需要ArrayStudio v3配置。')
            checked=planar_optimizer.validate(array,synthesis['options'])
            if not checked['ok']:raise ValueError('；'.join(checked['errors']))
            value['planar']=checked['options']
    elif any(k in value for k in ('geometry','referenceGHz','frequencyGHz','nx')):
        value={'array':value}
    if set(value)-set(default_configuration()):raise ValueError('未知相控阵配置字段：'+','.join(sorted(set(value)-set(default_configuration()))))
    output=default_configuration()
    if 'array' in value:
        if type(value['array']) is not dict or set(value['array'])-set(engine.defaults()):raise ValueError('阵列设置字段无效。')
        output['array']=engine.normalized_config(value['array'])
    if type(output['array']['name']) is not str or not 1<=len(output['array']['name'])<=64:raise ValueError('方案名称须为1～64字。')
    # Omitted optimizer settings get sensible frequency defaults; explicit saved
    # frequency lists are never silently changed or interpreted in other units.
    f=output['array']['referenceGHz']
    output['planar']['frequencies']=sorted(set(max(.1,min(300,f*s)) for s in (.9,1,1.1)))
    for name in ('scan','linear','planar'):
        if name in value:
            if type(value[name]) is not dict or set(value[name])-set(output[name]):raise ValueError(name+'设置字段无效。')
            output[name].update(value[name])
    output['mapSize']=value.get('mapSize',81)
    if type(output['mapSize']) is not int or not 21<=output['mapSize']<=301:raise ValueError('显示u–v网格须为21～301的整数。')
    return output


def validate_configuration(value=None):
    try:
        config=normalize_configuration(value);array=config['array'];elements=engine.build_elements(array)
        warnings=[]
        for name,check in [('一维优化',linear_optimizer.validate(config['linear'])),('二维优化',planar_optimizer.validate(array,config['planar']))]:
            if not check['ok']:warnings.extend(name+'：'+error for error in check['errors'])
        return dict(valid=True,errors=[],warnings=warnings,metrics={'elementCount':len(elements),'referenceWavelengthM':engine.C/(array['referenceGHz']*1e9),'analysisWavelengthM':engine.C/(array['frequencyGHz']*1e9)})
    except (ValueError,TypeError,KeyError,OverflowError) as exc:return dict(valid=False,errors=[str(exc)],warnings=[],metrics={})


def _compute_display(configuration,cancelled):
        analysis=engine.evaluate(configuration['array'],cancelled=cancelled)
        grid=engine.uv_map(configuration['array'],analysis['elements'],configuration['mapSize'],cancelled=cancelled)
        arrays={'cut_x':np.array([[p['angle'],p['db']] for p in analysis.pop('cutX')],float),
                'cut_y':np.array([[p['angle'],p['db']] for p in analysis.pop('cutY')],float),
                'uv_u':np.array(grid['u']), 'uv_v':np.array(grid['v']),
                'uv_db':np.array([[math.nan if x is None else x for x in row] for row in grid['values']],float),
                'elements':np.array([[p[k] for k in ('x','y','amplitude','phaseDeg')] for p in analysis['elements']],float)}
        return analysis,arrays


def _display(configuration,cancelled,workers):
    values,plan=ordered_map(lambda _:_compute_display(configuration,cancelled),[None],workers=workers,cancelled=cancelled)
    return *values[0],plan


def execute(kind,configuration=None,*,progress=None,cancel=None,workers=None):
    if kind not in ('analysis','scan','optimize1d','optimize2d'):raise ValueError('未知相控阵任务。')
    workers=requested_workers(workers)
    config=normalize_configuration(configuration);check_cancelled(cancel)
    if progress:progress({'stage':'准备相控阵计算','progress':0})
    def numerical_progress(event):
        if progress:progress({**event,'progress':.85*event.get('progress',0)})
    result={'format':'arraylab.phased.result','schema_version':1,'app_version':__version__,'action':kind,'configuration':deepcopy(config),
            'source':{'name':'ArrayStudio','version':engine.SOURCE_VERSION,'archive_sha256':engine.SOURCE_ARCHIVE_SHA256},
            'scan':None,'synthesis':None,'notices':['坐标以参考波长λ0表示；X/Y为方向角，u=sin(X)、v=sin(Y)，不是传统方位/俯仰角。',
                '理想标量远场分析，不含互耦、全波电磁、射频硬件或实际辐射增益。',
                '切面分析与原始CSV保留至−300 dB数值下限；绘图范围和u–v显示下限不参与旁瓣判定。数值下限不是物理噪声底。']}
    display=deepcopy(config)
    if kind=='scan':
        result['scan']=engine.scan_study(config['array'],config['scan'],workers=workers,cancelled=cancel,
            progress=(lambda done,total:numerical_progress({'stage':'扫描评估','progress':done/total})) if progress else None)
    elif kind=='optimize1d':
        values,_=ordered_map(lambda _:linear_optimizer.optimize(config['array'],config['linear'],cancelled=cancel,progress=numerical_progress),[None],workers=workers,cancelled=cancel)
        result['synthesis']=values[0];display['array']=result['synthesis']['config']
    elif kind=='optimize2d':
        result['synthesis']=planar_optimizer.optimize(config['array'],config['planar'],workers=workers,cancelled=cancel,progress=numerical_progress)
        synthesis=result['synthesis'];first=synthesis['after']['cases'][0]
        display['array']={**synthesis['config'],'frequencyGHz':first['frequencyGHz'],'steerX':first['scanX'],'steerY':first['scanY']}
        result['displayCase']=0
    if progress:progress({'stage':'生成可浏览的结果快照','progress':.88 if kind!='analysis' else .1})
    result['analysis'],arrays,result['execution']=_display(display,cancel,workers)
    if kind=='optimize2d':
        synthesis=result['synthesis'];configs=[{**config,'array':{**synthesis['config'],'frequencyGHz':case['frequencyGHz'],'steerX':case['scanX'],'steerY':case['scanY']}} for case in synthesis['after']['cases']]
        values,_=ordered_map(lambda cfg:_compute_display(cfg,cancel),configs,workers=workers,
                             work_units=len(configs)*config['mapSize']**2*len(synthesis['config']['custom']),cancelled=cancel,
                             progress=(lambda done,total:progress({'stage':'保存各工况方向图','progress':.9+.09*done/total})) if progress else None)
        result['caseAnalyses']=[]
        for index,(analysis,case_arrays) in enumerate(values):
            result['caseAnalyses'].append(analysis)
            for key in ('cut_x','cut_y','uv_db','elements'):arrays[f'case_{index}_{key}']=case_arrays[key]
        result['notices'].append('显示X/Y切面固定另一方向的命令余弦；逐工况约束表中的HPBW则穿过实际二维期望峰。两者定义分别保留。')
    result['metrics']=result['synthesis']['after'] if result['synthesis'] else result['analysis']['metrics']
    result['warnings']=result['analysis']['warnings']
    if result['synthesis']:
        initial=result['synthesis'].get('baselineConfig')
        if initial:arrays['initial_elements']=np.array([[p[k] for k in ('x','y','amplitude','phaseDeg')] for p in initial['custom']],float)
    result['array_metadata']={}
    for key,array in arrays.items():
        if key.endswith(('cut_x','cut_y')):meta={'columns':['direction_angle_deg','relative_power_db']}
        elif key.endswith('elements'):meta={'columns':['x_lambda0','y_lambda0','relative_voltage','phase_deg'],
            'phase_reference':'additional input phase' if key=='initial_elements' else 'actual feed phase at displayed analysis condition'}
        elif key.endswith('uv_db'):meta={'axes':['v','u'],'unit':'dB','masked_value':'NaN outside visible disk'}
        else:meta={'unit':'direction cosine'}
        result['array_metadata'][key]={**meta,'shape':list(array.shape)}
    check_cancelled(cancel)
    if progress:progress({'stage':'相控阵计算完成','progress':1})
    return result,arrays


def presets():
    base=default_configuration();base['array'].update(name='16元均匀直线阵',geometry='linear',nx=16,ny=1)
    taylor=deepcopy(base);taylor['array'].update(name='32元Taylor直线阵',nx=32,window='taylor')
    squint=deepcopy(base);squint['array'].update(name='固定移相频率斜视',steerX=40,frequencyGHz=12)
    planar=default_configuration();planar['array'].update(name='二维联合综合基准',nx=5,ny=5)
    planar['planar'].update(count=8,gridNx=5,gridNy=5,scans=[{'x':0,'y':0}],frequencies=[10],iterations=100,targetSll=-5,maxHpbwX=60,maxHpbwY=60,mainlobeRadiusU=.5,mainlobeRadiusV=.5)
    continuous=default_configuration();continuous['array']['name']='连续二维坐标与幅度';continuous['planar']=planar_optimizer.continuous_defaults()
    return [dict(id=key,name=value['array']['name'],configuration=value) for key,value in [('uniform',base),('taylor',taylor),('squint',squint),('grid',planar),('continuous',continuous)]]


def apply_selected_case(result,index=0):
    """Return explicit candidate inputs; selecting a row does not calculate."""
    config=deepcopy(result['configuration']);synthesis=result.get('synthesis')
    if synthesis:
        config['array']=deepcopy(synthesis['config'])
        cases=synthesis['after'].get('cases',[])
        if cases:
            if type(index) is not int or not 0<=index<len(cases):raise ValueError('综合工况序号无效。')
            case=cases[index];config['array'].update(frequencyGHz=case['frequencyGHz'],steerX=case['scanX'],steerY=case['scanY'])
    else:config['array']=deepcopy(result['analysis']['config'])
    return normalize_configuration(config)


def array_snapshot(result,index=0):
    """Explicit coordinates in metres plus source convention, not automatic handoff."""
    analysis=result['analysis']
    if result.get('caseAnalyses'):
        if type(index) is not int or not 0<=index<len(result['caseAnalyses']):raise ValueError('综合工况序号无效。')
        analysis=result['caseAnalyses'][index]
    c=analysis['config'];scale=engine.C/(c['referenceGHz']*1e9)
    return {'coordinate_convention':'x-y aperture; +z broadside; X/Y direction angles',
            'reference_frequency_ghz':c['referenceGHz'],'frequency_ghz':c['frequencyGHz'],
            'coordinates_m':[[p['x']*scale,p['y']*scale,0.] for p in analysis['elements']],
            'amplitudes':[p['amplitude'] for p in analysis['elements']],
            'phases_deg':[p['phaseDeg'] for p in analysis['elements']],
            'steer_u':math.sin(c['steerX']*engine.RAD),'steer_v':math.sin(c['steerY']*engine.RAD)}


def record_compatibility(result):
    source=result.get('source',{}) if isinstance(result,dict) else {}
    if not isinstance(source,dict):source={}
    version=source.get('version')
    current=version==engine.SOURCE_VERSION and source.get('archive_sha256')==engine.SOURCE_ARCHIVE_SHA256
    return {'current':current,'source_version':version,
            'message':'' if current else '此相控阵记录来自旧版或未知版本算法；原结果与馈相快照保持不变。请使用当前版本重新计算并复核旁瓣、波束宽度及约束结论。'}
