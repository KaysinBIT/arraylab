"""Qt-free operations for the shared ArrayLab simulation workspace."""
from copy import deepcopy
import time

from arraylab import __version__ as APP_VERSION
from .provenance import SOURCE_VERSION,SOURCE_ARCHIVE_SHA256
from .config import (default_project,normalize_project,validate_project,
                     acquisition_plan,project_hash,presets as source_presets)

ACTIONS=('budget','simulate','statistics','inverse')


def default_configuration():
    return normalize_configuration(default_project())


def normalize_configuration(configuration=None):
    project=normalize_project(default_project() if configuration is None else configuration)
    try:
        project['scene']['platform'].setdefault('height_mode','absolute')
        if not isinstance(project['scene']['targets'],list):raise ValueError('目标须为列表。')
        for target in project['scene']['targets']:target.setdefault('height_mode','absolute')
    except (KeyError,TypeError,AttributeError) as exc:
        raise ValueError('场景平台和目标结构无效。') from exc
    return project


def validate_configuration(configuration):
    from arraylab.shared.simulation_io import clean_json
    return clean_json(validate_project(configuration),complex_style='object')


def presets():
    return [{'id':key,'name':row['name'],'description':row['description'],
             'configuration':normalize_configuration(row['project'])} for key,row in source_presets().items()]


def execute(kind,configuration=None,*,progress=None,cancel=None):
    """Return a complete immutable result and ndarray dictionary.

    Budget only inspects the configuration. Signal, statistical and inverse
    operations are explicit and retain PulseLab's independent receive chain.
    Cancellation raises InterruptedError and never returns a partial result.
    """
    if kind not in ACTIONS:raise ValueError('未知脉冲雷达操作：'+str(kind))
    if cancel and cancel():raise InterruptedError('计算已取消')
    project=normalize_configuration(configuration)
    started=time.perf_counter()
    validation=validate_project(project)
    if kind=='budget':
        result={**validation,'kind':'budget','assumptions':[
            '解析预算是设计检查，不是检测概率或硬件性能验证。',
            '实际回波仿真、独立统计和反向搜索须分别显式运行。']}
        arrays={}
    else:
        if not validation['valid']:raise ValueError('；'.join(validation['errors']))
        if progress:progress(0.,'核对工程与资源预算')
        if kind=='simulate':
            from .engine import simulate
            result,arrays=simulate(project,progress=progress,cancel=cancel)
        else:
            from .analysis import statistics,inverse_design
            fn=statistics if kind=='statistics' else inverse_design
            result,arrays=fn(project,progress=progress,cancel=cancel)
        result={**result,'valid':True,'kind':kind,'validation':validation}
    if cancel and cancel():raise InterruptedError('计算已取消')
    result.update(configuration=project,config_hash=project_hash(project),input_hash=project_hash(project),
        project_name=project['name'],version=SOURCE_VERSION,algorithm_version=SOURCE_VERSION,
        app_version=APP_VERSION,schema_version=1,seed=project['seed'],elapsed_s=time.perf_counter()-started,
        source={'name':'PulseLab','version':SOURCE_VERSION,'archive_sha256':SOURCE_ARCHIVE_SHA256})
    if validation['valid']:result['acquisition']=acquisition_plan(project)
    if kind=='inverse' and result.get('array_source'):
        actual=array_configuration(result)
        result['waveform']=actual['waveform']['kind']
        result['range_axis_definition']=('表观零多普勒匹配滤波延迟距离' if result['waveform']=='lfm' else
            '相干频率–时间匹配的物理距离假设' if result['waveform']=='stepped' else '匹配滤波延迟距离')
    result['array_units']={'raw':'sqrt(W)','compressed':'sqrt(W)','rd_power':'W','cfar_threshold':'W',
        'cfar_noise':'W','cfar_mask':'boolean','receive_valid':'boolean','ranges_m':'m','velocities_mps':'m/s',
        'waveform_time_s':'s','waveform_real':'normalized amplitude','waveform_imag':'normalized amplitude',
        'pulse_times_s':'s','pulse_frequencies_hz':'Hz','channel_positions_m':'m',
        'angle_azimuth_deg':'deg','angle_elevation_deg':'deg','angle_spectrum_db':'dB relative'}
    result['array_manifest']={key:{'shape':list(value.shape),'dtype':str(value.dtype)} for key,value in arrays.items()}
    from arraylab.shared.simulation_io import clean_json
    return clean_json(result,complex_style='object'),arrays


def run(project=None,action='simulate',**kwargs):
    """Convenience spelling for direct Python callers."""
    return execute(action,project,**kwargs)


def apply_inverse_candidate(result,index=0):
    """Return the complete candidate including its declared validation scene."""
    candidates=result.get('candidates',[])
    if result.get('kind')!='inverse' or type(index) is not int or not 0<=index<len(candidates):
        raise ValueError('请选择有效的脉冲反向设计候选。')
    project=candidates[index].get('project')
    if type(project) is not dict:raise ValueError('该候选缺少完整工程配置。')
    return normalize_configuration(project)


def array_configuration(result):
    """Recover exact inputs for inverse-design companion arrays, without rerun."""
    source=result.get('array_source')
    if result.get('kind')!='inverse' or not source:return deepcopy(result['configuration'])
    rank=source.get('candidate_rank')
    if type(rank) is not int:raise ValueError('候选数组来源序号无效。')
    candidate=next((row for row in result.get('candidates',[]) if row.get('rank')==rank),None)
    if candidate is None or type(candidate.get('project')) is not dict:raise ValueError('候选数组来源缺少对应工程。')
    project=deepcopy(candidate['project'])
    project['statistics']['seed']=source['statistics_seed']
    if project_hash(project)!=source['config_hash']:raise ValueError('候选数组来源配置摘要不一致。')
    return project
