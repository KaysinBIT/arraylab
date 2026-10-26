"""Explicit composition of independent radar engines and shared job transport."""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import math
import hashlib
import tempfile
import shutil
import csv

from arraylab import __version__
from arraylab.framework.parallel import check_cancelled,requested_workers,set_workers,execution_plan
from arraylab.framework.process import temporary_directory,run_owned
from arraylab.shared.simulation_io import read_json,write_json,read_npz,write_npz,configuration_digest,clean_json
from arraylab.framework.cli_support import Parser


FEATURES=('phased','equation','pulse')


def provider(feature):
    if feature=='phased':
        from arraylab.features import phased
        return phased
    if feature=='equation':
        from arraylab.features import equation
        return equation
    if feature=='pulse':
        from arraylab.features import pulse
        return pulse
    raise ValueError('未知雷达模块：'+str(feature))


def memory_budget(configuration):
    requested=configuration.get('resources',{}).get('memory_mib',configuration.get('experiment',{}).get('memory_mib',1024))
    if type(requested) not in (int,float) or not 128<=requested<=16384:
        raise ValueError('计算内存预算须为128–16384 MiB。')
    import psutil
    return int(min(requested*1024**2,psutil.virtual_memory().available*.8))


def calculate(feature,action,configuration,*,workers=0,progress=None,cancel=None):
    module=provider(feature);configuration=module.normalize_configuration(configuration)
    validation=module.validate_configuration(configuration)
    if not validation.get('valid',False):raise ValueError('；'.join(map(str,validation.get('errors',[]))))
    set_workers(requested_workers(workers));check_cancelled(cancel)
    if feature=='phased':
        result,arrays=module.execute(action,configuration,progress=progress,cancel=cancel)
        execution={'requested_workers':requested_workers(workers),'scope':'independent array conditions; native BLAS single-threaded'}
    else:
        w=configuration['waveform'];a=configuration['array']
        events=int(w.get('pulses',configuration.get('timing',{}).get('pulses',1)))
        if w.get('kind')=='stepped':events=int(w['frequency_steps']*configuration['timing']['groups'])
        samples=int(w.get('samples',max(1,w.get('sample_rate_hz',1)*configuration.get('timing',{}).get('receive_end_s',1))))
        execution=execution_plan(8,events*int(a['rx_count'])*samples,workers=workers)
        execution={**execution,'backend':'native_libraries','scope':'bounded BLAS/SciPy FFT; trial and random-stream order preserved'}
        execution.update(native_blas_threads=execution['workers'],fft_workers=execution['workers'])
        from threadpoolctl import threadpool_limits
        from scipy.fft import set_workers as fft_workers
        with threadpool_limits(limits=execution['workers'],user_api='blas'),fft_workers(execution['workers']):
            result,arrays=module.execute(action,configuration,progress=progress,cancel=cancel)
    result=clean_json(result,complex_style='pair' if feature=='equation' else 'object')
    check_cancelled(cancel)
    envelope={'format':'arraylab.simulation.result','schema_version':1,'app_version':__version__,
              'feature':feature,'action':action,'configuration':configuration,
              'configuration_sha256':configuration_digest(configuration),'result':result,'execution':execution,
              'array_manifest':{key:{'shape':list(value.shape),'dtype':str(value.dtype)} for key,value in arrays.items()}}
    return envelope,arrays


def validate_record(record,arrays,*,expected_feature=None,expected_action=None):
    if (type(record) is not dict or record.get('format')!='arraylab.simulation.result' or
            type(record.get('schema_version')) is not int or record['schema_version']!=1 or
            record.get('feature') not in FEATURES or type(record.get('result')) is not dict):
        raise ValueError('计算记录类型、版本或结果格式无效。')
    feature=record['feature'];module=provider(feature)
    if expected_feature is not None and feature!=expected_feature:raise ValueError('记录来自不同模块。')
    actions={key for key,title in module.ACTIONS}
    if type(record.get('action')) is not str or record['action'] not in actions or expected_action is not None and record['action']!=expected_action:
        raise ValueError('记录操作类型不匹配。')
    config=record_configuration(feature,record.get('configuration'))
    if configuration_digest(config)!=record.get('configuration_sha256'):raise ValueError('记录输入摘要不匹配。')
    inner=record['result'].get('configuration',record['result'].get('config'))
    if inner is None or configuration_digest(inner)!=record['configuration_sha256']:
        raise ValueError('结果与外层输入快照不一致。')
    if record['result'].get('action',record['result'].get('kind'))!=record['action']:
        raise ValueError('结果类型与记录操作不一致。')
    manifest=record.get('array_manifest')
    if type(manifest) is not dict or set(manifest)!=set(arrays):raise ValueError('记录缺少匹配的数组清单。')
    for key,value in arrays.items():
        if value.dtype.hasobject or manifest[key]!={'shape':list(value.shape),'dtype':str(value.dtype)}:
            raise ValueError('结果数组的形状或类型不匹配。')
    state=record.get('view_state')
    if state is not None:
        if type(state) is not dict or type(state.get('view')) is not str or state['view'] not in {key for key,title in module.VIEWS}:
            raise ValueError('记录结果视图无效。')
        if type(state.get('table')) is not bool or type(state.get('scene_time')) not in (int,float) or not math.isfinite(state['scene_time']) or state['scene_time']<0:
            raise ValueError('记录显示状态无效。')
        selection=state.get('selection')
        if selection is not None and (type(selection) is not int or selection<0):raise ValueError('记录工况选择无效。')
        camera=state.get('scene_camera')
        if camera is not None:
            if feature!='pulse' or type(camera) is not dict:raise ValueError('场景视角记录无效。')
            for key in ('position','target'):
                value=camera.get(key)
                if type(value) is not list or len(value)!=3 or any(type(x) not in (int,float) or not math.isfinite(x) for x in value):raise ValueError('场景视角坐标无效。')
            zoom=camera.get('zoom')
            if type(zoom) not in (int,float) or not math.isfinite(zoom) or zoom<=0:raise ValueError('场景视角缩放无效。')
    return record


def record_configuration(feature,configuration):
    """Check a saved snapshot's structure without rerunning newer model limits."""
    module=provider(feature)
    normalize=getattr(module,'normalize_record_configuration',module.normalize_configuration)
    return normalize(configuration)


def record_compatibility(record):
    """Expose provenance warnings without rewriting saved scientific results."""
    if record is None:return {'current':True,'message':''}
    module=provider(record['feature']);inspect=getattr(module,'record_compatibility',None)
    return inspect(record['result']) if inspect else {'current':True,'message':''}


def load_configuration(feature,path):
    module=provider(feature);path=Path(path)
    if feature=='pulse' and path.suffix.lower()=='.prlab':return module.import_project(path)
    document=read_json(path)
    if type(document) is not dict:raise ValueError('工程须为JSON对象。')
    app=document.get('app')
    if app is not None and app!={'phased':'ArrayStudio','equation':'EquationLab','pulse':'PulseLab'}[feature]:
        raise ValueError('该文件来自另一种工具，请使用明确的参数导入入口。')
    if 'schema_version' in document and type(document['schema_version']) is not int:
        raise ValueError('工程格式版本须为整数。')
    if document.get('format')=='arraylab.simulation':
        if (type(document.get('schema_version')) is not int or document['schema_version']!=1 or document.get('feature')!=feature):
            raise ValueError('工程模块或版本不匹配。')
        document=document.get('configuration')
    else:
        formats={'phased':{'arraylab.phased','array-studio'},'equation':{'arraylab.equation'},'pulse':{'arraylab.pulse.project'}}
        if document.get('format') is not None and document['format'] not in formats[feature]:
            raise ValueError('该参数文件不属于当前模块。')
        if feature=='pulse':return module.import_project(path)
        if feature=='equation':return module.load_configuration(path)
    return module.normalize_configuration(document)


def load_record(path):
    path=Path(path)
    if path.suffix.lower()=='.npz':record,arrays=read_npz(path)
    else:
        document=read_json(path)
        if (type(document) is not dict or document.get('format')!='arraylab.simulation.export' or
                type(document.get('schema_version')) is not int or document['schema_version']!=1 or
                document.get('arrays_file')!='arrays.npz'):
            raise ValueError('不是可恢复的完整雷达记录。')
        filename=path.parent/'arrays.npz'
        with filename.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if digest!=document.get('arrays_sha256'):raise ValueError('结果数组摘要不匹配。')
        inner,arrays=read_npz(filename);record=document.get('record')
        if type(record) is not dict or inner!=record.get('result'):raise ValueError('数组元数据与记录不一致。')
    validate_record(record,arrays)
    return record,arrays


def export_result(path,record,arrays,*,plot=False):
    """One array archive shared by domain exports and the recoverable record."""
    validate_record(record,arrays)
    target=Path(path).expanduser().absolute()
    if target.exists() or target.is_symlink():raise FileExistsError('输出目录已存在，未覆盖。')
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.arraylab-export-',dir=target.parent));payload=staging/'result'
    try:
        module=provider(record['feature']);module.export_result(payload,record['result'],arrays)
        filename=payload/'arrays.npz'
        with filename.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        write_json(payload/'arraylab-record.json',{'format':'arraylab.simulation.export','schema_version':1,
                   'arrays_file':'arrays.npz','arrays_sha256':digest,'record':record})
        compatibility=record_compatibility(record)
        if not compatibility.get('current',False):
            write_json(payload/'model-version-notice.json',compatibility)
        state=record.get('view_state',{});selection=state.get('selection') or 0
        if record['feature']=='phased' and record['result'].get('caseAnalyses'):
            (payload/'elements.csv').unlink(missing_ok=True)
            for index in range(len(record['result']['caseAnalyses'])):
                columns,rows,note=module.table(record['result'],arrays,'elements',selection=index)
                with (payload/f'elements-case-{index+1}.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(columns);writer.writerows(rows)
        if state:
            selected=state['scene_time'] if state['view']=='scene' else selection
            columns,rows,note=module.table(record['result'],arrays,state['view'],selection=selected)
            with (payload/'selected-view.csv').open('w',encoding='utf-8-sig',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(columns);writer.writerows(rows)
            write_json(payload/'export-view.json',{'view':state,'note':note})
        if plot:
            import matplotlib
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            result=record['result'];feature=record['feature']
            if feature=='phased':views=['layout','cuts','uv']+(['scan'] if result.get('scan') else [])+(['optimization'] if result.get('synthesis') else [])
            elif feature=='equation':views=['budget']+(['rd'] if 'rd_power' in arrays else [])+(['snr'] if result['kind']=='snr' else [])+(['statistics'] if result['kind'] in ('statistics','calibration','roc','distance') else [])
            else:views=['scene']+(['rd','profiles','waveform'] if 'rd_power' in arrays else [])+(['statistics'] if result['kind']=='statistics' else [])+(['candidates'] if result.get('candidates') else [])
            with matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':9}):
                for view in views:
                    figure=Figure(figsize=(10,6),dpi=130);FigureCanvasAgg(figure)
                    selected=state.get('scene_time',0) if view=='scene' else selection
                    module.draw(figure,result,arrays,view,selection=selected);figure.savefig(payload/(view+'.png'));figure.clear()
        if target.exists() or target.is_symlink():raise FileExistsError('输出目录已被创建，未覆盖。')
        payload.rename(target)
    finally:
        if staging.exists() and staging.resolve().is_relative_to(target.parent.resolve()) and staging.name.startswith('.arraylab-export-'):
            shutil.rmtree(staging)
    return target


def export_process(path,record,arrays,*,progress=None,cancel=None,scene_png=None):
    """Keep Agg/font rendering out of the live Qt process and publish atomically."""
    validate_record(record,arrays)
    if scene_png is not None and (record['feature']!='pulse' or not isinstance(scene_png,bytes) or not scene_png.startswith(b'\x89PNG\r\n\x1a\n')):
        raise ValueError('三维场景图像无效。')
    target=Path(path).expanduser().absolute()
    if target.exists() or target.is_symlink():raise FileExistsError('输出目录已存在，未覆盖。')
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.arraylab-export-worker-',dir=target.parent))
    try:
        with temporary_directory(prefix='arraylab-export-input-') as temporary:
            folder=Path(temporary);request=folder/'record.npz'
            if progress:progress('准备完整结果快照…')
            write_npz(request,record,arrays,compressed=False);check_cancelled(cancel)
            if getattr(sys,'frozen',False):
                executable=Path(sys.executable).resolve().with_name('ArrayLab-cli.exe')
                command=[str(executable),'simulation-export-worker'];cwd=executable.parent
            else:
                from arraylab.framework.resources import source_root
                cwd=source_root();command=[sys.executable,'-m','arraylab','--headless','simulation-export-worker']
            env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLBACKEND='Agg')
            if getattr(sys,'frozen',False):env['PYINSTALLER_RESET_ENVIRONMENT']='1'
            else:env['PYTHONPATH']=str(cwd)
            run_owned(command+['--record',str(request),'--output',str(staging/'result'),'--caller-version',__version__],cwd=cwd,environment=env,folder=folder,
                      memory_limit_bytes=memory_budget(record['configuration']),progress=progress,cancelled=cancel)
            check_cancelled(cancel)
            if scene_png is not None:
                (staging/'result'/'scene.png').write_bytes(scene_png)
                write_json(staging/'result'/'scene-view.json',{'renderer':'PulseLab Three.js','view_state':record.get('view_state',{})})
            if target.exists() or target.is_symlink():raise FileExistsError('输出目录已被创建，未覆盖。')
            (staging/'result').rename(target)
        return target
    finally:
        if staging.exists() and staging.resolve().is_relative_to(target.parent.resolve()) and staging.name.startswith('.arraylab-export-worker-'):
            shutil.rmtree(staging)


def export_worker_main(argv=None):
    parser=Parser(description='Owned ArrayLab export worker',allow_abbrev=False)
    parser.add_argument('--record',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--caller-version',required=True)
    try:
        args=parser.parse_args(argv)
        import matplotlib;matplotlib.use('Agg',force=True)
        if args.caller_version!=__version__:raise ValueError('导出程序与界面版本不匹配。')
        record,arrays=load_record(args.record)
        print(json.dumps({'schema_version':1,'event':'progress','message':'正在导出数值、完整记录和PNG…'},ensure_ascii=False),file=sys.stderr,flush=True)
        export_result(args.output,record,arrays,plot=True)
        print(json.dumps({'status':'ok','app_version':__version__}));return 0
    except Exception as exc:
        print(json.dumps({'status':'error','error':{'type':type(exc).__name__,'message':str(exc)}},ensure_ascii=False));return 2


def run_process(feature,action,configuration,*,workers=0,progress=None,cancel=None):
    module=provider(feature);configuration=module.normalize_configuration(configuration)
    budget=memory_budget(configuration);check_cancelled(cancel)
    if getattr(sys,'frozen',False):
        executable=Path(sys.executable).resolve().with_name('ArrayLab-cli.exe')
        if not executable.is_file():raise ValueError('请完整解压包含ArrayLab-cli.exe的运行目录。')
        command=[str(executable),'simulation-worker'];cwd=executable.parent
    else:
        from arraylab.framework.resources import source_root
        cwd=source_root();command=[sys.executable,'-m','arraylab','--headless','simulation-worker']
    with temporary_directory(prefix='arraylab-radar-') as temporary:
        folder=Path(temporary);request=folder/'request.json';output=folder/'result.npz'
        write_json(request,{'feature':feature,'action':action,'configuration':configuration,'workers':requested_workers(workers)})
        env=os.environ.copy();env.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        if getattr(sys,'frozen',False):env['PYINSTALLER_RESET_ENVIRONMENT']='1'
        else:env['PYTHONPATH']=str(cwd)
        run_owned(command+['--request',str(request),'--result',str(output)],cwd=cwd,environment=env,
                  folder=folder,memory_limit_bytes=budget,progress=progress,cancelled=cancel)
        check_cancelled(cancel)
        result,arrays=read_npz(output,max_bytes=max(budget,64*1024**2))
        validate_record(result,arrays,expected_feature=feature,expected_action=action)
        if result.get('app_version')!=__version__:raise ValueError('计算程序与界面版本不同，请使用同一完整运行包。')
        if result.get('configuration_sha256')!=configuration_digest(configuration):raise ValueError('子进程返回的输入快照不一致。')
        return result,arrays


def worker_main(argv=None):
    parser=Parser(description='Owned ArrayLab numerical worker',allow_abbrev=False)
    parser.add_argument('--request',type=Path,required=True);parser.add_argument('--result',type=Path,required=True)
    try:
        args=parser.parse_args(argv);request=read_json(args.request)
        def progress(*values):
            if len(values)==1 and isinstance(values[0],dict):
                item=values[0];message=f"{max(0,min(100,item.get('progress',0)*100)):.0f}% · {item.get('stage','计算中')}"
            elif len(values)>=2 and isinstance(values[0],(int,float)) and isinstance(values[1],str):
                message=f'{max(0,min(100,values[0]*100)):.0f}% · {values[1]}'
            else:message=' · '.join(map(str,values))
            print(json.dumps({'schema_version':1,'event':'progress','message':message},ensure_ascii=False),file=sys.stderr,flush=True)
        result,arrays=calculate(request['feature'],request['action'],request['configuration'],workers=request.get('workers',0),progress=progress)
        write_npz(args.result,result,arrays,compressed=False)
        print(json.dumps({'status':'ok','app_version':__version__}));return 0
    except Exception as exc:
        print(json.dumps({'status':'error','error':{'type':type(exc).__name__,'message':str(exc)}},ensure_ascii=False));return 2


def cli_main(feature,argv=None):
    module=provider(feature)
    parser=Parser(description='ArrayLab '+feature,allow_abbrev=False)
    parser.add_argument('--config',type=Path);parser.add_argument('--action',default='analysis' if feature=='phased' else 'budget')
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--workers',type=int,default=0)
    parser.add_argument('--plot',action='store_true');parser.add_argument('--version',action='version',version='ArrayLab '+__version__)
    try:
        args=parser.parse_args(argv)
        if args.output_dir and args.output_dir.exists():raise FileExistsError('输出目录已存在，未覆盖。')
        configuration=module.default_configuration()
        if args.config:configuration=load_configuration(feature,args.config)
        result,arrays=calculate(feature,args.action,configuration,workers=args.workers)
        if args.output_dir:
            if args.plot:
                import matplotlib;matplotlib.use('Agg',force=True)
            export_result(args.output_dir,result,arrays,plot=args.plot)
        from arraylab.framework.cli_support import emit
        emit({'status':'ok','app_version':__version__,'simulation':result},sys.stdout);return 0
    except Exception as exc:
        from arraylab.framework.cli_support import emit
        emit({'status':'error','error':{'type':type(exc).__name__,'message':str(exc)}},sys.stderr)
        return 3 if isinstance(exc,FileExistsError) else 2
