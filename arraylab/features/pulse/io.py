"""Pulse project compatibility and native result exports; no HTTP server."""
import csv
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from arraylab.shared.io import write_text_atomic
from .config import normalize_project
from .reporting import render_html_report

MAX_PROJECT_BYTES=128*1024**2
MAX_UNPACKED_BYTES=256*1024**2


def _pairs(items):
    result={}
    for key,value in items:
        if key in result:raise ValueError('重复JSON字段：'+key)
        result[key]=value
    return result


def _loads(data):
    if isinstance(data,bytes):data=data.decode('utf-8-sig')
    return json.loads(data,object_pairs_hook=_pairs,
        parse_constant=lambda value:(_ for _ in ()).throw(ValueError('JSON含非有限数值：'+value)))


def import_project(path_or_bytes):
    if isinstance(path_or_bytes,(bytes,bytearray)):data=bytes(path_or_bytes)
    else:
        with Path(path_or_bytes).open('rb') as stream:data=stream.read(MAX_PROJECT_BYTES+1)
    if len(data)>MAX_PROJECT_BYTES:raise ValueError('工程包超过128 MiB。')
    try:
        if data[:2]==b'PK':
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                names=archive.namelist()
                if len(names)!=len(set(names)):raise ValueError('工程包包含重复文件名。')
                if sum(info.file_size for info in archive.infolist())>MAX_UNPACKED_BYTES:raise ValueError('解压后工程过大。')
                project=_loads(archive.read('project.json'))
                reference=project.get('scene',{}).get('terrain',{}).get('resource')
                if reference not in (None,'terrain.json'):raise ValueError('不支持的工程地形资源引用。')
                if reference=='terrain.json':
                    project['scene']['terrain']=_loads(archive.read('terrain.json'))
        else:project=_loads(data)
        if project.get('format')=='arraylab.pulse.project':
            if type(project.get('schema_version')) is not int or project['schema_version']!=1:raise ValueError('不支持的ArrayLab脉冲工程版本。')
            project=project['configuration']
        elif 'format' in project:
            raise ValueError('不支持此工程封装；请先按对应模块解析配置。')
        from .service import normalize_configuration
        project=normalize_configuration(project)
        json.dumps(project,allow_nan=False)
        return project
    except (ValueError,KeyError,TypeError,AttributeError,UnicodeError,zipfile.BadZipFile) as exc:
        raise ValueError('无法导入脉冲工程：'+str(exc)) from exc


def project_bytes(configuration):
    project=normalize_project(configuration);buffer=io.BytesIO();terrain=project['scene']['terrain']
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as archive:
        if terrain.get('kind')=='dem':
            archive.writestr('terrain.json',json.dumps(terrain,ensure_ascii=False,allow_nan=False))
            project['scene']['terrain']={'resource':'terrain.json'}
        archive.writestr('project.json',json.dumps(project,ensure_ascii=False,allow_nan=False))
    return buffer.getvalue()


def export_project(path,configuration):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix.lower()=='.json':
        from arraylab.shared.simulation_io import write_json
        write_json(path,normalize_project(configuration),complex_style='object')
    else:
        import os
        temporary=None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.'+path.name+'.',suffix='.tmp',delete=False) as stream:
                temporary=Path(stream.name);stream.write(project_bytes(configuration));stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,path)
        finally:
            if temporary is not None:temporary.unlink(missing_ok=True)
    return path


def csv_text(result,arrays):
    buffer=io.StringIO();writer=csv.writer(buffer)
    if result.get('kind')=='statistics':
        writer.writerow(['trial_index','target_cut_detected','noise_cut_false_alarm','target_associated'])
        writer.writerows(np.asarray(arrays.get('trial_results',[])).tolist())
    elif result.get('kind')=='inverse':
        writer.writerow(['rank','passed','reason','average_power_w','cpi_s','range_resolution_m','pd_estimate','pd_lower95','pfa_estimate','pfa_upper95'])
        for candidate in result.get('candidates',[]):
            metrics=candidate['metrics'];verification=candidate.get('verification',{})
            writer.writerow([candidate['rank'],candidate['passed'],candidate['reason'],metrics.get('average_power_w'),metrics.get('cpi_s'),metrics.get('range_resolution_m'),verification.get('pd_estimate'),verification.get('pd_ci95',['',''])[0],verification.get('pfa_estimate'),verification.get('pfa_ci95',['',''])[1]])
    elif result.get('kind')=='budget':
        writer.writerow(['metric','value'])
        writer.writerows((key,value) for key,value in result.get('metrics',{}).items() if not isinstance(value,(dict,list)))
    else:
        keys=['range_m','velocity_mps','azimuth_deg','elevation_deg','snr_db'];writer.writerow(keys)
        writer.writerows([row.get(key,'') for key in keys] for row in result.get('detections',[]))
    return '\ufeff'+buffer.getvalue()


def export_result(path,result,arrays):
    from arraylab.shared.simulation_io import write_json,write_npz
    path=Path(path).expanduser().absolute()
    if path.exists() or path.is_symlink():raise FileExistsError('输出目录已存在，未覆盖。')
    path.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.arraylab-pulse-',dir=path.parent))
    try:
        write_json(staging/'summary.json',result,complex_style='object')
        export_project(staging/'project.json',result['configuration'])
        export_project(staging/'project.prlab',result['configuration'])
        write_npz(staging/'arrays.npz',result,arrays,complex_style='object')
        write_text_atomic(staging/'results.csv',csv_text(result,arrays))
        # The portable project carries large DEM grids; do not duplicate those
        # input arrays in the human-readable report's details section.
        write_text_atomic(staging/'report.html',render_html_report({k:v for k,v in result.items() if k!='configuration'}))
        if path.exists() or path.is_symlink():raise FileExistsError('输出目录已被创建，未覆盖。')
        staging.rename(path)
    finally:
        if staging.exists():
            # Only the six files created inside this generated sibling folder.
            for child in staging.iterdir():child.unlink()
            staging.rmdir()
    return path
