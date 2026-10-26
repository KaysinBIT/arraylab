"""Configuration and artifact exchange without an HTTP or browser dependency."""
from pathlib import Path
import math
import tempfile

import numpy as np

from arraylab.shared.io import write_text_atomic
from arraylab.shared.simulation_io import clean_json, read_json, write_json, write_npz, read_npz
from .service import normalize_configuration, ACTIONS
from .config import config_hash
from .reporting import render_report, result_csv


def save_configuration(path, configuration):
    document={'format':'arraylab.equation','schema_version':1,
              'configuration':normalize_configuration(configuration)}
    return write_json(path,document,complex_style='pair')


def load_configuration(path):
    path=Path(path)
    if path.suffix.lower()=='.npz':
        document,_=read_npz(path)
    else:
        document=read_json(path,max_bytes=2*1024*1024)
    if isinstance(document,dict) and document.get('kind') in ACTIONS and 'config' in document:
        config=normalize_configuration(document['config'])
        if document.get('config_hash') is not None and document['config_hash']!=config_hash(config):
            raise ValueError('雷达方程结果中的配置摘要不匹配。')
        return config
    return normalize_configuration(document)


def plot_data(result, arrays):
    """Bounded display data; original numerical arrays remain untouched."""
    out={'detections':result.get('detections',[]),'points':result.get('points',[])}
    if 'rd_power' in arrays:
        power=np.asarray(arrays['rd_power']);vs=max(1,math.ceil(power.shape[0]/128));rs=max(1,math.ceil(power.shape[1]/512))
        out.update(ranges_m=np.asarray(arrays['ranges_m'])[::rs],velocities_mps=np.asarray(arrays['velocities_mps'])[::vs],
                   rd_power_db=10*np.log10(np.maximum(power[::vs,::rs],1e-300)),
                   display_stride=[vs,rs],full_shape=list(power.shape))
        if 'noise_power' in arrays:
            noise=np.broadcast_to(arrays['noise_power'],power.shape)
            out['snr_db']=10*np.log10(np.maximum(power[::vs,::rs]/np.maximum(noise[::vs,::rs],1e-300),1e-300))
        if 'cfar_threshold' in arrays:
            out['threshold_db']=10*np.log10(np.maximum(np.asarray(arrays['cfar_threshold'])[::vs,::rs],1e-300))
    if 'noise_trial_power' in arrays:
        out['noise_trial_power']=np.asarray(arrays['noise_trial_power'])[:10000]
    return clean_json(out,complex_style='pair')


def export_result(path, result, arrays, *, plot=False):
    """Write all source result formats to a new directory, without overwrites."""
    target=Path(path).expanduser().absolute()
    if target.exists() or target.is_symlink():raise FileExistsError('输出目录已存在，未覆盖。')
    config=normalize_configuration(result.get('config'))
    if result.get('config_hash')!=config_hash(config):raise ValueError('结果配置摘要不一致。')
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.arraylab-equation-',dir=target.parent))
    try:
        write_json(staging/'result.json',result,complex_style='pair')
        write_npz(staging/'arrays.npz',result,arrays,complex_style='pair')
        save_configuration(staging/'parameters.json',config)
        write_text_atomic(staging/'results.csv',result_csv(result))
        write_text_atomic(staging/'report.html',render_report(result,arrays))
        if plot:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from .presentation import draw, available_views
            for view in available_views(result,arrays):
                figure=Figure(figsize=(10,6),dpi=140);FigureCanvasAgg(figure)
                draw(figure,result,arrays,view)
                figure.savefig(staging/(view+'.png'));figure.clear()
        if target.exists() or target.is_symlink():raise FileExistsError('输出目录已存在，未覆盖。')
        staging.rename(target)
    finally:
        if staging.exists():
            for child in staging.iterdir():child.unlink()
            staging.rmdir()
    return target
