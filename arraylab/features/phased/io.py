"""Phased-specific tables; all atomic JSON and NPZ mechanics are shared."""
from pathlib import Path
import csv,uuid
from arraylab.shared.simulation_io import clean_json,read_json,write_json,write_npz
from .service import normalize_configuration

def load_configuration(path):return normalize_configuration(read_json(path))
def save_configuration(path,configuration):
    return write_json(path,{'format':'arraylab.phased','schema_version':1,'configuration':normalize_configuration(configuration)})

def export_result(path,result,arrays,plot='none'):
    """Write one new directory from a completed snapshot; never recompute."""
    allowed=('none','all','layout','cuts','uv','scan','optimization')
    if plot not in allowed:raise ValueError('相控阵图形导出选项无效。')
    target=Path(path)
    if target.exists() or target.is_symlink():raise FileExistsError('输出目录已存在，不覆盖。')
    target.parent.mkdir(parents=True,exist_ok=True)
    staging=target.parent/('.arraylab-phased-'+uuid.uuid4().hex);staging.mkdir()
    try:
        result=clean_json(result)
        write_json(staging/'result.json',result);save_configuration(staging/'parameters.json',result['configuration'])
        write_npz(staging/'arrays.npz',result,arrays)
        from .presentation import table
        for view in ('elements','scan','cases'):
            headers,rows,note=table(result,arrays,view)
            if rows:
                with (staging/(view+'.csv')).open('w',encoding='utf-8-sig',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(headers);writer.writerows(rows)
        if plot!='none':
            import matplotlib
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from .presentation import draw
            views=['layout','cuts','uv']+(['scan'] if result['scan'] else [])+(['optimization'] if result['synthesis'] else []) if plot=='all' else [plot]
            with matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans']}):
                for view in views:
                    figure=Figure(figsize=(10,7),dpi=140);FigureCanvasAgg(figure)
                    draw(figure,result,arrays,view);figure.savefig(staging/(view+'.png'));figure.clear()
        if target.exists() or target.is_symlink():raise FileExistsError('输出目录已被创建，不覆盖。')
        staging.rename(target)
    finally:
        if staging.exists():
            for item in staging.iterdir():item.unlink()
            staging.rmdir()
    return target
