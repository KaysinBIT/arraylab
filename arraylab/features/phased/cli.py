"""Headless phased-array analysis and synthesis; native JSON/NPZ output."""
import sys
from pathlib import Path
from arraylab import __version__
from .service import execute,default_configuration,normalize_configuration

def main(argv=None):
    from arraylab.framework.cli_support import Parser,emit
    try:
        p=Parser(description='ArrayLab phased array analysis, scan and synthesis',allow_abbrev=False)
        p.add_argument('--config',type=Path);p.add_argument('--action',choices=['analysis','scan','optimize1d','optimize2d'],default='analysis')
        p.add_argument('--output-dir',type=Path);p.add_argument('--workers',type=int,default=0)
        p.add_argument('--plot',choices=('none','all','layout','cuts','uv','scan','optimization'),default='none')
        p.add_argument('--headless',action='store_true');p.add_argument('--version',action='version',version=f'ArrayLab {__version__} (phased)')
        args=p.parse_args(argv)
        if args.plot!='none' and not args.output_dir:raise ValueError('--plot需要--output-dir。')
        if args.output_dir and args.output_dir.exists():raise FileExistsError('输出目录已存在，不覆盖。')
        from arraylab.shared.simulation_io import read_json
        configuration=read_json(args.config) if args.config else default_configuration()
        result,arrays=execute(args.action,configuration,workers=args.workers)
        from arraylab.shared.simulation_io import clean_json
        result=clean_json(result)
        if args.output_dir:
            from .io import export_result
            export_result(args.output_dir,result,arrays,plot=args.plot)
        emit({'status':'ok','phased':result,'array_keys':list(arrays)},sys.stdout);return 0
    except FileExistsError as exc:code,kind,error=3,'output_exists',str(exc)
    except (ValueError,TypeError,KeyError,OSError) as exc:code,kind,error=2,'input',str(exc)
    emit({'status':'error','error':{'type':kind,'message':error}},sys.stderr);return code

if __name__=='__main__':raise SystemExit(main())
