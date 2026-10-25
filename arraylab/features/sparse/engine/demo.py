"""Legacy sparse command options and complete offline report renderer."""
import os
# Bound numerical-library worker counts before importing NumPy.
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse
import csv
from datetime import datetime
import html
import json
from pathlib import Path
import sys

import numpy as np
from arraylab.features.sparse.engine.core import Config, run, scan_uv, structured_arrays, coarray
from arraylab.features.sparse.engine.scenarios import LIGHT_MM_GHZ, case_config, scenarios
from arraylab.features.sparse.engine.elements import MODELS, build_element
from arraylab.features.sparse.engine.runtime_paths import data_directory
from arraylab.features.sparse.engine.phase_quantization import excitation_phases
from arraylab.features.sparse.engine.reporting import export_bundle, _export_bundle






def parser():
    p=argparse.ArgumentParser(description='Standalone planar sparse-array research demo',
        epilog='JSON headless control: python demo.py --headless --help (or python headless.py --help)')
    p.add_argument('--gui',action='store_true',help='Open independent desktop experiment window')
    p.add_argument('--output',type=Path,help='Create a new result directory; never overwrite')
    p.add_argument('--weight-method',choices=['none','minimax','taylor','chebyshev','gaussian','uniform'],default='none')
    p.add_argument('--weight-source',choices=['dense','regular','random','greedy','annealing','l1-topk','l1-weighted'],default='annealing')
    p.add_argument('--weight-dr-db',type=float,default=20.)
    p.add_argument('--weight-sll',type=float,default=30.)
    p.add_argument('--weight-nbar',type=int,default=4)
    p.add_argument('--weight-sigma',type=float,default=.35)
    defaults=Config()
    for key in Config.__dataclass_fields__:
        value=getattr(defaults,key)
        choices={'hole_shape':['none','rectangle','circle'],'element_model':list(MODELS),
                 'element_import_mode':['grid','axisymmetric'],'element_dipole_axis':['x','y','z']}
        extra={'choices':choices[key]} if key in choices else {}
        p.add_argument('--'+key.replace('_','-'),type=type(value),default=value,**extra)
    return p


def main():
    if len(sys.argv)>1 and sys.argv[1]=='--headless':
        from arraylab.features.sparse.engine.headless import main as control_main
        return control_main(sys.argv[2:])
    p=parser(); args=p.parse_args()
    values={k:getattr(args,k) for k in Config.__dataclass_fields__}
    try: config=Config(**values).validate()
    except ValueError as exc: p.error(str(exc))
    from arraylab.features.sparse.engine.weighting import Weighting, add_weighting
    settings=Weighting(method=args.weight_method,dr_db=args.weight_dr_db,sll_db=args.weight_sll,nbar=args.weight_nbar,sigma=args.weight_sigma)
    if args.weight_method!='none':
        try: settings.validate(config)
        except ValueError as exc: p.error(str(exc))
    if args.gui:
        p.error('Open the native ArrayLab workbench and press F8 for sparse synthesis; no standalone sparse GUI is provided.')
    target=args.output or data_directory()/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')
    if target.exists():
        print(f'Output already exists: {target}',file=sys.stderr); return 3
    try:
        bundle=run(config,lambda msg:print(msg,file=sys.stderr,flush=True))
        if args.weight_method!='none':
            source_name={'dense':'Hollow uniform' if config.hole_shape!='none' else 'Dense reference',
                'regular':'Regular thinning','random':'Random symmetric','greedy':'Greedy exchange',
                'annealing':'Simulated annealing','l1-topk':'L1 top-K uniform','l1-weighted':'L1 weighted solution'}[args.weight_source]
            source_index=0 if args.weight_source=='dense' else next((i for i,r in enumerate(bundle['results']) if r['name']==source_name),None)
            if source_index is None: raise ValueError('Selected weighting source is unavailable for this geometry')
            bundle=add_weighting(bundle,source_index,settings,lambda msg:print(msg,file=sys.stderr,flush=True))
        export_bundle(bundle,target)
    except (OSError,RuntimeError,ValueError) as exc:
        print(str(exc),file=sys.stderr); return 1
    print(json.dumps({'output':str(target.resolve()),'results':[
        {'method':r['name'],**r['metrics'],'phase_quantization':r.get('phase_quantization',{}).get('metrics')} for r in bundle['results']]},ensure_ascii=False,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
