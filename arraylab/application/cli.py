"""Concrete CLI composition; framework owns only route mechanics."""
import sys
from arraylab.framework.registry import CliRoute, dispatch
def _waveform():
    from arraylab.features.waveform import main
    return main
def _mimo():
    from arraylab.features.mimo import main
    return main
def _subarray():
    from arraylab.features.subarray import main
    return main
def _sparse():
    from arraylab.features.sparse import main
    return main
def _study():
    from arraylab.features.studies import main
    return main
def _planar():
    from arraylab.features.planar import main
    return main
def _default():
    from .linear_feed_cli import main
    return main
def _simulation(feature):
    def load():
        from .simulation import cli_main
        return lambda argv:cli_main(feature,argv)
    return load
def _simulation_worker():
    from .simulation import worker_main
    return worker_main
def _simulation_export_worker():
    from .simulation import export_worker_main
    return export_worker_main
def _routed(argv):return [v for v in argv if v!='--headless']
def _command(name):return lambda argv:bool(_routed(argv)) and _routed(argv)[0]==name
def _flag(*names):return lambda argv:any(v in names or any(v.startswith(n+'=') for n in names) for v in argv)
def _subcommand(argv):return _routed(argv)[1:]
def _sparse_args(argv):
    result=_routed(argv);result.remove('--sparse');return result
CLI_ROUTES=(
    CliRoute('simulation-export-worker',_command('simulation-export-worker'),_simulation_export_worker,_subcommand),
    CliRoute('simulation-worker',_command('simulation-worker'),_simulation_worker,_subcommand),
    CliRoute('phased-command',_command('phased'),_simulation('phased'),_subcommand),
    CliRoute('equation-command',_command('equation'),_simulation('equation'),_subcommand),
    CliRoute('pulse-command',_command('pulse'),_simulation('pulse'),_subcommand),
    CliRoute('waveform-command',_command('waveform'),_waveform,_subcommand),
    CliRoute('waveform-config',_flag('--waveform-config'),_waveform,_routed),
    CliRoute('mimo-command',_command('mimo'),_mimo,_subcommand),
    CliRoute('mimo-config',_flag('--mimo-config'),_mimo,_routed),
    CliRoute('subarray-command',_command('subarray'),_subarray,_subcommand),
    CliRoute('sparse',lambda argv:'--sparse' in argv,_sparse,_sparse_args),
    CliRoute('study',_flag('--study','--study-config'),_study),
    CliRoute('planar',lambda argv:any(v in ('--planar','--planar-config') or v.startswith('--planar-config=') for v in argv),_planar),
)
def main(argv=None):return dispatch(CLI_ROUTES,list(sys.argv[1:] if argv is None else argv),_default)
def __getattr__(name):
    if name.startswith("__"): raise AttributeError(name)
    from . import linear_feed_cli
    return getattr(linear_feed_cli,name)
if __name__=='__main__':raise SystemExit(main())
