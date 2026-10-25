"""Qt-free command line for the integrated stable waveform capabilities."""
import argparse
import json
from pathlib import Path
import sys
from . import __version__


class Parser(argparse.ArgumentParser):
    def error(self, message): raise ValueError(message)


def main(argv=None):
    try:
        p = Parser(description='ArrayLab FMCW TDM/DDM, A/B and distance recovery', allow_abbrev=False)
        configs = p.add_mutually_exclusive_group()
        configs.add_argument('--config', type=Path)
        configs.add_argument('--waveform-config', type=Path)
        p.add_argument('--output-dir', type=Path)
        p.add_argument('--plot', choices=('none','all','overview','phase','spectra','rd','range_recovery','pri_recovery'), default='none')
        p.add_argument('--workers', type=int, default=0)
        p.add_argument('--headless', action='store_true', help=argparse.SUPPRESS)
        p.add_argument('--version', action='version', version=f'ArrayLab {__version__} (waveform headless)')
        args = p.parse_args(argv)
        if args.plot != 'none' and args.output_dir is None: raise ValueError('--plot需要--output-dir。')
        if args.output_dir is not None and (args.output_dir.exists() or args.output_dir.is_symlink()):
            raise FileExistsError('输出目录已存在，未覆盖。')
        if sys.stdout is None and args.output_dir is None: raise ValueError('此入口没有标准输出，请指定--output-dir。')
        from .waveform import calculate_waveform, default_waveform_configuration
        from .waveform_io import load_waveform_configuration, export_waveform_directory
        from .parallel import requested_workers
        requested_workers(args.workers)
        file = args.config or args.waveform_config
        config = load_waveform_configuration(file) if file else default_waveform_configuration()
        report = calculate_waveform(config, workers=args.workers)
        result = {'status': 'ok', 'app_version': __version__, 'waveform': report}
        if args.output_dir:
            if args.plot != 'none':
                import matplotlib
                matplotlib.use('Agg', force=True)
            result['output_directory'] = str(export_waveform_directory(args.output_dir, report, args.plot))
        from .cli import emit
        emit(result, sys.stdout); return 0
    except FileExistsError as exc: code, category, message = 3, 'output_exists', str(exc)
    except (ValueError, TypeError, FileNotFoundError) as exc: code, category, message = 2, 'input', str(exc)
    except Exception as exc: code, category, message = 1, 'runtime', f'{type(exc).__name__}: {exc}'
    from .cli import emit
    emit({'status': 'error', 'error': {'type': category, 'message': message}}, sys.stderr)
    return code


if __name__ == '__main__': raise SystemExit(main())
