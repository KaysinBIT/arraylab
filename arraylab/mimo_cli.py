"""Headless radar MIMO analysis; no Qt imports or GUI startup."""
import argparse
import json
from pathlib import Path
import sys

from . import __version__


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def parser():
    result = _Parser(description='ArrayLab radar MIMO virtual-array geometry and response', allow_abbrev=False)
    config = result.add_mutually_exclusive_group()
    config.add_argument('--config', type=Path, help='Versioned MIMO parameter JSON')
    config.add_argument('--mimo-config', type=Path, help='Alias for --config')
    result.add_argument('--output-dir', type=Path, help='NEW directory; existing output is never overwritten')
    result.add_argument('--plot', choices=('none', 'layout', 'cut', 'map', 'all'), default='none')
    result.add_argument('--workers', type=int, default=0, help='0=auto, 1=serial, 2..8=bounded parallel')
    result.add_argument('--headless', action='store_true', help=argparse.SUPPRESS)
    result.add_argument('--version', action='version', version=f'ArrayLab {__version__} (MIMO headless)')
    return result


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'): stream.reconfigure(encoding='utf-8')
    try:
        args = parser().parse_args(sys.argv[1:] if argv is None else argv)
        if args.plot != 'none' and args.output_dir is None:
            raise ValueError('--plot requires --output-dir')
        from .parallel import requested_workers
        requested_workers(args.workers)
        if args.output_dir is not None and (args.output_dir.exists() or args.output_dir.is_symlink()):
            raise FileExistsError('输出目录已存在，未覆盖：' + str(args.output_dir))
        from .mimo import MIMOParameters, calculate_mimo
        from .mimo_io import load_mimo_configuration, export_mimo_directory
        source = args.config or args.mimo_config
        parameters = load_mimo_configuration(source) if source is not None else MIMOParameters()
        report = calculate_mimo(parameters, workers=args.workers)
        result = {'status': 'ok', 'app_version': __version__, 'mimo': report}
        if args.output_dir is not None:
            if args.plot != 'none':
                import matplotlib
                matplotlib.use('Agg', force=True)
            directory = export_mimo_directory(args.output_dir, report, args.plot)
            result['output_directory'] = str(directory)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except FileExistsError as exc:
        code, error = 3, str(exc)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        code, error = 2, str(exc)
    except Exception as exc:
        code, error = 1, f'{type(exc).__name__}: {exc}'
    category = {1: 'runtime', 2: 'input', 3: 'output_exists'}[code]
    print(json.dumps({'status': 'error', 'error': {'type': category, 'message': error},
                      'exit_code': code}, ensure_ascii=False), file=sys.stderr)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
