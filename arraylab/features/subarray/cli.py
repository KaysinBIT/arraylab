"""Qt-free command-line interface for continuous subarray-center design."""
import argparse
import json
from pathlib import Path
import sys

from arraylab.framework.cli_support import emit, Parser
from arraylab.framework.parallel import ComputationCancelled
from arraylab.features.subarray.service import SubarrayConfig, export_subarray, load_config, optimize_subarrays, save_config


def main(argv=None):
    parser = Parser(description='Optimize continuous planar subarray center positions; internally steerable elements', allow_abbrev=False)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path, help='New output directory')
    parser.add_argument('--plot', choices=('none', 'layout', 'pattern', 'cuts', '3d', 'all'), default='none')
    parser.add_argument('--workers', type=int, default=0, help='0: automatic, 1: serial, 2–8: bounded threads')
    parser.add_argument('--write-example', type=Path)
    try:
        args = parser.parse_args(argv)
        if args.write_example:
            if args.write_example.exists():
                raise ValueError('示例文件已存在。')
            save_config(args.write_example, SubarrayConfig())
            emit({'ok': True, 'configuration': str(args.write_example)}, sys.stdout)
            return 0
        if args.output is None:
            raise ValueError('请指定 --output 新目录，或使用 --write-example。')
        if args.output.exists():
            raise ValueError('输出目录已存在；请选择新目录。')
        config = load_config(args.config) if args.config else SubarrayConfig()
        report = optimize_subarrays(config, workers=args.workers)
        export_subarray(args.output, report, plot=args.plot)
        emit({'ok': True, 'output': str(args.output.absolute()),
                          'subarray_count': config.count, 'element_count': config.count*config.nx*config.ny,
                          'initial_psl_db': report['initial']['worst_psl_db'],
                          'best_psl_db': report['best']['worst_psl_db'], 'execution': report['execution']}, sys.stdout)
        return 0
    except (ValueError, OSError, ArithmeticError, ComputationCancelled) as exc:
        emit({'ok': False, 'error': str(exc)}, sys.stdout)
        return 2
    except ImportError as exc:
        emit({'ok': False, 'error': '绘图导出需要安装 plotting 依赖：'+str(exc)}, sys.stdout)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
