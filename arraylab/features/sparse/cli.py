"""ArrayLab entry point for the integrated Sparse2D JSON control API."""
import sys
from pathlib import Path

from arraylab import __version__
import arraylab.features.sparse.engine.headless as headless


def main(argv=None):
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ['--version']:
        print(f'ArrayLab {__version__} (Sparse2D engine 1.3.0)'); return 0
    if argv and argv[0] == 'planar':
        try:
            parser = headless.ControlParser(description='将选中稀疏方案及工况的理想布局/幅度转为平面阵电气设置')
            parser.add_argument('--input', required=True)
            parser.add_argument('--index', required=True, type=int, help='0-based result row')
            parser.add_argument('--case-index', default=0, type=int, help='0-based scenario index')
            parser.add_argument('--output', required=True, type=Path, help='New directory')
            args = parser.parse_args(argv[1:])
            from arraylab.features.sparse.planar_bridge import export_planar_handoff
            provenance = export_planar_handoff(args.output, headless.load_bundle(args.input), args.index, args.case_index)
            print(headless.dumps({'ok': True, 'output': str(args.output.absolute()), 'source': provenance})); return 0
        except Exception as exc:
            code, error = headless._error(exc)
            print(headless.dumps({'ok': False, 'error': error, 'exit_code': code})); return code
    return headless.main(argv)
