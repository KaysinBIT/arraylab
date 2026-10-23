"""Headless electrical scan studies using saved linear or planar designs."""
from dataclasses import asdict
from pathlib import Path
import sys

from . import __version__
from .cli import Parser, UsageError, emit
from .feed import load_feed_configuration
from .planar import load_planar_configuration
from .study import StudySettings, load_study_configuration, run_study, export_study_directory


def main(argv=None):
    try:
        parser = Parser(description="ArrayLab electrical scan study", allow_abbrev=False)
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--study", choices=["scan"], help="Multi-angle electrical scan; no error statistics")
        parser.add_argument("--version", action="version", version=f"ArrayLab {__version__} (electrical scan)")
        source = parser.add_mutually_exclusive_group(required=True)
        source.add_argument("--study-config", type=Path, help="Saved complete scan study configuration")
        source.add_argument("--feed-config", type=Path, help="Saved linear feed electrical design")
        source.add_argument("--planar-config", type=Path, help="Saved rectangular/arbitrary planar design")
        parser.add_argument("--study-count", dest="count", type=int, help="2..200 scan points, inclusive endpoints")
        parser.add_argument("--workers", type=int, default=0, help="0=auto up to 4; 1=serial; 2..8=bounded parallel scan points")
        parser.add_argument("--scan-axis", choices=["theta", "phi"], help="phi is available only for planar arrays")
        parser.add_argument("--scan-start", dest="start_deg", type=float)
        parser.add_argument("--scan-stop", dest="stop_deg", type=float)
        parser.add_argument("--output-dir", type=Path, help="New result directory; existing directories are never overwritten")
        parser.add_argument("--plot", choices=["none", "all"], default="none", help="Agg PNG export (no display required)")
        args = parser.parse_args(argv)
        if args.plot != "none" and args.output_dir is None:
            raise UsageError("--plot需要--output-dir。")
        if sys.stdout is None and args.output_dir is None:
            raise UsageError("此入口没有标准输出，请指定--output-dir。")
        if args.output_dir is not None and args.output_dir.exists():
            raise FileExistsError("输出目录已存在，请选择新目录。")
        if args.study_config:
            p, settings, study = load_study_configuration(args.study_config)
        else:
            p, settings = load_feed_configuration(args.feed_config) if args.feed_config else load_planar_configuration(args.planar_config)
            study = StudySettings()
        values = asdict(study)
        for key in values:
            if getattr(args, key) is not None:
                values[key] = getattr(args, key)
        report = run_study(p, settings, StudySettings(**values), workers=args.workers)
        if args.output_dir is not None:
            export_study_directory(args.output_dir, report, plot=args.plot == "all")
        emit({"status": "ok", "mode": "headless_scan_study", "study": report,
              "output_directory": str(args.output_dir.absolute()) if args.output_dir else None}, sys.stdout)
        return 0
    except (ValueError, TypeError, OverflowError, UnicodeError) as exc:
        emit({"status": "error", "error": {"type": "input", "message": str(exc)}}, sys.stderr); return 2
    except OSError as exc:
        emit({"status": "error", "error": {"type": "io", "message": str(exc)}}, sys.stderr); return 3
    except Exception as exc:
        emit({"status": "error", "error": {"type": "runtime", "message": str(exc)}}, sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
