"""Headless rectangular/arbitrary planar-array electrical design."""
from dataclasses import asdict
from pathlib import Path
import sys

from .cli import parser as base_parser, UsageError, emit, feed_settings_from_args
from .planar import PlanarParameters, load_planar_configuration, read_coordinates_csv, design_planar, export_planar_directory


def main(argv=None):
    try:
        parser = base_parser(planar=True)
        parser.description = "ArrayLab planar electrical design (isotropic point elements)"
        parser.add_argument("--planar", action="store_true", help="Use the planar-array workflow")
        parser.add_argument("--planar-config", type=Path, help="Planar electrical configuration JSON")
        parser.add_argument("--geometry", choices=["rectangular", "arbitrary"])
        parser.add_argument("--nx", type=int); parser.add_argument("--ny", type=int)
        parser.add_argument("--spacing-x", type=float); parser.add_argument("--spacing-y", type=float)
        parser.add_argument("--coordinate-unit", choices=["lambda", "mm"])
        parser.add_argument("--coordinates", type=Path, help="element,x,y CSV; implies arbitrary geometry")
        parser.add_argument("--scan-theta", dest="scan_theta_deg", type=float, help="Target beam theta, 0..90 deg from +z; use --feed-phase scan for compensation")
        parser.add_argument("--scan-phi", dest="scan_phi_deg", type=float, help="Target beam phi, -180..180 deg from +x towards +y")
        parser.add_argument("--cut-phi", dest="cut_phi_deg", type=float, help="Azimuth of signed-theta 2-D cut")
        args = parser.parse_args(argv)
        from .parallel import requested_workers
        requested_workers(args.workers)
        if args.config or args.feed_config:
            raise UsageError("平面阵使用--planar-config，不能同时使用直线阵--config或--feed-config。")
        if any(getattr(args, key) is not None for key in ("elements", "spacing", "spacing_unit", "scan_deg")) or args.include_phase:
            raise UsageError("平面阵请使用--nx/--ny、--spacing-x/y、--scan-theta/phi或坐标CSV；不使用直线阵参数及--include-phase。")
        if args.plot != "none" and args.output_dir is None:
            raise UsageError("--plot需要--output-dir。")
        if sys.stdout is None and args.output_dir is None:
            raise UsageError("此入口没有标准输出，请指定--output-dir。")
        p, settings = load_planar_configuration(args.planar_config) if args.planar_config else (PlanarParameters(), None)
        values = asdict(p)
        for key in values:
            if key != "coordinates" and getattr(args, key, None) is not None:
                values[key] = getattr(args, key)
        if args.geometry == "rectangular": values["coordinates"] = ()
        if args.coordinates:
            if args.geometry == "rectangular": raise UsageError("坐标CSV不能与rectangular类型同时指定。")
            values.update(geometry="arbitrary", coordinates=read_coordinates_csv(args.coordinates))
        p = PlanarParameters(**values); p.validate()
        settings = feed_settings_from_args(args, p.array_parameters(), settings)
        report = design_planar(p, settings, workers=args.workers)
        if args.output_dir is not None: export_planar_directory(args.output_dir, report, args.plot)
        emit({"status":"ok","mode":"headless_planar","planar":report,
              "output_directory":str(args.output_dir.absolute()) if args.output_dir else None}, sys.stdout)
        return 0
    except (ValueError,TypeError,OverflowError,UnicodeError) as exc:
        emit({"status":"error","error":{"type":"input","message":str(exc)}},sys.stderr); return 2
    except OSError as exc:
        emit({"status":"error","error":{"type":"io","message":str(exc)}},sys.stderr); return 3
    except Exception as exc:
        emit({"status":"error","error":{"type":"runtime","message":str(exc)}},sys.stderr); return 1


if __name__ == "__main__":
    raise SystemExit(main())
