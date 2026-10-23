"""GUI-free command line execution. No Qt or GUI backend is imported here."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import uuid

from . import __version__


class UsageError(ValueError):
    pass


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise UsageError(message)


def port_list(value):
    if not value.strip():
        return ()
    try:
        return tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise UsageError("失效端口须为逗号分隔的整数，如1,3。") from exc


def parser(planar=False):
    p = Parser(description="ArrayLab headless uniform linear-array analysis", allow_abbrev=False)
    if not planar:
        p.add_argument("--study", choices=["scan"], help="Electrical multi-angle scan: use --study scan --help")
        p.add_argument("--study-config", type=Path, help="Saved multi-angle electrical scan configuration")
        p.add_argument("--planar", action="store_true", help="Planar arrays: use --planar --help for geometry and plot options")
        p.add_argument("--planar-config", type=Path, help="Run a saved planar electrical configuration")
    p.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--version", action="version", version=f"ArrayLab {__version__} (headless)")
    p.add_argument("--config", type=Path, help="Versioned analysis JSON or legacy parameter JSON; explicit arguments override it")
    p.add_argument("--feed-config", type=Path, help="Feed impedance configuration JSON; cannot combine with --config")
    p.add_argument("--feed-network", choices=["corporate", "series", "symmetric_series"], help="Quarter-wave feed impedances; symmetric_series requires an even element count")
    p.add_argument("--feed-input-ohm", type=float, help="Real input resistance (default: 50)")
    p.add_argument("--feed-junction-ohm", type=float, help="Real internal junction resistance (default: 50)")
    p.add_argument("--feed-load-ohm", type=float, help="Equal real element load resistance (default: 50)")
    p.add_argument("--feed-min-line-ohm", type=float, help="Advisory line impedance lower bound (default: 20)")
    p.add_argument("--feed-max-line-ohm", type=float, help="Advisory line impedance upper bound (default: 120)")
    p.add_argument("--feed-phase", dest="feed_phase_mode", choices=["none", "equalize", "scan"], help="Terminal phase compensation (new design default: scan; legacy v1: none)")
    p.add_argument("--feed-weighting", choices=["uniform", "chebyshev", "taylor", "custom"], help="Target amplitude taper")
    p.add_argument("--feed-sidelobe-db", type=float, help="Chebyshev/Taylor sidelobe parameter, positive dB")
    p.add_argument("--feed-taylor-nbar", type=int, help="Taylor near-in sidelobe count")
    p.add_argument("--feed-excitation", type=Path, help="Strict element,amplitude,phase_deg CSV; phase is an additional target offset")
    p.add_argument("--feed-phase-bits", type=int, help="0=continuous; 1..12 phase-control bits")
    p.add_argument("--feed-amplitude-error-db", type=float, help="Gaussian amplitude error standard deviation in dB")
    p.add_argument("--feed-phase-error-deg", type=float, help="Gaussian phase error standard deviation in degrees")
    p.add_argument("--feed-failed-ports", type=port_list, help="Comma-separated matched nonradiating port numbers")
    p.add_argument("--feed-seed", type=int, help="Seed for one reproducible error realization")
    p.add_argument("--elements", type=int, help="Number of isotropic elements, 1..200 (default: 8)")
    p.add_argument("--frequency-ghz", type=float, help="Operating frequency in GHz (default: 3)")
    p.add_argument("--spacing", type=float, help="Center-to-center spacing (default: 0.5)")
    p.add_argument("--spacing-unit", choices=["lambda", "mm"], help="Spacing unit (default: lambda)")
    p.add_argument("--scan-deg", type=float, help="Signed scan angle from +z towards +x (default: 0)")
    p.add_argument("--step-deg", type=float, help="Pattern sampling interval, degrees (default: 0.1)")
    p.add_argument("--output-dir", type=Path, help="NEW directory for summary.json, pattern.dat and excitation.csv; never overwrite an existing directory")
    p.add_argument("--include-phase", action="store_true", help="Include the optional phase column in pattern.dat")
    p.add_argument("--plot", choices=["none", "cut", "map", "3d", "all"] if planar else ["none", "rectangular", "polar", "both"], default="none",
                   help="Write PNG plots using Agg; requires --output-dir (default: none)")
    return p


def load_parameters(args, base=None):
    from .core import ArrayParameters
    from .configuration import load_configuration, parameters_from_values
    values = asdict(load_configuration(args.config) if args.config else base or ArrayParameters())
    for key in values:
        supplied = getattr(args, key, None)
        if supplied is not None:
            values[key] = supplied
    return parameters_from_values(values)


def summary(result):
    p = result.parameters
    return {
        "schema_version": 1,
        "app_version": __version__,
        "status": "ok",
        "mode": "headless",
        "parameters": asdict(p),
        "model": {
            "array": "uniform_linear", "element": "isotropic", "amplitude": "uniform",
            "mutual_coupling": False, "ground_plane": False, "array_axis": "x",
            "cut": "x-z, z>=0", "angle_reference": "from +z towards +x",
            "phase_reference": "array_center", "pattern_normalization": "theoretical_peak_0_dB",
            "data_floor_db": -300, "directivity_integral": "full_sphere",
        },
        "metrics": {
            "directivity_dbi": result.directivity_dbi,
            "directivity_linear": 10 ** (result.directivity_dbi / 10),
            "main_lobe_angle_deg": result.peaks_deg[0] if len(result.peaks_deg) == 1 else None,
            "equal_peak_angles_deg": list(result.peaks_deg),
            "half_power_beamwidth_deg": result.beamwidth_deg,
            "half_power_edges_deg": list(result.half_power_deg) if result.half_power_deg else None,
            "wavelength_mm": p.wavelength_mm,
            "spacing_mm": p.spacing_mm,
            "spacing_lambda": p.spacing_lambda,
            "progressive_phase_deg": -360 * p.spacing_lambda * math.sin(math.radians(p.scan_deg)),
            "sample_count": len(result.angles_deg),
        },
        "notices": list(result.notices),
        "output_directory": None,
        "files": {},
    }


def render_plots(result, directory, mode, prefix="", patterns=None):
    # This is a file renderer, not a hidden/offscreen Qt application. Do not
    # import pyplot, widgets.py or any Qt backend, even when MPLBACKEND=QtAgg.
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import numpy as np

    kinds = ["rectangular", "polar"] if mode == "both" else [mode]
    for kind in kinds:
        with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.usetex": False}):
            figure = Figure(figsize=(8, 5), dpi=150, layout="constrained")
            FigureCanvasAgg(figure)
            polar = kind == "polar"
            ax = figure.add_subplot(111, projection="polar" if polar else None)
            curves = [(None, result.pattern_db)] if patterns is None else [
                ("Target", patterns["target"]["pattern_db"]), ("Uncompensated", patterns["uncompensated"]["pattern_db"]),
                ("Output", patterns["output"]["pattern_db"])]
            for (label, values), color in zip(curves, ["#087fa8", "#d77a28", "#6c5bc4"]):
                data = np.maximum(values, -40)
                ax.plot(np.deg2rad(result.angles_deg) if polar else result.angles_deg,
                        data + 40 if polar else data, color=color, linewidth=1.8, label=label)
            if patterns is not None:
                ax.legend(fontsize=8)
            if polar:
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(-1)
                ax.set_thetamin(-90)
                ax.set_thetamax(90)
                ax.set_ylim(0, 40)
                ax.set_yticks([0, 10, 20, 30, 40])
                ax.set_yticklabels([])
                for level in [-30, -20, -10, 0]:
                    ax.text(math.radians(-45), level + 39.4, str(level), ha="right", va="center", fontsize=8,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85, "pad": .8})
                ax.set_xlabel("Normalized pattern (dB); theta from +z towards +x")
            else:
                ax.set_xlim(-90, 90)
                ax.set_ylim(-40, 2)
                ax.set_xlabel("Theta (deg), from +z towards +x")
                ax.set_ylabel("Target coherent-sum reference (dB)" if patterns else "Normalized pattern (dB)")
            p = result.parameters
            ax.set_title(f"ULA: N={p.elements}, d={p.spacing_lambda:g} lambda, scan={p.scan_deg:g} deg")
            ax.grid(True, color="#dce5ed", linewidth=.75)
            figure.savefig(directory / f"{prefix}{kind}.png")
            figure.clear()


def save_outputs(result, report, args):
    import numpy as np
    from .dataio import write_pattern

    target = args.output_dir.expanduser().absolute()
    if target.exists():
        raise FileExistsError("输出目录已存在，请指定新的目录，避免覆盖已有结果。")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / (".arraylab-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        write_pattern(staging / "pattern.dat", result.angles_deg, result.pattern_db,
                      result.phase_deg if args.include_phase else None)
        excitation = np.column_stack((np.arange(1, result.parameters.elements + 1), result.positions_mm,
                                      np.ones(result.parameters.elements), result.excitation_phase_deg))
        np.savetxt(staging / "excitation.csv", excitation, delimiter=",", header="element,x_mm,amplitude,phase_deg",
                   comments="", fmt=["%d", "%.12g", "%.12g", "%.12g"])
        files = {"summary": "summary.json", "pattern": "pattern.dat", "excitation": "excitation.csv"}
        if "feed_network" in report:
            from .feed import write_feed_files
            files.update(write_feed_files(staging, report["feed_network"]))
        if args.plot != "none":
            render_plots(result, staging, args.plot)
            for kind in (["rectangular", "polar"] if args.plot == "both" else [args.plot]):
                files[kind + "_plot"] = kind + ".png"
            if "feed_network" in report:
                render_plots(result, staging, args.plot, prefix="feed-", patterns=report["feed_network"]["electrical"]["patterns"])
                for kind in (["rectangular", "polar"] if args.plot == "both" else [args.plot]):
                    files["feed_" + kind + "_plot"] = "feed-" + kind + ".png"
        report["output_directory"] = str(target)
        report["files"] = files
        (staging / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        # All outputs are complete before the directory becomes visible.
        staging.rename(target)
    finally:
        if staging.exists():
            # Only flat files created in this call; no recursive deletion.
            for item in staging.iterdir():
                item.unlink()
            staging.rmdir()


def emit(value, stream):
    if stream is not None:
        # ASCII-safe JSON on Windows pipes; JSON files are UTF-8 human-readable.
        stream.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
        stream.flush()


def feed_settings_from_args(args, parameters, base=None):
    from .feed import FeedSettings
    values = asdict(base or FeedSettings())
    if args.feed_network:
        values["topology"] = args.feed_network
    for key in values:
        if key != "topology" and getattr(args, "feed_" + key, None) is not None:
            values[key] = getattr(args, "feed_" + key)
    if args.feed_weighting and args.feed_weighting != "custom":
        values.update(custom_amplitudes=(), custom_phases_deg=())
    if args.feed_excitation:
        if args.feed_weighting not in (None, "custom"):
            raise UsageError("--feed-excitation不能与非custom幅度分布同时使用。")
        from .excitation import read_excitation_csv
        amplitude, phase = read_excitation_csv(args.feed_excitation, parameters.elements)
        values.update(weighting="custom", custom_amplitudes=amplitude, custom_phases_deg=phase)
    settings = FeedSettings(**values)
    settings.validate()
    return settings


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(value in ("--study", "--study-config") or value.startswith(("--study=", "--study-config=")) for value in argv):
        from .study_cli import main as study_main
        return study_main(argv)
    if any(value == "--planar" or value == "--planar-config" or value.startswith("--planar-config=") for value in argv):
        from .planar_cli import main as planar_main
        return planar_main(argv)
    try:
        args = parser().parse_args(argv)
        if args.plot != "none" and args.output_dir is None:
            raise UsageError("--plot需要同时指定--output-dir。")
        if sys.stdout is None and args.output_dir is None:
            raise UsageError("此入口没有标准输出，请指定--output-dir或使用ArrayLab-cli.exe。")
        base, feed_settings = None, None
        if args.feed_config:
            if args.config:
                raise UsageError("--feed-config不能与--config同时使用。")
            from .feed import load_feed_configuration
            base, feed_settings = load_feed_configuration(args.feed_config)
        parameters = load_parameters(args, base)
        if args.feed_network or feed_settings is not None:
            from .feed import design_feed
            feed_settings = feed_settings_from_args(args, parameters, feed_settings)
        elif any(value is not None for key, value in vars(args).items() if key.startswith("feed_") and key not in ("feed_network", "feed_config")):
            raise UsageError("馈电阻抗选项需要--feed-network或--feed-config。")
        from .core import calculate
        result = calculate(parameters)
        report = summary(result)
        if feed_settings is not None:
            report["feed_network"] = design_feed(parameters, feed_settings)
        if args.output_dir is not None:
            save_outputs(result, report, args)
        emit(report, sys.stdout)
        return 0
    except (ValueError, TypeError, OverflowError, UnicodeError) as exc:
        emit({"status": "error", "error": {"type": "input", "message": str(exc)}}, sys.stderr)
        return 2
    except OSError as exc:
        emit({"status": "error", "error": {"type": "io", "message": str(exc)}}, sys.stderr)
        return 3
    except Exception as exc:
        emit({"status": "error", "error": {"type": "runtime", "message": str(exc)}}, sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
