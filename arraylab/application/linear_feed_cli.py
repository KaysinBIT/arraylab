"""GUI-free command line execution. No Qt or GUI backend is imported here."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import uuid

from arraylab import __version__
from arraylab.framework.cli_support import Parser, UsageError, emit
from arraylab.features.feed import port_list, parser, feed_settings_from_args










def load_parameters(args, base=None):
    from arraylab.features.linear import ArrayParameters
    from arraylab.features.linear import load_configuration, parameters_from_values
    values = asdict(load_configuration(args.config) if args.config else base or ArrayParameters())
    for key in values:
        supplied = getattr(args, key, None)
        if supplied is not None:
            values[key] = supplied
    return parameters_from_values(values)


def summary(result):
    from arraylab.features.linear import linear_metrics
    derived = linear_metrics(result)
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
            "sphere_theta_reference": "0..180 degrees from +z",
            "sphere_phi_reference": "-180..180 degrees from +x towards +y",
            "sphere_sampling_step_deg": 2,
        },
        "metrics": {
            "peak_sidelobe_db": derived.get("peak_sidelobe_db"),
            "sidelobe_main_lobe_bounds_deg": derived.get("main_lobe_bounds_deg"),
            "sidelobe_method": "sampled_xz_cut_outside_nearest_main_lobe_minima",
            "gain_dbi": None,
            "gain_status": "not_calculated_missing_radiation_efficiency",
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


def plot_kinds(mode):
    return ["rectangular", "polar", "3d"] if mode == "all" else ["rectangular", "polar"] if mode == "both" else [mode]


def render_plots(result, directory, mode, prefix="", patterns=None, feed_report=None):
    # This is a file renderer, not a hidden/offscreen Qt application. Do not
    # import pyplot, widgets.py or any Qt backend, even when MPLBACKEND=QtAgg.
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    import numpy as np

    from arraylab.features.linear import linear_metrics, metric_lines, sphere_pattern, sphere_from_feed, draw_sphere
    from arraylab.shared.plotting.metrics import draw_metric_footer
    kinds = plot_kinds(mode)
    for kind in kinds:
        with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.usetex": False}):
            figure = Figure(figsize=(8, 5), dpi=150, layout="constrained")
            FigureCanvasAgg(figure)
            metrics = linear_metrics(result if feed_report is None else None, feed_report, sphere=kind == "3d")
            if kind == "3d":
                sphere = sphere_from_feed(feed_report) if feed_report is not None else sphere_pattern(result.parameters)
                draw_sphere(figure, sphere)
                draw_metric_footer(figure, metric_lines(metrics, sphere=True, label="Feed output" if feed_report else "Linear array"))
                figure.savefig(directory / f"{prefix}{kind}.png"); figure.clear()
                continue
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
            edges = metrics.get("half_power_edges_deg")
            if edges is not None and metrics.get("peak_level_db") is not None:
                threshold = metrics["peak_level_db"] + 10 * math.log10(.5)
                ax.plot(np.deg2rad(edges) if polar else edges,
                        [threshold + (40 if polar else 0)] * 2, "o--", color="#7a8b9a", lw=1, markersize=3)
            ax.grid(True, color="#dce5ed", linewidth=.75)
            draw_metric_footer(figure, metric_lines(metrics, label="Feed output" if feed_report else "Linear array"))
            figure.savefig(directory / f"{prefix}{kind}.png")
            figure.clear()


def save_outputs(result, report, args):
    import numpy as np
    from arraylab.features.linear import write_pattern

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
        from arraylab.features.linear import sphere_pattern, write_sphere_csv
        write_sphere_csv(staging / "sphere-pattern.csv", sphere_pattern(result.parameters))
        files["sphere_pattern"] = "sphere-pattern.csv"
        if "feed_network" in report:
            from arraylab.features.feed import write_feed_files
            files.update(write_feed_files(staging, report["feed_network"]))
        if args.plot != "none":
            render_plots(result, staging, args.plot)
            for kind in plot_kinds(args.plot):
                files[kind + "_plot"] = kind + ".png"
            if "feed_network" in report:
                render_plots(result, staging, args.plot, prefix="feed-", patterns=report["feed_network"]["electrical"]["patterns"], feed_report=report["feed_network"])
                for kind in plot_kinds(args.plot):
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






def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parser().parse_args(argv)
        from arraylab.framework.parallel import set_workers
        set_workers(args.workers)
        if args.plot != "none" and args.output_dir is None:
            raise UsageError("--plot需要同时指定--output-dir。")
        if sys.stdout is None and args.output_dir is None:
            raise UsageError("此入口没有标准输出，请指定--output-dir或使用ArrayLab-cli.exe。")
        base, feed_settings = None, None
        if args.feed_config:
            if args.config:
                raise UsageError("--feed-config不能与--config同时使用。")
            from arraylab.features.feed import load_feed_configuration
            base, feed_settings = load_feed_configuration(args.feed_config)
        parameters = load_parameters(args, base)
        if args.feed_network or feed_settings is not None:
            from arraylab.features.feed import design_feed
            feed_settings = feed_settings_from_args(args, parameters, feed_settings)
        elif any(value is not None for key, value in vars(args).items() if key.startswith("feed_") and key not in ("feed_network", "feed_config")):
            raise UsageError("馈电阻抗选项需要--feed-network或--feed-config。")
        from arraylab.features.linear import calculate
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
