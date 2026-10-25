"""Existing linear/planar feed CLI options; command syntax is unchanged."""
from pathlib import Path
from dataclasses import asdict
import argparse
from arraylab import __version__
from arraylab.framework.cli_support import Parser, UsageError

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
        p.add_argument("--sparse", action="store_true", help="Sparse planar arrays: --sparse --help (or --sparse planar --help)")
        p.add_argument("--mimo-config", metavar="JSON", help="Independent radar Tx/Rx geometry and virtual beam: mimo --help")
        p.add_argument("--waveform-config", metavar="JSON", help="FMCW, A/B, TDM/DDM and distance recovery: waveform --help")
        p.add_argument("--study", choices=["scan"], help="Electrical multi-angle scan: use --study scan --help")
        p.add_argument("--study-config", type=Path, help="Saved multi-angle electrical scan configuration")
        p.add_argument("--planar", action="store_true", help="Planar arrays: use --planar --help for geometry and plot options")
        p.add_argument("--planar-config", type=Path, help="Run a saved planar electrical configuration")
    p.add_argument("--headless", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--version", action="version", version=f"ArrayLab {__version__} (headless)")
    p.add_argument("--config", type=Path, help="Versioned analysis JSON or legacy parameter JSON; explicit arguments override it")
    p.add_argument("--workers", type=int, default=0, help="Calculation workers: 0=auto up to 4, 1=serial, 2..8=bounded parallel for supported independent tasks")
    p.add_argument("--feed-config", type=Path, help="Feed impedance configuration JSON; cannot combine with --config")
    p.add_argument("--feed-network", choices=["corporate", "series", "symmetric_series"], help="Quarter-wave feed impedances; symmetric_series requires an even element count")
    p.add_argument("--feed-input-ohm", type=float, help="Real input resistance (default: 50)")
    p.add_argument("--feed-junction-ohm", type=float, help="Real internal junction resistance (default: 50)")
    p.add_argument("--feed-load-ohm", type=float, help="Equal real element load resistance (default: 50)")
    p.add_argument("--feed-min-line-ohm", type=float, help="Advisory line impedance lower bound (default: 20)")
    p.add_argument("--feed-max-line-ohm", type=float, help="Advisory line impedance upper bound (default: 120)")
    p.add_argument("--feed-phase", dest="feed_phase_mode", choices=["none", "equalize", "scan"], help="scan: compensate towards requested beam angles; equalize: broadside; none: raw network (new default: scan; legacy v1: none)")
    p.add_argument("--feed-weighting", choices=["uniform", "chebyshev", "taylor", "custom"], help="Target amplitude taper")
    p.add_argument("--feed-sidelobe-db", type=float, help="Chebyshev/Taylor sidelobe parameter, positive dB")
    p.add_argument("--feed-taylor-nbar", type=int, help="Taylor near-in sidelobe count")
    p.add_argument("--feed-excitation", type=Path, help="Strict element,amplitude,phase_deg CSV; phase is an additional target offset")
    branch_group = p.add_mutually_exclusive_group()
    branch_group.add_argument("--feed-branch-delays", type=Path, help="Complete section,delay_deg CSV for matched delays after each 90-degree transformer")
    branch_group.add_argument("--clear-feed-branch-delays", dest="feed_clear_branch_delays", action="store_true", default=None,
                              help="Clear saved branch delay values and their geometry/topology binding")
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
    p.add_argument("--plot", choices=["none", "cut", "map", "3d", "all"] if planar else ["none", "rectangular", "polar", "both", "3d", "all"], default="none",
                   help="Write PNG plots using Agg; requires --output-dir (default: none)")
    return p


def feed_settings_from_args(args, parameters, base=None):
    from arraylab.features.feed.model import FeedSettings, with_branch_delays, clear_branch_delays
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
        from arraylab.features.feed.excitation import read_excitation_csv
        amplitude, phase = read_excitation_csv(args.feed_excitation, parameters.elements)
        values.update(weighting="custom", custom_amplitudes=amplitude, custom_phases_deg=phase)
    settings = FeedSettings(**values)
    if getattr(args, 'feed_clear_branch_delays', None):
        settings = clear_branch_delays(settings)
    if getattr(args, 'feed_branch_delays', None) is not None:
        from arraylab.features.feed.branch_io import read_branch_delays_csv
        settings = with_branch_delays(clear_branch_delays(settings), parameters.elements,
                                     read_branch_delays_csv(args.feed_branch_delays, parameters.elements))
    settings.validate()
    return settings
