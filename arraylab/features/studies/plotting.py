"""Shared scan-study drawing, with a GUI-independent PNG renderer."""
import numpy as np


def draw_study(figure, report, view="response", selected=0):
    figure.clear()
    rows = report["records"]
    axis = report["configuration"]["study_settings"]["scan_axis"]
    scans = [row[f"scan_{axis}_deg"] for row in rows]
    ax = figure.add_subplot(111)
    if view == "map":
        data = np.array([row["pattern_db"] for row in rows])
        mesh = ax.pcolormesh(report["angles_deg"], scans, np.maximum(data, -60),
                             shading="nearest", vmin=-60, vmax=max(0, float(np.max(data))), cmap="viridis")
        figure.colorbar(mesh, ax=ax, label="Relative field level (dB)")
        ax.set_xlabel("Cut signed theta (deg)"); ax.set_ylabel(f"Commanded {axis} (deg)")
        ax.set_xlim(report["angles_deg"][0], report["angles_deg"][-1]); ax.set_ylim(scans[0], scans[-1])
        ax.set_title(f"Fixed cut phi = {report['cut_phi_deg']:g} deg")
    elif view == "pattern":
        row = rows[selected]
        ax.plot(report["angles_deg"], np.maximum(row["pattern_db"], -60), color="#087fa8")
        ax.set_xlabel("Cut signed theta (deg)"); ax.set_ylabel("Relative field level (dB)")
        ax.set_title(f"Scan #{row['index']}: theta={row['scan_theta_deg']:g}, phi={row['scan_phi_deg']:g} deg")
        ax.set_ylim(-60, max(1, max(row["pattern_db"]) + 1)); ax.grid(alpha=.25)
    else:
        directivity = [row["metrics"]["target_direction_directivity_dbi"] for row in rows]
        ax.plot(scans, [np.nan if v is None else v for v in directivity], "o-", color="#087fa8", label="Target-direction D (dBi)")
        ax.plot(scans, [max(-60, row["metrics"]["target_direction_level_db"]) for row in rows], "s--", color="#cf762b", label="Target field level (dB)")
        ax.set_xlabel(f"Commanded {axis} (deg)"); ax.set_ylabel("dB / dBi (see legend)")
        ax.legend(fontsize=8); ax.grid(alpha=.25)
        ax.set_title("Target direction only; not peak gain or pointing accuracy")
    figure.subplots_adjust(left=.12, right=.9 if view == "map" else .97, bottom=.2, top=.86)
    if view == 'pattern':
        from arraylab.shared.plotting.metrics import sampled_cut_metrics, metrics_text, draw_metric_footer
        summary = sampled_cut_metrics(report['angles_deg'], row['pattern_db'])
        lines = metrics_text(summary)
        value = row['metrics']['target_direction_directivity_dbi']
        target = 'n/a' if value is None else f'{value:.3g} dBi'
        lines[1] = 'Target-direction D ' + target + ' (not peak D) | Gain G unavailable (efficiency missing)'
        lines.append('Selected scan-point cut; sample estimates. PSL excludes first bounding minima.')
        draw_metric_footer(figure, lines)


def render_study_plot(path, report):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    with matplotlib.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "text.usetex": False}):
        figure = Figure(figsize=(9, 5), dpi=150)
        canvas = FigureCanvasAgg(figure)
        draw_study(figure, report)
        canvas.print_png(str(path))
