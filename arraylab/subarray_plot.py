"""Subarray result plots shared by the native workbench and headless exports."""
import numpy as np

from .subarray import SubarrayConfig, expanded_coordinates, normalized_field
from .pattern_metrics import metrics_text, draw_metric_footer
from .surface_plot import clear_figure, register_surface


def render_subarray_figure(figure, report, view='layout', case_index=0, chinese=False):
    if view not in ('layout', 'pattern', 'cuts', '3d'):
        raise ValueError('未知子阵图形类型。')
    clear_figure(figure)
    config = SubarrayConfig(**report['config'])
    best = report['best']
    case = best['cases'][case_index]
    centers = np.array(best['centers'])
    xy = expanded_coordinates(config, centers)
    if view == 'layout':
        ax = figure.add_subplot(111)
        ids = np.repeat(np.arange(config.count), config.nx*config.ny)
        ax.scatter(xy[:, 0], xy[:, 1], c=ids, cmap='tab20', s=30, edgecolor='white', linewidth=.3)
        ax.scatter(centers[:, 0], centers[:, 1], c='#222222', marker='+', s=65, label='Optimized centers')
        initial = np.array(report['initial']['centers'])
        ax.scatter(initial[:, 0], initial[:, 1], facecolors='none', edgecolors='#888888', s=55, label='Initial centers')
        if config.count <= 24:
            for index, point in enumerate(centers, 1):
                ax.annotate(str(index), point, xytext=(4, 5), textcoords='offset points', fontsize=8)
        from matplotlib.patches import Rectangle, Circle
        if config.hole_shape == 'rectangle':
            ax.add_patch(Rectangle((-config.hole_x/2, -config.hole_y/2), config.hole_x, config.hole_y,
                                   facecolor='#e8cccc', edgecolor='#aa6666', alpha=.5))
        elif config.hole_shape == 'circle':
            ax.add_patch(Circle((0, 0), config.hole_x/2, facecolor='#e8cccc', edgecolor='#aa6666', alpha=.5))
        ax.set(xlabel='x / reference wavelength', ylabel='y / reference wavelength',
               title=f'{config.count} subarrays / {len(xy)} elements',
               xlim=(-config.aperture_x*.55, config.aperture_x*.55),
               ylim=(-config.aperture_y*.55, config.aperture_y*.55))
        ax.set_aspect('equal'); ax.grid(True, alpha=.25); ax.legend(fontsize=8)
    elif view == 'pattern':
        ax = figure.add_subplot(111)
        db = np.array([[np.nan if value is None else value for value in row] for row in case['map_db']])
        mesh = ax.pcolormesh(case['axis_uv'], case['axis_uv'], np.maximum(db, -40), cmap='viridis', vmin=-40, vmax=0, shading='auto')
        theta, phi = np.deg2rad([case['theta'], case['phi']])
        u, v = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((u-config.exclusion_u, v-config.exclusion_v), 2*config.exclusion_u,
                               2*config.exclusion_v, fill=False, edgecolor='white', linestyle='--', linewidth=1.))
        ax.plot(u, v, '+', color='red', ms=9)
        ax.set(xlabel='u = sin(theta) cos(phi)', ylabel='v = sin(theta) sin(phi)',
               title='Upper-hemisphere field / coherent peak (dB)', xlim=(-1, 1), ylim=(-1, 1))
        ax.set_aspect('equal'); figure.colorbar(mesh, ax=ax, label='Relative field (dB)')
    elif view == 'cuts':
        ax = figure.add_subplot(111)
        for index, label in enumerate(('Meridional great circle', 'Transverse great circle')):
            ax.plot(case['cut_angles_deg'], np.maximum(case['cuts_db'][index], -60), label=label)
            cut = case['beamwidths']['cuts'][index]
            edges = cut['half_power_edges_deg']
            if edges:
                ax.plot(edges, [10*np.log10(.5)]*2, 'o--', ms=4, lw=1)
        ax.set(xlabel='Angular displacement from steering direction (deg)', ylabel='Relative field (dB)',
               title='Orthogonal great-circle cuts through steering direction', xlim=(-90, 90), ylim=(-60, 2))
        ax.grid(True, alpha=.25); ax.legend(fontsize=8)
    else:
        ax = figure.add_subplot(111, projection='3d')
        theta, phi = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(-np.pi, np.pi, 121), indexing='ij')
        uv = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi)], axis=-1)
        radius = abs(normalized_field(config, centers, case, uv))
        from matplotlib import colormaps
        db = 20*np.log10(np.maximum(radius, 1e-15))
        colors = colormaps['viridis'](np.clip((db+40)/40, 0, 1))
        x,y,z=radius*uv[:, :, 0],radius*uv[:, :, 1],radius*np.cos(theta)
        surface=ax.plot_surface(x,y,z,
                        facecolors=colors, rstride=1, cstride=2, linewidth=0, shade=False, antialiased=False)
        register_surface(figure,ax,surface,x,y,z,colors)
        ax.set(xlabel='x', ylabel='y', zlabel='z', xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1),
               title='Full sphere: radius = relative field magnitude')
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=22, azim=-50)
        from matplotlib.ticker import MaxNLocator
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):axis.set_major_locator(MaxNLocator(3))
    lines = metrics_text(case['metrics'], chinese=chinese)
    lines = [line.replace('Cut PSL', 'Sampled disk PSL').replace('切面 PSL', '空间采样 PSL')
             .replace('Cut HPBW', 'Meridional HPBW').replace('切面 3 dB 宽度', '子午切面 3 dB 宽度') for line in lines]
    lines.insert(0, f"{case['frequency_ghz']:g} GHz | theta={case['theta']:g}, phi={case['phi']:g} deg | {config.count} subarrays / {len(xy)} elements")
    lines.append(f"PSL: sampled upper disk, exclusion half-widths u={config.exclusion_u:g}, v={config.exclusion_v:g}; peak not unique")
    if report.get('sampling', {}).get('resolution_warning'):
        lines.append('Sampling resolution warning: increase evaluation grid for this electrical aperture.')
    draw_metric_footer(figure, lines)
    return ax


def render_subarray_plots(directory, report, mode='all'):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    modes = ('layout', 'pattern', 'cuts', '3d') if mode == 'all' else (mode,)
    for index, _ in enumerate(report['best']['cases']):
        for view in modes:
            if view == 'layout' and index:
                continue
            with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.usetex': False}):
                figure = Figure(figsize=(9, 7), dpi=140)
                FigureCanvasAgg(figure)
                render_subarray_figure(figure, report, view, index)
                figure.savefig(directory/f'subarray-{view}-case-{index}.png')
                figure.clear()
