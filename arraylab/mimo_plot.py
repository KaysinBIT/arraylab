"""Qt-free figures for a saved coherent radar MIMO receive response."""
from pathlib import Path

import numpy as np

from .pattern_metrics import draw_metric_footer
from .plot_layout import responsive_subplots
from .surface_plot import clear_figure


VIEWS = ('layout', 'cut', 'map')


def _number(value, digits=4):
    return 'n/a' if value is None else f'{float(value):.{digits}g}'


def mimo_metric_lines(report, chinese=False):
    """Keep sampled cut quantities separate from theoretical steering facts."""
    p, metrics = report['parameters'], report['metrics']
    cut = metrics['cut']
    widths = metrics.get('orthogonal_beamwidths_deg', [None, None])
    if chinese:
        lines = [f"虚拟接收合成响应 | {p['frequency_ghz']:g} GHz | 目标 θ={p['theta_deg']:g}°，φ={p['phi_deg']:g}°",
                 f"当前切面 φ={report['patterns']['cut']['phi_deg']:g}° | 3 dB宽度 {_number(cut.get('half_power_beamwidth_deg'))}° | PSL {_number(cut.get('peak_sidelobe_db'))} dB",
                 f"目标方向两条正交大圆3 dB宽度：{_number(widths[0])}° / {_number(widths[1])}°",
                 '切面指标为采样估计；未定义显示n/a。相干目标峰不保证唯一；不是实际天线辐射增益。']
    else:
        lines = [f"Virtual receive response | {p['frequency_ghz']:g} GHz | steering {p['theta_deg']:g}/{p['phi_deg']:g} deg",
                 f"Cut phi={report['patterns']['cut']['phi_deg']:g} deg | 3 dB width {_number(cut.get('half_power_beamwidth_deg'))} deg | PSL {_number(cut.get('peak_sidelobe_db'))} dB",
                 f"Steering-centered great-circle 3 dB widths: {_number(widths[0])} / {_number(widths[1])} deg",
                 'Sampled cut metrics; n/a means undefined. Steering peak may be nonunique. Not radiated antenna gain.']
    return lines


def draw_mimo_figure(figure, report, view='layout', chinese=False):
    """Draw only saved result arrays; view changes never invoke the solver."""
    if view not in VIEWS:
        raise ValueError('Unknown MIMO view: ' + str(view))
    clear_figure(figure)
    if hasattr(figure, '_arraylab_responsive_plots'):
        del figure._arraylab_responsive_plots
    figure.set_layout_engine('constrained', w_pad=.07, h_pad=.08, wspace=.12)
    geometry = report['geometry']
    if view == 'layout':
        physical = figure.add_subplot(121)
        virtual = figure.add_subplot(122)
        for side, marker, color, label in (('tx', '^', '#d4a145', 'Tx'), ('rx', 'o', '#236caa', 'Rx')):
            xy = np.asarray(geometry[side]['positions_lambda'], dtype=float)
            amplitudes = np.asarray(geometry[side]['weights'], dtype=float)
            sizes = 25 + 55 * amplitudes / amplitudes.max()
            physical.scatter(xy[:, 0], xy[:, 1], marker=marker, s=sizes, label=label,
                             facecolors=color if side == 'tx' else 'none', edgecolors=color, linewidths=1.3)
            if len(xy) <= 12:
                for index, (x, y) in enumerate(xy, 1):
                    physical.annotate(f'{label}{index}', (x, y), xytext=(4, 5 if side == 'tx' else -11),
                                      textcoords='offset points', fontsize=7, color=color)
        physical.set_title('Tx / Rx物理位置' if chinese else 'Physical Tx / Rx positions', fontsize=10)
        physical.legend(fontsize=8, loc='best')
        rows = geometry['virtual_unique']
        x, y = np.array([[r['x_lambda'], r['y_lambda']] for r in rows], dtype=float).T
        multiplicity = np.array([r['multiplicity'] for r in rows], dtype=float)
        amplitudes = np.array([r['weight'] for r in rows], dtype=float)
        dots = virtual.scatter(x, y, s=25 + 55 * amplitudes / amplitudes.max(), c=multiplicity,
                               cmap='viridis', edgecolors='#285f88', linewidths=.5)
        if len(rows) <= 16:
            for row in rows:
                virtual.annotate(f"{row['index']}" + (f" ×{row['multiplicity']}" if row['multiplicity'] > 1 else ''),
                                 (row['x_lambda'], row['y_lambda']), xytext=(4, 4), textcoords='offset points', fontsize=7)
        virtual.set_title('虚拟位置\nTx + Rx' if chinese else 'Virtual positions\nTx + Rx', fontsize=10)
        if int(multiplicity.max()) > 1:
            colorbar = figure.colorbar(dots, ax=virtual, shrink=.8, pad=.02)
            colorbar.set_label('通道重数' if chinese else 'Multiplicity', fontsize=8)
            colorbar.ax.tick_params(labelsize=7)
        for axis in (physical, virtual):
            axis.set(xlabel='x / λ', ylabel='y / λ')
            axis.set_aspect('equal', adjustable='datalim'); axis.grid(alpha=.25)
            axis.tick_params(labelsize=8)
        counts = report['counts']
        lines = [f"Tx {counts['tx_elements']} · Rx {counts['rx_elements']} · "
                 + (f"配对通道 {counts['tx_rx_channels']} · 独特虚拟位置 {counts['unique_virtual_positions']}" if chinese else
                    f"pair channels {counts['tx_rx_channels']} · unique positions {counts['unique_virtual_positions']}"),
                 (f"重复通道 {counts['redundant_channels']}（{100*counts['redundancy_ratio']:.3g}%）| 最大重数 {counts['max_multiplicity']}" if chinese else
                  f"Redundant channels {counts['redundant_channels']} ({100*counts['redundancy_ratio']:.3g}%) | maximum multiplicity {counts['max_multiplicity']}"),
                 ('重复位置保留全部通道权重；点大小为数字合成幅度。RF链数未由阵元/通道数推定。' if chinese else
                  'Coincident positions retain all channel weights. Dot size: digital amplitude. RF-chain count is not inferred.')]
        draw_metric_footer(figure, lines)
        responsive_subplots(figure, [physical, virtual], 2, column_width=230, row_height=230)
    elif view == 'cut':
        axis = figure.add_subplot(111)
        cut = report['patterns']['cut']
        for key, name, color in (('tx_db', 'Tx数字因子' if chinese else 'Tx digital factor', '#d4a145'),
                                 ('rx_db', 'Rx数字因子' if chinese else 'Rx digital factor', '#236caa'),
                                 ('virtual_db', '虚拟数字波束' if chinese else 'Virtual digital beam', '#008f9c')):
            axis.plot(cut['angles_deg'], np.maximum(np.asarray(cut[key], dtype=float), -80), color=color,
                      label=name, lw=1.8 if key == 'virtual_db' else 1.1)
        axis.axhline(-3.0102999566, color='#9aa9b4', ls=':', lw=.9)
        axis.set(xlim=(-90, 90), ylim=(-60, 2), xlabel='θ (°)' if chinese else 'Signed theta (deg)',
                 ylabel='相干目标参考 / dB' if chinese else 'Relative to coherent target (dB)',
                 title=(f"MIMO切面 · φ={cut['phi_deg']:g}°" if chinese else f"MIMO cut | phi={cut['phi_deg']:g} deg"))
        axis.grid(alpha=.3); axis.legend(fontsize=8)
        draw_metric_footer(figure, mimo_metric_lines(report, chinese))
    else:
        axis = figure.add_subplot(111)
        saved = report['patterns']['map']
        u, v = np.asarray(saved['axis_u']), np.asarray(saved['axis_v'])
        values = np.asarray(saved['virtual_db'], dtype=float)
        mesh = axis.pcolormesh(u, v, np.maximum(values, -40), vmin=-40, vmax=0, cmap='viridis', shading='auto')
        p = report['parameters']; theta, phi = np.deg2rad([p['theta_deg'], p['phi_deg']])
        axis.plot(np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), '+', color='white', ms=9)
        axis.set(xlabel='u = sinθ cosφ', ylabel='v = sinθ sinφ', xlim=(-1, 1), ylim=(-1, 1), aspect='equal',
                 title='虚拟数字波束\n可见u-v域' if chinese else 'Virtual digital beam\nVisible u-v disk')
        bar = figure.colorbar(mesh, ax=axis, shrink=.85, pad=.03)
        bar.set_label('dB', fontsize=8); bar.ax.tick_params(labelsize=8)
        draw_metric_footer(figure, mimo_metric_lines(report, chinese))
    return figure


def render_mimo_plots(directory, report, mode='all'):
    """Export saved MIMO arrays through Agg, without importing Qt."""
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    if mode == 'none':
        return
    kinds = VIEWS if mode == 'all' else (mode,)
    if any(kind not in VIEWS for kind in kinds):
        raise ValueError('MIMO plot must be none, all, layout, cut or map.')
    for kind in kinds:
        with matplotlib.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.usetex': False}):
            figure = Figure(figsize=(9, 6), dpi=140)
            FigureCanvasAgg(figure)
            draw_mimo_figure(figure, report, kind)
            figure.savefig(Path(directory) / f'mimo-{kind}.png')
            figure.clear()
