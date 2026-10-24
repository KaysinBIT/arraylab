"""Responsive subplot placement; independent of Qt for image exports."""
from __future__ import annotations

import math


def _resize(event):
    reflow_plot_layout(event.canvas.figure)


def reflow_plot_layout(figure):
    """Reflow existing artists only, without recomputing scientific results."""
    state = getattr(figure, '_arraylab_responsive_plots', None)
    if state is None or any(ax not in figure.axes for ax in state['axes']):
        return
    canvas = figure.canvas
    registry = canvas.callbacks
    previous = getattr(canvas, '_arraylab_plot_reflow_registry', None)
    if previous is not registry:
        cid = getattr(canvas, '_arraylab_plot_reflow_cid', None)
        if previous is not None and cid is not None:
            previous.disconnect(cid)
        # Figures are prepared before the desktop installs its shared canvas.
        # Reusing the same callback function preserves one connection when the
        # Figure moves from its original canvas to that live Qt canvas.
        canvas._arraylab_plot_reflow_registry = registry
        canvas._arraylab_plot_reflow_cid = registry.connect('resize_event', _resize)
    width = figure.get_figwidth() * 100
    columns = min(state['columns'], max(1, int(width / state['column_width'])))
    rows = math.ceil(len(state['axes']) / columns)
    shape = (rows, columns)
    if state.get('shape') != shape:
        grid = figure.add_gridspec(*shape)
        for index, ax in enumerate(state['axes']):
            ax.set_subplotspec(grid[index])
        state['shape'] = shape
    if hasattr(canvas, 'setMinimumHeight'):
        # Only desktop widgets need scrollable room. Export dimensions remain
        # exactly those supplied by the caller.
        canvas.setMinimumHeight(max(360, rows * state['row_height'] + 80))
        # A minimum-size change during Qt's resize notification can resize the
        # widget before another resize event arrives. Keep this frame's Figure
        # dimensions in sync, rather than painting a short image at the top.
        ratio = canvas.device_pixel_ratio
        figure.set_size_inches(canvas.width() * ratio / figure.dpi,
                               canvas.height() * ratio / figure.dpi, forward=False)


def responsive_subplots(figure, axes, columns, *, column_width=310, row_height=210):
    """Keep separate panels readable when a desktop plot becomes narrow.

    Called after all axes and colorbars have been created. Colorbars should be
    created with constrained layout active, so they stay attached to their
    parent axes when the subplot grid changes.
    """
    figure._arraylab_responsive_plots = {
        'axes': list(axes), 'columns': columns, 'column_width': column_width,
        'row_height': row_height,
    }
    reflow_plot_layout(figure)
