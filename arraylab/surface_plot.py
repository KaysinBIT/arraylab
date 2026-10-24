"""Qt-free lifecycle helpers for scientific 3D surfaces."""


def clear_figure(figure):
    """Disconnect discarded Axes3D mouse callbacks before clearing its artists.

    Matplotlib stores weak callbacks, but old axes can remain alive in reference
    cycles until garbage collection. They must not receive later drag events.
    """
    reset = getattr(figure.canvas, 'reset_interaction', None)
    if reset is not None:
        reset()
    old_axes = [axis for axis in figure.axes if getattr(axis, 'name', None) == '3d']
    registry = figure.canvas.callbacks
    for callbacks in list(registry.callbacks.values()):
        for cid, reference in list(callbacks.items()):
            callback = reference()
            if any(getattr(callback, '__self__', None) is axis for axis in old_axes):
                registry.disconnect(cid)
    figure.clear()


def register_surface(figure, axis, surface, x, y, z, colors):
    """Enable desktop-only previews; Agg exports keep the original full surface."""
    register = getattr(figure.canvas, 'register_surface', None)
    if register is not None:
        register(axis, surface, x, y, z, colors)
    return surface
