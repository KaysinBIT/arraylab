"""Bounded redraws for 3D interaction; scientific grids stay unchanged.

Only the temporary mouse-interaction display may use a coarser surface.  An
explicit draw, resize, print, navigation action, or completed gesture restores
the original artists and the normal Matplotlib drawing path.
"""
from dataclasses import dataclass
import math
import time
import weakref

import numpy as np
from PySide6.QtCore import QTimer
from matplotlib.backend_bases import DrawEvent
from .resize_canvas import ResizeAwareFigureCanvas


@dataclass
class _Surface:
    full: weakref.ReferenceType
    xyz: tuple
    colors: np.ndarray
    preview: weakref.ReferenceType | None = None
    visible_before: bool | None = None


def _preview_grid(x, y, z, facecolors, max_faces=600):
    """Keep principal cuts and a sampled peak within a bounded existing grid."""
    xyz = tuple(np.asarray(value) for value in (x, y, z))
    shape = xyz[0].shape
    if len(shape) != 2 or min(shape) < 2 or any(value.shape != shape for value in xyz):
        raise ValueError("3D surface coordinates must have matching two-dimensional grids.")
    colors = np.asarray(facecolors)
    if colors.ndim != 3 or colors.shape[2] not in (3, 4) or colors.shape[:2] not in (shape, (shape[0]-1, shape[1]-1)):
        raise ValueError("3D surface colors must match the grid vertices or faces.")
    nr, nc = shape[0]-1, shape[1]-1
    radius = np.hypot(np.hypot(xyz[0], xyz[1]), xyz[2])
    peak = np.unravel_index(np.argmax(np.where(np.isfinite(radius), radius, -np.inf)), shape)
    required_rows = {0, nr, round(nr/2), int(peak[0])}
    required_cols = {0, nc, *(round(nc*fraction) for fraction in (.25, .5, .75)), int(peak[1])}
    minimum_rows, minimum_cols = len(required_rows)-1, len(required_cols)-1
    if type(max_faces) is not int or max_faces < minimum_rows*minimum_cols:
        raise ValueError('The preview face budget cannot retain the principal cuts and sampled peak.')
    rows = min(nr, max_faces//minimum_cols,
               max(minimum_rows, int(math.sqrt(max_faces*nr/nc))))
    cols = min(nc, max_faces//rows)

    def indices(last, intervals, required):
        selected = sorted(required | set(np.linspace(0, last, intervals+1).round().astype(int)))
        # Critical cuts replace the nearest ordinary samples instead of adding
        # faces beyond the budget. All remaining coordinates are original data.
        while len(selected) > intervals+1:
            removable = [i for i in range(1, len(selected)-1) if selected[i] not in required]
            remove = min(removable, key=lambda i: min(selected[i]-selected[i-1], selected[i+1]-selected[i]))
            selected.pop(remove)
        return np.asarray(selected, dtype=int)

    ri = indices(nr, rows, required_rows)
    ci = indices(nc, cols, required_cols)
    selected = np.ix_(ri, ci)
    color_selection = np.ix_(np.minimum(ri, colors.shape[0]-1), np.minimum(ci, colors.shape[1]-1))
    return tuple(value[selected].copy() for value in xyz), colors[color_selection].copy()


class InteractiveFigureCanvas(ResizeAwareFigureCanvas):
    """Keep the latest 3D camera, with one pending frame and no motion backlog.

    Register optional display-only surface previews with :meth:`register_surface`.
    Existing Matplotlib Axes3D callbacks remain responsible for drag rotation,
    pan, and zoom.  Two-dimensional canvas behavior is inherited unchanged.
    """

    frame_interval_ms = 33
    wheel_settle_ms = 150

    def __init__(self, figure=None):
        # QWidget construction can dispatch resizeEvent before __init__ returns.
        self._interaction_ready = False
        super().__init__(figure)
        self._surfaces = weakref.WeakKeyDictionary()
        self._active_axes = None
        self._background = None
        self._background_key = None
        self._wheel_changed = False
        self._in_mouse_event = 0
        self._in_scroll_event = False
        self._interaction_painting = False
        self._last_frame = 0.0
        self._closed = False
        self._event_registry = None
        self._event_ids = []
        self.interaction_draw_count = 0
        self.full_draw_count = 0
        self.last_interaction_error = None
        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.timeout.connect(self._draw_interaction)
        self._wheel_timer = QTimer(self)
        self._wheel_timer.setSingleShot(True)
        self._wheel_timer.timeout.connect(self.finish_interaction)
        self._interaction_ready = True
        self.setToolTip('三维视图：左键拖动旋转，右键拖动或滚轮缩放。\n拖动中为预览，松开恢复完整细节。')
        self._synchronize()

    @property
    def interaction_active(self):
        return self._active_axes is not None and self._active_axes() is not None

    def _synchronize(self):
        """A workbench may replace canvas.figure or clear its axes in place."""
        registry = self.callbacks
        if registry is not self._event_registry:
            self._disconnect_events()
            self._event_registry = registry
            self._event_ids = [registry.connect(name, callback) for name, callback in (
                ('button_press_event', self._on_press),
                ('button_release_event', self._on_release),
                ('scroll_event', self._on_scroll))]
        active = self._active_axes() if self._active_axes else None
        if active is not None and active not in self.figure.axes:
            self.finish_interaction(redraw=False)
        for ax, records in list(self._surfaces.items()):
            if ax not in self.figure.axes:
                del self._surfaces[ax]
            else:
                records[:] = [record for record in records if record.full() in ax.collections]

    def _disconnect_events(self):
        if self._event_registry is not None:
            for cid in self._event_ids:
                self._event_registry.disconnect(cid)
        self._event_ids = []
        self._event_registry = None

    def register_surface(self, ax, full_surface, x, y, z, facecolors):
        """Register a <=600-face preview, without altering the supplied surface."""
        if ax not in self.figure.axes or getattr(ax, 'name', None) != '3d' or full_surface not in ax.collections:
            raise ValueError('The registered 3D surface must belong to this canvas.')
        self.finish_interaction(redraw=False)
        self._synchronize()
        xyz, colors = _preview_grid(x, y, z, facecolors)
        records = self._surfaces.setdefault(ax, [])
        for record in records[:]:
            if record.full() is full_surface:
                previous = record.preview() if record.preview else None
                if previous is not None and previous in ax.collections:
                    previous.remove()
                records.remove(record)
        records.append(_Surface(weakref.ref(full_surface), xyz, colors))

    def _use_previews(self, ax):
        for record in self._surfaces.get(ax, ()):
            full = record.full()
            if full is None or full not in ax.collections:
                continue
            preview = record.preview() if record.preview else None
            if preview is None or preview not in ax.collections:
                view = ax._get_view()
                try:
                    preview = ax.plot_surface(*record.xyz, facecolors=record.colors,
                        rstride=1, cstride=1, linewidth=0, antialiased=False,
                        shade=False, alpha=full.get_alpha())
                    preview.set_visible(False)
                finally:
                    ax._set_view(view)
                record.preview = weakref.ref(preview)
            record.visible_before = full.get_visible()
            full.set_visible(False)
            preview.set_visible(record.visible_before)

    def _restore_surfaces(self):
        for records in self._surfaces.values():
            for record in records:
                full = record.full()
                preview = record.preview() if record.preview else None
                if preview is not None:
                    preview.set_visible(False)
                if record.visible_before is not None and full is not None:
                    full.set_visible(record.visible_before)
                record.visible_before = None

    def _begin_interaction(self, ax):
        self._synchronize()
        if self._closed or ax not in self.figure.axes or getattr(ax, 'name', None) != '3d':
            return False
        if self._active_axes and self._active_axes() is ax:
            return True
        self.finish_interaction(redraw=False)
        self._active_axes = weakref.ref(ax)
        self._use_previews(ax)
        self._background = None
        self._background_key = None
        # Cancel a previously queued full repaint. Its singleShot callback will
        # observe this flag; only the bounded interaction timer now draws.
        self._draw_pending = False
        return True

    def _on_press(self, event):
        if event.button in (1, 2, 3) and event.inaxes is not None:
            self._begin_interaction(event.inaxes)

    def _on_release(self, event):
        if self.interaction_active:
            self.finish_interaction()

    def _on_scroll(self, event):
        ax = event.inaxes
        if (ax is None or getattr(ax, 'name', None) != '3d' or
                ax.get_navigate_mode() is not None or not math.isfinite(event.step)):
            return
        if not event.step or not self._begin_interaction(ax):
            return
        toolbar = getattr(self, 'toolbar', None)
        if not self._wheel_changed and toolbar is not None and toolbar._nav_stack() is None:
            toolbar.push_current()
        scale = 1.15**(-max(-20.0, min(20.0, event.step)))
        # Public setters preserve the data center and inverted limit direction.
        for getter, setter in ((ax.get_xlim3d, ax.set_xlim3d),
                               (ax.get_ylim3d, ax.set_ylim3d),
                               (ax.get_zlim3d, ax.set_zlim3d)):
            lo, hi = getter()
            center, half = (lo+hi)/2, (hi-lo)*scale/2
            if abs(half) > 1e-12 and math.isfinite(half):
                setter(center-half, center+half, auto=None)
        if toolbar is not None:
            if self._wheel_changed:
                # Replace the current gesture's last snapshot, without applying
                # the older view to the axes or accumulating a step per notch.
                toolbar._nav_stack.back()
            toolbar.push_current()
        self._wheel_changed = True
        self._in_scroll_event = True
        try:
            self.draw_idle()
        finally:
            self._in_scroll_event = False
        self._wheel_timer.start(self.wheel_settle_ms)

    def mouseMoveEvent(self, event):
        self._in_mouse_event += 1
        try:
            super().mouseMoveEvent(event)
        finally:
            self._in_mouse_event -= 1

    def draw_idle(self):
        if not getattr(self, '_interaction_ready', False):
            return super().draw_idle()
        if self._interaction_painting:
            return
        self._synchronize()
        if self.interaction_active and (self._in_mouse_event or self._in_scroll_event):
            if not self._frame_timer.isActive():
                delay = max(0, self.frame_interval_ms-int((time.monotonic()-self._last_frame)*1000))
                self._frame_timer.start(delay)
            return
        # Home, programmatic view changes, and ordinary redraw requests require
        # the complete image; they must not inherit an unfinished gesture.
        if self.interaction_active:
            self.finish_interaction(redraw=False)
        return super().draw_idle()

    def _draw_interaction(self):
        frame_started = time.monotonic()
        self._synchronize()
        ax = self._active_axes() if self._active_axes else None
        if ax is None or self._closed or self.width() <= 0 or self.height() <= 0:
            return
        try:
            self._interaction_painting = True
            renderer = self.get_renderer()
            key = (id(self.figure), id(renderer), tuple(self.figure.axes),
                   tuple(self.figure.bbox.bounds), self.figure.dpi)
            if self._background is None or key != self._background_key:
                # Capture static decorations once per gesture. Temporarily
                # disable layout so hiding an axes does not reposition it.
                engine = self.figure.get_layout_engine()
                visible = ax.get_visible()
                try:
                    ax.set_visible(False)
                    self.figure._layout_engine = None
                    renderer.clear()
                    with self.callbacks.blocked(signal='draw_event'):
                        self.figure.draw(renderer)
                    self._background = self.copy_from_bbox(self.figure.bbox)
                    self._background_key = key
                finally:
                    ax.set_visible(visible)
                    self.figure._layout_engine = engine
            self.restore_region(self._background)
            ax.draw(renderer)
            # The interval is between frame starts, not an extra pause after
            # rendering. A slower frame must not incur another full 33 ms wait.
            self._last_frame = frame_started
            self.interaction_draw_count += 1
            DrawEvent('draw_event', self, renderer)._process()
            self.update()
        except Exception as exc:
            # Timer callbacks must never strand hidden full-resolution artists.
            self.last_interaction_error = f'{type(exc).__name__}: {exc}'[:500]
            self.finish_interaction(redraw=False)
            super().draw_idle()
        finally:
            self._interaction_painting = False

    def finish_interaction(self, *, redraw=True):
        if not getattr(self, '_interaction_ready', False):
            return
        active = self.interaction_active
        self._frame_timer.stop()
        self._wheel_timer.stop()
        self._restore_surfaces()
        self._active_axes = None
        self._background = None
        self._background_key = None
        self._wheel_changed = False
        if active and redraw and not self._closed:
            super().draw_idle()

    def reset_interaction(self):
        """Stop gestures and release previews before replacing/clearing axes.

        This deliberately does not schedule a draw. The caller can construct the
        next figure first and issue its normal draw request afterwards.
        """
        if not getattr(self, '_interaction_ready', False):
            return
        self.finish_interaction(redraw=False)
        for ax, records in list(self._surfaces.items()):
            for record in records:
                preview = record.preview() if record.preview else None
                if preview is not None and preview in ax.collections:
                    preview.remove()
        self._surfaces.clear()
        self._draw_pending = False

    def draw(self):
        if getattr(self, '_interaction_ready', False):
            self.finish_interaction(redraw=False)
            self._synchronize()
            self.full_draw_count += 1
        return super().draw()

    def print_figure(self, *args, **kwargs):
        self.finish_interaction(redraw=False)
        try:
            return super().print_figure(*args, **kwargs)
        finally:
            self._background = None
            self._background_key = None

    def resizeEvent(self, event):
        if getattr(self, '_interaction_ready', False):
            self.finish_interaction(redraw=False)
        return super().resizeEvent(event)

    def closeEvent(self, event):
        if getattr(self, '_interaction_ready', False):
            self.reset_interaction()
            self._closed = True
            self._disconnect_events()
        return super().closeEvent(event)

    def showEvent(self, event):
        if getattr(self, '_interaction_ready', False):
            self._closed = False
            self._synchronize()
        super().showEvent(event)
        if getattr(self, '_interaction_ready', False):
            self.draw_idle()

    def hideEvent(self, event):
        if getattr(self, '_interaction_ready', False):
            self.finish_interaction(redraw=False)
        return super().hideEvent(event)
