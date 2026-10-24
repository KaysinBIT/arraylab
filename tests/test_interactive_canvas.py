"""Synthetic Qt event regression; real-window timing is a separate smoke test."""
from io import BytesIO
import os

import numpy as np
import pytest

pytest.importorskip('PySide6')

from PySide6.QtCore import QEvent, QEventLoop, QPointF, QTimer, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure

from arraylab.interactive_canvas import InteractiveFigureCanvas, _preview_grid


@pytest.fixture(scope='module')
def application():
    # This unit module does not claim desktop/OS input acceptance. Explicitly
    # selected Windows runs retain that platform; ordinary unit runs are hidden.
    previous = os.environ.get('QT_QPA_PLATFORM')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    app = QApplication.instance() or QApplication([])
    yield app
    if previous is None:
        os.environ.pop('QT_QPA_PLATFORM', None)


def settle(milliseconds=60):
    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


@pytest.fixture
def canvas(application):
    figure = Figure(figsize=(6, 5), dpi=100, layout='constrained')
    item = InteractiveFigureCanvas(figure)
    ax = figure.add_subplot(111, projection='3d')
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(-np.pi, np.pi, 121), indexing='ij')
    x, y, z = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    colors = np.ones((*x.shape, 4)); colors[..., :3] = (.2, .5, .7)
    surface = ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=2, shade=False)
    item.register_surface(ax, surface, x, y, z, colors)
    figure.text(.5, .02, 'Unchanged scientific metrics', ha='center')
    item.draw()
    yield item, ax, surface
    item.close()
    application.processEvents()


def send_mouse(canvas, kind, x, y, pressed=True):
    local = QPointF(x, canvas.height()-y)
    event = QMouseEvent(kind, local, local,
        Qt.MouseButton.NoButton if kind == QEvent.Type.MouseMove else Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton if pressed else Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(canvas, event)


@pytest.mark.parametrize('shape', [(61, 121), (181, 361), (2, 10001), (10001, 2)])
def test_preview_uses_only_existing_samples_and_bounds_faces(shape):
    x = np.arange(np.prod(shape)).reshape(shape)
    colors = np.ones((*shape, 4))
    xyz, sampled_colors = _preview_grid(x, x+1, x+2, colors)
    assert (xyz[0].shape[0]-1)*(xyz[0].shape[1]-1) <= 600
    assert xyz[0][0, 0] == x[0, 0] and xyz[0][-1, -1] == x[-1, -1]
    assert np.isin(xyz[0], x).all()
    assert sampled_colors.shape == (*xyz[0].shape, 4)


@pytest.mark.parametrize('shape', [(61, 121), (181, 361), (2, 10001), (10001, 2)])
def test_preview_preserves_principal_cut_indices_and_existing_peak(shape):
    x, y = np.indices(shape)
    z = np.zeros(shape)
    peak = (shape[0]//3, shape[1]//3)
    z[peak] = 1e6
    colors = np.ones((*shape, 4))
    colors[peak] = (.1, .2, .3, .4)
    xyz, sampled_colors = _preview_grid(x, y, z, colors)
    assert (xyz[0].shape[0]-1)*(xyz[0].shape[1]-1) <= 600
    assert {0, shape[0]-1, round((shape[0]-1)/2), peak[0]} <= set(xyz[0][:, 0])
    assert {0, shape[1]-1, *(round((shape[1]-1)*f) for f in (.25, .5, .75)), peak[1]} <= set(xyz[1][0, :])
    selected_peak = np.unravel_index(np.argmax(xyz[2]), xyz[2].shape)
    assert (xyz[0][selected_peak], xyz[1][selected_peak], xyz[2][selected_peak]) == (*peak, 1e6)
    np.testing.assert_array_equal(sampled_colors[selected_peak], colors[peak])
    assert z[peak] == 1e6 and np.count_nonzero(z) == 1


def test_preview_keeps_narrow_broadside_main_lobe_circle():
    theta, phi = np.meshgrid(np.linspace(0, np.pi, 61), np.linspace(-np.pi, np.pi, 121), indexing='ij')
    radius = np.exp(-(25*np.sin(theta)*np.cos(phi))**2)
    x, y, z = radius*np.sin(theta)*np.cos(phi), radius*np.sin(theta)*np.sin(phi), radius*np.cos(theta)
    xyz, _ = _preview_grid(x, y, z, np.ones((*x.shape, 4)))
    for sign in (-1, 1):
        assert np.any(np.isclose(xyz[0], 0) & np.isclose(xyz[1], sign) & np.isclose(xyz[2], 0))


def test_mouse_frames_coalesce_restore_full_surface_and_keep_camera(canvas):
    item, ax, surface = canvas
    engine = item.figure.get_layout_engine()
    x, y = ax.bbox.x0+ax.bbox.width/2, ax.bbox.y0+ax.bbox.height/2
    initial = (ax.elev, ax.azim)
    frames = []
    item.mpl_connect('draw_event', lambda event: frames.append(event))
    send_mouse(item, QEvent.Type.MouseButtonPress, x, y)
    assert item.interaction_active and not surface.get_visible()
    full_before = item.full_draw_count
    for step in range(1, 21):
        send_mouse(item, QEvent.Type.MouseMove, x+step, y+step/2)
    assert item._frame_timer.isActive()
    settle()
    assert item.interaction_draw_count == 1
    assert len(frames) == 1  # Static-background capture is not a rendered frame.
    assert item.full_draw_count == full_before
    assert (ax.elev, ax.azim) != initial
    camera = (ax.elev, ax.azim)
    assert item.figure.get_layout_engine() is engine
    send_mouse(item, QEvent.Type.MouseButtonRelease, x+20, y+10, False)
    settle()
    assert surface.get_visible() and not item.interaction_active
    assert item.full_draw_count == full_before+1
    assert (ax.elev, ax.azim) == camera


def test_wheel_burst_has_one_final_history_view_and_home_restores(canvas):
    item, ax, surface = canvas
    toolbar = NavigationToolbar2QT(item, None)
    initial = ax.get_xlim3d()
    x, y = ax.bbox.x0+ax.bbox.width/2, ax.bbox.y0+ax.bbox.height/2
    for _ in range(3):
        MouseEvent('scroll_event', item, x, y, step=1)._process()
    assert ax.get_xlim3d()[1]-ax.get_xlim3d()[0] < initial[1]-initial[0]
    assert len(toolbar._nav_stack) == 2
    assert item.interaction_active
    # Home while the wheel timer is still pending must restore the full display.
    toolbar.home()
    settle()
    assert not item.interaction_active and surface.get_visible()
    assert ax.get_xlim3d() == initial
    # Matplotlib Home appends the home view; Back returns to the wheel result.
    toolbar.back()
    settle()
    assert ax.get_xlim3d()[1]-ax.get_xlim3d()[0] < initial[1]-initial[0]
    toolbar.forward(); settle()
    assert ax.get_xlim3d() == initial
    toolbar.close()


def test_external_draw_and_export_restore_scientific_surface(canvas):
    item, ax, surface = canvas
    item._begin_interaction(ax)
    assert not surface.get_visible()
    item.draw()
    assert surface.get_visible() and not item.interaction_active
    item._begin_interaction(ax)
    destination = BytesIO()
    item.print_figure(destination, format='png')
    assert destination.getvalue().startswith(b'\x89PNG')
    assert surface.get_visible() and not item.interaction_active
    assert not item._frame_timer.isActive() and not item._wheel_timer.isActive()


def test_reset_clear_replace_and_close_do_not_keep_pending_gestures(canvas):
    item, ax, surface = canvas
    item._begin_interaction(ax)
    item._frame_timer.start(30)
    item.reset_interaction()
    assert surface.get_visible() and not item._surfaces
    assert not item._frame_timer.isActive() and item._background is None
    item.figure.clear()
    new_ax = item.figure.add_subplot(111)
    new_ax.plot([0, 1], [0, 1])
    item.draw_idle(); settle()
    assert not item.interaction_active and item.figure.axes == [new_ax]
    old_registry = item.callbacks
    item.figure = Figure(figsize=(4, 3)); item.figure.set_canvas(item)
    item.figure.add_subplot(111).plot([1, 2], [2, 3])
    item.draw()
    assert item._event_registry is item.callbacks and item.callbacks is not old_registry
    item.close()
    assert item._event_registry is None and not item._frame_timer.isActive()


def test_hidden_view_restores_full_surface_before_being_shown_again(canvas):
    item, ax, surface = canvas
    item.show(); settle()
    item._begin_interaction(ax)
    item._frame_timer.start(30)
    item.hide()
    assert not item.interaction_active and surface.get_visible()
    assert not item._frame_timer.isActive()
    count = item.full_draw_count
    item.show(); settle()
    assert item.full_draw_count > count and surface.get_visible()


def test_workbench_picture_capture_finishes_preview_without_changing_clipboard(canvas):
    from arraylab.workflow import picture_pixmap
    item, ax, surface = canvas
    item._begin_interaction(ax)
    image = picture_pixmap(item)
    assert not image.isNull()
    assert surface.get_visible() and not item.interaction_active
