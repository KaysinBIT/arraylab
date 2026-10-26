"""Resize preview lifecycle and full-quality output, without business solvers."""
from io import BytesIO
import os

import numpy as np
import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from matplotlib.figure import Figure
from PIL import Image

from arraylab.shared.qt.resize_canvas import ResizeAwareFigureCanvas
from arraylab.shared.qt.canvas import InteractiveFigureCanvas
from arraylab.shared.plotting.layout import responsive_subplots
from arraylab.shared.plotting.metrics import draw_metric_footer


@pytest.fixture(scope='module')
def app():
    os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
    return QApplication.instance() or QApplication([])


def settle(ms=180):
    loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec()


def make_canvas(cls=ResizeAwareFigureCanvas):
    canvas=cls(Figure(figsize=(8,5),dpi=100,layout='constrained'))
    canvas.resize(800,500)
    ax=canvas.figure.add_subplot(111);line=ax.plot([0,1,2],[2,4,3])[0]
    ax.set_xlim(.2,1.8);ax.set_ylim(1,5)
    canvas.show();settle();canvas.draw()
    return canvas,ax,line


@pytest.mark.parametrize('cls',[ResizeAwareFigureCanvas,InteractiveFigureCanvas])
def test_resize_burst_has_preview_then_one_full_render(app,cls):
    canvas,ax,line=make_canvas(cls)
    try:
        camera=(ax.get_xlim(),ax.get_ylim());data=line.get_xydata().copy();events=[]
        canvas.mpl_connect('draw_event',lambda event:events.append(event))
        for width in range(810,910,10):
            canvas.resize(width,510);app.processEvents()
        assert canvas.resize_pending and not events
        assert canvas.resize_preview_paints>0 and canvas._resize_image is not None
        settle()
        assert not canvas.resize_pending and len(events)==1
        assert canvas._resize_image is None and not canvas._resize_timer.isActive()
        assert canvas.renderer.width==canvas.width()*canvas.device_pixel_ratio
        assert canvas.renderer.height==canvas.height()*canvas.device_pixel_ratio
        assert (ax.get_xlim(),ax.get_ylim())==camera
        np.testing.assert_array_equal(line.get_xydata(),data)
    finally:canvas.close()


@pytest.mark.parametrize('ratio',[1.,1.25,2.])
def test_forced_draw_applies_current_size_and_pixel_ratio(app,ratio):
    canvas,ax,line=make_canvas()
    try:
        canvas._set_device_pixel_ratio(ratio)
        canvas.resize(764,516);app.processEvents()
        assert canvas.resize_pending
        canvas.draw()
        assert not canvas.resize_pending
        assert canvas.renderer.width==pytest.approx(764*ratio)
        assert canvas.renderer.height==pytest.approx(516*ratio)
    finally:canvas.close()


def test_copy_and_export_flush_pending_preview_at_requested_resolution(app):
    from arraylab.application.desktop.workflow import picture_pixmap
    canvas,ax,line=make_canvas()
    try:
        canvas.resize(720,480);app.processEvents();assert canvas.resize_pending
        image=picture_pixmap(canvas)
        assert not canvas.resize_pending and not image.isNull()
        assert image.width()/image.devicePixelRatio()==canvas.width()
        canvas.resize(740,490);app.processEvents();assert canvas.resize_pending
        destination=BytesIO();canvas.print_figure(destination,format='png',dpi=200)
        assert not canvas.resize_pending
        image=Image.open(destination)
        assert image.size==(1480,980)
        assert canvas._resize_image is None
    finally:canvas.close()


def test_figure_replacement_discards_old_preview_after_settle(app):
    canvas,_,_=make_canvas()
    try:
        canvas.resize(710,450);app.processEvents();assert canvas.resize_pending
        old=canvas.figure
        figure=Figure(figsize=(7,4.5),dpi=100,layout='constrained')
        line=figure.add_subplot(111).plot([0,1],[10,30])[0]
        canvas.figure=figure;figure.set_canvas(canvas);canvas.draw_idle()
        settle()
        assert canvas.figure is figure and canvas.figure is not old
        assert canvas._resize_image is None and not canvas.resize_pending
        np.testing.assert_array_equal(line.get_ydata(),[10,30])
        assert canvas.renderer.width==canvas.width()*canvas.device_pixel_ratio
    finally:canvas.close()


def test_responsive_axes_footer_and_navigation_survive_settle(app):
    canvas=ResizeAwareFigureCanvas(Figure(figsize=(10,6),dpi=100,layout='constrained'))
    try:
        axes=canvas.figure.subplots(1,2)
        artists=[ax.plot([0,1],[i,i+1])[0] for i,ax in enumerate(axes)]
        for ax in axes:ax.set_xlabel('Angle / deg');ax.set_ylabel('Magnitude / dB')
        draw_metric_footer(canvas.figure,['Scientific values remain unchanged','Final full-resolution layout'])
        responsive_subplots(canvas.figure,axes,2,column_width=310)
        canvas.resize(1000,600);canvas.show();settle();canvas.draw()
        canvas.resize(500,600);app.processEvents();settle()
        assert canvas.figure._arraylab_responsive_plots['shape']==(2,1)
        renderer=canvas.get_renderer();footer=canvas.figure._arraylab_metric_footer.get_window_extent(renderer)
        assert all(not ax.get_tightbbox(renderer).overlaps(footer) for ax in axes)
        assert [ax.lines[0] for ax in axes]==artists
        canvas.resize(1000,600);app.processEvents();settle()
        assert canvas.figure._arraylab_responsive_plots['shape']==(1,2)
    finally:canvas.close()


def test_close_pending_resize_cleans_timer_and_reopen_repaints(app):
    canvas,ax,line=make_canvas()
    try:
        canvas.resize(750,510);app.processEvents();assert canvas.resize_pending
        canvas.close()
        assert not canvas._resize_timer.isActive() and canvas._resize_image is None
        canvas.show();settle()
        assert not canvas.resize_pending and canvas.renderer.width==750
        np.testing.assert_array_equal(line.get_ydata(),[2,4,3])
    finally:canvas.close()


def test_first_resize_without_rendered_image_is_safe(app):
    canvas=ResizeAwareFigureCanvas(Figure())
    try:
        canvas.figure.add_subplot(111).plot([0,1],[0,1])
        canvas.resize(650,400);canvas.show();app.processEvents();settle()
        assert not canvas.resize_pending and hasattr(canvas,'renderer')
        assert canvas.renderer.width==650
    finally:canvas.close()


def test_plot_mouse_press_flushes_resize_before_coordinate_mapping(app):
    canvas,ax,line=make_canvas()
    try:
        canvas.resize(750,510);app.processEvents();assert canvas.resize_pending
        events=[];canvas.mpl_connect('button_press_event',lambda event:events.append(event))
        QTest.mousePress(canvas,Qt.MouseButton.LeftButton,pos=QPoint(300,200))
        assert not canvas.resize_pending and canvas.renderer.width==750
        assert len(events)==1 and events[0].x==300
        QTest.mouseRelease(canvas,Qt.MouseButton.LeftButton,pos=QPoint(300,200))
    finally:canvas.close()


def test_hiding_preview_stops_background_resize_render(app):
    canvas,ax,line=make_canvas()
    try:
        canvas.resize(750,510);app.processEvents();assert canvas.resize_pending
        frames=[];canvas.mpl_connect('draw_event',lambda event:frames.append(event))
        canvas.hide();settle()
        assert not canvas.resize_pending and not frames
        assert not canvas._resize_timer.isActive() and canvas._resize_image is None
        canvas.show();settle()
        assert len(frames)==1 and canvas.renderer.width==750
    finally:canvas.close()


def test_resize_during_3d_gesture_restores_surface_and_retains_camera(app):
    canvas=InteractiveFigureCanvas(Figure(figsize=(6,5),layout='constrained'))
    try:
        ax=canvas.figure.add_subplot(111,projection='3d')
        x,y=np.meshgrid(np.linspace(-1,1,41),np.linspace(-1,1,21));z=x*x+y*y
        colors=np.ones((*x.shape,4));colors[...,:3]=(.2,.5,.7)
        surface=ax.plot_surface(x,y,z,facecolors=colors,shade=False)
        canvas.register_surface(ax,surface,x,y,z,colors)
        canvas.show();settle();canvas.draw()
        ax.view_init(27,43);camera=(ax.elev,ax.azim)
        canvas._begin_interaction(ax);assert not surface.get_visible()
        canvas._frame_timer.start(30)
        canvas.resize(750,510);app.processEvents()
        assert canvas.resize_pending and not canvas.interaction_active
        assert surface.get_visible() and not canvas._frame_timer.isActive()
        settle()
        assert not canvas.resize_pending and surface.get_visible()
        assert (ax.elev,ax.azim)==camera
    finally:canvas.close()
