"""Scoped resize acceptance using real Qt dock-separator events."""
from contextlib import ExitStack
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import json
import statistics
import time
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, QPoint, Qt
from PySide6.QtTest import QTest

from arraylab.shared.qt.resize_canvas import ResizeAwareFigureCanvas
from arraylab.shared.plotting.metrics import draw_metric_footer
from arraylab.shared.plotting.layout import responsive_subplots
from arraylab.shared.plotting.surface import clear_figure
from arraylab.features.waveform import default_waveform_configuration, calculate_waveform


def settle(app, milliseconds=200):
    loop=QEventLoop();QTimer.singleShot(milliseconds,loop.quit);loop.exec();app.processEvents()


def dock_drag(app, window, canvas, steps=40):
    draws=[];times=[];original=canvas.draw
    def measured(*args,**kwargs):
        start=time.perf_counter()
        try:return original(*args,**kwargs)
        finally:draws.append((time.perf_counter()-start)*1000)
    geometry=window.nav_dock.geometry()
    origin=QPoint(geometry.right()+2,geometry.top()+200)
    widths=[window.nav_dock.width()];preview_start=canvas.resize_preview_paints
    with patch.object(canvas,'draw',measured):
        started=time.perf_counter();QTest.mousePress(window,Qt.MouseButton.LeftButton,pos=origin)
        for index in range(1,steps+1):
            tick=time.perf_counter()
            QTest.mouseMove(window,origin+QPoint(index*4,0),delay=1);app.processEvents()
            times.append((time.perf_counter()-tick)*1000);widths.append(window.nav_dock.width())
        during=len(draws);duration=time.perf_counter()-started
        QTest.mouseRelease(window,Qt.MouseButton.LeftButton,pos=origin+QPoint(steps*4,0))
        settle(app,400)
    return {'steps':steps,'dock_width_min':min(widths),'dock_width_max':max(widths),
            'drag_seconds':duration,'move_median_ms':statistics.median(times),'move_max_ms':max(times),
            'full_draws_during_drag':during,'full_draws_total':len(draws),
            'preview_paints':canvas.resize_preview_paints-preview_start,
            'final_canvas_size':[canvas.width(),canvas.height()]}


def exercise_resize(app, window, output, report, check):
    # Windows can honor the launcher's SW_HIDE for its first top-level window.
    # Use a separate isolated probe, as other native desktop scopes do, so real
    # paint timings are not accidentally measured on an unexposed startup HWND.
    old_quit=app.quitOnLastWindowClosed();app.setQuitOnLastWindowClosed(False)
    probe=None
    try:
        window.hide()
        probe=type(window)(auto_compute=False)
        probe.show();probe.raise_();probe.activateWindow();settle(app,250)
        check(probe.windowHandle().isExposed(),'resize_native_probe_is_exposed')
        return _exercise_resize(app,probe,output,report,check)
    finally:
        if probe is not None:probe.close();probe.deleteLater()
        app.setQuitOnLastWindowClosed(old_quit)


def _exercise_resize(app, window, output, report, check):
    output=Path(output);directory=output/'resize-ui';directory.mkdir(parents=True,exist_ok=True)
    window.resize(1500,1000);window.activate_task('waveform')
    config=default_waveform_configuration();config['params'].update(txCount=4,rxCount=4)
    result=calculate_waveform(config,workers=1)
    panel=window.waveform_panel;panel.accept_result(result,restore_inputs=True)
    settle(app,350);panel.canvas.draw();settle(app)
    original=json.dumps(panel.report,sort_keys=True,allow_nan=False)
    figure=panel.figure;axes=tuple(figure.axes);artists=tuple(tuple(ax.lines) for ax in axes)
    limits=[(ax.get_xlim(),ax.get_ylim()) for ax in axes]
    samples=[[line.get_xydata().copy() for line in group] for group in artists]
    with ExitStack() as stack:
        for owner,name in [(window,'compute'),(window,'compute_linear'),(panel,'start'),
                           (window.feed_panel,'generate'),(window.planar_panel,'generate'),
                           (window.sparse_panel,'start'),(window.subarray_panel,'start'),(window.mimo_panel,'start')]:
            stack.enter_context(patch.object(owner,name,side_effect=AssertionError('Resize must not calculate')))
        timing=dock_drag(app,window,panel.canvas)
        report['resize']={'timing':timing,'visible':window.isVisible(),'exposed':window.windowHandle().isExposed()}
        check(timing['dock_width_max']-timing['dock_width_min']>=100,'resize_real_dock_separator_changes_width')
        check(timing['preview_paints']>0 and timing['full_draws_during_drag']<=2,'resize_drag_uses_image_preview_without_full_frame_backlog')
        check(timing['full_draws_total']<=3 and not panel.canvas.resize_pending,'resize_final_frame_settles_once_or_bounded_layout_followup')
        check(panel.figure is figure and tuple(figure.axes)==axes,'resize_keeps_existing_figure_and_axes')
        check([(ax.get_xlim(),ax.get_ylim()) for ax in axes]==limits,'resize_keeps_zoom_and_axis_limits')
        check(json.dumps(panel.report,sort_keys=True,allow_nan=False)==original,'resize_keeps_full_scientific_snapshot')
        check(all(np.array_equal(line.get_xydata(),saved) for group,data in zip(artists,samples) for line,saved in zip(group,data)),
              'resize_keeps_original_plot_samples')
        renderer=panel.canvas.get_renderer();footer=figure._arraylab_metric_footer.get_window_extent(renderer)
        check(all(not ax.get_tightbbox(renderer).overlaps(footer) for ax in axes),'resize_final_labels_do_not_overlap_metric_footer')
        check(abs(renderer.width-panel.canvas.width()*panel.canvas.device_pixel_ratio)<1 and
              abs(renderer.height-panel.canvas.height()*panel.canvas.device_pixel_ratio)<1,'resize_final_renderer_matches_dpi_and_widget_size')
        # Figure replacement and export during the pending interval must flush it.
        window.resizeDocks([window.nav_dock],[220],Qt.Orientation.Horizontal);app.processEvents()
        panel.view.setCurrentIndex(panel.view.findData('spectra'));settle(app,350)
        check(panel.view.currentData()=='spectra' and not panel.canvas.resize_pending,'resize_view_switch_discards_old_preview')
        window.resizeDocks([window.nav_dock],[260],Qt.Orientation.Horizontal);app.processEvents()
        image=window.current_picture()
        from arraylab.application.desktop.workflow import picture_pixmap
        captured=picture_pixmap(image)
        check(not captured.isNull() and not panel.canvas.resize_pending,'resize_picture_copy_forces_full_resolution')
        destination=BytesIO();panel.canvas.print_figure(destination,format='png',dpi=160)
        check(destination.getvalue().startswith(b'\x89PNG'),'resize_png_export_keeps_normal_rendering_path')
        panel.view.setCurrentIndex(panel.view.findData('overview'));settle(app,250)
        window.grab().save(str(directory/'waveform-after-drag.png'))
        # The other workbenches share this canvas path. Synthetic artists test
        # their real container resize hooks without running their algorithms.
        canvases={'linear':window.plot.canvas,'feed':window.feed_panel.pattern_plot.canvas,
                  'planar':window.planar_panel.canvas,'sparse':window.sparse_panel.canvas,
                  'subarray':window.subarray_panel.canvas,'mimo':window.mimo_panel.canvas}
        for task,canvas in canvases.items():
            window.activate_task(task);settle(app,200)
            clear_figure(canvas.figure)
            ax=canvas.figure.add_subplot(111);line=ax.plot([0,1,2],[2,5,3])[0]
            ax.set(xlabel='Angle / deg',ylabel='Magnitude / dB')
            draw_metric_footer(canvas.figure,['Resize verification: synthetic samples'])
            responsive_subplots(canvas.figure,[ax],1)
            # Feed/linear compact mode owns a public axes reference.
            if task=='linear':window.plot.axes=ax;window.plot.three_dimensional=False
            if task=='feed':window.feed_panel.pattern_plot.axes=ax;window.feed_panel.pattern_plot.three_dimensional=False
            canvas.draw();settle(app)
            with patch.object(canvas.figure,'draw',wraps=canvas.figure.draw) as painting:
                for width in (200,210,220,230,240,250):
                    window.resizeDocks([window.nav_dock],[width],Qt.Orientation.Horizontal);app.processEvents()
                settle(app,350)
                check(painting.call_count<=3,'resize_'+task+'_bounded_final_renders')
            check(isinstance(canvas,ResizeAwareFigureCanvas) and not canvas.resize_pending,
                  'resize_'+task+'_uses_settled_shared_canvas')
            check(canvas.figure.axes[0] is ax and np.array_equal(line.get_ydata(),[2,5,3]),
                  'resize_'+task+'_preserves_artists_and_samples')
    report['resize']={'scope':'Real waveform dock drag; other workbenches use synthetic artists, no numerical regressions',
                      'qt_platform':app.platformName(),'uses_real_user_history':False,'timing':timing,
                      'screenshot':str(directory/'waveform-after-drag.png')}
