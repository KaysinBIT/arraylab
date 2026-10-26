"""Native Windows/frozen acceptance limited to the restored 3D scene."""
from pathlib import Path
import time
import json
from PySide6.QtCore import QEventLoop,QTimer,Qt,QPoint,QPointF
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest


def settle(app,ms=100):
    loop=QEventLoop();QTimer.singleShot(ms,loop.quit);loop.exec();app.processEvents()


def wait(app,condition,seconds=30):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        if condition():return
        settle(app,30)
    raise AssertionError('Scene operation timed out')


def exercise_pulse_scene(app,window,output,report,check):
    from arraylab.application.simulation import calculate,export_process,load_record
    from arraylab.shared.simulation_io import write_json
    from arraylab import __version__
    quit_on_close=app.quitOnLastWindowClosed();app.setQuitOnLastWindowClosed(False)
    window.hide();probe=type(window)(auto_compute=False);probe.resize(1440,900)
    probe.show();probe.raise_();probe.activateWindow();settle(app,300)
    try:
        panel=probe.pulse_panel;config=panel.backend.default_configuration()
        config['scene']['terrain'].update(enabled=True,kind='hills',resolution=17)
        record,arrays=calculate('pulse','budget',config,workers=1)
        panel.accept_result(record,arrays,restore_inputs=True);probe.activate_task('pulse')
        panel.view.setCurrentIndex(panel.view.findData('scene'));scene=panel.scene_widget
        wait(app,lambda:scene.ready and bool(scene.get_state(sync=True).get('actors')))
        state=scene.get_state(sync=True);report['renderer']=state;report['app_version']=__version__
        check(state.get('supported') is True and state['terrain_triangles']>0,'original_WebGL_terrain_and_actors')
        check(probe.current_picture() is scene and not panel.canvas.isVisible(),'native_scene_picture_route')
        geometry=state['geometries'];panel.scene_time.setValue(1.25);settle(app,250)
        current=scene.get_state(sync=True)
        check(abs(current['time_s']-1.25)<.002 and current['geometries']==geometry,'time_changes_do_not_rebuild_scene')
        before=current['camera'];surface=scene.view.focusProxy() or scene.view
        start=QPoint(surface.width()//2,surface.height()//2)
        QTest.mousePress(surface,Qt.MouseButton.LeftButton,pos=start)
        for i in range(1,7):QTest.mouseMove(surface,start+QPoint(12*i,5*i),delay=20)
        QTest.mouseRelease(surface,Qt.MouseButton.LeftButton,pos=start+QPoint(72,30));settle(app,350)
        rotated=scene.get_state(sync=True)['camera'];check(rotated!=before,'actual_mouse_orbit')
        wheel=QWheelEvent(QPointF(start),QPointF(surface.mapToGlobal(start)),QPoint(),QPoint(0,120),Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
        app.sendEvent(surface,wheel);settle(app,300)
        check(scene.get_state(sync=True)['camera']!=rotated,'actual_mouse_zoom')
        probe.reset_plot_view();settle(app,300)
        restored=scene.get_state(sync=True)['camera'];check(all(abs(a-b)<1e-6 for a,b in zip(before['position'],restored['position'])),'stable_camera_reset')
        scene.play_button.click();settle(app,250);state=scene.get_state(sync=True)
        check(state['playing'] and state['time_s']>1.25,'native_play_and_time_sync')
        probe.activate_task('home');settle(app,200)
        check(not scene.get_state(sync=True)['playing'],'switch_task_pauses_animation')
        probe.activate_task('pulse');panel.scene_time.setValue(1.5);settle(app,200);scene.pause()
        record=panel.record_snapshot();path=output/'scene-record.npz';panel.save_record_path(path)
        scene.reset_camera();panel.scene_time.setValue(0);probe.load_simulation_record_async(path)
        wait(app,lambda:getattr(probe,'_record_reader',None) is None);settle(app,250)
        check(abs(panel.scene_time.value()-1.5)<.002 and panel.record_snapshot()['view_state']['scene_camera']==record['view_state']['scene_camera'],'camera_and_time_record_roundtrip')
        png=scene.png_bytes();(output/'original-scene.png').write_bytes(png)
        check(png.startswith(b'\x89PNG') and len(png)>10000,'original_style_PNG_capture')
        check(not scene.capture_pixmap().isNull(),'toolbar_picture_capture')
        target=output/'export';record=panel.record_snapshot()
        panel._launch(lambda progress,cancel:export_process(target,record,arrays,progress=progress,cancel=cancel,scene_png=png),'导出结果')
        wait(app,lambda:panel.thread is None,60)
        check((target/'scene.png').read_bytes()==png and (target/'scene-view.json').is_file(),'export_keeps_original_scene_pixels')
        restored,_=load_record(target/'arraylab-record.json')
        check(restored['view_state']==record['view_state'],'export_preserves_scene_view_metadata')
        probe.grab().save(str(output/'native-pulse-scene.png'))
        scene.shutdown();check(not scene.ready and not scene.timer.isActive(),'scene_dispose_and_timer_cleanup')
        report['numerical_tests_rerun']=False
    finally:
        if probe.pulse_panel.thread is not None:
            probe.pulse_panel.cancel();wait(app,lambda:probe.pulse_panel.thread is None,30)
        probe.close();probe.deleteLater();app.processEvents();app.setQuitOnLastWindowClosed(quit_on_close)
