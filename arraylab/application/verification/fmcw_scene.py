"""Focused acceptance for the FMCW scene and the relocated shared viewport."""
from copy import deepcopy
from pathlib import Path
from .pulse_scene import settle,wait


def exercise_fmcw_scene(app,window,output,report,check):
    from arraylab.features.waveform import default_waveform_configuration,calculate_waveform
    from arraylab.features.waveform.io import load_waveform_record
    from arraylab.application.simulation import calculate
    from arraylab import __version__
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    quit_on_close=app.quitOnLastWindowClosed();app.setQuitOnLastWindowClosed(False)
    window.hide();probe=type(window)(auto_compute=False);probe.resize(1440,960)
    probe.show();probe.raise_();probe.activateWindow();settle(app,250)
    try:
        panel=probe.waveform_panel;config=default_waveform_configuration()
        config['params'].update(samples=64,rangeFft=128,loops=8,dopplerFft=16,targetRange=12.,targetVelocity=1.)
        config['profileB'].update(samples=64,rangeFft=128,loops=8,dopplerFft=16)
        config['abOptions'].update(enabled=True,layout='blocks',switchGapUs=5.)
        saved=calculate_waveform(config,workers=1);before=deepcopy(saved)
        panel.accept_result(saved,restore_inputs=True);probe.activate_task('waveform')
        check(probe.task_title.text()==probe.waveform_tree_item.text(0)=='FMCW雷达波形设计','renamed_task_header_and_tree')
        check(probe.task_actions['waveform'].text()=='FMCW雷达波形设计' and 'FMCW' in probe.compute_button.text(),'renamed_menu_and_toolbar')
        panel.view.setCurrentIndex(panel.view.findData('scene'));scene=panel.scene_widget
        wait(app,lambda:scene.ready and len(scene.get_state(sync=True).get('actors',[]))==2)
        state=scene.get_state(sync=True)
        check(state['supported'] and state['terrain_triangles']==0,'shared_offline_scene_without_invented_terrain')
        check(state['beam_direction'] is None and state['pointing_direction'] is not None,'unknown_width_uses_known_pointing_arrow')
        check(max(map(abs,state['camera']['position']))<100,'near_range_auto_frame')
        panel.scene_target_az.setValue(20);panel.scene_target_el.setValue(10)
        panel.scene_motion.setCurrentIndex(panel.scene_motion.findData('receding'));panel.scene_time.setValue(1.25);settle(app,200)
        a=scene.get_state(sync=True);panel.result_profile.setCurrentIndex(panel.result_profile.findData('B'));settle(app,200)
        b=scene.get_state(sync=True)
        check(a['actors'][1]['position']==b['actors'][1]['position'] and a['camera']==b['camera'],'AB_share_target_and_camera')
        check(saved==before,'scene_controls_preserve_numerical_snapshot')
        panel.inputs['targetRange'].setText('999');panel.redraw();settle(app,120)
        check(scene.get_state(sync=True)['actors'][1]['position']==b['actors'][1]['position'],'uncomputed_inputs_do_not_move_saved_target')
        scene.play_button.click();settle(app,250);check(scene.get_state(sync=True)['playing'],'preview_play')
        probe.activate_task('home');settle(app,150);check(not scene.get_state(sync=True)['playing'],'hidden_preview_pauses')
        probe.activate_task('waveform');panel.scene_time.setValue(1.5);settle(app,150);scene.pause()
        expected=panel.scene_state();path=output/'fmcw-scene-record.json';panel.save_record_path(path)
        panel.scene_target_az.setValue(-30);panel.scene_time.setValue(0);panel.load_record_path(path);settle(app,200)
        check(panel.scene_state()==expected,'scene_record_restores_camera_time_and_options')
        check(probe.current_picture() is scene and not scene.capture_pixmap().isNull(),'native_picture_route')
        png=scene.png_bytes();(output/'fmcw-scene.png').write_bytes(png)
        target=output/'export';panel.export_path(target)
        check((target/'scene.png').read_bytes()==png,'export_retains_original_scene_pixels')
        check(load_waveform_record(target/'waveform-record.json')['scene_state']==expected,'export_record_keeps_scene_state')
        probe.grab().save(str(output/'native-fmcw-scene.png'))
        # Only verify the shared component relocation, not other Pulse algorithms.
        pulse=probe.pulse_panel;pr,pa=calculate('pulse','budget',pulse.backend.default_configuration(),workers=1)
        pulse.accept_result(pr,pa,restore_inputs=True);probe.activate_task('pulse');pulse.view.setCurrentIndex(pulse.view.findData('scene'))
        wait(app,lambda:pulse.scene_widget.ready and bool(pulse.scene_widget.get_state(sync=True).get('actors')))
        check(type(scene) is type(pulse.scene_widget) and pulse.scene_widget.get_state(sync=True)['beam_direction'] is not None,'pulse_retains_shared_scene_default_beam')
        report.update(app_version=__version__,numerical_regression_suite=False)
    finally:
        probe.close();probe.deleteLater();app.processEvents();app.setQuitOnLastWindowClosed(quit_on_close)
