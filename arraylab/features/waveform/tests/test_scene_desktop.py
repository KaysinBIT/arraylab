"""Focused native FMCW radar-scene and affected navigation/record acceptance."""
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUTPUT=ROOT/'tmp/fmcw-scene-checks'


def configuration():
    from arraylab.features.waveform import default_waveform_configuration
    value=default_waveform_configuration()
    value['params'].update(samples=64,rangeFft=128,loops=8,dopplerFft=16,targetRange=12.,targetVelocity=1.)
    value['profileB'].update(samples=64,rangeFft=128,loops=8,dopplerFft=16)
    value['abOptions'].update(enabled=True,layout='blocks',switchGapUs=5.)
    return value


def settle(app,milliseconds=100):
    from PySide6.QtCore import QEventLoop,QTimer
    loop=QEventLoop();QTimer.singleShot(milliseconds,loop.quit);loop.exec();app.processEvents()


def wait(app,predicate,timeout=20):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=predicate()
        if value:return value
        settle(app,40)
    raise AssertionError('The offline FMCW scene did not reach its expected state')


def select(control,value):
    index=control.findData(value);assert index>=0,value;control.setCurrentIndex(index)


@pytest.fixture(scope='module')
def desktop():
    if os.name!='nt':pytest.skip('Focused real Windows Qt/WebGL acceptance')
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    settings={'QT_QPA_PLATFORM':'windows','MPLCONFIGDIR':str(OUTPUT/'mpl'),
              'ARRAYLAB_HISTORY_DIR':str(OUTPUT/'linear'),'ARRAYLAB_SPARSE_HISTORY_DIR':str(OUTPUT/'sparse'),
              'TEMP':str(OUTPUT/'temp'),'TMP':str(OUTPUT/'temp'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    for key in ('MPLCONFIGDIR','ARRAYLAB_HISTORY_DIR','ARRAYLAB_SPARSE_HISTORY_DIR','TEMP'):Path(settings[key]).mkdir(parents=True,exist_ok=True)
    old={key:os.environ.get(key) for key in settings};os.environ.update(settings)
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    from arraylab.application.desktop.window import MainWindow
    from arraylab.features.waveform import calculate_waveform
    from arraylab.framework.parallel import requested_workers,set_workers
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts,True)
    app=QApplication.instance() or QApplication([]);assert app.platformName()=='windows'
    app.setStyle('Fusion');app.setQuitOnLastWindowClosed(False)
    workers=requested_workers();set_workers(1)
    report=calculate_waveform(configuration(),workers=1)
    window=MainWindow(auto_compute=False);window.resize(1440,900);window.show();window.raise_();window.activateWindow()
    window.activate_task('waveform');panel=window.waveform_panel
    try:yield app,window,panel,report
    finally:
        if panel.thread is not None:panel.cancel();wait(app,lambda:panel.thread is None)
        window.close();window.deleteLater();settle(app,150);set_workers(workers)
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


@pytest.fixture
def scene(desktop,monkeypatch):
    app,window,panel,report=desktop
    from arraylab.features.waveform import desktop as implementation,model
    def forbidden(*args,**kwargs):pytest.fail('Scene interaction started a waveform calculation')
    monkeypatch.setattr(implementation,'calculate_waveform',forbidden);monkeypatch.setattr(model,'calculate_waveform',forbidden)
    window.activate_task('waveform');panel.accept_result(deepcopy(report),restore_inputs=True,view='overview',profile='A')
    panel.scene_time.setValue(0.);panel.scene_target_az.setValue(0.);panel.scene_target_el.setValue(0.);select(panel.scene_motion,'stationary')
    select(panel.view,'scene');wait(app,lambda:panel.scene_widget is not None and panel.scene_widget.ready)
    widget=panel.scene_widget;widget.pause();widget.reset_camera();settle(app,150)
    state=widget.get_state(sync=True);assert state.get('supported') and not state.get('error'),state
    yield app,window,panel,widget
    widget.pause();select(panel.view,'overview');settle(app,100)


def test_fmcw_name_menu_toolbar_and_scene_tree_entry(scene):
    from arraylab.application.desktop.task_registry import task_spec
    app,window,panel,widget=scene
    assert task_spec('waveform').title=='FMCW雷达波形设计'
    assert window.task_title.text()=='FMCW雷达波形设计'
    assert window.waveform_tree_item.text(0)=='FMCW雷达波形设计'
    assert window.task_actions['waveform'].text()=='FMCW雷达波形设计'
    assert '计算FMCW波形' in window.compute_button.text()
    item=window._task_nav_nodes[('waveform','result','scene')]
    select(panel.view,'overview');window.activate_task_tree_item(item)
    assert panel.view.currentData()=='scene' and window.tree.currentItem() is item


def test_fmcw_scene_uses_shared_native_renderer_and_real_png(scene):
    from PIL import Image
    from arraylab.shared.qt.radar_scene import SceneWidget
    from arraylab.features.pulse.desktop_scene import SceneWidget as PulseSceneWidget
    app,window,panel,widget=scene
    assert isinstance(widget,SceneWidget) and PulseSceneWidget is SceneWidget
    state=widget.get_state(sync=True)
    assert state['mesh_count']>0 and len(state['actors'])==2
    # This fixture is a linear aperture: no invented elevation width or cone.
    assert state['beam_direction'] is None
    raw=widget.png_bytes();assert raw.startswith(b'\x89PNG\r\n\x1a\n')
    image=np.asarray(Image.open(BytesIO(raw)).convert('RGB'))
    assert image.shape[0]>=300 and image.shape[1]>=400 and np.std(image)>10
    assert len(np.unique(image.reshape(-1,3),axis=0))>30
    (OUTPUT/'fmcw-scene.png').write_bytes(raw);assert window.grab().save(str(OUTPUT/'fmcw-native-window.png'))


def test_known_fmcw_pointing_and_near_range_framing_without_invented_beamwidth(scene):
    app,window,panel,widget=scene;saved=deepcopy(panel.report)
    state=widget.get_state(sync=True)
    assert state['beam_direction'] is None and state['pointing_direction'] is not None
    geometry=panel.report['configuration'].get('geometry') or {}
    az,el=np.deg2rad([geometry.get('steerAzDeg',0.),geometry.get('steerElDeg',0.)])
    # World(x,y,z) maps to renderer(x,z,-y); use the saved antenna direction.
    expected=[np.cos(el)*np.cos(az),np.sin(el),-np.cos(el)*np.sin(az)]
    assert np.allclose(state['pointing_direction'],expected,atol=1e-12)
    assert max(abs(x) for x in state['camera']['position'])<100.
    assert np.allclose(state['actors'][1]['position'],[12.,0.,0.])
    panel.scene_target_az.setValue(25.);panel.scene_target_el.setValue(10.);settle(app,100)
    assert np.allclose(widget.get_state(sync=True)['pointing_direction'],expected,atol=1e-12)
    assert panel.report==saved


def test_fmcw_ab_keeps_the_target_and_camera_and_display_does_not_mutate_report(scene):
    app,window,panel,widget=scene
    before=deepcopy(panel.report);config=deepcopy(panel.config())
    panel.scene_target_az.setValue(25.);panel.scene_target_el.setValue(10.);select(panel.scene_motion,'receding');panel.scene_time.setValue(1.25)
    settle(app,150);state_a=widget.get_state(sync=True)
    camera=deepcopy(state_a['camera']);camera['position'][0]+=15.;widget.restore_camera(camera);settle(app,100)
    select(panel.result_profile,'B');settle(app,150);state_b=widget.get_state(sync=True)
    assert state_b['time_s']==pytest.approx(1.25)
    assert np.allclose(state_b['actors'][1]['position'],state_a['actors'][1]['position'])
    assert np.allclose(state_b['camera']['position'],camera['position'])
    assert panel.report==before and panel.config()==config
    assert panel.scene_target_az.value()==25. and panel.scene_target_el.value()==10.
    assert panel.scene_motion.currentData()=='receding'


def test_unsolved_fmcw_inputs_do_not_replace_the_saved_scene(scene):
    app,window,panel,widget=scene
    state=widget.get_state(sync=True);saved=deepcopy(panel.report)
    panel.inputs['targetRange'].setText('999');panel.inputs['targetVelocity'].setText('9')
    panel.redraw();settle(app,100);after=widget.get_state(sync=True)
    assert panel.config()['params']['targetRange']==999.
    assert np.allclose(after['actors'][1]['position'],state['actors'][1]['position'])
    assert panel.report==saved


def test_fmcw_record_restores_scene_time_camera_and_display_assumptions(scene,tmp_path):
    from arraylab.features.waveform.io import load_waveform_record,save_waveform_record
    app,window,panel,widget=scene
    panel.scene_target_az.setValue(-20.);panel.scene_target_el.setValue(12.);select(panel.scene_motion,'approaching');panel.scene_time.setValue(1.5)
    select(panel.result_profile,'B');settle(app,150)
    camera=deepcopy(widget.get_state(sync=True)['camera']);camera['position'][1]+=10.;widget.restore_camera(camera);settle(app,100)
    path=tmp_path/'scene-record.json';panel.save_record_path(path)
    document=json.loads(path.read_text(encoding='utf-8'));record=load_waveform_record(path)
    assert document['schema_version']==2 and record['scene_state']['time_s']==1.5
    assert record['scene_state']['target_az_deg']==-20. and record['scene_state']['target_el_deg']==12.
    assert record['scene_state']['motion']=='approaching'
    panel.scene_target_az.setValue(0.);panel.scene_target_el.setValue(0.);select(panel.scene_motion,'stationary');panel.scene_time.setValue(0.);widget.reset_camera()
    panel.load_record_path(path);settle(app,150)
    assert panel.view.currentData()=='scene' and panel.result_profile.currentData()=='B'
    assert panel.scene_time.value()==1.5 and panel.scene_target_az.value()==-20. and panel.scene_target_el.value()==12.
    assert panel.scene_motion.currentData()=='approaching'
    assert np.allclose(widget.get_state(sync=True)['camera']['position'],camera['position'])
    old=tmp_path/'old-record.json';save_waveform_record(old,panel.report,view='overview',profile='A')
    assert json.loads(old.read_text(encoding='utf-8'))['schema_version']==1
    panel.load_record_path(old);assert panel.view.currentData()=='overview' and panel.result_profile.currentData()=='A'


def test_fmcw_scene_picture_routing_play_pause_and_other_plots(scene):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    app,window,panel,widget=scene
    assert window.current_picture() is widget
    picture=widget.capture_pixmap();assert not picture.isNull()
    QTest.mouseClick(widget.play_button,Qt.MouseButton.LeftButton);wait(app,lambda:panel.scene_time.value()>.1)
    select(panel.view,'rd');settle(app,150)
    state=widget.get_state(sync=True);assert not state['visible'] and not state['playing']
    assert window.current_picture() is panel.canvas and panel.canvas.isVisible()
    count=state['render_count'];settle(app,200);assert widget.get_state(sync=True)['render_count']==count
    select(panel.view,'scene');settle(app,100);assert window.current_picture() is widget
    window.reset_plot_view();settle(app,100);assert widget.get_state(sync=True)['visible']


def test_fmcw_scene_export_uses_selected_group_actual_png_and_metadata(scene,tmp_path):
    app,window,panel,widget=scene
    select(panel.result_profile,'B');panel.scene_target_az.setValue(15.);panel.scene_time.setValue(1.25);settle(app,150)
    png=widget.png_bytes();target=tmp_path/'scene-export';panel.export_path(target)
    assert (target/'scene.png').read_bytes()==png
    note=json.loads((target/'scene-view.json').read_text(encoding='utf-8'))
    assert note['profile']=='B' and note['scene_state']['target_az_deg']==15.
    assert note['scene_state']['time_s']==1.25
    assert (target/'summary.json').is_file() and panel.last_export==target


def test_pulse_scene_still_uses_shared_widget_after_switching_tasks(scene):
    from arraylab.shared.qt.radar_scene import SceneWidget
    app,window,panel,widget=scene
    window.activate_task('pulse');pulse=window.pulse_panel;select(pulse.view,'scene')
    wait(app,lambda:pulse.scene_widget is not None and pulse.scene_widget.ready)
    assert isinstance(pulse.scene_widget,SceneWidget)
    assert pulse.scene_widget.get_state(sync=True).get('supported')
    assert not widget.get_state(sync=True)['visible']
    window.activate_task('waveform');settle(app,150)
    assert panel.scene_widget is widget and widget.get_state(sync=True)['visible']
    assert not pulse.scene_widget.get_state(sync=True)['visible']
