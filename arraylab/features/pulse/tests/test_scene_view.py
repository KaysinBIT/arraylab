"""Focused checks for the original PulseLab scene retained inside ArrayLab.

These are real Windows Qt/WebGL checks, not browser-DOM mocks. They only
exercise scene geometry, viewport state and the directly affected record path.
"""
from copy import deepcopy
from io import BytesIO
from pathlib import Path
import json
import os
import time

import numpy as np
import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUTPUT=ROOT/'tmp/pulse-scene-checks'


def scene_project():
    from arraylab.features.pulse import default_configuration
    p=default_configuration()
    p['scene']['terrain'].update(enabled=True,kind='flat',extent_m=[1000.,1000.],resolution=17,base_height_m=80.)
    p['scene']['platform'].update(position_m=[100.,0.,50.],velocity_mps=[30.,0.,0.],height_mode='agl',attitude_deg=[20.,10.,-15.])
    p['array'].update(pointing_deg=[15.,-5.],scan_rate_deg_s=[2.,1.])
    p['scene']['targets']=[{'id':'目标A','position_m':[500.,100.,150.],'velocity_mps':[0.,10.,0.],
                           'rcs_m2':1.,'fluctuation':'none','trajectory':[],'height_mode':'absolute'}]
    return p


def test_scene_payload_preserves_input_units_and_terrain_without_signal_run(monkeypatch):
    from arraylab.features.pulse import scene_payload,engine
    from arraylab.features.pulse.scene import pose_with_terrain,terrain_grid
    monkeypatch.setattr(engine,'simulate',lambda *a,**k:pytest.fail('Scene display must not calculate a signal'))
    project=scene_project();before=deepcopy(project)
    result={'kind':'budget','configuration':project}
    payload=scene_payload(result,1.25)
    assert project==before and payload['project']==project
    assert payload['time_s']==1.25 and payload['grid'] is not None
    assert np.all(np.asarray(payload['grid']['z_m'])==80.)
    position,_=pose_with_terrain(payload['project']['scene']['platform'],1.25,terrain_grid(project))
    assert np.allclose(position,[137.5,0.,130.])
    json.dumps(payload,ensure_ascii=False,allow_nan=False)


def test_scene_payload_uses_inverse_companion_configuration(monkeypatch):
    from arraylab.features.pulse import scene_payload,service
    input_config=scene_project();candidate=deepcopy(input_config);candidate['scene']['platform']['position_m'][0]=300.
    result={'kind':'inverse','configuration':input_config,'array_source':{'candidate_rank':3}}
    calls=[]
    def selected(value):calls.append(value);return candidate
    monkeypatch.setattr(service,'array_configuration',selected)
    payload=scene_payload(result,2.)
    assert calls==[result] and payload['project']==candidate
    assert '#3' in payload['caption']


@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_scene_payload_rejects_nonfinite_time(bad):
    from arraylab.features.pulse import scene_payload
    with pytest.raises(ValueError):scene_payload({'kind':'budget','configuration':scene_project()},bad)


def settle(app,milliseconds=50):
    from PySide6.QtCore import QEventLoop,QTimer
    loop=QEventLoop();QTimer.singleShot(milliseconds,loop.quit);loop.exec();app.processEvents()


def wait(app,predicate,timeout=20):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=predicate()
        if value:return value
        settle(app)
    raise AssertionError('Scene did not reach its expected state within the deadline')


@pytest.fixture(scope='module')
def qt():
    if os.name!='nt':pytest.skip('This focused real desktop acceptance is Windows-only')
    pytest.importorskip('PySide6.QtWebEngineWidgets')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    env={'QT_QPA_PLATFORM':'windows','MPLCONFIGDIR':str(OUTPUT/'mpl'),
         'ARRAYLAB_HISTORY_DIR':str(OUTPUT/'history'),'ARRAYLAB_SPARSE_HISTORY_DIR':str(OUTPUT/'sparse'),
         'TEMP':str(OUTPUT/'temp'),'TMP':str(OUTPUT/'temp'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    for key in ('MPLCONFIGDIR','ARRAYLAB_HISTORY_DIR','ARRAYLAB_SPARSE_HISTORY_DIR','TEMP'):Path(env[key]).mkdir(exist_ok=True,parents=True)
    previous={k:os.environ.get(k) for k in env};os.environ.update(env)
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts,True)
    app=QApplication.instance() or QApplication([]);assert app.platformName()=='windows'
    app.setQuitOnLastWindowClosed(False)
    try:yield app
    finally:
        app.processEvents()
        for key,value in previous.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


@pytest.fixture
def scene(qt):
    from arraylab.features.pulse import SceneWidget,scene_payload
    from PySide6.QtWidgets import QMainWindow
    host=QMainWindow();widget=SceneWidget();host.setCentralWidget(widget);host.resize(1000,720)
    host.show();host.raise_();host.activateWindow()
    payload=scene_payload({'kind':'budget','configuration':scene_project()},0.)
    widget.set_scene(payload,key='scene-check')
    wait(qt,lambda:widget.get_state(sync=True).get('ready'))
    state=widget.get_state(sync=True)
    assert state.get('supported') is not False,state
    try:yield host,widget,payload
    finally:
        widget.shutdown();host.close();host.deleteLater();settle(qt,100)


def test_original_scene_has_real_terrain_mesh_and_png(qt,scene):
    from PIL import Image
    host,widget,payload=scene
    state=widget.get_state(sync=True)
    assert state.get('terrain_triangles',0)>0 and state.get('mesh_count',0)>0,state
    raw=widget.png_bytes();assert raw.startswith(b'\x89PNG\r\n\x1a\n')
    (OUTPUT/'original-scene.png').write_bytes(raw)
    pixels=np.asarray(Image.open(BytesIO(raw)).convert('RGB'))
    assert pixels.shape[0]>=300 and pixels.shape[1]>=400
    # The original viewport is dark blue, with lit terrain/actors rather than a blank canvas.
    assert pixels[:20,:20].mean()<100 and np.std(pixels)>15
    assert len(np.unique(pixels.reshape(-1,3),axis=0))>50
    assert host.grab().save(str(OUTPUT/'scene-native-window.png'))


def test_scene_time_changes_without_rebuilding_mesh_or_resetting_camera(qt,scene):
    from arraylab.features.pulse.scene import rotation_matrix,pose_with_terrain,terrain_grid
    from scipy.spatial.transform import Rotation
    host,widget,payload=scene;before=widget.get_state(sync=True)
    widget.set_time(1.5);settle(qt,150);after=widget.get_state(sync=True)
    assert after['time_s']==pytest.approx(1.5,abs=.005)
    assert after['mesh_count']==before['mesh_count']
    assert after['terrain_triangles']==before['terrain_triangles']
    assert after['camera']==before['camera']
    project=payload['project'];obj=project['scene']['platform'];grid=terrain_grid(project)
    position,attitude=pose_with_terrain(obj,1.5,grid)
    transform=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
    platform=after['actors'][0]
    assert platform['visible'] and np.allclose(platform['position'],transform@position)
    assert np.allclose(Rotation.from_quat(platform['quaternion']).as_matrix(),transform@rotation_matrix(attitude)@transform.T,atol=1e-12)
    az,el=np.deg2rad(np.asarray(project['array']['pointing_deg'])+1.5*np.asarray(project['array']['scan_rate_deg_s']))
    direction=np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
    assert np.allclose(after['beam_direction'],transform@rotation_matrix(attitude)@direction,atol=1e-12)


def test_scene_unknown_agl_positions_are_hidden_instead_of_put_on_zero_ground(qt,scene):
    host,widget,payload=scene;changed=deepcopy(payload)
    changed['project']['scene']['platform']['velocity_mps']=[300.,0.,0.]
    widget.set_scene(changed,key='leaving-terrain');widget.set_time(5.)
    settle(qt,150);state=widget.get_state(sync=True)
    assert state['unknown_actors']==1 and state['actors'][0]['visible'] is False
    assert state['actors'][1]['visible'] is True


def test_scene_camera_mouse_rotation_zoom_and_reset(qt,scene):
    from PySide6.QtCore import Qt,QPoint,QPointF
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtTest import QTest
    host,widget,payload=scene;before=widget.get_state(sync=True)['camera']
    view=widget.view
    target=view.focusProxy() or view
    center=QPoint(target.width()//2,target.height()//2)
    QTest.mousePress(target,Qt.MouseButton.LeftButton,pos=center)
    for index in range(1,9):QTest.mouseMove(target,center+QPoint(index*10,index*3),delay=10)
    QTest.mouseRelease(target,Qt.MouseButton.LeftButton,pos=center+QPoint(80,24));settle(qt,500)
    rotated=widget.get_state(sync=True)['camera'];assert rotated!=before
    wheel=QWheelEvent(QPointF(center),QPointF(target.mapToGlobal(center)),QPoint(0,0),QPoint(0,120),
                      Qt.MouseButton.NoButton,Qt.KeyboardModifier.NoModifier,Qt.ScrollPhase.NoScrollPhase,False)
    qt.sendEvent(target,wheel);settle(qt,400)
    zoomed=widget.get_state(sync=True)['camera'];assert zoomed!=rotated
    widget.reset_camera();settle(qt,300)
    reset=widget.get_state(sync=True)['camera']
    assert np.allclose(reset['position'],before['position']) and np.allclose(reset['target'],before['target'])


@pytest.fixture
def pulse_panel(qt,monkeypatch):
    from arraylab.application.desktop.window import MainWindow
    from arraylab.application.simulation import calculate
    from arraylab.features.pulse import engine
    record,arrays=calculate('pulse','budget',scene_project(),workers=1)
    monkeypatch.setattr(engine,'simulate',lambda *a,**k:pytest.fail('Scene display calculated a new signal'))
    host=MainWindow(auto_compute=False);host.resize(1440,900);host.show();host.raise_();host.activateWindow()
    host.activate_task('pulse');panel=host.pulse_panel;panel.accept_result(record,arrays,restore_inputs=True)
    panel.view.setCurrentIndex(panel.view.findData('scene'))
    wait(qt,lambda:panel.scene_widget is not None and panel.scene_widget.get_state(sync=True).get('ready'))
    try:yield host,panel
    finally:
        host.close();host.deleteLater();settle(qt,200)


def test_scene_record_restores_camera_and_time_without_recalculation(qt,pulse_panel,tmp_path):
    host,panel=pulse_panel;widget=panel.scene_widget
    panel.scene_time.setValue(1.25);settle(qt,100)
    camera=deepcopy(widget.get_state(sync=True)['camera']);camera['position'][0]+=250.;camera['position'][1]+=100.
    from arraylab.features.pulse import scene_payload
    widget.set_scene(scene_payload(panel.report['result'],1.25),key='record-camera',camera=camera);settle(qt,100)
    path=tmp_path/'scene-record.npz';panel.save_record_path(path)
    saved=panel.record_snapshot()
    assert np.allclose(saved['view_state']['scene_camera']['position'],camera['position'])
    widget.reset_camera();panel.scene_time.setValue(0.);settle(qt,100)
    panel.load_record_path(path);settle(qt,200)
    assert panel.view.currentData()=='scene' and panel.scene_time.value()==1.25
    restored=widget.get_state(sync=True)['camera']
    assert np.allclose(restored['position'],camera['position']) and np.allclose(restored['target'],camera['target'])
    # Existing records predate scene_camera. They remain readable without rerun.
    from arraylab.shared.simulation_io import write_npz
    legacy=deepcopy(saved);legacy['view_state'].pop('scene_camera')
    old_path=tmp_path/'old-scene-record.npz';write_npz(old_path,legacy,panel.arrays)
    panel.load_record_path(old_path);settle(qt,100)
    assert panel.scene_time.value()==1.25 and panel.view.currentData()=='scene'


def test_scene_playback_updates_time_and_stops_rendering_when_hidden(qt,pulse_panel):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    host,panel=pulse_panel;widget=panel.scene_widget
    QTest.mouseClick(widget.play_button,Qt.MouseButton.LeftButton)
    wait(qt,lambda:widget.get_state(sync=True).get('playing'))
    wait(qt,lambda:panel.scene_time.value()>.1)
    panel.view.setCurrentIndex(panel.view.findData('metrics'));settle(qt,150)
    first=widget.get_state(sync=True);assert not first['visible'] and not first['playing']
    settle(qt,250);second=widget.get_state(sync=True)
    assert second['render_count']==first['render_count']
    panel.view.setCurrentIndex(panel.view.findData('scene'));settle(qt,150)
    assert widget.get_state(sync=True)['visible'] and not widget.get_state(sync=True)['playing']


def test_scene_export_keeps_the_actual_three_canvas_and_snapshot(qt,pulse_panel,tmp_path):
    from arraylab.application.simulation import export_process,load_record
    host,panel=pulse_panel;panel.scene_time.setValue(1.25);settle(qt,150)
    record=panel.record_snapshot();png=panel.scene_widget.png_bytes()
    target=tmp_path/'export'
    export_process(target,record,panel.arrays,scene_png=png)
    assert (target/'scene.png').read_bytes()==png
    note=json.loads((target/'scene-view.json').read_text(encoding='utf-8'))
    assert note=={'renderer':'PulseLab Three.js','view_state':record['view_state']}
    restored,arrays=load_record(target/'arraylab-record.json')
    assert restored['view_state']==record['view_state'] and set(arrays)==set(panel.arrays)
