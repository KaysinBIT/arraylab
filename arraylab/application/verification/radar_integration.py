"""Focused native/frozen acceptance for the three newly integrated engines."""
from pathlib import Path
from copy import deepcopy
import time
import sys

import numpy as np
from PySide6.QtCore import QEventLoop,QTimer,Qt
from PySide6.QtTest import QTest


def settle(app,milliseconds=30):
    loop=QEventLoop();QTimer.singleShot(milliseconds,loop.quit);loop.exec();app.processEvents()


def wait(app,predicate,timeout=90):
    deadline=time.monotonic()+timeout
    while not predicate() and time.monotonic()<deadline:settle(app)
    if not predicate():raise AssertionError('Radar operation did not finish in time')


def exercise_radar_integration(app,window,output,report,check):
    # Windows honors SW_HIDE on the first top-level window launched by a helper.
    # A fresh probe receives real shortcut/paint events, as in resize acceptance.
    quit_on_close=app.quitOnLastWindowClosed();app.setQuitOnLastWindowClosed(False)
    window.hide();probe=type(window)(auto_compute=False)
    try:
        probe.resize(1440,900);probe.show();probe.raise_();probe.activateWindow();settle(app,300)
        _exercise(app,probe,output,report,check)
    finally:
        probe.close();probe.deleteLater();app.processEvents();app.setQuitOnLastWindowClosed(quit_on_close)


def _exercise(app,window,output,report,check):
    from arraylab.application.simulation import export_process,load_record
    from arraylab.features.pulse import import_dem,export_project,import_project
    from arraylab import __version__
    from arraylab.framework.parallel import set_workers,requested_workers
    report['app_version']=__version__;previous=requested_workers();set_workers(1)
    try:
        check(window.tabs.count()==14,'three_native_workbenches_registered')
        check(not window.equation_panel.array_import_button.isEnabled(),'no_uncomputed_array_import')
        for feature,action in [('phased','analysis'),('equation','simulation'),('pulse','simulate')]:
            panel=window.simulation_panels[feature];window.activate_task(feature)
            config=panel.backend.default_configuration()
            if feature=='phased':config['array'].update(geometry='linear',nx=8,ny=1);config['mapSize']=21
            panel.apply_configuration(config);panel.action.setCurrentIndex(panel.action.findData(action))
            settle(app,100);QTest.keyClick(window,Qt.Key.Key_F5);settle(app)
            check(panel.thread is not None,feature+'_F5_started_owned_worker')
            wait(app,lambda:panel.thread is None)
            check(panel.report is not None and panel.report['action']==action,feature+'_worker_returned_complete_result: '+panel.state.text())
            check(bool(panel.arrays) and all(not x.dtype.hasobject for x in panel.arrays.values()),feature+'_numeric_arrays')
            if feature=='pulse':panel.view.setCurrentIndex(panel.view.findData('scene'));panel.scene_time.setValue(1.5)
            settle(app,400)
            window.grab().save(str(output/(feature+'-native.png')))
            path=output/(feature+'-record.npz');panel.save_record_path(path)
            expected=panel.record_snapshot();window.activate_task('home');window.load_simulation_record_async(path)
            wait(app,lambda:getattr(window,'_record_reader',None) is None)
            check(window.active_task()==feature and panel.record_snapshot()['view_state']==expected['view_state'],feature+'_global_record_restore')
            target=output/(feature+'-export');record,arrays=panel.record_snapshot(),panel.arrays
            panel._launch(lambda progress,cancel,p=target,r=record,a=arrays:export_process(p,r,a,progress=progress,cancel=cancel),'导出结果')
            wait(app,lambda:panel.thread is None)
            check((target/'arraylab-record.json').is_file(),feature+'_owned_export: '+panel.state.text())
            restored,saved=load_record(target/'arraylab-record.json')
            check(restored==record and set(saved)==set(arrays),feature+'_export_record_identity')
            check(all(np.array_equal(saved[key],value,equal_nan=True) for key,value in arrays.items()),feature+'_all_arrays_roundtrip')
            check(bool(list(target.glob('*.png'))),feature+'_png_export')
        # Real GDAL/PROJ resource resolution in the portable runtime.
        examples=Path(sys.executable).parent/'examples' if getattr(sys,'frozen',False) else Path(__file__).resolve().parents[3]/'examples'
        terrain=import_dem(examples/'pulse-demo-hill-utm.tif',{'max_size':17,'height_unit':'m','vertical_datum':'synthetic'})
        check(terrain['metadata']['source_epsg']==32650,'GeoTIFF_projection_and_resources')
        config=window.pulse_panel.configuration();config['scene']['terrain']=terrain
        project=output/'terrain.prlab';export_project(project,config);restored=import_project(project)
        check(all(restored['scene']['terrain'][key]==value for key,value in terrain.items()),'portable_project_retains_DEM')
        # Start an actual long optimizer; cancellation keeps the completed result.
        panel=window.phased_panel;window.activate_task('phased');old=panel.report
        config=deepcopy(panel.backend.default_configuration());config['planar'].update(mode='continuous',iterations=5000)
        panel.apply_configuration(config);panel.action.setCurrentIndex(panel.action.findData('optimize2d'))
        QTest.keyClick(window,Qt.Key.Key_F5);settle(app,300)
        check(panel.thread is not None,'long_optimization_started')
        QTest.mouseClick(window.cancel_button,Qt.MouseButton.LeftButton);wait(app,lambda:panel.thread is None,20)
        check(panel.report is old and '取消' in panel.state.text(),'cancel_retains_previous_result')
        window.activate_task('equation');settle(app,200)
        check(window.compute_button.isEnabled(),'controls_restored_after_worker_cleanup')
    finally:
        for panel in window.simulation_panels.values():
            if panel.thread is not None:
                panel.cancel();wait(app,lambda:panel.thread is None,30)
        set_workers(previous)
