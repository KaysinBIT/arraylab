"""Scoped acceptance of synchronized model versions and real legacy snapshots."""
from pathlib import Path
from copy import deepcopy
import os
import sys
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
from .pulse_scene import settle,wait


def exercise_upstream_sync(app,window,output,report,check):
    from arraylab import __version__
    from arraylab.application.simulation import load_record,export_process,record_compatibility
    fixture=os.environ.get('ARRAYLAB_UPSTREAM_FIXTURES')
    if fixture:fixture=Path(fixture)
    elif not getattr(sys,'frozen',False):fixture=Path(__file__).resolve().parents[1]/'tests/fixtures/upstream_sync'
    else:raise ValueError('此专项验收需通过ARRAYLAB_UPSTREAM_FIXTURES指定旧版合成记录目录。')
    quit_on_close=app.quitOnLastWindowClosed();app.setQuitOnLastWindowClosed(False)
    window.hide();probe=type(window)(auto_compute=False);probe.resize(1440,940)
    probe.show();probe.raise_();probe.activateWindow();settle(app,250)
    try:
        for feature,name,source_version,action in [
            ('phased','phased-analysis-v121.npz','1.2.2','analysis'),
            ('equation','equation-calibration-v100.npz','1.0.1','simulation')]:
            original,original_arrays=load_record(fixture/name);immutable=deepcopy(original['result'])
            probe.load_simulation_record_async(fixture/name)
            wait(app,lambda:getattr(probe,'_record_reader',None) is None)
            panel=probe.simulation_panels[feature]
            check(probe.active_task()==feature and panel.report['result']==immutable,feature+'_legacy_original_snapshot')
            check(panel.snapshot_notice.isVisible() and not record_compatibility(panel.report)['current'],feature+'_persistent_old_model_notice')
            if feature=='equation':
                check(not panel.calibration_button.isEnabled(),'old_ready_calibration_button_disabled')
                try:panel.backend.apply_calibration(panel.parameter_widget.value(),panel.report['result'])
                except ValueError:rejected=True
                else:rejected=False
                check(rejected,'old_calibration_service_rejected')
                target=output/'legacy-equation-export';legacy=panel.record_snapshot()
                panel._launch(lambda progress,cancel:export_process(target,legacy,original_arrays,progress=progress,cancel=cancel),'导出结果')
                wait(app,lambda:panel.thread is None,60)
                saved,saved_arrays=load_record(target/'arraylab-record.json')
                check(saved['result']==immutable and (target/'model-version-notice.json').is_file(),'legacy_export_preserves_budget_and_adds_notice')
            config=panel.backend.default_configuration()
            if feature=='phased':config['array'].update(geometry='linear',nx=6,ny=1,steerX=30,phaseBits=2);config['mapSize']=21
            panel.apply_configuration(config);panel.action.setCurrentIndex(panel.action.findData(action))
            check(panel.snapshot_notice.isVisible(),feature+'_editing_does_not_relabel_history')
            painted=[];original_draw=panel.canvas.draw
            def traced_draw(*args,_canvas=panel.canvas,_draw=original_draw,_painted=painted,**kwargs):
                figure=_canvas.figure;value=_draw(*args,**kwargs);_painted.append(figure);return value
            panel.canvas.draw=traced_draw
            QTest.keyClick(probe,Qt.Key.Key_F5);settle(app,30)
            check(panel.thread is not None,feature+'_real_worker_started')
            wait(app,lambda:panel.thread is None,60)
            check(panel.report['app_version']==__version__ and panel.report['result']['source']['version']==source_version,
                  feature+'_new_worker_source_version: '+panel.state.text())
            check(record_compatibility(panel.report)['current'] and not panel.snapshot_notice.isVisible(),feature+'_new_result_clears_old_notice')
            wait(app,lambda:any(figure is panel.figure for figure in painted) and not panel.canvas.resize_pending)
            settle(app,200);panel.canvas.draw=original_draw
            check(any(figure is panel.figure for figure in painted),feature+'_current_figure_painted')
            path=output/(feature+'-current.npz');panel.save_record_path(path)
            reopened,_=load_record(path)
            check(reopened['result']==panel.report['result'] and record_compatibility(reopened)['current'],feature+'_new_record_roundtrip')
            probe.grab().save(str(output/(feature+'-current.png')))
        report.update(app_version=__version__,legacy_fixtures_are_data_only=True,full_numerical_suite=False)
    finally:
        for panel in probe.simulation_panels.values():
            if panel.thread is not None:panel.cancel();wait(app,lambda:panel.thread is None,30)
        probe.close();probe.deleteLater();app.processEvents();app.setQuitOnLastWindowClosed(quit_on_close)
