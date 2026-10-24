"""Exercise integrated sparse controls, snapshots, exports and planar transfer."""
from dataclasses import asdict
import json
import sys
import time

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox, QMainWindow, QDialog, QFileDialog, QLineEdit

from .sparse2d.core import Config
from .sparse2d.headless import jsonable, source_index
from .sparse2d.runtime_paths import resource_path


def exercise_sparse(app, window, output, report, check):
    window.activate_task("home")
    QTest.mouseClick(window.sparse_button, Qt.MouseButton.LeftButton); app.processEvents()
    sparse = window.sparse_panel
    check(sparse.isVisible() and sparse.bundle is None, 'sparse_workbench_opens_without_auto_calculation')
    check(not sparse.isWindow() and sparse.window() is window and window.tabs.currentWidget() is sparse,
          'sparse_is_native_page_inside_main_workbench')
    check(window.tree.currentItem() is window.sparse_tree_item,'native_sparse_navigation_tracks_active_page')
    check(window.inspector_stack.currentWidget() is sparse.parameter_widget and not window.result_bar.isVisible(),
          'sparse_parameters_use_shared_inspector_and_hide_linear_result_selector')
    check('arraylab.sparse2d.gui' not in sys.modules and 'arraylab.sparse2d.detail_view' not in sys.modules,
          'legacy_sparse_GUI_is_not_imported')
    check([w for w in app.topLevelWidgets() if isinstance(w,QMainWindow) and w.isVisible()]==[window],
          'only_one_visible_application_main_window')
    check(resource_path('MANUAL.md').is_file() and resource_path('examples/element_grid_multifrequency.csv').is_file(),
          'sparse_help_and_element_example_resources_available')
    config = Config(nx=4,ny=4,k=8,iterations=6,opt_grid=25,eval_grid=81,hole_shape='circle',hole_x=.8,
                    min_spacing=.5,scan_angles='0,0;30,35',frequencies_ghz='9,10',phase_bits=3)
    sparse.set_config(config)
    check(sparse.bundle is None, 'sparse_setting_inputs_does_not_compute')

    def wait_finished():
        errors=[]
        started=time.monotonic();worker=sparse.thread;job_kind=getattr(sparse,'_job_kind',None);timed_out=[]
        loop=QEventLoop();poll=QTimer();deadline=QTimer();deadline.setSingleShot(True)
        def check_state():
            modal=QApplication.activeModalWidget()
            if isinstance(modal,QMessageBox):errors.append(modal.text());modal.accept()
            if sparse.thread is None:loop.quit()
        poll.timeout.connect(check_state);deadline.timeout.connect(loop.quit)
        deadline.timeout.connect(lambda:timed_out.append(True))
        poll.start(30);deadline.start(65000)
        if sparse.thread is not None:loop.exec()
        poll.stop();deadline.stop()
        if sparse.thread is not None or errors or sparse._last_error:
            diagnostic={'check_index':len(report['checks']),'job_kind':job_kind,
                        'elapsed_seconds':time.monotonic()-started,'deadline_reached':bool(timed_out),
                        'thread_is_none':sparse.thread is None,'busy':sparse._busy,
                        'worker_error':getattr(worker,'error',None),
                        'worker_cancelled':getattr(worker,'cancelled',None),
                        'last_error':sparse._last_error,'state':sparse.state.text(),
                        'modal_errors':errors,'selected_row':sparse.table.currentRow(),
                        'weight_method':sparse.weight_method.currentData()}
            try:diagnostic['worker_running']=worker.isRunning() if worker is not None else None
            except RuntimeError:diagnostic['worker_running']='Qt object already deleted'
            screenshot=output/f"sparse-worker-failure-{len(report['checks'])}.png"
            try:
                diagnostic['screenshot_saved']=window.grab().save(str(screenshot))
                diagnostic['screenshot']=screenshot.name
            except Exception as exc:diagnostic['screenshot_error']=f'{type(exc).__name__}: {exc}'
            report.setdefault('sparse_worker_failures',[]).append(diagnostic)
        check(sparse.thread is None and not errors and not sparse._last_error,
              'sparse_worker_completes_without_error_' + str(len(report['checks'])) + (' '+str(errors) if errors else ''))

    linear_sequence=window.sequence
    QTest.keyClick(window,Qt.Key.Key_F5);app.processEvents()
    check(sparse._busy and not sparse.planar_button.isEnabled(), 'sparse_workflow_locks_handoff_while_running')
    check(window.sequence==linear_sequence,'F5_in_sparse_page_does_not_compute_linear_array')
    window.close();app.processEvents()
    check(window.isVisible(), 'sparse_worker_prevents_parent_window_destruction')
    wait_finished()
    bundle=sparse.bundle
    check(len(bundle['scenarios'])==4 and bundle['candidate_counts']['forbidden']>0
          and all(r['metrics']['spacing_violations']==0 for r in bundle['results']),
          'sparse_joint_cases_hole_and_spacing_constraints')
    check(all(r['phase_quantization']['bits']==3 for r in bundle['results']), 'sparse_quantization_replay_preserved')
    index=source_index(bundle,'annealing');sparse.table.selectRow(index);app.processEvents()
    sparse.case_selector.setCurrentIndex(3);app.processEvents()
    check(sparse.case_selector.currentIndex()==3, 'sparse_selected_case_switch')
    for kind in ('layout','pattern','cuts','coarray','cases','quantization','element','comparison','structures'):
        sparse.view.setCurrentIndex(sparse.view.findData(kind));sparse.canvas.draw();app.processEvents()
        check(bool(sparse.figure.axes),'native_sparse_'+kind+'_view')
    check(sparse.port_table.rowCount()==8 and sparse.case_table.rowCount()==4,'native_sparse_amplitude_phase_and_case_tables')
    sparse.view.setCurrentIndex(sparse.view.findData('cuts'));app.processEvents();sparse.canvas.draw()
    axes=sparse.figure.axes[0];limits=axes.get_xlim();box=axes.get_window_extent()
    point=QPoint(int((box.x0+box.x1)/2),int(sparse.canvas.height()-(box.y0+box.y1)/2))
    sparse.toolbar.pan()
    QTest.mousePress(sparse.canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
    QTest.mouseMove(sparse.canvas,point+QPoint(35,0),30)
    QTest.mouseRelease(sparse.canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point+QPoint(35,0));app.processEvents()
    check(not np.allclose(axes.get_xlim(),limits),'native_sparse_pan_works_after_replacing_plot')
    sparse.toolbar.pan();sparse.toolbar.home();app.processEvents()
    check(np.allclose(axes.get_xlim(),limits),'native_sparse_toolbar_home_restores_limits')
    sparse.toolbar.pan();sparse.view.setCurrentIndex(sparse.view.findData('layout'));app.processEvents();sparse.canvas.draw()
    axes=sparse.figure.axes[0];limits=axes.get_xlim();box=axes.get_window_extent()
    point=QPoint(int((box.x0+box.x1)/2),int(sparse.canvas.height()-(box.y0+box.y1)/2))
    sparse.toolbar.pan()
    QTest.mousePress(sparse.canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point)
    QTest.mouseMove(sparse.canvas,point+QPoint(25,0),30)
    QTest.mouseRelease(sparse.canvas,Qt.MouseButton.LeftButton,Qt.KeyboardModifier.NoModifier,point+QPoint(25,0));app.processEvents()
    check(not np.allclose(axes.get_xlim(),limits),'native_sparse_changing_view_releases_previous_pan_lock')
    sparse.toolbar.pan();sparse.toolbar.home()
    sparse.inputs['k'].setValue(10)
    check(sparse.bundle is bundle and sparse.bundle['config']['k']==8, 'sparse_pending_inputs_do_not_change_results')
    sparse.weight_method.setCurrentIndex(sparse.weight_method.findData('gaussian'))
    sparse.weight_dr.setValue(10);sparse.weight_sigma.setValue(.5)
    sparse.tabs.setCurrentIndex(sparse.weight_tab);app.processEvents()
    QTest.mouseClick(sparse.weight_button,Qt.MouseButton.LeftButton);wait_finished()
    weighted=sparse.bundle;row=weighted['results'][-1]
    check(len(weighted['results'])==len(bundle['results'])+1
          and np.array_equal(row['indices'],bundle['results'][index]['indices'])
          and weighted['config']['k']==8, 'sparse_reweight_preserves_source_support_and_snapshot')
    check('当前输入尚未综合' in sparse.state.text(),'native_sparse_weighted_result_keeps_pending_input_warning')
    sparse.case_selector.setCurrentIndex(3);app.processEvents()
    saved=output/'sparse-selected-export';sparse.export_path(saved)
    exported=json.loads((saved/'summary.json').read_text(encoding='utf-8'))
    check(exported['config']==weighted['config'] and exported['results'][-1]['weights']==row['weights'].tolist()
          and (saved/'report.html').is_file(), 'sparse_integrated_complete_report_export')
    (output/'sparse-gui-config.json').write_text(json.dumps({'schema_version':1,'config':asdict(config),'weighting':[]}),encoding='utf-8')
    report['sparse_results'] = {'unweighted':jsonable(bundle),'weighted':jsonable(weighted),'selected_index':len(weighted['results'])-1,'case_index':3}
    sparse.tabs.setCurrentIndex(0);sparse.view.setCurrentIndex(sparse.view.findData('pattern'));app.processEvents();sparse.canvas.draw()
    check(window.grab().save(str(output/'sparse-workbench.png')), 'sparse_integrated_screenshot_saved')
    old_size=window.size();window.resize(1100,740);app.processEvents();QTest.qWait(100);app.processEvents();sparse.canvas.draw()
    check(window.grab().save(str(output/'sparse-workbench-small.png')),'native_sparse_small_screenshot_saved')
    check(sparse.canvas.width()>200 and sparse.canvas.height()>100 and sparse.parameter_widget.isVisible(),
          'native_sparse_small_workbench_keeps_plot_and_parameters_visible')
    window.resize(old_size);app.processEvents()
    old_planar=window.planar_panel.report
    QTest.mouseClick(sparse.planar_button,Qt.MouseButton.LeftButton);app.processEvents()
    planar=window.planar_panel;p,s=planar.read_parameters(),planar.read_settings()
    check(not sparse.isVisible() and window.tabs.currentIndex()==5 and planar.report is old_planar,
          'sparse_handoff_loads_inputs_preserving_previous_electrical_result')
    check(window.inspector_stack.currentIndex()==2 and not window.result_bar.isVisible(),
          'leaving_sparse_restores_inspector_and_planar_result_context')
    np.testing.assert_allclose(np.array(p.coordinates),row['xy']*299.792458/config.reference_ghz,atol=1e-12)
    check(p.elements==8 and p.coordinate_unit=='mm' and p.frequency_ghz==10
          and p.scan_theta_deg==30 and p.scan_phi_deg==35 and s.phase_bits==0 and s.weighting=='custom',
          'sparse_handoff_coordinates_case_and_continuous_new_network')
    planar.save_path(output/'sparse-planar-config.json')
    QTest.mouseClick(window.compute_button,Qt.MouseButton.LeftButton);app.processEvents()
    check(planar.report is not old_planar and len(planar.report['network']['ports'])==8,
          'sparse_layout_generates_planar_feed_design')
    report['sparse_planar_result']=planar.report
    window.open_sparse();app.processEvents()
    check(window.sparse_panel is sparse and sparse.bundle is weighted, 'sparse_reopen_retains_session_results')
    check(window.inspector_stack.currentWidget() is sparse.parameter_widget,'returning_to_sparse_restores_sparse_inspector')
    sparse.restore_parameters();check(sparse.config_from_inputs()==Config(**weighted['config']),'native_sparse_restore_computed_parameters')
    native_config=output.resolve()/'native-sparse-settings.json';sparse.inputs['seed'].setText(str(2**32-1))
    def accept_settings_file():
        modal=QApplication.activeModalWidget()
        if isinstance(modal,QFileDialog):modal.findChild(QLineEdit,'fileNameEdit').setText(str(native_config));modal.accept()
    window.ribbon.setCurrentIndex(0);app.processEvents()
    QTimer.singleShot(150,accept_settings_file);QTest.mouseClick(window.save_scheme_button,Qt.MouseButton.LeftButton)
    check(json.loads(native_config.read_text(encoding='utf-8'))['config']['seed']==2**32-1,
          'shared_save_scheme_uses_sparse_inputs_and_full_uint32_seed')
    sparse.inputs['k'].setValue(10)
    QTimer.singleShot(150,accept_settings_file);QTest.mouseClick(window.load_scheme_button,Qt.MouseButton.LeftButton)
    check(sparse.inputs['k'].value()==8 and sparse.bundle is weighted,'shared_open_scheme_restores_sparse_inputs_without_replacing_results')
    bad=json.loads(native_config.read_text(encoding='utf-8'));bad['weighting']=[{'method':'uniform'}]
    bad_path=output/'native-sparse-queued-weighting.json';bad_path.write_text(json.dumps(bad),encoding='utf-8')
    before=sparse.config_from_inputs()
    try:sparse.load_settings_path(bad_path)
    except ValueError:pass
    else:raise AssertionError('queued_weighting_silently_ignored')
    check(sparse.config_from_inputs()==before,'native_sparse_rejects_unsupported_queued_actions_without_mutation')
    sparse.restore_parameters();window.ribbon.setCurrentIndex(1)
    old_bundle=sparse.bundle;sparse.load_element_file(resource_path('examples/element_grid_multifrequency.csv'));sparse.preview_element()
    check(sparse.element_model.currentData()=='imported' and sparse.bundle is old_bundle and '预览' in sparse.state.text(),
          'native_sparse_element_import_preview_keeps_existing_results')
    sparse.restore_parameters();sparse.start();sparse.cancel();wait_finished()
    check(sparse.bundle is old_bundle and '取消' in sparse.state.text(),'native_sparse_cancel_preserves_previous_results')
    dialogs=[]
    def close_plot():
        modal=QApplication.activeModalWidget()
        if isinstance(modal,QDialog):dialogs.append(modal.windowTitle());modal.accept()
    QTimer.singleShot(150,close_plot);QTest.mouseClick(sparse.max_button,Qt.MouseButton.LeftButton)
    check(len(dialogs)==1,'native_sparse_explicit_plot_enlargement_dialog')
    window.tabs.setCurrentIndex(0);app.processEvents()
