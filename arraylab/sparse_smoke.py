"""Exercise integrated sparse controls, snapshots, exports and planar transfer."""
from dataclasses import asdict
import json

import numpy as np
from PySide6.QtCore import QEventLoop, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox

from .sparse2d.core import Config
from .sparse2d.headless import jsonable, source_index
from .sparse2d.runtime_paths import resource_path


def exercise_sparse(app, window, output, report, check):
    window.ribbon.setCurrentIndex(1)
    QTest.mouseClick(window.sparse_button, Qt.MouseButton.LeftButton); app.processEvents()
    sparse = window.sparse_window
    check(sparse.isVisible() and sparse.bundle is None, 'sparse_workbench_opens_without_auto_calculation')
    check(resource_path('MANUAL.md').is_file() and resource_path('examples/element_grid_multifrequency.csv').is_file(),
          'sparse_help_and_element_example_resources_available')
    config = Config(nx=4,ny=4,k=8,iterations=6,opt_grid=25,eval_grid=81,hole_shape='circle',hole_x=.8,
                    min_spacing=.5,scan_angles='0,0;30,35',frequencies_ghz='9,10',phase_bits=3)
    sparse.set_config(config)
    check(sparse.bundle is None, 'sparse_setting_inputs_does_not_compute')

    def wait_finished():
        errors=[]
        loop=QEventLoop();poll=QTimer();deadline=QTimer();deadline.setSingleShot(True)
        def check_state():
            modal=QApplication.activeModalWidget()
            if isinstance(modal,QMessageBox):errors.append(modal.text());modal.accept()
            if sparse.thread is None:loop.quit()
        poll.timeout.connect(check_state);deadline.timeout.connect(loop.quit)
        poll.start(30);deadline.start(65000)
        if sparse.thread is not None:loop.exec()
        poll.stop();deadline.stop()
        check(sparse.thread is None and not errors and not sparse._last_error,
              'sparse_worker_completes_without_error_' + str(len(report['checks'])) + (' '+str(errors) if errors else ''))

    QTest.mouseClick(sparse.run_button,Qt.MouseButton.LeftButton)
    check(sparse._busy and not sparse.planar_button.isEnabled(), 'sparse_workflow_locks_handoff_while_running')
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
    sparse.case_selector.setCurrentIndex(3);app.processEvents()
    saved=output/'sparse-selected-export';sparse.export_path(saved)
    exported=json.loads((saved/'summary.json').read_text(encoding='utf-8'))
    check(exported['config']==weighted['config'] and exported['results'][-1]['weights']==row['weights'].tolist()
          and (saved/'report.html').is_file(), 'sparse_integrated_complete_report_export')
    (output/'sparse-gui-config.json').write_text(json.dumps({'schema_version':1,'config':asdict(config),'weighting':[]}),encoding='utf-8')
    report['sparse_results'] = {'unweighted':jsonable(bundle),'weighted':jsonable(weighted),'selected_index':len(weighted['results'])-1,'case_index':3}
    sparse.tabs.setCurrentIndex(0);app.processEvents()
    check(sparse.grab().save(str(output/'sparse-workbench.png')), 'sparse_integrated_screenshot_saved')
    old_planar=window.planar_panel.report
    QTest.mouseClick(sparse.planar_button,Qt.MouseButton.LeftButton);app.processEvents()
    planar=window.planar_panel;p,s=planar.read_parameters(),planar.read_settings()
    check(not sparse.isVisible() and window.tabs.currentIndex()==5 and planar.report is old_planar,
          'sparse_handoff_loads_inputs_preserving_previous_electrical_result')
    np.testing.assert_allclose(np.array(p.coordinates),row['xy']*299.792458/config.reference_ghz,atol=1e-12)
    check(p.elements==8 and p.coordinate_unit=='mm' and p.frequency_ghz==10
          and p.scan_theta_deg==30 and p.scan_phi_deg==35 and s.phase_bits==0 and s.weighting=='custom',
          'sparse_handoff_coordinates_case_and_continuous_new_network')
    planar.save_path(output/'sparse-planar-config.json')
    QTest.mouseClick(planar.design_button,Qt.MouseButton.LeftButton);app.processEvents()
    check(planar.report is not old_planar and len(planar.report['network']['ports'])==8,
          'sparse_layout_generates_planar_feed_design')
    report['sparse_planar_result']=planar.report
    window.open_sparse();app.processEvents()
    check(window.sparse_window is sparse and sparse.bundle is weighted, 'sparse_reopen_retains_session_results')
    sparse.close();app.processEvents()
