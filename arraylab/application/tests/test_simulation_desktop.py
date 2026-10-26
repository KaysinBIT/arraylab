"""Native integration checks for the three shared simulation panels only."""
from copy import deepcopy
from pathlib import Path
import os,time,json
from unittest.mock import patch

import numpy as np
import pytest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
OUTPUT=ROOT/'tmp/merge-radar/ui-checks'


@pytest.fixture(scope='module')
def ui():
    pytest.importorskip('PySide6')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    settings={'QT_QPA_PLATFORM':'offscreen','MPLCONFIGDIR':str(OUTPUT/'mpl'),
              'ARRAYLAB_HISTORY_DIR':str(OUTPUT/'linear'),'ARRAYLAB_SPARSE_HISTORY_DIR':str(OUTPUT/'sparse'),
              'TEMP':str(OUTPUT/'temp'),'TMP':str(OUTPUT/'temp'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    for key in ('MPLCONFIGDIR','ARRAYLAB_HISTORY_DIR','ARRAYLAB_SPARSE_HISTORY_DIR','TEMP'):Path(settings[key]).mkdir(parents=True,exist_ok=True)
    old={k:os.environ.get(k) for k in settings};os.environ.update(settings)
    from PySide6.QtWidgets import QApplication,QMainWindow,QSplitter
    from PySide6.QtGui import QFontDatabase,QFont
    from PySide6.QtCore import Qt
    import matplotlib
    from arraylab.application.desktop.simulation_panel import SimulationPanel
    from arraylab.application.simulation import calculate
    from arraylab.framework.parallel import requested_workers,set_workers
    app=QApplication.instance() or QApplication([]);assert app.platformName()=='offscreen'
    if not QFontDatabase.families():
        font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
        if font.is_file():QFontDatabase.addApplicationFont(str(font));app.setFont(QFont('Microsoft YaHei',9))
    app.setQuitOnLastWindowClosed(False);previous=requested_workers();set_workers(1)
    panels={};reports={};windows=[]
    context=matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans']});context.__enter__()
    for feature in ('phased','equation','pulse'):
        host=QMainWindow();host.setWindowTitle('ArrayLab 原生组件验收 · '+feature);host.resize(1280,820)
        host.last_directory=str(OUTPUT);host._workflow_ready=True;host.simulation_panels={}
        host.refresh_count=0
        def refresh(h=host):h.refresh_count+=1
        host.refresh_actions=refresh;host.simulation_busy=lambda *_:False
        panel=SimulationPanel(host,feature);host.simulation_panels[feature]=panel
        split=QSplitter(Qt.Orientation.Horizontal);split.addWidget(panel);split.addWidget(panel.parameter_widget);split.setSizes([900,360]);host.setCentralWidget(split)
        panels[feature]=(host,panel);windows.append(host)
        config=panel.backend.default_configuration()
        if feature=='phased':config['array'].update(geometry='linear',nx=8,ny=1);config['mapSize']=21
        reports[feature]=calculate(feature,'analysis' if feature=='phased' else 'budget',config,workers=1)
    try:yield app,panels,reports
    finally:
        for host,panel in panels.values():
            if panel.thread is not None:panel.cancel();wait_idle(app,panel)
            host.close();host.deleteLater()
        app.processEvents();set_workers(previous);context.__exit__(None,None,None)
        for key,value in old.items():
            if value is None:os.environ.pop(key,None)
            else:os.environ[key]=value


def settle(app,seconds=.15):
    from PySide6.QtCore import QEventLoop,QTimer
    loop=QEventLoop();QTimer.singleShot(int(seconds*1000),loop.quit);loop.exec();app.processEvents()


def wait_idle(app,panel,timeout=15):
    deadline=time.monotonic()+timeout
    while panel.thread is not None and time.monotonic()<deadline:settle(app,.02)
    assert panel.thread is None and not panel._busy,panel.state.text()


def reset(ui,feature):
    app,panels,reports=ui;host,panel=panels[feature]
    for other,_ in panels.values():other.hide()
    panel.apply_configuration(panel.backend.default_configuration());host.show();settle(app)
    return app,host,panel,reports[feature]


@pytest.mark.parametrize('feature,path,expected',[('phased','array.phaseMode','固定'),('equation','waveform.kind','FMCW'),('pulse','waveform.kind','LFM')])
def test_metadata_uses_native_chinese_enum_controls(ui,feature,path,expected):
    from PySide6.QtWidgets import QComboBox
    app,host,panel,_=reset(ui,feature)
    editor=panel.parameter_widget.editors[path]
    assert isinstance(editor.control,QComboBox)
    assert expected.lower() in ' '.join(editor.control.itemText(i) for i in range(editor.control.count())).lower()
    assert editor.control.accessibleName()!=path


@pytest.mark.parametrize('feature,frequency,bandwidth,time_path',[
    ('equation','waveform.fc_hz','waveform.bandwidth_hz','waveform.ramp_s'),
    ('pulse','waveform.fc_hz','waveform.sample_rate_hz','waveform.pulse_width_s')])
def test_si_units_display_in_engineering_units_and_roundtrip_exactly(ui,feature,frequency,bandwidth,time_path):
    app,host,panel,_=reset(ui,feature);original=panel.configuration()
    editors=panel.parameter_widget.editors
    assert float(editors[frequency].control.text())==original['waveform']['fc_hz']/1e9
    assert '(GHz)' in editors[frequency].control.accessibleName()
    assert '(MHz)' in editors[bandwidth].control.accessibleName()
    assert '(μs)' in editors[time_path].control.accessibleName()
    assert panel.parameter_widget.value()==original
    editors[frequency].control.setText('12.3456789012345')
    assert panel.parameter_widget.value()['waveform']['fc_hz']==pytest.approx(12345678901.2345,rel=1e-15)
    editors[time_path].control.setText('2.125')
    assert panel.parameter_widget.value()['waveform'][time_path.split('.')[-1]]==pytest.approx(2.125e-6,rel=1e-15)


@pytest.mark.parametrize('feature,name_key',[('equation','name'),('pulse','id')])
def test_target_lists_are_edited_in_native_dialogs(ui,feature,name_key):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication,QDialogButtonBox
    from arraylab.shared.qt.configuration_form import ConfigurationForm,ObjectListEditor
    app,host,panel,_=reset(ui,feature)
    editor=panel.parameter_widget.editors['scene.targets'];assert isinstance(editor,ObjectListEditor)
    before=deepcopy(editor.value());seen=[]
    def edit_dialog():
        dialog=QApplication.activeModalWidget();assert dialog is not None
        form=dialog.findChild(ConfigurationForm);control=form.editors['scene.targets[].'+name_key].control
        control.setText('原生目标编辑');seen.append(dialog.windowTitle())
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()
    QTimer.singleShot(20,edit_dialog);editor.list.setCurrentRow(0);editor.edit()
    assert seen and editor.value()[0][name_key]=='原生目标编辑'
    for key in before[0]:
        if key!=name_key:assert editor.value()[0][key]==before[0][key]


@pytest.mark.parametrize('feature',['phased','equation','pulse'])
def test_action_and_view_switches_do_not_solve_or_edit_inputs(ui,feature):
    app,host,panel,_=reset(ui,feature);before=panel.configuration();previous=panel.report
    with patch.object(panel.backend,'execute',side_effect=AssertionError('View started a solver')) as solver,patch('arraylab.application.desktop.simulation_panel.run_process',side_effect=AssertionError('View started a process')) as process:
        for i in range(panel.action.count()):panel.action.setCurrentIndex(i)
        for i in range(panel.view.count()):panel.view.setCurrentIndex(i);settle(app,.01)
        assert solver.call_count==process.call_count==0
    assert panel.configuration()==before and panel.report is previous


@pytest.mark.parametrize('feature',['phased','equation','pulse'])
def test_completed_result_and_npz_restore_without_recalculation(ui,feature):
    app,host,panel,(report,arrays)=reset(ui,feature)
    panel.accept_result(report,arrays,restore_inputs=True);path=OUTPUT/(feature+'-record.npz');panel.save_record_path(path)
    assert panel.report is report and panel.configuration()==report['configuration']
    with patch.object(panel.backend,'execute',side_effect=AssertionError('Restore calculated')) as solver:
        panel.load_record_path(path)
    assert solver.call_count==0 and panel.configuration()==report['configuration']
    for key in arrays:np.testing.assert_equal(panel.arrays[key],arrays[key])
    assert '未重新' in panel.state.text()


@pytest.mark.parametrize('feature,path',[('phased','array.frequencyGHz'),('equation','waveform.fc_hz'),('pulse','waveform.fc_hz')])
def test_invalid_pending_edit_does_not_block_valid_record_restore(ui,feature,path):
    from arraylab.shared.simulation_io import write_npz
    app,host,panel,(report,arrays)=reset(ui,feature);target=OUTPUT/(feature+'-valid-with-invalid-editor.npz')
    write_npz(target,report,arrays)
    panel.parameter_widget.editors[path].control.setText('无效输入')
    panel.load_record_path(target)
    assert panel.configuration()==report['configuration']
    assert panel.parameter_widget.editors[path].control.text()!='无效输入'


def test_shared_qthread_path_releases_busy_and_accepts_result(ui):
    app,host,panel,(report,arrays)=reset(ui,'phased');panel.apply_configuration(report['configuration'])
    with patch('arraylab.application.desktop.simulation_panel.run_process',return_value=(report,arrays)) as process:
        panel.start('analysis');assert panel._busy and not panel.parameter_widget.isEnabled();wait_idle(app,panel)
    assert process.call_count==1 and panel.report is report and panel.parameter_widget.isEnabled()


def test_failed_record_render_restores_raw_edit_tab_and_selected_case(ui):
    from arraylab.application.simulation import calculate
    app,host,panel,(new_report,new_arrays)=reset(ui,'phased')
    config=panel.backend.default_configuration();config['array'].update(geometry='linear',nx=8,ny=1);config['mapSize']=21
    config['planar'].update(count=8,gridNx=5,gridNy=5,iterations=0,scans=[{'x':0,'y':0}],frequencies=[9,11],targetSll=-5,
        maxHpbwX=60,maxHpbwY=60,mainlobeRadiusU=.5,mainlobeRadiusV=.5)
    old_report,old_arrays=calculate('phased','optimize2d',config,workers=1)
    panel.accept_result(old_report,old_arrays,restore_inputs=True);panel.selection.setCurrentIndex(1)
    panel.parameter_widget.setCurrentIndex(2);panel.parameter_widget.editors['array.frequencyGHz'].control.setText('等待修正')
    raw=panel.parameter_widget.snapshot();chosen=panel.selection.currentData();choices=[panel.selection.itemText(i) for i in range(panel.selection.count())]
    original=panel.backend.draw;calls=[]
    def fail_once(*args,**kwargs):
        calls.append(1)
        if len(calls)==1:raise RuntimeError('synthetic render failure')
        return original(*args,**kwargs)
    with patch.object(panel.backend,'draw',side_effect=fail_once),pytest.raises(RuntimeError,match='synthetic render failure'):
        panel.accept_result(new_report,new_arrays,restore_inputs=True)
    assert panel.report is old_report and panel.arrays is old_arrays
    assert panel.parameter_widget.snapshot()==raw
    assert [panel.selection.itemText(i) for i in range(panel.selection.count())]==choices
    assert panel.selection.currentData()==chosen


@pytest.mark.parametrize('feature',['phased','equation','pulse'])
def test_native_json_project_save_and_reopen_without_compute(ui,feature):
    app,host,panel,_=reset(ui,feature);config=panel.configuration();path=OUTPUT/(feature+'-project.json')
    panel.save_path(path)
    name='array.name' if feature=='phased' else 'name'
    panel.parameter_widget.editors[name].control.setText('未计算的临时输入')
    with patch.object(panel.backend,'execute',side_effect=AssertionError('Project load computed')) as solver:
        panel.open_path(path)
    assert panel.configuration()==config and solver.call_count==0


@pytest.mark.parametrize('schema',[True,2,1.5])
def test_unknown_shared_project_schema_is_rejected_without_editing(ui,schema):
    from arraylab.shared.simulation_io import write_json
    app,host,panel,_=reset(ui,'phased');before=panel.parameter_widget.snapshot();path=OUTPUT/('bad-schema-'+str(schema)+'.json')
    write_json(path,{'format':'arraylab.simulation','schema_version':schema,'feature':'phased','configuration':panel.configuration()})
    with pytest.raises(ValueError):panel.open_path(path)
    assert panel.parameter_widget.snapshot()==before


def test_context_actions_stay_disabled_for_unrelated_results_and_busy_import(ui):
    from PySide6.QtWidgets import QFileDialog
    for feature in ('phased','equation','pulse'):
        app,host,panel,(report,arrays)=reset(ui,feature);panel.accept_result(report,arrays,restore_inputs=True)
        if feature in ('phased','pulse'):assert not panel.apply_button.isEnabled()
        if feature=='equation':assert not panel.calibration_button.isEnabled()
        panel._busy=True
        try:
            panel.update_context_actions()
            for i in range(panel.integration_bar.count()):
                widget=panel.integration_bar.itemAt(i).widget()
                if widget is not None:assert not widget.isEnabled()
            if feature=='pulse':
                with patch.object(QFileDialog,'getOpenFileName',side_effect=AssertionError('Busy import opened dialog')):
                    panel.import_terrain()
        finally:panel._busy=False;panel.update_context_actions()


@pytest.fixture(scope='module')
def full_window(ui):
    from arraylab.application.desktop.window import MainWindow
    app,panels,reports=ui
    for host,_ in panels.values():host.hide()
    window=MainWindow(auto_compute=False);window.resize(1536,900);window.show();window.raise_();window.activateWindow();settle(app,.2)
    try:yield app,window
    finally:
        for panel in window.simulation_panels.values():
            if panel.thread is not None:panel.cancel();wait_idle(app,panel,30)
        window.close();window.deleteLater();app.processEvents()


@pytest.mark.parametrize('feature',['phased','equation','pulse'])
def test_main_window_f5_runs_real_owned_worker_and_routes_record(full_window,feature):
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    app,window=full_window;window.activate_task(feature);panel=window.simulation_panels[feature]
    config=panel.backend.default_configuration()
    if feature=='phased':config['array'].update(geometry='linear',nx=8,ny=1);config['mapSize']=21
    panel.apply_configuration(config);panel.action.setCurrentIndex(0);window.activateWindow();window.raise_();settle(app,.1)
    assert window.inspector_stack.currentWidget() is panel.parameter_widget
    assert panel.action.currentText() in window.compute_button.text()
    QTest.keyClick(window,Qt.Key.Key_F5);assert panel.thread is not None
    wait_idle(app,panel,40)
    assert panel.report is not None and panel.report['feature']==feature,panel.state.text()
    assert '完成' in panel.state.text()
    path=OUTPUT/(feature+'-full-window-record.npz');window.save_history_path(path)
    window.activate_task('home')
    with patch.object(panel.backend,'execute',side_effect=AssertionError('Global history read computed')):
        window.load_history_path(path)
    assert window.active_task()==feature and panel.configuration()==panel.report['configuration']
    assert window.tree.currentItem() is not None
    settle(app,.5)
    if panel.canvas.isVisible():panel.canvas.draw()
    assert window.grab().save(str(OUTPUT/(feature+'-full-window.png')))


def test_main_window_cancel_reaps_real_worker_and_keeps_previous_result(full_window):
    import psutil
    from PySide6.QtTest import QTest
    from PySide6.QtCore import Qt
    app,window=full_window;window.activate_task('phased');panel=window.phased_panel
    assert panel.report is not None
    previous=panel.report;config=panel.backend.default_configuration();config['mapSize']=21;config['planar']['iterations']=5000
    panel.apply_configuration(config);panel.action.setCurrentIndex(panel.action.findData('optimize2d'))
    children={p.pid for p in psutil.Process().children(recursive=True)}
    window.activateWindow();QTest.keyClick(window,Qt.Key.Key_F5)
    deadline=time.monotonic()+20
    while panel.thread is not None and time.monotonic()<deadline:
        settle(app,.03)
        if 'search' in panel.state.text() or '搜索' in panel.state.text():break
    active=[p for p in psutil.Process().children(recursive=True) if p.pid not in children]
    assert active and panel.thread is not None,panel.state.text()
    identities=[(p.pid,p.create_time()) for p in active]
    window.cancel_button.click();wait_idle(app,panel,20)
    assert panel.report is previous and '取消' in panel.state.text()
    for pid,created in identities:
        try:process=psutil.Process(pid)
        except psutil.NoSuchProcess:continue
        assert process.create_time()!=created or not process.is_running(),pid


@pytest.mark.parametrize('feature',['phased','equation'])
def test_restored_plot_eventually_paints_without_recomputing(ui,full_window,feature):
    from arraylab.shared.simulation_io import write_npz
    app,window=full_window;panel=window.simulation_panels[feature]
    path=OUTPUT/(feature+'-visual-restore-record.npz')
    write_npz(path,*ui[2][feature])
    with patch.object(panel.backend,'execute',side_effect=AssertionError('Visual restore computed')) as solve:
        window.load_history_path(path);settle(app,.6)
        assert not panel.canvas.resize_pending
        sample=panel.canvas.grab().toImage().pixelColor(5,5)
        assert sample.red()>180 and sample.green()>180 and sample.blue()>180
        assert solve.call_count==0
    assert window.grab().save(str(OUTPUT/(feature+'-full-window-settled.png')))


@pytest.mark.parametrize('feature',['phased','equation','pulse'])
def test_small_window_form_labels_do_not_overlap_and_all_pages_scroll(ui,feature):
    from PySide6.QtWidgets import QFormLayout,QScrollArea
    from PySide6.QtCore import QPoint,QRect
    app,host,panel,_=reset(ui,feature);host.resize(1100,640);host.centralWidget().setSizes([740,360]);settle(app,.2)
    audited=0
    for index in range(panel.parameter_widget.count()):
        panel.parameter_widget.setCurrentIndex(index);settle(app,.01)
        page=panel.parameter_widget.widget(index);assert isinstance(page,QScrollArea)
        assert page.horizontalScrollBar().maximum()==0,(feature,index,page.horizontalScrollBar().maximum())
        for form in page.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                label=form.itemAt(row,QFormLayout.ItemRole.LabelRole);field=form.itemAt(row,QFormLayout.ItemRole.FieldRole)
                if label is None or field is None or label.widget() is None or field.widget() is None:continue
                a,b=label.widget(),field.widget();ra=QRect(a.mapTo(host,QPoint(0,0)),a.size());rb=QRect(b.mapTo(host,QPoint(0,0)),b.size())
                assert not ra.intersects(rb),(feature,index,a.text());audited+=1
    assert audited>10
    panel.parameter_widget.setCurrentIndex(0);settle(app,.15)
    assert host.grab().save(str(OUTPUT/(feature+'-1100x640.png')))
    host.resize(1536,900);host.centralWidget().setSizes([1150,380]);settle(app,.2)
    assert host.grab().save(str(OUTPUT/(feature+'-1536x900.png')))


@pytest.mark.parametrize('feature',['phased','equation','pulse'])
def test_saved_project_restores_selected_operation_without_running(ui,feature):
    app,host,panel,_=reset(ui,feature)
    panel.action.setCurrentIndex(panel.action.count()-1);operation=panel.action.currentData()
    path=OUTPUT/(feature+'-operation-project.json');panel.save_path(path)
    assert json.loads(path.read_text(encoding='utf-8'))['operation']==operation
    panel.action.setCurrentIndex(0)
    with patch.object(panel.backend,'execute',side_effect=AssertionError('Operation restore computed')):
        panel.open_path(path)
    assert panel.action.currentData()==operation


def test_source_project_starts_with_analysis_instead_of_previous_operation(ui):
    app,host,panel,_=reset(ui,'phased')
    panel.action.setCurrentIndex(panel.action.findData('optimize2d'))
    with patch.object(panel.backend,'execute',side_effect=AssertionError('Source import computed')):
        panel.open_path(ROOT/'examples/phased-uniform-linear.json')
    assert panel.action.currentData()=='analysis'


@pytest.mark.parametrize('feature',['phased','equation','pulse'])
def test_saved_record_restores_operation_view_case_table_and_scene_time(ui,feature):
    from arraylab.application.simulation import calculate
    from arraylab.shared.simulation_io import read_npz
    app,host,panel,(report,arrays)=reset(ui,feature)
    if feature=='phased':
        config=deepcopy(report['configuration'])
        config['planar'].update(count=8,gridNx=5,gridNy=5,iterations=0,scans=[{'x':0,'y':0}],frequencies=[9,11],targetSll=-5,
            maxHpbwX=60,maxHpbwY=60,mainlobeRadiusU=.5,mainlobeRadiusV=.5)
        report,arrays=calculate(feature,'optimize2d',config,workers=1)
    panel.accept_result(report,arrays,restore_inputs=True)
    wanted={'phased':'cuts','equation':'budget','pulse':'scene'}[feature]
    panel.view.setCurrentIndex(panel.view.findData(wanted))
    if feature=='phased':panel.selection.setCurrentIndex(1)
    panel.table_toggle.setChecked(feature!='pulse');panel.scene_time.setValue(3.125)
    expected=panel.record_snapshot()['view_state'];original=deepcopy(report)
    path=OUTPUT/(feature+'-view-state.npz');panel.save_record_path(path)
    stored,_=read_npz(path)
    assert stored['view_state']==expected and report==original and 'view_state' not in report
    panel.action.setCurrentIndex((panel.action.currentIndex()+1)%panel.action.count())
    panel.view.setCurrentIndex((panel.view.currentIndex()+1)%panel.view.count())
    panel.selection.setCurrentIndex(0);panel.table_toggle.setChecked(not expected['table']);panel.scene_time.setValue(.5)
    with patch.object(panel.backend,'execute',side_effect=AssertionError('Display restore computed')):
        panel.load_record_path(path)
    assert panel.action.currentData()==report['action']
    assert panel.record_snapshot()['view_state']==expected


def test_cancel_after_function_returns_does_not_replace_previous_result(ui):
    from threading import Event
    app,host,panel,(report,arrays)=reset(ui,'phased');panel.accept_result(report,arrays,restore_inputs=True)
    previous=panel.report;started=Event();release=Event();returned=deepcopy(report)
    def finish_after_stop(progress,cancel):
        started.set();assert release.wait(5);return returned,arrays
    panel._launch(finish_after_stop,'计算')
    try:
        assert started.wait(5)
        panel.cancel();release.set();wait_idle(app,panel)
        assert panel.report is previous and '取消' in panel.state.text()
        assert panel.parameter_widget.isEnabled() and panel.action.isEnabled()
    finally:
        release.set()
        if panel.thread is not None:wait_idle(app,panel)


@pytest.mark.parametrize('feature',['phased','equation','pulse'])
def test_mode_specific_controls_disable_without_discarding_values_or_solving(ui,feature):
    app,host,panel,_=reset(ui,feature);editors=panel.parameter_widget.editors
    def choice(path,value):
        control=editors[path].control;index=control.findData(value);assert index>=0,(path,value)
        control.setCurrentIndex(index);panel.update_input_availability()
    def enabled(*paths):
        for path in paths:assert editors[path].isEnabled(),path
    def disabled(*paths):
        for path in paths:assert not editors[path].isEnabled(),path
    def unchanged(values):
        for path,value in values.items():assert editors[path].value()==value,path
    with patch.object(panel.backend,'execute',side_effect=AssertionError('Mode switch computed')):
        if feature=='phased':
            held={p:deepcopy(editors[p].value()) for p in ('array.custom','array.nx','array.ny','planar.initialPositions','planar.width')}
            choice('array.geometry','custom');enabled('array.custom');disabled('array.nx','array.ny','array.dx','array.dy','array.window')
            choice('array.geometry','rectangular');enabled('array.nx','array.ny','array.dx','array.dy');disabled('array.custom')
            choice('array.geometry','linear');disabled('array.ny','array.dy')
            choice('array.window','uniform');disabled('array.sidelobeDb')
            choice('array.window','taylor');enabled('array.sidelobeDb')
            choice('planar.mode','continuous');enabled('planar.width','planar.height','planar.amplitudeMin','planar.initialization');disabled('planar.gridNx','planar.gridNy','planar.dx','planar.dy')
            choice('planar.initialization','provided');enabled('planar.initialPositions')
            choice('planar.initialization','generated');disabled('planar.initialPositions')
            choice('planar.mode','grid');enabled('planar.gridNx','planar.gridNy');disabled('planar.width','planar.height','planar.amplitudeMin','planar.initialPositions')
        elif feature=='equation':
            held={p:deepcopy(editors[p].value()) for p in ('waveform.ramp_s','waveform.pulse_width_s','radar.system_temperature_k','propagation.custom_paths','clutter.cnr_db')}
            choice('waveform.kind','fmcw');enabled('waveform.ramp_s','waveform.adc_start_s');disabled('waveform.pulse_width_s')
            choice('waveform.kind','lfm');enabled('waveform.pulse_width_s');disabled('waveform.ramp_s','waveform.adc_start_s')
            choice('array.mode','ddm');enabled('array.ddm_slots')
            choice('array.mode','simo');disabled('array.ddm_slots')
            choice('radar.noise_mode','tsys');enabled('radar.system_temperature_k');disabled('radar.noise_figure_db','radar.antenna_temperature_k')
            choice('radar.noise_mode','nf');disabled('radar.system_temperature_k');enabled('radar.noise_figure_db','radar.antenna_temperature_k')
            choice('propagation.model','two_ray');enabled('propagation.epsilon_r','propagation.conductivity_s_m');disabled('propagation.custom_paths')
            choice('propagation.model','custom');enabled('propagation.custom_paths');disabled('propagation.epsilon_r','propagation.conductivity_s_m')
            for state in (False,True):
                editors['propagation.atmosphere.enabled'].control.setChecked(state);panel.update_input_availability()
                for path in editors:
                    if path.startswith('propagation.atmosphere.') and not path.endswith('.enabled'):assert editors[path].isEnabled()==state,path
            choice('clutter.kind','none');disabled('clutter.cnr_db')
            choice('clutter.kind','gaussian');enabled('clutter.cnr_db')
        else:
            held={p:deepcopy(editors[p].value()) for p in ('waveform.phase_code','waveform.bandwidth_hz','timing.pri_sequence_s','array.tx_positions_m','array.rx_positions_m')}
            choice('waveform.kind','stepped');enabled('waveform.frequency_steps','waveform.frequency_step_hz','timing.groups');disabled('timing.pulses','waveform.phase_code','waveform.bandwidth_hz')
            choice('waveform.kind','phase');enabled('waveform.phase_code','timing.pulses');disabled('waveform.frequency_steps','waveform.frequency_step_hz','timing.groups','waveform.bandwidth_hz')
            choice('waveform.kind','lfm');enabled('waveform.bandwidth_hz');disabled('waveform.phase_code')
            choice('timing.mode','uniform');disabled('timing.pri_sequence_s')
            choice('timing.mode','staggered');enabled('timing.pri_sequence_s')
            choice('array.geometry','custom');enabled('array.tx_positions_m','array.rx_positions_m');disabled('array.spacing_lambda')
            choice('array.geometry','ula');enabled('array.spacing_lambda');disabled('array.tx_positions_m','array.rx_positions_m')
            for state in (False,True):
                editors['processing.doa.enabled'].control.setChecked(state);panel.update_input_availability()
                for path in editors:
                    if path.startswith('processing.doa.') and not path.endswith('.enabled'):assert editors[path].isEnabled()==state,path
        unchanged(held)


def test_failed_record_restore_preserves_operation_and_extended_scene_time(ui):
    from arraylab.application.simulation import calculate
    app,host,panel,(new_report,new_arrays)=reset(ui,'pulse')
    config=deepcopy(new_report['configuration'])
    config['scene']['platform']['trajectory']=[{'time_s':0.,'position_m':[0.,0.,10.]},
                                                {'time_s':20.,'position_m':[20.,0.,10.]}]
    report,arrays=calculate('pulse','budget',config,workers=1)
    panel.accept_result(report,arrays,restore_inputs=True)
    panel.view.setCurrentIndex(panel.view.findData('scene'));panel.table_toggle.setChecked(False)
    panel.scene_time.setValue(15.125);panel.action.setCurrentIndex(panel.action.findData('statistics'))
    original_state=panel.record_snapshot()['view_state'];original_action=panel.action.currentData()
    original_maximum=panel.scene_time.maximum();original=panel.backend.draw;calls=[]
    def fail_once(*args,**kwargs):
        calls.append(1)
        if len(calls)==1:raise RuntimeError('extended scene restore failure')
        return original(*args,**kwargs)
    # Force the incoming valid budget snapshot to draw, so failure occurs after its time range is installed.
    incoming={**new_report,'view_state':{'view':'scene','selection':None,'table':False,'scene_time':1.}}
    with patch.object(panel.backend,'draw',side_effect=fail_once),pytest.raises(RuntimeError,match='extended scene restore failure'):
        panel.accept_result(incoming,new_arrays,restore_inputs=True)
    assert panel.report is report and panel.arrays is arrays
    assert panel.action.currentData()==original_action
    assert panel.scene_time.maximum()==original_maximum
    assert panel.record_snapshot()['view_state']==original_state


@pytest.mark.parametrize('feature',['phased','equation','pulse'])
def test_main_window_integration_actions_stay_in_two_columns_without_overlap(full_window,feature):
    from PySide6.QtCore import QPoint,QRect
    app,window=full_window;window.resize(1100,740);window.activate_task(feature);settle(app,.15)
    panel=window.simulation_panels[feature];bar=panel.integration_bar
    assert bar.columnCount()<=2
    rects=[]
    for index in range(bar.count()):
        button=bar.itemAt(index).widget();assert button is not None and button.isVisible()
        rectangle=QRect(button.mapTo(panel,QPoint(0,0)),button.size())
        assert panel.rect().contains(rectangle),(feature,button.text(),rectangle,panel.rect())
        assert button.width()>=button.fontMetrics().horizontalAdvance(button.text())+8,(feature,button.text())
        for other in rects:assert not rectangle.intersects(other),(feature,button.text())
        rects.append(rectangle)
    assert rects
    assert window.grab().save(str(OUTPUT/(feature+'-integration-grid.png')))


def wait_record_idle(app,window,timeout=15):
    deadline=time.monotonic()+timeout
    while getattr(window,'_record_reader',None) is not None and time.monotonic()<deadline:settle(app,.02)
    assert getattr(window,'_record_reader',None) is None,window.statusBar().currentMessage()


@pytest.mark.parametrize('feature,kind',[('phased','npz'),('equation','sidecar'),('pulse','npz')])
def test_global_record_dialog_restores_async_snapshot_and_routes_task(ui,full_window,tmp_path,feature,kind):
    from PySide6.QtWidgets import QFileDialog,QMessageBox
    from arraylab.application.simulation import export_result
    from arraylab.shared.simulation_io import write_npz
    app,window=full_window;panel=window.simulation_panels[feature]
    report,arrays=ui[2][feature];view={'phased':'cuts','equation':'budget','pulse':'scene'}[feature]
    report={**report,'view_state':{'view':view,'selection':None,'table':False,'scene_time':2.125}}
    if kind=='npz':
        path=tmp_path/(feature+'.npz');write_npz(path,report,arrays)
    else:
        directory=export_result(tmp_path/'sidecar',report,arrays);path=directory/'arraylab-record.json'
    window.activate_task('home')
    with patch.object(QFileDialog,'getOpenFileName',return_value=(str(path),'')),patch.object(QMessageBox,'warning') as warning,patch.object(panel.backend,'execute',side_effect=AssertionError('Async restore computed')):
        window.load_history_dialog()
        assert window._record_reader is not None and not window._record_reader.cancellable
        assert not window.compute_button.isEnabled() and not window.history_open_button.isEnabled()
        assert not window.inspector_stack.isEnabled()
        wait_record_idle(app,window)
    assert not warning.called and window.active_task()==feature
    assert panel.report==report and panel.action.currentData()==report['action']
    assert panel.record_snapshot()['view_state']==report['view_state']
    assert window.inspector_stack.isEnabled() and window.compute_button.isEnabled()
    for key in arrays:np.testing.assert_equal(panel.arrays[key],arrays[key])


@pytest.mark.parametrize('kind',['npz','sidecar'])
def test_global_async_bad_record_keeps_result_and_pending_inputs(ui,full_window,tmp_path,kind):
    from PySide6.QtWidgets import QMessageBox
    from arraylab.application.simulation import export_result
    app,window=full_window;window.activate_task('phased');panel=window.phased_panel
    panel.accept_result(*ui[2]['phased'],restore_inputs=True)
    panel.parameter_widget.editors['array.frequencyGHz'].control.setText('待修输入')
    old_report=panel.report;old_arrays=panel.arrays;old_form=panel.parameter_widget.snapshot()
    if kind=='npz':path=tmp_path/'bad.npz';path.write_bytes(b'not an array archive')
    else:
        directory=export_result(tmp_path/'bad-sidecar',*ui[2]['phased']);path=directory/'arraylab-record.json'
        content=json.loads(path.read_text(encoding='utf-8'));content['arrays_sha256']='0'*64
        path.write_text(json.dumps(content),encoding='utf-8')
    with patch.object(QMessageBox,'warning') as warning,patch.object(panel.backend,'execute',side_effect=AssertionError('Bad record computed')):
        window.load_simulation_record_async(path);wait_record_idle(app,window)
    assert warning.call_count==1 and '原结果保留' in warning.call_args.args[1]
    assert panel.report is old_report and panel.arrays is old_arrays
    assert panel.parameter_widget.snapshot()==old_form and window.active_task()=='phased'
    assert window.compute_button.isEnabled() and window.inspector_stack.isEnabled()


def test_window_close_waits_for_global_record_reader_to_finish(ui,tmp_path):
    from threading import Event
    from PySide6.QtWidgets import QMessageBox
    from arraylab.application.desktop.window import MainWindow
    from arraylab.application.simulation import load_record
    from arraylab.shared.simulation_io import write_npz
    app=ui[0];window=MainWindow(auto_compute=False);window.show()
    path=tmp_path/'closing.npz';write_npz(path,*ui[2]['pulse'])
    started=Event();release=Event()
    def delayed_read(filename):
        started.set();assert release.wait(10);return load_record(filename)
    try:
        with patch('arraylab.application.simulation.load_record',side_effect=delayed_read),patch.object(QMessageBox,'warning') as warning:
            window.load_simulation_record_async(path);assert started.wait(5)
            reader=window._record_reader;window.close();settle(app,.03)
            assert window.isVisible() and window._record_close_pending and reader.isRunning()
            release.set();wait_record_idle(app,window);settle(app,.05)
            assert not warning.called and not window.isVisible()
            assert window.pulse_panel.report==ui[2]['pulse'][0]
    finally:
        release.set();wait_record_idle(app,window);window.close();window.deleteLater();app.processEvents()


@pytest.mark.parametrize('feature,page,inspector',[('phased',11,6),('equation',12,7),('pulse',13,8)])
def test_integrated_task_tree_routes_native_views_parameters_and_distinct_icons(full_window,feature,page,inspector):
    from PySide6.QtCore import Qt
    from arraylab.shared.qt.icons import workbench_icon
    app,window=full_window;panel=window.simulation_panels[feature]
    panel.apply_configuration(panel.backend.default_configuration());before=panel.configuration()
    with patch.object(panel.backend,'execute',side_effect=AssertionError('Navigation computed')),patch('arraylab.application.desktop.simulation_panel.run_process',side_effect=AssertionError('Navigation launched a process')):
        window.activate_task(feature)
        assert window.tabs.currentIndex()==page and window.inspector_stack.currentIndex()==inspector
        root=getattr(window,feature+'_tree_item')
        assert tuple(root.data(0,Qt.ItemDataRole.UserRole))==('view',page) and not root.icon(0).isNull()
        for index in range(panel.view.count()):
            view=panel.view.itemData(index);item=window._task_nav_nodes[(feature,'result',view)]
            window.activate_task_tree_item(item)
            assert panel.view.currentData()==view and window.tree.currentItem() is item
        for index in range(panel.parameter_widget.count()):
            item=window._task_nav_nodes[(feature,'parameter',index)];window.activate_task_tree_item(item)
            assert panel.parameter_widget.currentIndex()==index and window.tree.currentItem() is item
        assert panel.configuration()==before
    rendered=[]
    for name in ('phased','equation','pulse'):
        picture=workbench_icon(name).pixmap(32,32).toImage();assert not picture.isNull()
        rendered.append(bytes(picture.constBits()))
    assert len(set(rendered))==3


def test_handoff_availability_survives_input_and_view_changes_and_global_busy(ui,full_window):
    app,window=full_window;phased=window.phased_panel;equation=window.equation_panel;pulse=window.pulse_panel
    saved={key:panel.report for key,panel in window.simulation_panels.items()}
    try:
        for panel in window.simulation_panels.values():panel.report=None
        window.activate_task('equation');window.refresh_actions()
        assert not equation.array_import_button.isEnabled() and not equation.waveform_import_button.isEnabled()
        assert not phased.planar_transfer_button.isEnabled()
        for update in (equation.mark_dirty,equation.redraw):
            update()
            assert not equation.array_import_button.isEnabled() and not equation.waveform_import_button.isEnabled()
        phased.report=ui[2]['phased'][0];window.refresh_actions()
        assert equation.array_import_button.isEnabled() and phased.planar_transfer_button.isEnabled()
        assert all(a.isEnabled() for a in equation.array_import_actions['phased'])
        assert all(not a.isEnabled() for name,actions in equation.array_import_actions.items() if name!='phased' for a in actions)
        pulse._busy=True;window.refresh_actions();equation.redraw()
        assert not window.compute_button.isEnabled() and not window.run_action.isEnabled()
        assert not equation.array_import_button.isEnabled() and not phased.planar_transfer_button.isEnabled()
    finally:
        pulse._busy=False
        for key,panel in window.simulation_panels.items():panel.report=saved[key]
        window.refresh_actions()


def test_phased_analysis_success_shows_cuts_but_record_restores_saved_view(ui):
    app,host,panel,(report,arrays)=reset(ui,'phased')
    panel.view.setCurrentIndex(panel.view.findData('scan'))
    with patch.object(panel.backend,'execute',side_effect=AssertionError('Result presentation computed')):
        panel.accept_result(report,arrays)
        assert panel.view.currentData()=='cuts' and panel.report is report
        saved={**report,'view_state':{'view':'uv','selection':None,'table':False,'scene_time':0.}}
        panel.accept_result(saved,arrays,restore_inputs=True)
        assert panel.view.currentData()=='uv' and panel.report is saved
        assert panel.record_snapshot()['view_state']==saved['view_state']
