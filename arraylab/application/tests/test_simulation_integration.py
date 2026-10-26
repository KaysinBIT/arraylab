"""Targeted contracts for shared radar forms, records and owned CLI workers."""
from copy import deepcopy
import json
import csv
import hashlib
from io import BytesIO
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pytest

from arraylab import __version__
from arraylab.application import simulation
from arraylab.framework.parallel import ComputationCancelled,requested_workers,set_workers
from arraylab.shared.simulation_io import write_json,write_npz,read_npz,configuration_digest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path,monkeypatch):
    folder=tmp_path/'runtime';folder.mkdir()
    for key,suffix in [('TEMP','temp'),('TMP','temp'),('MPLCONFIGDIR','mpl'),
                       ('ARRAYLAB_HISTORY_DIR','linear'),('ARRAYLAB_SPARSE_HISTORY_DIR','sparse')]:
        path=folder/suffix;path.mkdir(exist_ok=True);monkeypatch.setenv(key,str(path))
    monkeypatch.setattr(tempfile,'tempdir',str(folder/'temp'))
    monkeypatch.setenv('OPENBLAS_NUM_THREADS','1');monkeypatch.setenv('OMP_NUM_THREADS','1')
    previous=requested_workers()
    yield folder
    set_workers(previous)


@pytest.fixture(scope='module')
def application():
    pytest.importorskip('PySide6')
    os.environ['QT_QPA_PLATFORM']='offscreen'
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFontDatabase,QFont
    app=QApplication.instance() or QApplication([])
    assert app.platformName()=='offscreen'
    if not QFontDatabase.families():
        font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
        if font.is_file():QFontDatabase.addApplicationFont(str(font));app.setFont(QFont('Microsoft YaHei',9))
    app.setQuitOnLastWindowClosed(False)
    import matplotlib
    with matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans']}):
        yield app


def tiny_equation():
    from arraylab.features.equation.tests.test_integration import small
    return small()


@pytest.mark.parametrize('feature',['phased','equation','pulse'])
def test_default_forms_roundtrip_typed_values_and_use_metadata(application,feature):
    from arraylab.application.desktop.simulation_panel import form_metadata
    from arraylab.shared.qt.configuration_form import ConfigurationForm
    from PySide6.QtWidgets import QComboBox
    module=simulation.provider(feature);configuration=module.default_configuration();before=deepcopy(configuration)
    metadata=form_metadata(module);assert metadata['GROUPS'] and metadata['LABELS'] and metadata['CHOICES']
    form=ConfigurationForm(configuration,metadata)
    try:
        assert form.value()==configuration and configuration==before
        assert form.count()>1 and form.editors
        for path in metadata['CHOICES']:
            if path in form.editors:assert isinstance(form.editors[path].control,QComboBox)
    finally:form.close();form.deleteLater();application.processEvents()


@pytest.mark.parametrize('feature',['equation','pulse'])
def test_engineering_units_convert_to_si_only_after_edit(application,feature):
    from arraylab.application.desktop.simulation_panel import form_metadata
    from arraylab.shared.qt.configuration_form import ConfigurationForm
    module=simulation.provider(feature);configuration=module.default_configuration();metadata=form_metadata(module)
    form=ConfigurationForm(configuration,metadata)
    try:
        carrier=form.editors['waveform.fc_hz'];pulse=form.editors['waveform.pulse_width_s']
        assert 'GHz' in carrier.control.accessibleName() and 'μs' in pulse.control.accessibleName()
        assert float(carrier.control.text())==pytest.approx(configuration['waveform']['fc_hz']/1e9)
        carrier.control.setText('79.125');pulse.control.setText('2.5')
        value=form.value()
        assert value['waveform']['fc_hz']==pytest.approx(79.125e9)
        assert value['waveform']['pulse_width_s']==pytest.approx(2.5e-6)
        assert configuration['waveform']['fc_hz']!=value['waveform']['fc_hz']
    finally:form.close();form.deleteLater();application.processEvents()


def test_target_templates_row_enums_and_empty_path_addition(application,monkeypatch):
    from arraylab.application.desktop.simulation_panel import form_metadata
    from arraylab.shared.qt.configuration_form import ConfigurationForm,ObjectListEditor
    from PySide6.QtWidgets import QComboBox
    module=simulation.provider('equation');configuration=module.default_configuration();metadata=form_metadata(module)
    form=ConfigurationForm(configuration,metadata)
    try:
        paths=form.editors['propagation.custom_paths'];assert isinstance(paths,ObjectListEditor)
        assert not paths.rows and paths.template
        monkeypatch.setattr(paths,'_dialog',lambda row:{**row,'attenuation_db':17.})
        paths.add();saved=form.value()
        assert saved['propagation']['custom_paths'][0]['attenuation_db']==17.
        assert not configuration['propagation']['custom_paths']
        row=ConfigurationForm(metadata['LIST_DEFAULTS']['scene.targets'],metadata,prefix='scene.targets[]')
        try:
            control=row.editors['scene.targets[].swerling'].control
            assert isinstance(control,QComboBox) and control.count()==5
            control.setCurrentIndex(control.findData(4));assert row.value()['swerling']==4
        finally:row.close();row.deleteLater()
    finally:form.close();form.deleteLater();application.processEvents()


def test_resources_are_readonly_and_invalid_form_snapshot_restores_exactly(application):
    from arraylab.application.desktop.simulation_panel import form_metadata
    from arraylab.shared.qt.configuration_form import ConfigurationForm,ResourceValue
    module=simulation.provider('equation');form=ConfigurationForm(module.default_configuration(),form_metadata(module))
    try:
        form.setCurrentIndex(3);control=form.editors['waveform.samples'].control;control.setText('unfinished-')
        snapshot=form.snapshot()
        with pytest.raises(ValueError):form.value()
        form.set_value(module.default_configuration());form.restore_snapshot(snapshot)
        assert form.currentIndex()==3 and form.editors['waveform.samples'].control.text()=='unfinished-'
        assert form.editors['detection.calibration_factor'].control.isReadOnly()
        assert 'detection.calibration_hash' not in form.editors
    finally:form.close();form.deleteLater()
    config={'scene':{'terrain':{'heights_m':[[1.,None],[2.,3.]],'valid':[[True,False],[True,True]],
                                 'x_m':[0.,1.],'y_m':[0.,1.],'metadata':{'crs':'synthetic'}}}}
    terrain=ConfigurationForm(config)
    try:
        assert all(isinstance(editor,ResourceValue) for editor in terrain.editors.values())
        assert terrain.value()==config
        result=terrain.value();result['scene']['terrain']['heights_m'][0][0]=999
        assert terrain.value()==config
    finally:terrain.close();terrain.deleteLater();application.processEvents()


@pytest.mark.parametrize('feature,action',[('phased','analysis'),('equation','budget'),('pulse','budget')])
def test_direct_calculation_envelope_is_qt_free(feature,action):
    code=f"""
import sys
from arraylab.application.simulation import calculate,provider,validate_record
p=provider({feature!r}).default_configuration()
r,a=calculate({feature!r},{action!r},p,workers=1)
validate_record(r,a,expected_feature={feature!r},expected_action={action!r})
assert not any(n.startswith(('PySide6','fastapi','uvicorn')) for n in sys.modules)
"""
    process=subprocess.run([sys.executable,'-c',code],cwd=ROOT,
        env={**os.environ,'QT_QPA_PLATFORM':'invalid-test-platform'},capture_output=True,timeout=25)
    assert process.returncode==0,process.stderr


@pytest.fixture
def equation_record():
    return simulation.calculate('equation','budget',tiny_equation(),workers=1)


@pytest.mark.parametrize('mutation',['schema_bool','feature','action','result_list','inner_config','outer_hash','array_manifest'])
def test_corrupted_record_header_or_input_binding_is_rejected(equation_record,mutation):
    original,arrays=equation_record;record=deepcopy(original)
    if mutation=='schema_bool':record['schema_version']=True
    elif mutation=='feature':record['feature']='pulse'
    elif mutation=='action':record['action']='unknown'
    elif mutation=='result_list':record['result']=[]
    elif mutation=='inner_config':record['result']['config']['radar']['power_w']*=2
    elif mutation=='outer_hash':record['configuration_sha256']='0'*64
    elif mutation=='array_manifest':record['array_manifest']={'unexpected':{'shape':[1],'dtype':'float64'}}
    with pytest.raises(ValueError):simulation.validate_record(record,arrays,expected_feature='equation')


def test_saved_record_loads_without_solver_even_if_current_inputs_are_invalid(application,tmp_path,equation_record,monkeypatch):
    from PySide6.QtWidgets import QMainWindow
    from arraylab.application.desktop import simulation_panel
    host=QMainWindow();host._workflow_ready=False;host.last_directory=str(tmp_path)
    panel=simulation_panel.SimulationPanel(host,'equation');host.setCentralWidget(panel)
    panel.parameter_widget.setParent(host)
    try:
        report,arrays=equation_record;panel.accept_result(report,arrays,restore_inputs=True)
        target=tmp_path/'complete.npz';panel.save_record_path(target);before=target.read_bytes()
        panel.parameter_widget.editors['waveform.samples'].control.setText('invalid')
        def fail(*args,**kwargs):pytest.fail('Restoring must not run any solver')
        monkeypatch.setattr(simulation,'calculate',fail);monkeypatch.setattr(simulation_panel,'run_process',fail)
        monkeypatch.setattr(panel.backend,'execute',fail)
        panel.load_record_path(target)
        assert panel.report==report and panel.configuration()==report['configuration']
        assert target.read_bytes()==before and panel.thread is None
    finally:host.close();host.deleteLater();application.processEvents()


def test_failed_record_display_restores_raw_inputs_and_selected_tab(application,tmp_path,equation_record,monkeypatch):
    from PySide6.QtWidgets import QMainWindow
    from arraylab.application.desktop.simulation_panel import SimulationPanel
    host=QMainWindow();host._workflow_ready=False;host.last_directory=str(tmp_path)
    panel=SimulationPanel(host,'equation');host.setCentralWidget(panel);panel.parameter_widget.setParent(host)
    try:
        report,arrays=equation_record;panel.accept_result(report,arrays,restore_inputs=True)
        panel.parameter_widget.setCurrentIndex(4);panel.parameter_widget.editors['waveform.samples'].control.setText('unfinished')
        snapshot=panel.parameter_widget.snapshot();old=panel.report;real=panel.redraw;calls=[0]
        def fail_once(*args):
            calls[0]+=1
            if calls[0]==1:raise ValueError('simulated display failure')
            return real(*args)
        monkeypatch.setattr(panel,'redraw',fail_once)
        with pytest.raises(ValueError,match='display'):panel.accept_result(deepcopy(report),arrays,restore_inputs=True)
        assert panel.report is old and panel.parameter_widget.snapshot()==snapshot
    finally:host.close();host.deleteLater();application.processEvents()


def test_failed_record_display_restores_candidate_choices_and_selection(application,tmp_path,equation_record,monkeypatch):
    from PySide6.QtWidgets import QMainWindow
    from arraylab.application.desktop.simulation_panel import SimulationPanel
    host=QMainWindow();host._workflow_ready=False;host.last_directory=str(tmp_path)
    panel=SimulationPanel(host,'equation');host.setCentralWidget(panel);panel.parameter_widget.setParent(host)
    try:
        # A controlled provider fixture isolates generic candidate-navigation
        # rollback from any expensive inverse-search algorithm.
        monkeypatch.setattr(panel.backend,'selections',lambda r:r.get('test_choices',[]),raising=False)
        report,arrays=equation_record;report=deepcopy(report)
        report['result']['test_choices']=[(0,'候选甲'),(1,'候选乙')]
        panel.accept_result(report,arrays,restore_inputs=True);panel.selection.setCurrentIndex(1)
        old_choices=[(panel.selection.itemData(i),panel.selection.itemText(i)) for i in range(panel.selection.count())]
        incoming=deepcopy(report);incoming['result']['test_choices']=[(0,'新候选')]
        real=panel.redraw;calls=[0]
        def fail_once(*args):
            calls[0]+=1
            if calls[0]==1:raise ValueError('simulated candidate display failure')
            return real(*args)
        monkeypatch.setattr(panel,'redraw',fail_once)
        with pytest.raises(ValueError,match='candidate'):panel.accept_result(incoming,arrays,restore_inputs=True)
        assert panel.report is report
        assert [(panel.selection.itemData(i),panel.selection.itemText(i)) for i in range(panel.selection.count())]==old_choices
        assert panel.selection.currentData()==1
    finally:host.close();host.deleteLater();application.processEvents()


def test_memory_limit_is_bounded_by_available_ram(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil,'virtual_memory',lambda:SimpleNamespace(available=256*1024**2))
    assert simulation.memory_budget({'resources':{'memory_mib':512}})==int(256*1024**2*.8)
    with pytest.raises(ValueError):simulation.memory_budget({'resources':{'memory_mib':True}})


@pytest.mark.parametrize('change',['version','feature','action','format','configuration'])
def test_run_process_rejects_wrong_child_identity(equation_record,monkeypatch,change):
    envelope,arrays=equation_record;record=deepcopy(envelope)
    if change=='version':record['app_version']='other-version'
    elif change=='feature':record['feature']='pulse'
    elif change=='action':record['action']='simulation'
    elif change=='format':record['format']='unrelated'
    elif change=='configuration':record['configuration_sha256']='f'*64
    def fake_child(command,*,folder,**kwargs):write_npz(Path(folder)/'result.npz',record,arrays);return 0
    monkeypatch.setattr(simulation,'run_owned',fake_child)
    with pytest.raises(ValueError):simulation.run_process('equation','budget',envelope['configuration'])


@pytest.mark.parametrize('feature,action',[('phased','analysis'),('equation','budget'),('pulse','budget')])
def test_actual_cli_worker_preserves_feature_action_version_and_inputs(feature,action):
    config=simulation.provider(feature).default_configuration();before=deepcopy(config)
    result,arrays=simulation.run_process(feature,action,config,workers=1)
    assert config==before and result['feature']==feature and result['action']==action
    assert result['app_version']==__version__
    assert result['configuration_sha256']==configuration_digest(result['configuration'])
    simulation.validate_record(result,arrays,expected_feature=feature,expected_action=action)


def test_actual_worker_cancel_reaps_owned_processes(isolated_runtime,monkeypatch):
    import psutil
    from arraylab.framework import process as owned
    config=tiny_equation();config['experiment']['trials']=10000
    stop=threading.Event();children={};captured=[];real=owned.subprocess.Popen
    def capture(*args,**kwargs):
        proc=real(*args,**kwargs);captured.append(proc);return proc
    def progress(message):
        for proc in captured:
            try:
                parent=psutil.Process(proc.pid)
                for child in [parent,*parent.children(recursive=True)]:children[child.pid]=child.create_time()
            except psutil.NoSuchProcess:pass
        stop.set()
    monkeypatch.setattr(owned.subprocess,'Popen',capture)
    started=time.monotonic()
    with pytest.raises(ComputationCancelled):
        simulation.run_process('equation','statistics',config,workers=1,progress=progress,cancel=stop.is_set)
    assert stop.is_set() and captured and time.monotonic()-started<15
    assert all(proc.poll() is not None for proc in captured)
    for pid,created in children.items():
        assert not psutil.pid_exists(pid) or psutil.Process(pid).create_time()!=created
    assert not list((isolated_runtime/'temp').glob('arraylab-radar-*'))


def test_cli_worker_and_public_command_have_structured_errors(tmp_path):
    bad=tmp_path/'bad.json';write_json(bad,{'feature':'unknown','action':'budget','configuration':{}})
    result=tmp_path/'result.npz'
    command=[sys.executable,'-m','arraylab','--headless','simulation-worker','--request',str(bad),'--result',str(result)]
    process=subprocess.run(command,cwd=ROOT,env={**os.environ,'QT_QPA_PLATFORM':'invalid-test-platform'},capture_output=True,timeout=20)
    assert process.returncode==2 and json.loads(process.stdout)['status']=='error' and not result.exists()
    output=tmp_path/'existing';output.mkdir();(output/'keep').write_text('untouched')
    process=subprocess.run([sys.executable,'-m','arraylab','--headless','equation','--output-dir',str(output)],cwd=ROOT,
                           env={**os.environ,'QT_QPA_PLATFORM':'invalid-test-platform'},capture_output=True,timeout=20)
    assert process.returncode==3 and json.loads(process.stderr)['status']=='error'
    assert (output/'keep').read_text()=='untouched'


def csv_rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.reader(stream))


def assert_selected_png(path,provider,result,arrays,view,selection):
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from PIL import Image
    with matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans'],'font.size':9}):
        figure=Figure(figsize=(10,6),dpi=130);FigureCanvasAgg(figure)
        provider.draw(figure,result,arrays,view,selection=selection)
        buffer=BytesIO();figure.savefig(buffer,format='png')
        expected=np.asarray(Image.open(buffer).convert('RGBA'))
    actual=np.asarray(Image.open(path).convert('RGBA'))
    np.testing.assert_array_equal(actual,expected)


def test_export_bundle_real_process_preserves_old_version_and_one_array_archive(tmp_path,equation_record,monkeypatch):
    record,arrays=equation_record;record=deepcopy(record)
    old_version='0.25.0' if __version__!='0.25.0' else '0.25.0.dev4'
    record['app_version']=old_version;record['result']['arraylab_version']=old_version
    record['view_state']={'view':'budget','selection':None,'table':False,'scene_time':0.}
    target=tmp_path/'旧版本 导出';messages=[]
    simulation.export_process(target,record,arrays,progress=messages.append)
    assert messages and {p.name for p in target.rglob('*.npz')}=={'arrays.npz'}
    assert (target/'arraylab-record.json').is_file() and (target/'budget.png').read_bytes().startswith(b'\x89PNG')
    sidecar=json.loads((target/'arraylab-record.json').read_text(encoding='utf-8'))
    assert sidecar['arrays_sha256']==hashlib.sha256((target/'arrays.npz').read_bytes()).hexdigest()
    monkeypatch.setattr(simulation,'calculate',lambda *a,**k:pytest.fail('Loading must not recalculate'))
    loaded,restored=simulation.load_record(target/'arraylab-record.json')
    assert loaded==record and loaded['app_version']==old_version and restored.keys()==arrays.keys()
    assert not list(tmp_path.glob('.arraylab-export-*'))


@pytest.mark.parametrize('damage',['digest','inner_metadata','path','schema'])
def test_export_bundle_sidecar_rejects_corruption_without_recalculation(tmp_path,equation_record,damage,monkeypatch):
    record,arrays=equation_record;target=simulation.export_result(tmp_path/'export',record,arrays)
    path=target/'arraylab-record.json';sidecar=json.loads(path.read_text(encoding='utf-8'))
    if damage=='digest':
        with (target/'arrays.npz').open('ab') as stream:stream.write(b'corrupt')
    elif damage=='inner_metadata':sidecar['record']['result']['seed']+=1
    elif damage=='path':sidecar['arrays_file']='../outside.npz'
    else:sidecar['schema_version']=True
    write_json(path,sidecar)
    monkeypatch.setattr(simulation,'calculate',lambda *a,**k:pytest.fail('Restore must not calculate'))
    with pytest.raises(ValueError):simulation.load_record(path)


def test_export_bundle_failure_is_atomic_and_does_not_touch_siblings(tmp_path,equation_record,monkeypatch):
    record,arrays=equation_record;other=tmp_path/'keep.txt';other.write_text('retained')
    def fail(target,*args,**kwargs):
        Path(target).mkdir();(Path(target)/'partial.txt').write_text('incomplete');raise OSError('injected export failure')
    monkeypatch.setattr(simulation.provider('equation'),'export_result',fail)
    target=tmp_path/'failed'
    with pytest.raises(OSError,match='injected'):simulation.export_result(target,record,arrays)
    assert not target.exists() and other.read_text()=='retained'
    assert {p.name for p in tmp_path.iterdir()}=={'keep.txt','runtime'}


def test_export_bundle_selected_phased_case_matches_csv_and_png(tmp_path):
    module=simulation.provider('phased');config=module.default_configuration()
    config['array'].update(geometry='rectangular',nx=3,ny=3,referenceGHz=10.,frequencyGHz=10.)
    config['planar'].update(count=4,gridNx=3,gridNy=3,iterations=0,frequencies=[10.],
        scans=[{'x':0.,'y':0.},{'x':20.,'y':0.}],targetSll=0.,maxHpbwX=120.,maxHpbwY=120.)
    config['mapSize']=21
    record,arrays=simulation.calculate('phased','optimize2d',config,workers=1)
    record['view_state']={'view':'elements','selection':1,'table':True,'scene_time':0.}
    target=tmp_path/'工况2';simulation.export_process(target,record,arrays)
    headers,rows,note=module.table(record['result'],arrays,'elements',selection=1)
    expected=[headers]+[['' if value is None else str(value) for value in row] for row in rows]
    assert csv_rows(target/'selected-view.csv')==expected
    assert not (target/'elements.csv').exists()
    assert csv_rows(target/'elements-case-2.csv')==expected
    assert csv_rows(target/'elements-case-1.csv')!=expected
    assert_selected_png(target/'layout.png',module,record['result'],arrays,'layout',1)
    loaded,_=simulation.load_record(target/'arraylab-record.json');assert loaded['view_state']==record['view_state']


def test_export_bundle_selected_scene_time_matches_csv_and_png(tmp_path):
    from arraylab.features.pulse.tests.test_integration import small_project
    module=simulation.provider('pulse');config=small_project()
    config['scene']['targets'][0]['velocity_mps']=[3.,0.,0.]
    record,arrays=simulation.calculate('pulse','simulate',config,workers=1)
    record['view_state']={'view':'scene','selection':None,'table':False,'scene_time':1.25}
    target=tmp_path/'时刻1.25';simulation.export_process(target,record,arrays)
    headers,rows,note=module.table(record['result'],arrays,'scene',selection=1.25)
    assert csv_rows(target/'selected-view.csv')==[headers]+[['' if value is None else str(value) for value in row] for row in rows]
    assert '1.25' in json.loads((target/'export-view.json').read_text(encoding='utf-8'))['note']
    assert_selected_png(target/'scene.png',module,record['result'],arrays,'scene',1.25)


def test_export_bundle_actual_cancel_reaps_process_and_removes_staging(tmp_path,equation_record,isolated_runtime,monkeypatch):
    from arraylab.framework import process as owned
    record,arrays=equation_record;stop=threading.Event();spawned=[];original=owned.subprocess.Popen
    def capture(*args,**kwargs):
        child=original(*args,**kwargs);spawned.append(child);return child
    def progress(message):
        if '正在导出' in message:stop.set()
    monkeypatch.setattr(owned.subprocess,'Popen',capture)
    target=tmp_path/'cancelled'
    with pytest.raises(ComputationCancelled):simulation.export_process(target,record,arrays,progress=progress,cancel=stop.is_set)
    assert stop.is_set() and spawned and all(child.poll() is not None for child in spawned)
    assert not target.exists() and not list(tmp_path.glob('.arraylab-export-*'))
    assert not list((isolated_runtime/'temp').glob('arraylab-export-input-*'))


def test_export_bundle_rejects_wrong_export_worker_caller_version(tmp_path,equation_record):
    record,arrays=equation_record;path=tmp_path/'record.npz';write_npz(path,record,arrays)
    output=tmp_path/'wrong-version'
    process=subprocess.run([sys.executable,'-m','arraylab','--headless','simulation-export-worker','--record',str(path),
        '--output',str(output),'--caller-version','not-the-running-app'],cwd=ROOT,
        env={**os.environ,'QT_QPA_PLATFORM':'invalid-test-platform'},capture_output=True,timeout=25)
    assert process.returncode==2 and json.loads(process.stdout)['status']=='error'
    assert not output.exists()


def test_native_threads_apply_restore_and_workers_keep_same_numerical_output(monkeypatch):
    from threadpoolctl import threadpool_limits,threadpool_info
    from scipy.fft import set_workers as fft_workers,get_workers
    import scipy.linalg,scipy.signal
    module=simulation.provider('equation');original=module.execute;observed=[]
    def pools():return sorted((row['filepath'],row['num_threads']) for row in threadpool_info() if row['user_api']=='blas')
    def observe(*args,**kwargs):
        observed.append((pools(),get_workers()));return original(*args,**kwargs)
    monkeypatch.setattr(module,'execute',observe)
    def stable(value):
        if isinstance(value,dict):return {key:stable(item) for key,item in value.items() if key!='elapsed_s'}
        if isinstance(value,list):return [stable(item) for item in value]
        return value
    with threadpool_limits(limits=3,user_api='blas'),fft_workers(3):
        before=pools();results=[]
        for workers in (1,2):
            record,arrays=simulation.calculate('equation','simulation',tiny_equation(),workers=workers)
            actual=record['execution']['workers']
            assert all(count==actual for _,count in observed[-1][0]) and observed[-1][1]==actual
            assert record['execution']['native_blas_threads']==actual
            assert record['execution']['fft_workers']==actual
            assert pools()==before and get_workers()==3
            results.append((record,arrays))
        assert stable(results[0][0]['result'])==stable(results[1][0]['result'])
        assert results[0][1].keys()==results[1][1].keys()
        for key in results[0][1]:np.testing.assert_allclose(results[0][1][key],results[1][1][key],rtol=2e-12,atol=1e-28)
        monkeypatch.setattr(module,'execute',lambda *a,**k:(_ for _ in ()).throw(ValueError('injected calculation failure')))
        with pytest.raises(ValueError,match='injected'):simulation.calculate('equation','simulation',tiny_equation(),workers=2)
        assert pools()==before and get_workers()==3


def test_reporting_uses_frozen_budget_without_recalculating(equation_record,monkeypatch):
    from arraylab.features.equation import reporting
    from arraylab.features.equation import budget as budget_module
    stored=deepcopy(equation_record[0]['result'])
    def fail(*args,**kwargs):pytest.fail('HTML must use the stored independent budget')
    monkeypatch.setattr(reporting,'calculate',fail,raising=False);monkeypatch.setattr(budget_module,'calculate',fail)
    summary={'kind':'simulation','config':stored['config'],'config_hash':stored['config_hash'],
             'algorithm_version':stored['algorithm_version'],'seed':stored['seed'],'budget':stored,'warnings':[]}
    rendered=reporting.render_report(summary,{})
    assert f"{stored['output_snr_db']:.3f} dB" in rendered and 'ArrayLab' in rendered
    assert reporting.render_report(stored,{})
