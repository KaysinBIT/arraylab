"""Compatibility of real 0.26.0 snapshots with the repaired upstream models.

Fixtures came from the unmodified released EXE; they are not new calculations
with a rewritten source version. Numerical correctness is tested by the owning
features; this scope checks immutable records, UI semantics and handoffs only.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arraylab.application import simulation
from arraylab.shared.simulation_io import read_npz,write_npz,configuration_digest

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
FIXTURES=Path(__file__).with_name('fixtures')/'upstream_sync'
NAMES=('equation-simulation-v100','equation-invalid-band-budget-v100',
       'equation-calibration-v100','phased-analysis-v121')


def legacy(name):
    return read_npz(FIXTURES/(name+'.npz'))


def no_solver(*args,**kwargs):
    pytest.fail('Snapshot browsing/export must not execute the current solver')


def assert_arrays_equal(actual,expected):
    assert set(actual)==set(expected)
    for key in expected:
        assert actual[key].dtype==expected[key].dtype
        np.testing.assert_array_equal(actual[key],expected[key])


@pytest.mark.parametrize('name',NAMES)
def test_real_legacy_npz_keeps_origin_input_hash_and_arrays(name,monkeypatch):
    before,expected=legacy(name)
    manifest=json.loads((FIXTURES/'provenance.json').read_text(encoding='utf-8'))
    item=next(row for row in manifest['fixtures'] if row['file']==name+'.npz')
    assert hashlib.sha256((FIXTURES/(name+'.npz')).read_bytes()).hexdigest()==item['sha256']
    assert before['app_version']=='0.26.0'
    monkeypatch.setattr(simulation.provider(before['feature']),'execute',no_solver)
    loaded,arrays=simulation.load_record(FIXTURES/(name+'.npz'))
    assert loaded==before
    assert loaded['configuration_sha256']==item['config_sha256']==configuration_digest(loaded['configuration'])
    assert loaded['result']['source']==item['source']
    assert_arrays_equal(arrays,expected)
    compatibility=simulation.record_compatibility(loaded)
    assert compatibility['current'] is False
    assert compatibility['source_version']==item['source']['version']
    assert compatibility['message'] and ('重新' in compatibility['message'] or '旧版' in compatibility['message'])


def test_old_invalid_band_budget_can_be_read_but_new_calculation_stays_strict():
    record,arrays=legacy('equation-invalid-band-budget-v100')
    original=deepcopy(record['configuration'])
    assert simulation.record_configuration('equation',original)==original
    assert not simulation.provider('equation').validate_configuration(original)['valid']
    simulation.validate_record(record,arrays)
    with pytest.raises(ValueError):simulation.calculate('equation','budget',original,workers=1)
    assert original==record['configuration']


@pytest.mark.parametrize('feature',['equation','phased'])
def test_current_model_result_has_no_legacy_warning(feature):
    module=simulation.provider(feature);config=module.default_configuration()
    if feature=='phased':config['array'].update(geometry='linear',nx=4,ny=1);config['mapSize']=21
    record,arrays=simulation.calculate(feature,'analysis' if feature=='phased' else 'budget',config,workers=1)
    simulation.validate_record(record,arrays)
    status=simulation.record_compatibility(record)
    assert status['current'] is True and not status['message']


def test_real_old_ready_calibration_is_rejected_by_new_service():
    from arraylab.features.equation import apply_calibration
    record,_=legacy('equation-calibration-v100');before=deepcopy(record)
    assert record['result']['ready'] is True
    with pytest.raises(ValueError,match='旧版|重新校准|算法'):
        apply_calibration(record['configuration'],record['result'])
    assert record==before


@pytest.mark.parametrize('name',('equation-simulation-v100','phased-analysis-v121'))
def test_legacy_compatibility_does_not_bypass_digest_checks(name,tmp_path):
    record,arrays=legacy(name);record['configuration_sha256']='0'*64
    path=tmp_path/'bad.npz';write_npz(path,record,arrays)
    with pytest.raises(ValueError,match='摘要'):simulation.load_record(path)


@pytest.mark.parametrize('name',('equation-simulation-v100','equation-invalid-band-budget-v100','phased-analysis-v121'))
def test_old_exports_keep_saved_budget_source_and_arrays(name,tmp_path,monkeypatch):
    record,arrays=legacy(name);before=deepcopy(record)
    module=simulation.provider(record['feature'])
    monkeypatch.setattr(module,'execute',no_solver)
    if record['feature']=='equation':
        from arraylab.features.equation import service
        monkeypatch.setattr(service,'calculate',no_solver)
    target=tmp_path/'legacy-export'
    simulation.export_result(target,record,arrays,plot=False)
    loaded,restored=simulation.load_record(target/'arraylab-record.json')
    assert loaded==record==before;assert_arrays_equal(restored,arrays)
    notice=json.loads((target/'model-version-notice.json').read_text(encoding='utf-8'))
    assert notice['current'] is False and notice['source_version']==record['result']['source']['version']
    if record['feature']=='equation':
        saved=json.loads((target/'result.json').read_text(encoding='utf-8'))
        assert saved==record['result']
        budget=saved if saved['kind']=='budget' else saved['budget']
        assert f"{budget['output_snr_db']:.3f}" in (target/'report.html').read_text(encoding='utf-8')


def test_old_phased_actual_feeds_are_handed_off_without_current_reanalysis(monkeypatch):
    from arraylab.application.bridges.radar import apply_phased_to_planar
    from arraylab.features.phased import engine
    record,_=legacy('phased-analysis-v121');before=deepcopy(record)
    monkeypatch.setattr(engine,'evaluate',no_solver)
    host=SimpleNamespace(phased_panel=SimpleNamespace(_busy=False,thread=None,report=record,
                                                     selection=SimpleNamespace(currentData=lambda:0)))
    prepared=apply_phased_to_planar(host)
    config=record['result']['analysis']['config'];elements=record['result']['analysis']['elements']
    expected_phase=np.asarray([row['phaseDeg'] for row in elements])
    assert np.any(expected_phase!=0)
    wavelength=299792458/(config['frequencyGHz']*1e9)
    xyz=np.asarray(prepared['parameters'].coordinates)/1000
    u,v=np.sin(np.deg2rad([config['steerX'],config['steerY']]))
    restored_phase=-360*(xyz[:,0]*u+xyz[:,1]*v)/wavelength+np.asarray(prepared['settings'].custom_phases_deg)
    np.testing.assert_allclose((restored_phase-expected_phase+180)%360-180,0,atol=1e-10)
    assert prepared['settings'].phase_bits==0 and prepared['provenance']['actual_phase_preserved']
    assert record==before


@pytest.fixture(scope='module')
def app():
    pytest.importorskip('PySide6')
    output=ROOT/'tmp/sync-0261/app-compat-ui';output.mkdir(exist_ok=True,parents=True)
    settings={'QT_QPA_PLATFORM':'offscreen','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1',
              'MPLCONFIGDIR':str(output/'mpl'),'ARRAYLAB_HISTORY_DIR':str(output/'linear'),
              'ARRAYLAB_SPARSE_HISTORY_DIR':str(output/'sparse'),'TEMP':str(output/'temp'),'TMP':str(output/'temp')}
    for key in ('MPLCONFIGDIR','ARRAYLAB_HISTORY_DIR','ARRAYLAB_SPARSE_HISTORY_DIR','TEMP'):Path(settings[key]).mkdir(exist_ok=True,parents=True)
    old={k:os.environ.get(k) for k in settings};os.environ.update(settings)
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont,QFontDatabase
    import matplotlib
    application=QApplication.instance() or QApplication([]);application.setQuitOnLastWindowClosed(False)
    assert application.platformName()=='offscreen'
    if not QFontDatabase.families():
        font=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts/msyh.ttc'
        if font.is_file():QFontDatabase.addApplicationFont(str(font));application.setFont(QFont('Microsoft YaHei',9))
    with matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans']}):
        yield application
    for key,value in old.items():
        if value is None:os.environ.pop(key,None)
        else:os.environ[key]=value


@pytest.mark.parametrize('name',('equation-invalid-band-budget-v100','equation-calibration-v100','phased-analysis-v121'))
def test_ui_old_snapshot_notice_persists_and_new_results_clear_it(app,name,tmp_path,monkeypatch):
    from PySide6.QtWidgets import QMainWindow,QSplitter
    from arraylab.application.desktop.simulation_panel import SimulationPanel
    record,arrays=legacy(name);host=QMainWindow();host.last_directory=str(tmp_path);host._workflow_ready=False
    panel=SimulationPanel(host,record['feature']);split=QSplitter();split.addWidget(panel);split.addWidget(panel.parameter_widget)
    host.setCentralWidget(split);host.resize(1200,800);host.show();app.processEvents()
    try:
        original=deepcopy(record)
        with monkeypatch.context() as context:
            context.setattr(panel.backend,'execute',no_solver)
            panel.accept_result(record,arrays,restore_inputs=True)
            app.processEvents()
            assert panel.report==original and panel.parameter_widget.value()==original['configuration']
            assert not panel.snapshot_notice.isHidden() and panel.snapshot_notice.text()
            notice=panel.snapshot_notice.text()
            if record['feature']=='equation':
                assert not panel.calibration_button.isEnabled()
                panel.view.setCurrentIndex(panel.view.findData('budget'))
                panel.parameter_widget.editors['name'].control.setText('Uncalculated edit')
            else:panel.view.setCurrentIndex(panel.view.findData('elements'))
            app.processEvents()
            assert panel.snapshot_notice.text()==notice and not panel.snapshot_notice.isHidden()
            destination=tmp_path/'saved-again.npz';panel.save_record_path(destination)
            saved,restored=simulation.load_record(destination)
            assert saved['configuration']==original['configuration'] and saved['configuration_sha256']==original['configuration_sha256']
            assert saved['result']==original['result'];assert_arrays_equal(restored,arrays)
        config=panel.backend.default_configuration()
        if record['feature']=='phased':config['array'].update(geometry='linear',nx=4,ny=1);config['mapSize']=21
        fresh,fresh_arrays=simulation.calculate(record['feature'],'analysis' if record['feature']=='phased' else 'budget',config,workers=1)
        panel.accept_result(fresh,fresh_arrays,restore_inputs=True);app.processEvents()
        assert panel.snapshot_notice.isHidden() and not panel.snapshot_notice.text()
    finally:
        panel.shutdown_scene();host.close();host.deleteLater();app.processEvents()
