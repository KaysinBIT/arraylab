"""Focused 1.4 import, snapshot and native headless integration contracts."""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from arraylab.features.waveform import antenna, gain, model, io, sparse_design

ROOT = next(path for path in Path(__file__).resolve().parents if (path / 'pyproject.toml').is_file())


def small_configuration(enabled=False):
    config = model.default_waveform_configuration()
    config['params'].update(samples=64, rangeFft=128, loops=8, dopplerFft=16,
                            txCount=2, rxCount=2, targetRange=12, targetVelocity=0)
    config['profileB'].update(samples=64, rangeFft=128, loops=8, dopplerFft=16, startGHz=78)
    config['abOptions']['enabled'] = enabled
    config['antennaDesign']['sparseOptions'].update(txCount=2, rxCount=2, iterations=100,
        criterion='hpbw', targetResolutionDeg=40, scanMaxDeg=0, maxApertureLambda=5, maxSidelobeDb=-1)
    config['geometry'] = {'tx': [[0, 0], [1, 0]], 'rx': [[0, .5], [.5, .5]],
                          'steerAzDeg': 0, 'steerElDeg': 0, 'referenceFrequencyGHz': 77}
    return config


@pytest.fixture(scope='module')
def radar_report():
    return model.calculate_waveform(small_configuration(), workers=1)


@pytest.fixture(scope='module')
def evaluated_report():
    config = small_configuration()
    config['geometry'].pop('referenceFrequencyGHz')
    return model.calculate_waveform(config, workers=1, radar_action='evaluate')


def write_json(path, document):
    path.write_text(json.dumps(document, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def fail_engine(*args, **kwargs):
    pytest.fail('Import/restore/default analysis must not launch an optimizer')


def resign(document):
    payload = {key: document[key] for key in ('report', 'view', 'profile', 'comparisons')}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    document['sha256'] = hashlib.sha256(encoded).hexdigest()


def test_schema3_import_keeps_radar_settings_without_running_search(tmp_path, monkeypatch):
    config = small_configuration(True)
    config['activeProfile'] = 'B'
    config['antennaDesign']['sparseOptions'].update(criterion='dml', snapshots=256, sourceCorrelation=1)
    source = {'app': 'WaveLab', 'schemaVersion': 3, 'name': 'Source radar', **config}
    path = tmp_path / '原雷达方案.json'
    write_json(path, source)
    original = path.read_bytes()
    monkeypatch.setattr(sparse_design, 'design', fail_engine)
    monkeypatch.setattr(sparse_design, 'evaluate', fail_engine)
    monkeypatch.setattr(model, 'calculate_waveform', fail_engine)
    actual = io.load_waveform_configuration(path)
    assert actual == model.normalize_waveform_configuration(config)
    assert actual['geometry']['referenceFrequencyGHz'] == 77
    assert actual['antennaDesign']['sparseOptions']['criterion'] == 'dml'
    assert path.read_bytes() == original
    native = tmp_path / 'native.json'
    io.save_waveform_configuration(native, actual)
    assert json.loads(native.read_text(encoding='utf-8'))['schema_version'] == 2
    assert io.load_waveform_configuration(native) == actual


@pytest.mark.parametrize('version', [1, 2])
def test_old_parameter_envelopes_get_defaults_without_automatic_search(version, tmp_path, monkeypatch):
    old = {key: value for key, value in small_configuration().items() if key in model.LEGACY_CONFIG_KEYS}
    path = tmp_path / 'legacy.json'
    write_json(path, {'app': 'WaveLab', 'schemaVersion': version, **old})
    monkeypatch.setattr(sparse_design, 'design', fail_engine)
    actual = io.load_waveform_configuration(path)
    assert actual['geometry'] is None
    assert actual['antennaDesign']['section'] == 'layout'
    assert set(actual) == set(model.CONFIG_KEYS)


def test_legacy_result_is_restored_verbatim_without_any_engine(radar_report, tmp_path, monkeypatch):
    old = deepcopy(radar_report)
    old['schema_version'] = 1
    old['configuration'] = {key: value for key, value in old['configuration'].items() if key in model.LEGACY_CONFIG_KEYS}
    for key in ('antenna', 'gain_budget', 'sparse_design'):
        old.pop(key)
    old['source'] = {'name': 'WaveLab', 'version': '1.2', 'archive_sha256': '16cd6c2e1ad15f7ecb4ba0c15af957711d2427b972b0dee1fd2e07bc08d21ad4'}
    path = tmp_path / 'legacy-result.json'
    io.save_waveform_record(path, old, 'spectra')
    original = path.read_bytes()
    for owner, names in [(model, ['calculate_waveform']), (model.waveform_core, ['calculate', 'simulate']),
                         (model.waveform_ab, ['calculate']), (gain, ['calculate']),
                         (antenna, ['calculate']), (sparse_design, ['design', 'evaluate'])]:
        for name in names:
            monkeypatch.setattr(owner, name, fail_engine)
    restored = io.load_waveform_record(path)
    assert restored['report'] == old
    assert restored['report']['schema_version'] == 1
    assert 'geometry' not in restored['report']['configuration']
    assert path.read_bytes() == original


def test_default_analysis_computes_gain_and_geometry_but_never_searches(monkeypatch):
    monkeypatch.setattr(sparse_design, 'design', fail_engine)
    monkeypatch.setattr(sparse_design, 'evaluate', fail_engine)
    result = model.calculate_waveform(small_configuration(), workers=1)
    assert result['source']['version'] == '1.4' and result['schema_version'] == 2
    assert result['sparse_design'] is None
    assert result['antenna']['A']['valid']
    assert result['gain_budget']['valid']
    assert result['gain_budget']['scope'] == 'active'


def test_gain_scope_uses_selected_profile_or_maximum_ab_requirement():
    config = small_configuration(True)
    config['activeProfile'] = 'B'
    active = model.calculate_waveform(config, workers=1)
    assert [row['name'] for row in active['gain_budget']['profiles']] == ['B']
    config['antennaDesign']['gainScope'] = 'all'
    both = model.calculate_waveform(config, workers=1)
    budget = both['gain_budget']
    assert [row['name'] for row in budget['profiles']] == ['A', 'B']
    assert budget['requiredGainSumDb'] == max(row['requiredGainSumDb'] for row in budget['profiles'])
    assert [row['name'] for row in budget['waveformInputs']] == ['A', 'B']
    io.validate_waveform_report(both)


def test_reference_frequency_preserves_physical_aperture_between_ab():
    config = small_configuration(True)
    report = model.calculate_waveform(config, workers=1)
    reference_mm = 299.792458 / config['geometry']['referenceFrequencyGHz']
    for name in ('A', 'B'):
        row = report['antenna'][name]
        wavelength = report['analysis']['profiles'][name]['result']['metrics']['lambdaMm']
        assert row['tx'][1][0] * wavelength == pytest.approx(reference_mm)
        assert row['rx'][1][0] * wavelength == pytest.approx(.5 * reference_mm)
    assert report['antenna']['A']['tx'] != report['antenna']['B']['tx']
    assert report['configuration']['geometry'] == config['geometry']


def test_invalid_full_ab_frame_blocks_gain_before_calculation(monkeypatch):
    config = small_configuration(True)
    config['params']['frameMs'] = .01
    monkeypatch.setattr(gain, 'calculate', fail_engine)
    with pytest.raises(ValueError):
        model.calculate_waveform(config, workers=1)


def test_invalid_gain_and_mismatched_geometry_are_explicit_storable_states(tmp_path):
    config = small_configuration()
    config['antennaDesign']['gainOptions']['rangeM'] = 0
    config['geometry']['tx'].pop()
    report = model.calculate_waveform(config, workers=1)
    assert report['gain_budget']['status'] == 'invalid'
    assert not report['gain_budget']['valid'] and report['gain_budget']['requiredTxGainDbi'] is None
    assert not report['antenna']['A']['valid']
    path = tmp_path / 'invalid-radar-domains.json'
    io.save_waveform_record(path, report, 'gain_budget')
    assert io.load_waveform_record(path)['report'] == report


def test_explicit_evaluation_preserves_candidate_settings(evaluated_report):
    result = evaluated_report['sparse_design']
    assert result['valid'] and result['action'] == 'evaluate'
    assert result['search']['iterations'] == 0
    assert len(result['geometry']['tx']) == len(result['geometry']['rx']) == 2
    assert result['referenceFrequencyGHz'] == evaluated_report['analysis']['profiles']['A']['result']['metrics']['centerGHz']


def test_new_record_roundtrip_preserves_sparse_candidate_and_comparisons(evaluated_report, radar_report, tmp_path, monkeypatch):
    path = tmp_path / 'radar-record.json'
    comparison = [{'name': 'before', 'report': radar_report}]
    io.save_waveform_record(path, evaluated_report, 'sparse_design', comparison)
    before = path.read_bytes()
    for owner, name in [(model, 'calculate_waveform'), (sparse_design, 'design'), (sparse_design, 'evaluate'),
                        (gain, 'calculate'), (antenna, 'calculate')]:
        monkeypatch.setattr(owner, name, fail_engine)
    restored = io.load_waveform_record(path)
    assert restored == {'report': evaluated_report, 'view': 'sparse_design', 'profile': 'A', 'comparisons': comparison}
    assert path.read_bytes() == before


@pytest.mark.parametrize('mutation', ['digest', 'gain_scope', 'gain_input', 'gain_options', 'gain_status',
                                     'gain_missing_value', 'antenna_missing_geometry', 'sparse_action', 'sparse_frequency',
                                     'sparse_count', 'sparse_settings'])
def test_corrupt_radar_record_is_rejected_even_if_rehashed(evaluated_report, tmp_path, mutation):
    path = tmp_path / 'corrupt.json'
    io.save_waveform_record(path, evaluated_report, 'sparse_design')
    doc = json.loads(path.read_text(encoding='utf-8'))
    report = doc['report']
    if mutation == 'digest': doc['sha256'] = '0' * 64
    elif mutation == 'gain_scope': report['gain_budget']['scope'] = 'all'
    elif mutation == 'gain_input': report['gain_budget']['waveformInputs'][0]['params']['samples'] += 1
    elif mutation == 'gain_options': report['gain_budget']['options']['rangeM'] += 10
    elif mutation == 'gain_status': report['gain_budget']['status'] = 'fictional'
    elif mutation == 'gain_missing_value': report['gain_budget']['requiredTxGainDbi'] = None
    elif mutation == 'antenna_missing_geometry': report['antenna']['A'].pop('tx')
    elif mutation == 'sparse_action': report['sparse_design']['action'] = 'fictional'
    elif mutation == 'sparse_frequency': report['sparse_design']['referenceFrequencyGHz'] = -1
    elif mutation == 'sparse_count': report['sparse_design']['geometry']['tx'].pop()
    elif mutation == 'sparse_settings': report['sparse_design']['options']['seed'] += 1
    if mutation != 'digest': resign(doc)
    write_json(path, doc)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        io.load_waveform_record(path)
    assert path.read_bytes() == before


def test_export_includes_radar_snapshots_and_guards_existing_destination(evaluated_report, tmp_path):
    out = tmp_path / 'export'
    io.export_waveform_directory(out, evaluated_report, plot='none')
    for key in ('antenna', 'gain_budget', 'sparse_design'):
        exported = json.loads((out / (key.replace('_', '-') + '.json')).read_text(encoding='utf-8'))
        assert exported == evaluated_report[key]
    assert json.loads((out / 'summary.json').read_text(encoding='utf-8')) == evaluated_report
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    with pytest.raises(FileExistsError):
        io.export_waveform_directory(out, evaluated_report, plot='none')
    assert before == {path.name: path.read_bytes() for path in out.iterdir()}


def invoke_cli(arguments, tmp_path):
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MPLCONFIGDIR=str(tmp_path / 'mpl'), QT_QPA_PLATFORM='invalid-test-platform')
    # No Qt imports are permitted, including indirect imports introduced by radar code.
    program = '''import importlib.abc,sys,runpy
class BlockQt(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(('PySide','PyQt','tkinter')): raise RuntimeError('GUI import blocked: '+fullname)
sys.meta_path.insert(0,BlockQt())
sys.argv=['arraylab.waveform_cli']+sys.argv[1:]
runpy.run_module('arraylab.waveform_cli',run_name='__main__')
'''
    return subprocess.run([sys.executable, '-c', program, *map(str, arguments)], cwd=ROOT, env=env,
                          capture_output=True, text=True, encoding='utf-8', timeout=60)


@pytest.mark.parametrize('action', [None, 'evaluate', 'sparse'])
def test_headless_radar_actions_are_explicit_and_qt_free(tmp_path, action):
    config = small_configuration()
    config['geometry'].pop('referenceFrequencyGHz')
    path = tmp_path / 'params.json'
    io.save_waveform_configuration(path, config)
    args = ['--waveform-config', path, '--workers', '1']
    if action is not None: args += ['--radar-action', action]
    process = invoke_cli(args, tmp_path)
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)['waveform']
    assert result['schema_version'] == 2
    if action is None:
        assert result['sparse_design'] is None
    else:
        assert result['sparse_design']['valid']
        assert result['sparse_design']['action'] == action
    io.validate_waveform_report(result)


def test_cli_rejects_unknown_action_and_protects_existing_output(tmp_path):
    process = invoke_cli(['--radar-action', 'implicit'], tmp_path)
    assert process.returncode == 2 and not process.stdout
    assert json.loads(process.stderr)['error']['type'] == 'input'
    out = tmp_path / 'existing'
    out.mkdir()
    (out / 'keep.txt').write_bytes(b'keep')
    process = invoke_cli(['--radar-action', 'sparse', '--output-dir', out], tmp_path)
    assert process.returncode == 3 and not process.stdout
    assert (out / 'keep.txt').read_bytes() == b'keep'
    assert len(list(out.iterdir())) == 1


@pytest.fixture(scope='module')
def music_report():
    config = small_configuration()
    config['geometry'].pop('referenceFrequencyGHz')
    config['antennaDesign']['sparseOptions'].update(criterion='music', snapshots=16, trials=1,
        targetResolutionDeg=20, scanMaxDeg=0, scanGridStepDeg=.5)
    return model.calculate_waveform(config, workers=1, radar_action='evaluate')


def test_music_record_without_dml_diagnostics_restores(music_report, tmp_path):
    sr = music_report['sparse_design']['superresolution']
    assert sr['valid'] and sr['representative']['diagnostics'] is None
    path = tmp_path / 'music.json'
    io.save_waveform_record(path, music_report, 'sparse_design')
    assert io.load_waveform_record(path)['report'] == music_report


@pytest.mark.parametrize('mutation', ['metrics_empty', 'pattern_missing', 'pattern_empty',
                                     'pattern_type', 'pattern_coordinate', 'pattern_order',
                                     'scans_missing', 'scans_empty', 'scan_row', 'scan_resolved'])
def test_sparse_detail_corruption_rejected(evaluated_report, tmp_path, mutation):
    path = tmp_path / 'bad-sparse.json'
    io.save_waveform_record(path, evaluated_report, 'sparse_design')
    document = json.loads(path.read_text(encoding='utf-8'))
    sparse = document['report']['sparse_design']
    if mutation == 'metrics_empty': sparse['metrics'] = {}
    elif mutation == 'pattern_missing': sparse.pop('pattern')
    elif mutation == 'pattern_empty': sparse['pattern'] = []
    elif mutation == 'pattern_type': sparse['pattern'] = 'bad'
    elif mutation == 'pattern_coordinate': sparse['pattern'][0]['y'] = 'bad'
    elif mutation == 'pattern_order': sparse['pattern'][1]['x'] = sparse['pattern'][0]['x']
    elif mutation == 'scans_missing': sparse.pop('scanChecks')
    elif mutation == 'scans_empty': sparse['scanChecks'] = []
    elif mutation == 'scan_row': sparse['scanChecks'][0] = 'bad'
    elif mutation == 'scan_resolved': sparse['scanChecks'][0]['resolved'] = 'bad'
    resign(document)
    write_json(path, document)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        io.load_waveform_record(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize('mutation', ['sr_method', 'sr_valid', 'sr_validation_type',
                                     'sr_validation_row', 'sr_case_rate', 'sr_case_rmse',
                                     'sr_success_rate', 'sr_rep_type', 'sr_truth', 'sr_estimates',
                                     'sr_diagnostics', 'sr_subset'])
def test_superresolution_detail_corruption_rejected(music_report, tmp_path, mutation):
    path = tmp_path / 'bad-sr.json'
    io.save_waveform_record(path, music_report, 'sparse_design')
    document = json.loads(path.read_text(encoding='utf-8'))
    sr = document['report']['sparse_design']['superresolution']
    if mutation == 'sr_method': sr['method'] = 'fictional'
    elif mutation == 'sr_valid': sr['valid'] = 'bad'
    elif mutation == 'sr_validation_type': sr['validation'] = 'bad'
    elif mutation == 'sr_validation_row': sr['validation'][0] = 'bad'
    elif mutation == 'sr_case_rate': sr['validation'][0]['successRate'] = 'bad'
    elif mutation == 'sr_case_rmse': sr['validation'][0]['rmseDeg'] = 'bad'
    elif mutation == 'sr_success_rate': sr['successRate'] = 2
    elif mutation == 'sr_rep_type': sr['representative'] = 'bad'
    elif mutation == 'sr_truth': sr['representative']['truth'] = 'bad'
    elif mutation == 'sr_estimates': sr['representative']['estimates'] = ['bad']
    elif mutation == 'sr_diagnostics': sr['representative']['diagnostics'] = 'bad'
    elif mutation == 'sr_subset': sr['representative']['subset'] = 'bad'
    resign(document)
    write_json(path, document)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        io.load_waveform_record(path)
    assert path.read_bytes() == before


def test_over_48_positions_preserves_unsupported_complete_record(tmp_path, monkeypatch):
    config = small_configuration()
    config['params'].update(txCount=7, rxCount=8)
    config['geometry'] = {'tx': [[4*i, 0] for i in range(7)],
                          'rx': [[.5*i, .5] for i in range(8)], 'steerAzDeg': 0, 'steerElDeg': 0}
    config['antennaDesign']['sparseOptions'].update(txCount=7, rxCount=8, criterion='music',
        maxApertureLambda=30, snapshots=16, trials=1, targetResolutionDeg=20, scanMaxDeg=0)
    report = model.calculate_waveform(config, workers=1, radar_action='evaluate')
    sparse = report['sparse_design']
    assert sparse['valid'] and sparse['metrics']['uniqueCount'] == 56
    assert sparse['status'] == 'unmet'
    assert not sparse['superresolution']['valid']
    assert sparse['superresolution']['status'] == 'unsupported'
    assert sparse['scanChecks'] == []
    path = tmp_path / 'unsupported-56-positions.json'
    io.save_waveform_record(path, report, 'sparse_design')
    monkeypatch.setattr(sparse_design, 'evaluate', fail_engine)
    monkeypatch.setattr(sparse_design, 'design', fail_engine)
    monkeypatch.setattr(model, 'calculate_waveform', fail_engine)
    assert io.load_waveform_record(path)['report'] == report
