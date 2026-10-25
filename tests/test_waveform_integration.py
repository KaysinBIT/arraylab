"""New waveform orchestration, snapshot IO and genuinely Qt-free CLI paths."""
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from arraylab import waveform, waveform_io as io

ROOT = Path(__file__).resolve().parents[1]


def small_configuration(enabled=False, layout='blocks', mode='tdm'):
    config = waveform.default_waveform_configuration()
    config['params'].update(samples=64, rangeFft=128, loops=8, dopplerFft=16,
                            targetRange=12, targetVelocity=1, mode=mode, ddmSlots=4)
    config['profileB'].update(samples=64, rangeFft=128, loops=8, dopplerFft=16)
    config['abOptions'].update(enabled=enabled, layout=layout, switchGapUs=5)
    return config


@pytest.fixture(scope='module')
def single_report():
    return waveform.calculate_waveform(small_configuration(), workers=1)


@pytest.fixture(scope='module')
def ab_report():
    return waveform.calculate_waveform(small_configuration(True, 'alternating'), workers=1)


def canonical(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def resign(document):
    payload = {key: document[key] for key in ('report', 'view', 'profile', 'comparisons')}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    document['sha256'] = hashlib.sha256(encoded).hexdigest()


def test_default_workbench_keeps_both_independent_experiments_off(single_report):
    default = waveform.default_waveform_configuration()
    assert default['rangeOptions']['method'] == default['priOptions']['method'] == 'off'
    assert default['rangeOptions']['secondSampleRateMsps'] == 5
    assert default['rangeOptions']['searchMaxM'] == 120
    assert set(single_report['spectra']) == set(single_report['range_recovery']) == {'A'}
    assert single_report['range_recovery']['A']['status'] == single_report['pri_recovery']['status'] == 'off'
    assert single_report['analysis']['profiles']['B'] is None


@pytest.mark.parametrize('layout', ['blocks', 'alternating'])
def test_ab_physical_timing_and_shared_independent_fields(layout):
    config = small_configuration(True, layout)
    config['profileB'].update(txCount=9, rxCount=1, mode='ddm', targetRange=999)
    original = deepcopy(config)
    result = waveform.calculate_waveform(config, workers=1)
    a, b = result['configuration']['params'], result['configuration']['profileB']
    assert config == original
    for key in ('txCount', 'rxCount', 'mode', 'targetRange'):
        assert b[key] == a[key]
    assert a['slopeMHzUs'] == 12 and b['slopeMHzUs'] == 18
    metrics = result['analysis']['metrics']
    # Directly count physical chirps and switch gaps, without calling AB helpers.
    switches = 1 if layout == 'blocks' else 15
    assert metrics['burstMs'] == pytest.approx((8*3*38 + 8*3*48 + switches*5)/1000)
    if layout == 'blocks':
        assert metrics['slowTimeIntervalAUs'] == 114
        assert metrics['slowTimeIntervalBUs'] == 144
    else:
        assert metrics['slowTimeIntervalAUs'] == metrics['slowTimeIntervalBUs'] == 268
    assert set(result['spectra']) == {'A', 'B'}


def test_disabled_b_is_preserved_but_not_evaluated_and_active_group_is_a(tmp_path):
    config = small_configuration()
    config['profileB'].update(slopeMHzUs=0, samples=4095, rangeFft=16)
    config['activeProfile'] = 'B'
    result = waveform.calculate_waveform(config, workers=1)
    assert result['configuration']['activeProfile'] == 'A'
    assert result['configuration']['profileB']['slopeMHzUs'] == 0
    assert result['analysis']['profiles']['B'] is None and set(result['spectra']) == {'A'}
    path = tmp_path/'inactive-B.json'; io.save_waveform_record(path, result)
    assert io.load_waveform_record(path)['report'] == canonical(result)
    config['abOptions']['enabled'] = True
    with pytest.raises(ValueError): waveform.calculate_waveform(config, workers=1)


@pytest.mark.parametrize('version', [1, 2])
def test_wavelab_import_uses_only_confirmed_waveform_domains(version, tmp_path):
    source = {'app': 'WaveLab', 'schemaVersion': version, 'name': '独立输入',
              'params': small_configuration(True)['params'],
              'geometry': {'tx': [[9, 9]], 'rx': [[99, 99]]}, 'savedAt': 'source-metadata',
              'profileB': {'slopeMHzUs': 15, 'rampUs': 42, 'idleUs': 12, 'samples': 64,
                           'rangeFft': 128, 'loops': 8, 'dopplerFft': 16},
              'abOptions': {'enabled': True, 'layout': 'blocks', 'switchGapUs': 5},
              'activeProfile': 'B'}
    path = tmp_path/'WaveLab 方案.json'; path.write_text(json.dumps(source), encoding='utf-8')
    before = path.read_bytes(); config = io.load_waveform_configuration(path)
    assert path.read_bytes() == before
    assert 'geometry' not in config and set(config) == set(waveform.CONFIG_KEYS)
    assert config['abOptions']['enabled'] is (version == 2)
    assert config['activeProfile'] == ('A' if version == 1 else 'B')
    report = waveform.calculate_waveform(config, workers=1)
    assert set(report['spectra']) == ({'A'} if version == 1 else {'A', 'B'})
    target = tmp_path/'native.json'; io.save_waveform_configuration(target, config)
    assert io.load_waveform_configuration(target) == config


def test_record_and_comparison_restore_do_not_run_any_numerical_engine(single_report, ab_report, tmp_path):
    target = tmp_path/'完整 记录.json'
    comparison = [{'name': '单波形', 'report': single_report}, {'name': 'AB交替', 'report': ab_report}]
    io.save_waveform_record(target, ab_report, 'comparisons', comparison, 'B')
    before = target.read_bytes()
    def fail(*args, **kwargs): pytest.fail('Reading a snapshot must not recalculate')
    with patch.object(waveform, 'calculate_waveform', fail), \
         patch.object(waveform.waveform_core, 'calculate', fail), \
         patch.object(waveform.waveform_core, 'simulate', fail), \
         patch.object(waveform.waveform_ab, 'calculate', fail), \
         patch.object(waveform.waveform_ambiguity, 'simulate', fail), \
         patch.object(waveform.waveform_ambiguity, 'resolve', fail), \
         patch.object(waveform.waveform_pri, 'simulate', fail), \
         patch.object(waveform.waveform_pri, 'resolve', fail):
        restored = io.load_waveform_record(target)
    assert restored == canonical({'report': ab_report, 'view': 'comparisons', 'profile': 'B', 'comparisons': comparison})
    assert target.read_bytes() == before and io.is_waveform_record(target)


@pytest.mark.parametrize('change', ['digest', 'profile_shape', 'grid_shape', 'grid_axis',
                                   'config', 'snapshot_params', 'nan', 'view', 'inactive_group',
                                   'profile_result_type', 'inactive_b_result'])
def test_corrupt_record_is_rejected_without_touching_saved_file(single_report, tmp_path, change):
    target = tmp_path/'坏记录.json'; io.save_waveform_record(target, single_report)
    document = json.loads(target.read_text(encoding='utf-8'))
    if change == 'digest': document['sha256'] = '0'*64
    elif change == 'profile_shape': document['report']['spectra']['A']['rangeProfile'].pop()
    elif change == 'grid_shape':
        grid = document['report']['spectra']['A']['rangeDoppler']
        grid.update(width=1, ranges=[0], values=grid['values'][:grid['height']])
    elif change == 'grid_axis': document['report']['spectra']['A']['rangeDoppler']['ranges'][0] = 'invalid'
    elif change == 'config': document['report']['configuration']['params']['samples'] = 65
    elif change == 'snapshot_params': document['report']['analysis']['profiles']['A']['params']['samples'] = 65
    elif change == 'nan': document['report']['spectra']['A']['rangeProfile'][0]['y'] = float('nan')
    elif change == 'view': document['view'] = 'unknown'
    elif change == 'inactive_group': document['profile'] = 'B'
    elif change == 'profile_result_type': document['report']['analysis']['profiles']['A']['result'] = []
    elif change == 'inactive_b_result':
        document['report']['analysis']['profiles']['B'] = deepcopy(document['report']['analysis']['profiles']['A'])
    if change not in ('digest', 'nan'): resign(document)
    target.write_text(json.dumps(document), encoding='utf-8'); before = target.read_bytes()
    with pytest.raises(ValueError): io.load_waveform_record(target)
    assert target.read_bytes() == before


def test_wrong_ddm_phase_code_is_rejected_after_digest_is_recomputed(tmp_path):
    report = waveform.calculate_waveform(small_configuration(mode='ddm'), workers=1)
    target = tmp_path/'DDM 相位记录.json'; io.save_waveform_record(target, report)
    document = json.loads(target.read_text(encoding='utf-8'))
    phases = document['report']['analysis']['profiles']['A']['result']['metrics']['ddmPhasesDeg']
    assert phases == [0, 90, 180]
    phases[1] = 91
    resign(document); target.write_text(json.dumps(document), encoding='utf-8')
    before = target.read_bytes()
    with pytest.raises(ValueError, match='相位'):
        io.load_waveform_record(target)
    assert target.read_bytes() == before


@pytest.mark.parametrize('text', ['{"format":"x","format":"y"}',
    '{"format":"arraylab.waveform","schema_version":true,"configuration":{}}',
    '{"format":"arraylab.waveform","schema_version":2,"configuration":{}}',
    '{"format":"arraylab.waveform","schema_version":1,"configuration":{"params":{"extra":1}}}',
    '{"format":"arraylab.waveform","schema_version":1,"configuration":{"params":{"samples":NaN}}}',
    '{"app":"WaveLab","schemaVersion":3,"params":{}}', '[1,2,3]'])
def test_invalid_configuration_documents_are_rejected(tmp_path, text):
    path = tmp_path/'invalid.json'; path.write_text(text, encoding='utf-8'); before = path.read_bytes()
    with pytest.raises(ValueError): io.load_waveform_configuration(path)
    assert path.read_bytes() == before


def test_record_probe_and_file_size_bound(tmp_path, monkeypatch):
    path = tmp_path/'arbitrary.extension'
    path.write_text('{"note":"arraylab.waveform.record"}', encoding='utf-8')
    assert io.is_waveform_record(path) is False
    path.write_text('{"format":"arraylab.waveform.record"}', encoding='utf-8')
    assert io.is_waveform_record(path) is True
    monkeypatch.setattr(io, 'MAX_BYTES', 8)
    with pytest.raises(ValueError): io.is_waveform_record(path)


def test_atomic_save_failure_preserves_original_file_and_cleans_temporary(single_report, tmp_path, monkeypatch):
    path = tmp_path/'old.json'; path.write_bytes(b'keep-old')
    def fail(*args, **kwargs): raise OSError('injected disk failure')
    monkeypatch.setattr(io.os, 'replace', fail)
    with pytest.raises(OSError): io.save_waveform_record(path, single_report)
    assert path.read_bytes() == b'keep-old' and list(tmp_path.iterdir()) == [path]


def test_comparison_limit_and_unknown_fields_do_not_overwrite(single_report, tmp_path):
    path = tmp_path/'old.json'; path.write_bytes(b'keep-old')
    for comparisons in [[{'name': str(i), 'report': single_report} for i in range(9)],
                        [{'name': 'bad', 'report': single_report, 'extra': 1}]]:
        with pytest.raises(ValueError): io.save_waveform_record(path, single_report, comparisons=comparisons)
        assert path.read_bytes() == b'keep-old'


@pytest.mark.parametrize('kind', ['blocked', 'ambiguous', 'inconsistent'])
def test_pri_experiment_status_does_not_fail_valid_waveform(kind, tmp_path):
    config = small_configuration()
    options = {'method': 'dual_period'}
    options.update({'period1Us': 0} if kind == 'blocked' else
                   {'period1Us': 38, 'period2Us': 38} if kind == 'ambiguous' else {'searchMaxM': 1000})
    config['priOptions'].update(options)
    result = waveform.calculate_waveform(config, workers=1)
    assert result['analysis']['valid'] and result['pri_recovery']['status'] == kind
    assert result['pri_recovery']['rangeM'] is None
    path = tmp_path/(kind + '.json'); io.save_waveform_record(path, result)
    assert io.load_waveform_record(path)['report']['pri_recovery'] == result['pri_recovery']


def test_adc_inconsistent_experiment_is_preserved_in_complete_record(tmp_path):
    config = small_configuration()
    config['params'].update(sampleRateMsps=4, samples=64, rangeFft=1024, targetRange=60, targetVelocity=0)
    config['rangeOptions'].update(method='dual_rate', secondSampleRateMsps=5, searchMaxM=30)
    result = waveform.calculate_waveform(config, workers=1)
    assert result['analysis']['valid'] and result['range_recovery']['A']['status'] == 'inconsistent'
    path = tmp_path/'inconsistent-adc.json'; io.save_waveform_record(path, result)
    assert io.load_waveform_record(path)['report']['range_recovery'] == result['range_recovery']


def test_export_new_directory_and_existing_target_protection(single_report, tmp_path):
    out = tmp_path/'中文 导出'
    io.export_waveform_directory(out, single_report, plot='none')
    assert io.load_waveform_record(out/'waveform-record.json')['report'] == canonical(single_report)
    assert io.load_waveform_configuration(out/'parameters.json') == single_report['configuration']
    with (out/'A-doppler.csv').open(encoding='utf-8-sig', newline='') as stream: rows = list(csv.DictReader(stream))
    assert len(rows) == single_report['configuration']['params']['dopplerFft']
    assert float(rows[0]['velocity_kmh']) == pytest.approx(3.6*float(rows[0]['velocity_mps']))
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    with pytest.raises(FileExistsError): io.export_waveform_directory(out, single_report)
    assert before == {p.name: p.read_bytes() for p in out.iterdir()}


def test_failed_export_and_late_target_leave_no_partial_output(single_report, tmp_path, monkeypatch):
    out = tmp_path/'failed'
    def fail(*args, **kwargs): raise OSError('injected CSV failure')
    monkeypatch.setattr(io, '_csv', fail)
    with pytest.raises(OSError): io.export_waveform_directory(out, single_report, plot='none')
    assert list(tmp_path.iterdir()) == []
    original = io._write
    def create_late(path, document):
        original(path, document)
        if not out.exists(): out.mkdir(); (out/'keep.txt').write_bytes(b'other-work')
    monkeypatch.setattr(io, '_write', create_late)
    monkeypatch.setattr(io, '_csv', lambda *a, **k: None)
    with pytest.raises(FileExistsError): io.export_waveform_directory(out, single_report, plot='none')
    assert list(out.iterdir()) == [out/'keep.txt'] and (out/'keep.txt').read_bytes() == b'other-work'
    assert list(tmp_path.iterdir()) == [out]


def invoke(arguments, tmp_path, *, block_qt=False):
    env = os.environ.copy()
    env.update(PYTHONUTF8='1', MPLCONFIGDIR=str(tmp_path/'mpl'), MPLBACKEND='QtAgg',
               QT_QPA_PLATFORM='invalid-test-platform', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    if block_qt:
        bootstrap = '''import importlib.abc,sys,runpy
class BlockQt(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith(('PySide','PyQt','tkinter')):raise RuntimeError('GUI import blocked: '+fullname)
sys.meta_path.insert(0,BlockQt())
sys.argv=['arraylab.waveform_cli']+sys.argv[1:]
runpy.run_module('arraylab.waveform_cli',run_name='__main__')
'''
        command = [sys.executable, '-c', bootstrap, *map(str, arguments)]
    else:
        command = [sys.executable, '-m', 'arraylab.waveform_cli', *map(str, arguments)]
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', timeout=45)


def test_cli_json_csv_png_matches_saved_ab_report_without_qt(ab_report, tmp_path):
    path = tmp_path/'AB 参数.json'; io.save_waveform_configuration(path, ab_report['configuration'])
    out = tmp_path/'CLI 结果'
    result = invoke(['--waveform-config', path, '--output-dir', out, '--plot', 'all', '--workers', '1'], tmp_path, block_qt=True)
    assert result.returncode == 0, result.stderr
    actual = json.loads(result.stdout)
    assert actual['waveform'] == canonical(ab_report)
    assert io.load_waveform_record(out/'waveform-record.json')['report'] == actual['waveform']
    expected = {f'waveform-{name}-{view}.png' for name in actual['waveform']['spectra']
                for view in ('overview', 'phase', 'spectra', 'rd', 'range_recovery', 'pri_recovery')}
    assert {p.name for p in out.glob('*.png')} == expected
    assert all(p.stat().st_size > 1000 for p in out.glob('*.png'))
    assert all((out/f'{name}-range.csv').is_file() and (out/f'{name}-doppler.csv').is_file() for name in ('A', 'B'))


@pytest.mark.parametrize('arguments', [('--unknown',), ('--plot', 'all'), ('--workers', '9'),
                                     ('--workers', 'bad'), ('--plot', 'not-a-view')])
def test_cli_bad_flags_return_json_input_error(arguments, tmp_path):
    result = invoke(arguments, tmp_path)
    assert result.returncode == 2 and not result.stdout
    assert json.loads(result.stderr)['error']['type'] == 'input'


def test_cli_mutual_exclusion_and_existing_output_exit_code(tmp_path):
    path = tmp_path/'config.json'; io.save_waveform_configuration(path, small_configuration())
    result = invoke(['--config', path, '--waveform-config', path], tmp_path)
    assert result.returncode == 2 and json.loads(result.stderr)['error']['type'] == 'input'
    out = tmp_path/'existing'; out.mkdir(); (out/'keep.txt').write_bytes(b'keep')
    result = invoke(['--config', path, '--output-dir', out], tmp_path)
    assert result.returncode == 3 and not result.stdout
    assert json.loads(result.stderr)['error']['type'] == 'output_exists'
    assert (out/'keep.txt').read_bytes() == b'keep' and len(list(out.iterdir())) == 1


def test_cli_help_version_without_qt(tmp_path):
    for option in ('--help', '--version'):
        result = invoke([option], tmp_path, block_qt=True)
        assert result.returncode == 0 and not result.stderr and 'ArrayLab' in result.stdout
