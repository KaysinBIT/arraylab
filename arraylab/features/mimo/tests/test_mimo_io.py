from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from arraylab.features.mimo.model import MIMOParameters, calculate_mimo
import arraylab.features.mimo.io as io


@pytest.fixture(scope='module')
def report():
    return calculate_mimo(MIMOParameters(map_size=81, step_deg=2), workers=1)


def rewrite_checksum(document):
    payload = {'report': document['report'], 'view': document['view']}
    document['sha256'] = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


@pytest.mark.parametrize('parameters', [MIMOParameters(map_size=81, step_deg=2),
    MIMOParameters(dimension='planar', tx_positions=((0,0),(0,1)),
                   rx_positions=((0,0),(.5,0)), theta_deg=20, phi_deg=30, cut_phi_deg=30, map_size=81, step_deg=1),
    MIMOParameters(coordinate_unit='mm', tx_positions=((0,0),(12,0)), rx_positions=((0,0),(3,0)))])
def test_configuration_roundtrip(parameters, tmp_path):
    target = tmp_path/'中文 参数.json'
    io.save_mimo_configuration(target, parameters)
    assert io.load_mimo_configuration(target) == parameters
    assert not io.is_mimo_record(target)


@pytest.mark.parametrize('view', ['layout','cut','map','channels','unique'])
def test_record_roundtrip_preserves_saved_report_without_field_evaluation(report, tmp_path, monkeypatch, view):
    import arraylab.features.mimo.model as core
    target = tmp_path/'完整 记录.json'
    io.save_mimo_record(target, report, view)
    before = target.read_bytes()
    monkeypatch.setattr(core, 'calculate_mimo', lambda *a, **k: pytest.fail('History must not calculate'))
    # Keep a deliberately changed but finite metric; loading does not replace
    # saved science with a fresh numerical run or substitute its own results.
    document = json.loads(before)
    document['report']['metrics']['cut']['peak_sidelobe_db'] = -17.123456
    rewrite_checksum(document); target.write_text(json.dumps(document), encoding='utf-8')
    loaded = io.load_mimo_record(target)
    assert loaded == {'report': document['report'], 'view': view}
    assert io.is_mimo_record(target)
    assert target.read_bytes() == json.dumps(document).encode('utf-8')


def test_single_channel_and_planar_reports_roundtrip(tmp_path):
    for name, p in [('single', MIMOParameters(tx_positions=((0,0),), rx_positions=((0,0),), map_size=81, step_deg=5)),
                    ('planar', MIMOParameters(dimension='planar', tx_positions=((0,0),(1,0)),
                       rx_positions=((0,0),(0,.5)), theta_deg=20, phi_deg=35, cut_phi_deg=35, map_size=81, step_deg=2))]:
        report = calculate_mimo(p, workers=1)
        path = tmp_path/(name + '.json')
        io.save_mimo_record(path, report)
        assert io.load_mimo_record(path)['report'] == json.loads(json.dumps(report))


@pytest.mark.parametrize('change', ['digest','pair','count','parameters','shape','nan','outside_mask','inside_mask','sampling','view'])
def test_corrupt_or_inconsistent_record_rejected_without_mutation(report, tmp_path, change):
    target = tmp_path/'bad.json'
    io.save_mimo_record(target, report)
    doc = json.loads(target.read_text(encoding='utf-8'))
    if change == 'digest': doc['sha256'] = '0'*64
    elif change == 'pair': doc['report']['geometry']['pairs'][0]['rx'] = 2
    elif change == 'count': doc['report']['counts']['tx_rx_channels'] += 1
    elif change == 'parameters': del doc['report']['parameters']['frequency_ghz']
    elif change == 'shape': doc['report']['patterns']['cut']['virtual_db'].pop()
    elif change == 'nan': doc['report']['patterns']['cut']['virtual_db'][0] = float('nan')
    elif change == 'outside_mask': doc['report']['patterns']['map']['virtual_db'][0][0] = 0
    elif change == 'inside_mask': doc['report']['patterns']['map']['virtual_db'][40][40] = None
    elif change == 'sampling': doc['report']['patterns']['cut']['angles_deg'][1] += .1
    elif change == 'view': doc['view'] = 'unknown'
    if change not in ('digest','nan'): rewrite_checksum(doc)
    target.write_text(json.dumps(doc), encoding='utf-8'); before = target.read_bytes()
    with pytest.raises(ValueError): io.load_mimo_record(target)
    assert target.read_bytes() == before


@pytest.mark.parametrize('text', ['{"format":"arraylab.mimo","format":"arraylab.mimo"}',
    '{"format":"arraylab.mimo","schema_version":1,"parameters":{"frequency_ghz":NaN}}',
    '{"format":"arraylab.mimo","schema_version":true,"parameters":{}}',
    '{"format":"arraylab.mimo","schema_version":99,"parameters":{}}',
    '{"format":"arraylab.mimo","schema_version":1,"parameters":{"unexpected":3}}', '[1,2,3]'])
def test_malformed_configuration_rejected(tmp_path, text):
    target = tmp_path/'bad.json'; target.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError): io.load_mimo_configuration(target)


def test_probe_uses_root_format_and_bounded_reader(tmp_path, monkeypatch):
    target = tmp_path/'arbitrary-name.data'
    target.write_text('{"text":"arraylab.mimo.record"}', encoding='utf-8')
    assert not io.is_mimo_record(target)
    target.write_text('{"format":"arraylab.mimo.record"}', encoding='utf-8')
    assert io.is_mimo_record(target)
    monkeypatch.setattr(io, 'MAX_FILE_BYTES', 8)
    with pytest.raises(ValueError, match='上限'): io.is_mimo_record(target)


def test_atomic_record_replace_failure_preserves_previous_file(report, tmp_path, monkeypatch):
    target = tmp_path/'old.json'; target.write_bytes(b'previous-result')
    def fail(*args): raise OSError('injected disk failure')
    monkeypatch.setattr(io.os, 'replace', fail)
    with pytest.raises(OSError): io.save_mimo_record(target, report)
    assert target.read_bytes() == b'previous-result'
    assert list(tmp_path.iterdir()) == [target]


def test_export_files_match_saved_result(report, tmp_path):
    directory = io.export_mimo_directory(tmp_path/'中文 输出', report, plot='none')
    assert {p.name for p in directory.iterdir()} == {'summary.json','parameters.json','mimo-record.json',
                                                     'tx.csv','rx.csv','pairs.csv','unique.csv','cut.csv','map.csv'}
    assert io.load_mimo_record(directory/'mimo-record.json')['report'] == json.loads(json.dumps(report))
    assert io.load_mimo_configuration(directory/'parameters.json') == MIMOParameters(**report['parameters'])
    import csv
    with (directory/'pairs.csv').open(encoding='utf-8-sig', newline='') as stream:
        pairs = list(csv.DictReader(stream))
    assert len(pairs) == 8
    assert [(int(row['tx']),int(row['rx'])) for row in pairs] == [(t,r) for t in (1,2) for r in (1,2,3,4)]
    cut = np.genfromtxt(directory/'cut.csv', delimiter=',', skip_header=1, encoding='utf-8-sig')
    np.testing.assert_array_equal(cut[:,3], report['patterns']['cut']['virtual_db'])


def test_existing_export_and_late_failure_do_not_publish_partial_result(report, tmp_path, monkeypatch):
    target = tmp_path/'export'; target.mkdir(); (target/'sentinel').write_text('keep')
    with pytest.raises(FileExistsError): io.export_mimo_directory(target, report)
    assert (target/'sentinel').read_text() == 'keep'
    def fail(directory, *args):
        (directory/'partial.csv').write_text('partial')
        raise RuntimeError('injected plotting failure')
    monkeypatch.setattr(io, '_write_exports', fail)
    with pytest.raises(RuntimeError): io.export_mimo_directory(tmp_path/'new', report)
    assert not (tmp_path/'new').exists()
    assert not list(tmp_path.glob('.*.mimo-*'))


def test_destination_created_during_export_is_preserved(report, tmp_path, monkeypatch):
    target = tmp_path/'race'
    def race(directory, *args):
        (directory/'partial.csv').write_text('partial')
        target.mkdir(); (target/'sentinel').write_text('other-process')
    monkeypatch.setattr(io, '_write_exports', race)
    with pytest.raises(FileExistsError): io.export_mimo_directory(target, report, 'none')
    assert (target/'sentinel').read_text() == 'other-process'
    assert not list(tmp_path.glob('.*.mimo-*'))
