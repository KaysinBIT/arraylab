"""ArrayLab adapter contracts affected by the EquationLab 1.0.1 repair."""
from copy import deepcopy
import csv
import io as stringio

import numpy as np
import pytest

from arraylab.features.equation import service,io,presentation,reporting
from arraylab.features.equation.config import config_hash
from arraylab.features.equation.detection import calibration_fingerprint
from arraylab.shared.simulation_io import read_npz
from .test_integration import small


def test_old_record_preserves_inputs_even_if_new_execution_memory_guard_rejects(tmp_path,monkeypatch):
    p=small();result,arrays=service.execute('budget',p)
    old=deepcopy(result);old['config']['experiment']['memory_mib']=64
    old['config_hash']=config_hash(old['config']);old['algorithm_version']='1.0.0';old['source']['version']='1.0.0'
    before=deepcopy(old)
    assert service.normalize_record_configuration(old['config'])==old['config']
    with pytest.raises(ValueError,match='内存'):service.normalize_configuration(old['config'])
    assert service.record_compatibility(old)['requires_recompute']
    monkeypatch.setattr(service,'execute',lambda *a,**k:pytest.fail('History export must not calculate'))
    monkeypatch.setattr(reporting,'calculate',lambda *a,**k:pytest.fail('No replacement budget'),raising=False)
    target=io.export_result(tmp_path/'legacy',old,arrays)
    restored,loaded=read_npz(target/'arrays.npz')
    assert restored==old==before and not loaded
    text=(target/'report.html').read_text(encoding='utf-8')
    assert '1.0.0计算快照' in text


@pytest.mark.parametrize('algorithm',[None,'1.0.0','1.0.2'])
def test_old_or_unknown_calibration_cannot_be_applied_even_with_matching_input_fingerprint(algorithm):
    p=small();p['detection']['pfa']=.25
    record={'kind':'calibration','ready':True,'verified':False,'trials':80,'pfa':.25,'factor':3.,
            'calibration_hash':calibration_fingerprint(p),'algorithm_version':algorithm}
    with pytest.raises(ValueError,match='旧版或未知'):service.apply_calibration(p,record)


def test_new_source_identity_is_explicit_and_configuration_schema_is_unchanged():
    p=small();r,_=service.execute('budget',p)
    assert service.record_compatibility(r)['current']
    assert r['algorithm_version']==r['source']['version']=='1.0.1'
    assert r['source']['archive_sha256']=='1743985dae7a5ad60caaad72175251ece653cb844a3b2098e15870579cc463be'
    assert r['config']['schema_version']==1 and r['config']==p


def test_native_statistics_exposes_primary_background_and_pointwise_scope():
    p=small();target=deepcopy(p['scene']['targets'][0]);p['scene']['targets'][0]['rcs_m2']=1e-12
    p['scene']['targets'].append(target)
    result,arrays=service.execute('statistics',p)
    headers,rows,note=presentation.table(result,arrays,'statistics')
    assert any('仅首目标移除背景Pfa'==row[0] for row in rows)
    assert '逐项' in note and '严格验收' in note
    assert result['background_pfa']['trials']==p['experiment']['trials']


def test_saved_snr_comparison_uses_the_power_ratio_reference_and_retains_total_energy():
    result={'kind':'snr','energy_reference_snr_db':20.,'power_ratio_reference_snr_db':17.,
            'theoretical_snr_db':17.,'measured_snr_db':17.,'error_db':0.,'discrete_loss_db':0.,'sinr_db':16.,
            'noise_trials':4,'noise_calibration_passed':False,'energy_equation_passed':False}
    _,rows,note=presentation.table(result,{},'snr')
    assert rows[0][1]=='20' and rows[1][1]=='17' and '非相干检测增益' in note
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure=Figure();FigureCanvasAgg(figure);presentation.draw(figure,result,{},'snr')
    np.testing.assert_allclose([p.get_height() for p in figure.axes[0].patches],[17,17,17])


def test_detection_table_and_csv_label_total_cell_power_without_changing_legacy_value():
    result={'kind':'simulation','detections':[{'range_m':5.,'velocity_mps':1.,'power_dbw':-80.,'cell_snr_db':3.}]}
    columns,rows,note=presentation.table(result,{},'detections')
    assert '总功率' in columns[-1] and rows[0][-1]=='3' and '不是' in note
    output=list(csv.reader(stringio.StringIO(reporting.result_csv(result).lstrip('\ufeff'))))
    assert output[0][-1]=='cell_power_to_noise_db' and output[1][-1]=='3.0'


@pytest.mark.parametrize('kind',['statistics','roc','distance'])
def test_legacy_statistics_never_relabel_old_pd_as_new_fixed_cut_probability(kind):
    metric={'estimate':1.,'lower':.9,'upper':1.,'count':10,'trials':10}
    result={'kind':kind,'algorithm_version':'1.0.0','validated':True,'pd':deepcopy(metric),'pfa':deepcopy(metric),'association':deepcopy(metric),
            'points':[{'range_m':25.,'threshold_db':10.,'pd':deepcopy(metric),'pfa':deepcopy(metric),'status':'passed'}]}
    original=deepcopy(result);headers,rows,note=presentation.table(result,{},'statistics')
    assert result==original and '旧版' in note and '严格验收：通过' not in note
    if kind=='statistics':
        assert rows[0][0]=='旧版Pd（原口径）' and rows[0][3]=='1'
    else:
        assert headers[1]=='旧版Pd' and rows[0][1]=='1'
        output=list(csv.reader(stringio.StringIO(reporting.result_csv(result).lstrip('\ufeff'))))
        assert output[0][1]=='legacy_pd' and output[1][1]=='1.0'
        import matplotlib
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        with matplotlib.rc_context({'font.family':['Microsoft YaHei','DejaVu Sans']}):
            figure=Figure();FigureCanvasAgg(figure);presentation.draw(figure,result,{},'statistics')
        assert figure.axes[0].get_ylabel().startswith('旧版Pd')
