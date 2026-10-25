from dataclasses import replace
import json
import numpy as np
import pytest

from arraylab.features.feed.branch_io import read_branch_delays_csv
from arraylab.features.linear.model import ArrayParameters
from arraylab.features.feed.model import FeedSettings, with_branch_delays, save_feed_configuration
from test_cli import run_cli


def test_complete_branch_csv_reorders_by_identifier(tmp_path):
    path = tmp_path/'支路.csv'; path.write_text('section,delay_deg\nT2,360\nT1,45.5\n', encoding='utf-8-sig')
    assert read_branch_delays_csv(path, 2) == (45.5, 360.)


@pytest.mark.parametrize('body', [
    'section,delay_deg\nT1,45\n', 'section,delay_deg\nT1,45\nT1,0\n',
    'section,delay_deg\nT1,-1\nT2,0\n', 'section,delay_deg\nT1,nan\nT2,0\n',
    'section,delay_deg\nT1,3601\nT2,0\n', 'section,delay_deg\nT1,45,2\nT2,0\n',
    'section,delay_deg\nT01,45\nT2,0\n', 'section,length_mm\nT1,45\nT2,0\n',
])
def test_invalid_branch_csv_rejected(tmp_path, body):
    path=tmp_path/'bad.csv';path.write_text(body, encoding='utf-8')
    with pytest.raises(ValueError):read_branch_delays_csv(path,2)


def test_actual_headless_branch_delays_and_binding_reset(tmp_path):
    path=tmp_path/'支路.csv';path.write_text('section,delay_deg\nT1,45\nT2,120\n',encoding='utf-8')
    output=tmp_path/'结果'
    process=run_cli('--elements',2,'--feed-network','corporate','--feed-phase','none',
                    '--feed-branch-delays',path,'--output-dir',output,guard=True)
    assert process.returncode==0,process.stderr
    report=json.loads(process.stdout)['feed_network']
    assert report['schema_version']==3
    np.testing.assert_allclose([r['network_phase_deg'] for r in report['ports']],[-135.,-210.])
    np.testing.assert_allclose([r['output_phase_deg'] for r in report['ports']],[-135.,150.])
    assert 'matched_delay_deg' in (output/'feed-sections.csv').read_text(encoding='utf-8-sig').splitlines()[0]
    saved=output/'feed-config.json'
    assert run_cli('--feed-config',saved,'--elements',4,guard=True).returncode==2
    cleared=run_cli('--feed-config',saved,'--elements',4,'--clear-feed-branch-delays',guard=True)
    assert cleared.returncode==0,cleared.stderr
    assert json.loads(cleared.stdout)['feed_network']['settings']['branch_delays_deg']==[]
    assert run_cli('--feed-config',saved,'--feed-network','series',guard=True).returncode==2
    assert run_cli('--feed-branch-delays',path,guard=True).returncode==2
    assert run_cli('--feed-network','corporate','--feed-branch-delays',path,'--clear-feed-branch-delays',guard=True).returncode==2


def test_feed_configuration_can_rebind_explicit_new_csv(tmp_path):
    path=tmp_path/'old.json'
    save_feed_configuration(path,ArrayParameters(elements=2),with_branch_delays(FeedSettings(),2,[45.,0.]))
    csv=tmp_path/'new.csv';csv.write_text('section,delay_deg\n'+''.join(f'T{i},{i}\n' for i in range(1,7)),encoding='utf-8')
    process=run_cli('--feed-config',path,'--elements',4,'--feed-network','series','--feed-branch-delays',csv,guard=True)
    assert process.returncode==0,process.stderr
    settings=json.loads(process.stdout)['feed_network']['settings']
    assert settings['branch_delay_elements']==4 and settings['branch_delay_topology']=='series'
    assert settings['branch_delays_deg']==[1.,2.,3.,4.,5.,6.]
