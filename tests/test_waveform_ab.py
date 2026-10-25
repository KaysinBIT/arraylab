"""Physical scheduling and profile-isolation checks for new A/B support."""
from copy import deepcopy
import json
import math

import numpy as np
import pytest

from arraylab import waveform_core as core, waveform_ab as ab


def profiles(mode='tdm'):
    a = {**core.BASE, 'mode': mode, 'targetVelocity': 1}
    b = {**a, 'slopeMHzUs': 15, 'rampUs': 42, 'idleUs': 12}
    return a, b


def test_disabled_ab_preserves_single_profile_and_ignores_invalid_b():
    a, _ = profiles()
    r = ab.calculate(a, {'samples': -1})
    expected = core.calculate(a)
    assert r['valid'] and r['profiles']['B'] is None
    assert r['profiles']['A']['result'] == expected
    assert r['metrics']['burstMs'] == expected['metrics']['burstMs']
    assert r['metrics']['switchCount'] == 0


def test_disabled_profile_does_not_leak_unused_nonfinite_options_into_json():
    r = ab.calculate(*profiles(), {'enabled': False, 'switchGapUs': math.nan})
    assert r['valid'] and r['options']['switchGapUs'] is None
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('mode', ['tdm', 'ddm'])
@pytest.mark.parametrize('layout', ['blocks', 'alternating'])
def test_ab_budget_and_revisit_follow_the_actual_physical_schedule(mode, layout):
    a, b = profiles(mode); gap = 5
    r = ab.calculate(a, b, {'enabled': True, 'layout': layout, 'switchGapUs': gap})
    assert r['valid']
    count = a['txCount'] if mode == 'tdm' else 1
    ta, tb = 38, 54; loops = 128
    switches = 2 * loops - 1 if layout == 'alternating' else 1
    expected_burst = loops * count * (ta + tb) + switches * gap
    assert r['metrics']['burstMs'] == expected_burst / 1000
    assert r['metrics']['switchCount'] == switches
    assert r['metrics']['frameSamples'] == count * a['rxCount'] * (a['loops'] * a['samples'] + b['loops'] * b['samples'])
    for name, p, period in [('A', a, ta), ('B', b, tb)]:
        interval = count * (ta + tb) + 2 * gap if layout == 'alternating' else count * period
        m = r['profiles'][name]['result']['metrics']
        assert m['txIntervalUs'] == interval
        assert m['velocityResolutionMps'] == pytest.approx((299792458 / (m['centerGHz'] * 1e9)) / (2 * loops * interval * 1e-6))
    assert r['profiles']['A']['result']['metrics']['rangeResolutionM'] != r['profiles']['B']['result']['metrics']['rangeResolutionM']


def test_switch_and_idle_times_never_add_sampling_bandwidth_or_duty_energy():
    a, b = profiles('ddm')
    original = ab.calculate(a, b, {'enabled': True, 'layout': 'alternating', 'switchGapUs': 0})
    delayed = ab.calculate(a, b, {'enabled': True, 'layout': 'alternating', 'switchGapUs': 30})
    assert delayed['valid']
    assert delayed['metrics']['burstMs'] > original['metrics']['burstMs']
    assert delayed['metrics']['txRampDutyPercent'] == original['metrics']['txRampDutyPercent']
    assert delayed['metrics']['frameSamples'] == original['metrics']['frameSamples']
    for name in ('A', 'B'):
        for key in ('bandwidthMHz', 'observedBandwidthMHz', 'rangeResolutionM'):
            assert delayed['profiles'][name]['result']['metrics'][key] == original['profiles'][name]['result']['metrics'][key]


def test_blocks_allow_different_counts_but_alternating_rejects_nonuniform_sampling():
    a, b = profiles(); b['loops'] = 64
    assert ab.calculate(a, b, {'enabled': True, 'layout': 'blocks'})['valid']
    r = ab.calculate(a, b, {'enabled': True, 'layout': 'alternating'})
    assert not r['valid'] and any(c['key'] == 'loops' for c in r['checks'])


def test_frame_return_switch_is_reserved_separately():
    a, b = profiles('ddm'); gap = 5
    bare = (a['loops'] * 38 + b['loops'] * 54 + gap) / 1000
    a['frameMs'] = b['frameMs'] = bare
    r = ab.calculate(a, b, {'enabled': True, 'switchGapUs': gap})
    checks = {c['key']: c['status'] for c in r['checks']}
    assert checks['combinedFrame'] == 'pass' and checks['frameReturnGap'] == 'fail'


@pytest.mark.parametrize('options', [{'enabled': 1}, {'enabled': True, 'layout': 'wrong'},
                                      {'enabled': True, 'switchGapUs': -1},
                                      {'enabled': True, 'switchGapUs': math.inf},
                                      {'enabled': True, 'switchGapUs': 10**1000},
                                      {'enabled': True, 'unknown': 1}])
def test_bad_options_return_json_safe_failure(options):
    r = ab.calculate(*profiles(), options)
    assert not r['valid'] and r['errors']
    json.dumps(r, allow_nan=False)


def test_shared_profile_parameters_cannot_silently_diverge():
    a, b = profiles(); b['txCount'] = 4
    r = ab.calculate(a, b, {'enabled': True})
    assert not r['valid'] and any(c['key'] == 'shared.txCount' for c in r['checks'])


def test_preview_uses_real_timestamps_and_each_profile_own_ddm_index():
    a, b = profiles('ddm')
    r = ab.calculate(a, b, {'enabled': True, 'layout': 'alternating', 'switchGapUs': 5})
    events = r['previewEvents']
    assert [(e['profile'], e['chirpIndex']) for e in events[:4]] == [('A', 0), ('B', 0), ('A', 1), ('B', 1)]
    assert [e['startUs'] for e in events[:4]] == [0, 43, 102, 145]
    assert all(e['txIndex'] is None for e in events)
    assert len(events) <= 12
    # A's long chronological prefix does not hide B's separately timestamped group.
    block = ab.calculate(a, b, {'enabled': True, 'layout': 'blocks', 'switchGapUs': 5})
    assert {e['profile'] for e in block['previewEvents']} == {'A'}
    assert block['previewGroups'][1]['startUs'] == 128 * 38 + 5
    assert all(e['profile'] == 'B' for e in block['previewGroups'][1]['events'])


def test_ddm_ab_spectrum_uses_profile_specific_revisit_interval():
    a, b = profiles('ddm');a['window'] = b['window'] = 'rect';a['targetVelocity'] = b['targetVelocity'] = 0
    r = ab.calculate(a, b, {'enabled': True, 'layout': 'alternating'})
    for name in ('A', 'B'):
        p = r['profiles'][name]
        s = core.simulate(p['params'], p['acquisition'])
        y = np.array([row['y'] for row in s['dopplerProfile']])
        np.testing.assert_allclose(y[[0, 64, 96]], 0, atol=1e-10)
        assert y[32] <= -150


def test_profile_edit_does_not_mutate_input_or_other_profile_and_preview_is_bounded():
    a, b = profiles(); original = deepcopy((a, b))
    r = ab.calculate(a, b, {'enabled': True})
    r['profiles']['B']['params']['slopeMHzUs'] = 22
    assert (a, b) == original
    assert r['profiles']['A']['params']['slopeMHzUs'] == 12
    preview = ab.sequence_preview(a, b, {'enabled': True}, {'rounds': 999})
    assert preview['valid'] and len(preview['previewEvents']) <= 12
    assert all(len(group['events']) <= 6 for group in preview['previewGroups'])
