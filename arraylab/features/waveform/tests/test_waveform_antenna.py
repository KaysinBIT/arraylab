"""Analytic geometry, reference-frequency and source-coordinate contracts."""
from copy import deepcopy
import math

import pytest

from arraylab.features.waveform import antenna as a
from arraylab.framework.parallel import ComputationCancelled


def test_virtual_channels_and_duplicate_multiplicity():
    result = a.calculate(a.geometry_preset('ula', 3, 4))
    assert result['channelCount'] == result['uniqueCount'] == 12
    assert [p['x'] for p in result['virtual']] == [i*.5 for i in range(12)]
    assert result['apertureX'] == 5.5
    assert result['beamwidthElDeg'] is None
    compact = a.calculate(a.geometry_preset('compact', 2, 3))
    assert [p['multiplicity'] for p in compact['virtual']] == [1, 2, 2, 1]
    assert compact['channelCount'] == 6 and compact['uniqueCount'] == 4


@pytest.mark.parametrize('spacing,steer', [(.5, 0), (100, 0), (100, 17.123)])
def test_two_element_analytic_beamwidth_independent_of_plot_grid(spacing, steer):
    result = a.calculate({'tx': [[0, 0]], 'rx': [[0, 0], [spacing, 0]], 'steerAzDeg': steer})
    offset = math.acos(10**(-3/20))/(math.pi*spacing)
    center = math.sin(math.radians(steer))
    expected = math.degrees(math.asin(center+offset)-math.asin(center-offset))
    assert result['beamwidthAzDeg'] == pytest.approx(expected, abs=1e-7)
    assert all(math.isfinite(p['y']) and p['y'] <= 0 for p in result['azimuth'])


def test_hidden_subgrid_boundary_and_planar_steering():
    result = a.calculate({'tx': [[-100, 0], [100, 0]], 'rx': [[-100, 0], [100, 0]]})
    expected = 2*math.degrees(math.asin(math.acos(math.sqrt(10**(-3/20)))/(math.pi*200)))
    assert result['beamwidthAzDeg'] == pytest.approx(expected, abs=1e-7)
    assert next(p['y'] for p in result['azimuth'] if p['x'] == .25) > -3
    result = a.calculate({**a.geometry_preset('planar', 4, 6), 'steerAzDeg': 25, 'steerElDeg': 15})
    assert max(result['azimuth'], key=lambda p: p['y'])['x'] == 25
    assert max(result['elevation'], key=lambda p: p['y'])['x'] == 15


def test_reference_frequency_preserves_physical_coordinates_without_input_mutation():
    geometry = {**a.geometry_preset('ula', 3, 4), 'referenceFrequencyGHz': 77}
    original = deepcopy(geometry)
    effective = a.effective_geometry(geometry, 79)
    for key in ('tx', 'rx'):
        for before, after in zip(geometry[key], effective[key]):
            for b, v in zip(before, after):
                assert v/79 == pytest.approx(b/77)
    assert geometry == original
    effective['tx'][0][0] = 100
    assert geometry == original
    del geometry['referenceFrequencyGHz']
    assert a.effective_geometry(geometry, 79) == geometry
    applied = a.applied_geometry({'valid': True, 'geometry': geometry}, 77)
    assert applied['referenceFrequencyGHz'] == 77
    assert applied['steerAzDeg'] == applied['steerElDeg'] == 0


@pytest.mark.parametrize('config', [None, {'tx': [], 'rx': [[0, 0]]},
                                   {'tx': [[math.nan, 0]], 'rx': [[0, 0]]},
                                   {'tx': [[101, 0]], 'rx': [[0, 0]]},
                                   {'tx': [[0, 0]], 'rx': [[0, 0]], 'steerAzDeg': 81}])
def test_invalid_geometry_has_no_nan_result(config):
    with pytest.raises(ValueError):
        a.calculate(config)


def test_cancel_and_flat_single_channel():
    with pytest.raises(ComputationCancelled):
        a.calculate(a.geometry_preset('ula', 3, 4), cancelled=lambda: True)
    result = a.calculate({'tx': [[0, 0]], 'rx': [[0, 0]]})
    assert all(p['y'] == 0 for p in result['azimuth'])
    assert result['beamwidthAzDeg'] is result['beamwidthElDeg'] is None
