"""Relative port amplitudes must not change with their common unit scale."""
import math
import numpy as np
import pytest

from arraylab.features.linear.model import ArrayParameters
from arraylab.features.feed.model import FeedSettings,synthesize_impedance
from arraylab.features.feed.excitation import electrical_excitation


@pytest.mark.parametrize('scale',[1.,1e-200,1e200])
def test_equal_split_uses_hand_calculated_quarter_wave_impedance(scale):
    p=ArrayParameters(elements=2)
    r=synthesize_impedance(p,FeedSettings(),amplitude_override=[scale,scale])
    assert [x['power_fraction'] for x in r['ports']]==pytest.approx([.5,.5])
    assert [x['line_ohm'] for x in r['sections']]==pytest.approx([50*math.sqrt(2)]*2)


@pytest.mark.parametrize('topology',['corporate','series','symmetric_series'])
@pytest.mark.parametrize('scale',[1.,1e-200,1e200])
def test_weighted_power_and_scan_phases_ignore_common_amplitude_scale(topology,scale):
    p=ArrayParameters(elements=4,scan_deg=27)
    s=FeedSettings(topology=topology)
    values=np.array([1.,2.,2.,1.])*scale
    expected=np.array([1.,4.,4.,1.])/10
    r=synthesize_impedance(p,s,amplitude_override=values)
    target,raw,output,metrics=electrical_excitation(p,s,r['ports'],amplitude_override=values)
    phase=-2*np.pi*.5*(np.arange(4)-1.5)*math.sin(math.radians(27))
    assert [x['power_fraction'] for x in r['ports']]==pytest.approx(expected)
    assert np.isfinite([x['line_ohm'] for x in r['sections']]).all()
    assert target==pytest.approx(np.sqrt(expected)*np.exp(1j*phase),abs=1e-14)
    assert output==pytest.approx(target,abs=1e-14)
    assert np.sum(abs(output)**2)==pytest.approx(1.)


@pytest.mark.parametrize('values',[[],[1.],[-1.,1.],[0.,1.],[float('nan'),1.],
                                 [1+1j,1.],[1e-200,1.]])
def test_unusable_port_amplitudes_rejected_before_partial_results(values):
    p=ArrayParameters(elements=2);s=FeedSettings()
    with pytest.raises(ValueError):synthesize_impedance(p,s,amplitude_override=values)
    ports=[{'network_phase_deg':-90.} for _ in range(2)]
    with pytest.raises(ValueError):electrical_excitation(p,s,ports,amplitude_override=values)
