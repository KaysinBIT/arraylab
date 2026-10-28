"""Analytic half-power references independent of common complex field scale."""
import numpy as np
import pytest

from arraylab.shared.plotting.metrics import spherical_beamwidths


@pytest.mark.parametrize('scale',[1.,1e-200,1e200,1e-308,1e308])
@pytest.mark.parametrize('phase',[1.,(1+1j)/np.sqrt(2)])
def test_cosine_field_has_analytic_90_degree_width_at_every_finite_scale(scale,phase):
    # |cos(a)|²=1/2 at a=±45 degrees, independently of units and common phase.
    result=spherical_beamwidths(lambda xyz:scale*phase*xyz[:,2],0,0)
    assert result['orthogonal_beamwidths_deg']==pytest.approx([90.,90.],abs=1e-8)
    for cut in result['cuts']:
        assert cut['status']=='bounded_selected_lobe'
        assert cut['half_power_edges_deg']==pytest.approx([-45.,45.],abs=1e-8)


@pytest.mark.parametrize('scale',[1.,1e-200,1e200])
def test_half_wavelength_pair_matches_independent_cosine_reference(scale):
    # Two equal sources at x=±lambda/4: F=cos(pi*u/2); sin(theta_3dB)=1/2.
    result=spherical_beamwidths(lambda xyz:scale*np.cos(np.pi*xyz[:,0]/2),0,0,
                               flat_cuts=('transverse',))
    assert result['orthogonal_beamwidths_deg'][0]==pytest.approx(60.,abs=1e-8)
    assert result['orthogonal_beamwidths_deg'][1] is None
