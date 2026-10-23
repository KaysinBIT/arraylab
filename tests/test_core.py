import math

import numpy as np
import pytest
from scipy.integrate import quad

from arraylab.core import ArrayParameters, InputError, calculate, normalized_field_u
from arraylab.dataio import read_pattern, write_pattern


def direct_sum(u, p):
    x = (np.arange(p.elements) - (p.elements - 1) / 2) * p.spacing_lambda
    return np.exp(2j * np.pi * x * (u - np.sin(np.deg2rad(p.scan_deg)))).sum() / p.elements


@pytest.mark.parametrize("n,d,scan", [(1, .5, 0), (5, .4, 60), (8, .5, 0), (12, 1.2, -35), (20, .5, 0), (200, 10, 90)])
def test_field_matches_independent_complex_sum(n, d, scan):
    p = ArrayParameters(elements=n, spacing=d, scan_deg=scan)
    u = np.linspace(-1, 1, 201)
    np.testing.assert_allclose(normalized_field_u(u, p), [direct_sum(v, p) for v in u], atol=3e-12)


@pytest.mark.parametrize("n,d,scan", [(1, .3, 20), (5, .4, 60), (8, .5, 0), (12, 1.2, -35), (20, .5, 0)])
def test_directivity_against_independent_spherical_quadrature(n, d, scan):
    p = ArrayParameters(elements=n, spacing=d, scan_deg=scan)
    integral, error = quad(lambda u: abs(direct_sum(u, p)) ** 2, -1, 1, epsabs=1e-11, limit=600)
    expected = 10 * math.log10(2 / integral)
    assert error < 1e-8
    assert calculate(p).directivity_dbi == pytest.approx(expected, abs=1e-9)


def test_manual_validation_p48_and_p49():
    # Original manual values are rounded: 7.1 dB and 13.0 dB.
    assert calculate(ArrayParameters(elements=5, spacing=.4, scan_deg=60)).directivity_dbi == pytest.approx(7.1, abs=.05)
    assert calculate(ArrayParameters(elements=20)).directivity_dbi == pytest.approx(13.0, abs=.05)


def test_n_halfwave_directivity_exact_and_beamwidth():
    p = ArrayParameters(elements=8)
    r = calculate(p)
    assert r.directivity_dbi == pytest.approx(10 * math.log10(8), abs=1e-12)
    assert r.peaks_deg == (0,)
    assert 12.7 < r.beamwidth_deg < 12.9
    for angle in r.half_power_deg:
        assert abs(direct_sum(np.sin(np.deg2rad(angle)), p)) ** 2 == pytest.approx(.5, abs=1e-11)


def test_boundary_and_multiple_lobes_have_no_misleading_width():
    for p in [ArrayParameters(elements=1), ArrayParameters(spacing=1), ArrayParameters(scan_deg=90), ArrayParameters(spacing=.01)]:
        r = calculate(p)
        assert r.beamwidth_deg is None
        assert r.notices
    assert calculate(ArrayParameters(spacing=1)).peaks_deg == (-90, 0, 90)


def test_metric_independent_of_plot_sampling_and_scan_sign():
    a = calculate(ArrayParameters(scan_deg=23.45, step_deg=5))
    b = calculate(ArrayParameters(scan_deg=23.45, step_deg=.01))
    c = calculate(ArrayParameters(scan_deg=-23.45, step_deg=.01))
    assert a.beamwidth_deg == b.beamwidth_deg
    assert a.directivity_dbi == b.directivity_dbi
    assert a.peaks_deg[0] == pytest.approx(23.45)
    assert c.directivity_dbi == pytest.approx(b.directivity_dbi)
    np.testing.assert_allclose(b.pattern_db, c.pattern_db[::-1], atol=1e-5)


def test_units_are_equivalent_and_sampling_includes_endpoints():
    p = ArrayParameters(spacing=.4, frequency_ghz=7.5, step_deg=.7)
    mm = ArrayParameters(spacing=p.spacing_mm, spacing_unit="mm", frequency_ghz=7.5, step_deg=.7)
    r, s = calculate(p), calculate(mm)
    np.testing.assert_allclose(r.pattern_db, s.pattern_db, atol=1e-8)
    assert r.angles_deg[0] == -90
    assert r.angles_deg[-1] == 90
    assert np.all(np.diff(r.angles_deg) > 0)


@pytest.mark.parametrize("kwargs", [
    {"elements": 0}, {"elements": 201}, {"elements": 2.1}, {"elements": True},
    {"frequency_ghz": 0}, {"frequency_ghz": float("nan")}, {"spacing": 0},
    {"spacing": 11}, {"scan_deg": 91}, {"step_deg": .001}, {"spacing_unit": "cm"},
])
def test_bad_inputs(kwargs):
    with pytest.raises(InputError):
        calculate(ArrayParameters(**kwargs))


def test_data_roundtrip_unicode_path(tmp_path):
    r = calculate(ArrayParameters(scan_deg=30))
    path = tmp_path / "中文 路径.dat"
    write_pattern(path, r.angles_deg, r.pattern_db, r.phase_deg)
    theta, db, phase = read_pattern(path)
    np.testing.assert_allclose(theta, r.angles_deg, atol=1e-9)
    np.testing.assert_allclose(db, r.pattern_db, atol=1e-8)
    np.testing.assert_allclose(phase, r.phase_deg)
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("contents", ["0 0\n0 -1", "1 0\n0 -1", "0 NaN\n1 0", "angle db\n0 0", "0 1 2 3\n1 2 3 4"])
def test_bad_imports(tmp_path, contents):
    path = tmp_path / "bad.dat"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError):
        read_pattern(path)
