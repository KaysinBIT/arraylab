"""Pointing reports must distinguish requested angles from electrical output."""
import json
import math

import numpy as np
import pytest

from arraylab.features.linear.model import ArrayParameters
from arraylab.features.feed.model import FeedSettings, design_feed
from arraylab.features.planar.model import PlanarParameters, design_planar
from arraylab.shared.numerics.pointing import angular_separation
from test_cli import run_cli


@pytest.mark.parametrize("topology", ["corporate", "series", "symmetric_series"])
@pytest.mark.parametrize("angle", [-37.25, 0, 56.5])
def test_line_pointing_and_independent_port_phase(topology, angle):
    p = ArrayParameters(elements=8, spacing=.45, scan_deg=angle, step_deg=2)
    r = design_feed(p, FeedSettings(topology=topology))
    summary = r["electrical"]["pointing"]
    assert summary["requested_theta_deg"] == summary["target_theta_deg"] == angle
    assert summary["output_cut_peak_deg"] == pytest.approx(angle, abs=1e-7)
    assert summary["output_cut_offset_deg"] == pytest.approx(0, abs=1e-7)
    assert summary["status"] == "unique_cut_peak"
    expected = np.exp(-2j*np.pi*(np.arange(8)-3.5)*.45*math.sin(math.radians(angle)))
    actual = np.array([complex(port["output_real"], port["output_imag"]) for port in r["ports"]])
    np.testing.assert_allclose(actual/abs(actual), expected, atol=1e-12)


def test_modes_do_not_report_request_as_output():
    p = ArrayParameters(scan_deg=31)
    none = design_feed(p, FeedSettings(phase_mode="none"))["electrical"]["pointing"]
    equal = design_feed(p, FeedSettings(phase_mode="equalize"))["electrical"]["pointing"]
    assert none["target_theta_deg"] == 31 and not none["compensation_enabled"]
    assert none["output_cut_peak_deg"] == pytest.approx(0, abs=1e-8)
    assert none["output_cut_offset_deg"] == pytest.approx(-31, abs=1e-8)
    assert equal["requested_theta_deg"] == 31 and equal["target_theta_deg"] == 0
    assert equal["output_cut_offset_deg"] == pytest.approx(0, abs=1e-8)


@pytest.mark.parametrize("p,s,status", [
    (ArrayParameters(elements=1), FeedSettings(), "undefined_cut_peak"),
    (ArrayParameters(), FeedSettings(failed_ports=tuple(range(1,9))), "undefined_cut_peak"),
    (ArrayParameters(spacing=1), FeedSettings(), "multiple_cut_peaks"),
    (ArrayParameters(scan_deg=90), FeedSettings(), "multiple_cut_peaks"),
])
def test_no_fabricated_unique_line_angle(p, s, status):
    summary = design_feed(p,s)["electrical"]["pointing"]
    assert summary["status"] == status
    assert summary["output_cut_peak_deg"] is None and summary["output_cut_offset_deg"] is None
    if status == "multiple_cut_peaks": assert len(summary["output_cut_peak_angles_deg"]) >= 2


def test_custom_phase_changes_output_without_changing_requested_angle():
    # Independently impose an additional phase ramp that steers 10 degrees to 25.
    x = (np.arange(8)-3.5)*.5
    offsets = -360*x*(math.sin(math.radians(25))-math.sin(math.radians(10)))
    s = FeedSettings(weighting="custom", custom_amplitudes=(1,)*8, custom_phases_deg=tuple(offsets))
    summary = design_feed(ArrayParameters(scan_deg=10),s)["electrical"]["pointing"]
    assert summary["target_theta_deg"] == 10
    assert summary["output_cut_peak_deg"] == pytest.approx(25, abs=1e-7)
    assert summary["output_cut_offset_deg"] == pytest.approx(15, abs=1e-7)


@pytest.mark.parametrize("topology", ["corporate", "series", "symmetric_series"])
@pytest.mark.parametrize("geometry", ["rectangular", "arbitrary"])
def test_planar_pointing_actual_coordinates(topology, geometry):
    p = PlanarParameters(geometry=geometry, nx=2, ny=2, scan_theta_deg=42, scan_phi_deg=-67,
                         coordinates=((-0.5,-0.1),(-0.1,0.4),(0.1,-0.3),(0.5,0.2)) if geometry=="arbitrary" else ())
    r = design_planar(p, FeedSettings(topology=topology))
    summary = r["electrical"]["pointing"]
    assert summary["target_theta_deg"] == 42 and summary["target_phi_deg"] == -67
    assert summary["representative_peak_separation_deg"] == pytest.approx(0, abs=1e-8)
    assert summary["peak_is_not_claimed_unique"] is True
    assert summary["peak_search_method"] == "coherent_bound"
    direction = np.array([math.sin(math.radians(42))*math.cos(math.radians(-67)),
                          math.sin(math.radians(42))*math.sin(math.radians(-67))])
    ports = r["network"]["ports"]
    xy = np.array([[port["x_lambda"],port["y_lambda"]] for port in ports])
    actual = np.array([complex(port["output_real"],port["output_imag"]) for port in ports])
    expected = np.exp(-2j*np.pi*((xy-xy.mean(axis=0))@direction))
    # A common phase is irrelevant; compare inter-port phase independently.
    np.testing.assert_allclose((actual/abs(actual))/(actual[0]/abs(actual[0])), expected/expected[0], atol=1e-12)


@pytest.mark.parametrize("theta_a,phi_a,theta_b,phi_b,expected", [
    (0,None,0,170,0), (90,179,90,-179,2), (90,0,90,180,180), (20,60,50,60,30),
])
def test_spherical_distance_pole_and_seam(theta_a,phi_a,theta_b,phi_b,expected):
    assert angular_separation(theta_a,phi_a,theta_b,phi_b) == pytest.approx(expected, abs=1e-12)


def test_planar_equalize_and_failed_output():
    p = PlanarParameters(nx=2, ny=2, scan_theta_deg=35, scan_phi_deg=130)
    equal = design_planar(p, FeedSettings(phase_mode="equalize"))["electrical"]["pointing"]
    assert equal["requested_theta_deg"] == 35 and equal["target_theta_deg"] == 0
    assert equal["target_phi_deg"] is None and equal["output_representative_phi_deg"] is None
    assert equal["representative_peak_separation_deg"] == pytest.approx(0, abs=1e-8)
    failed = design_planar(p, FeedSettings(failed_ports=(1,2,3,4)))["electrical"]["pointing"]
    assert failed["output_representative_theta_deg"] is None
    assert failed["representative_peak_separation_deg"] is None


def test_quantized_report_keeps_peak_estimate_identity():
    p = PlanarParameters(nx=2, ny=2, scan_theta_deg=28, scan_phi_deg=51)
    r = design_planar(p, FeedSettings(phase_bits=2))
    summary = r["electrical"]["pointing"]; m = r["electrical"]["patterns"]["output"]["metrics"]
    assert summary["peak_search_method"] == m["peak_search_method"] == "disk_grid_multistart_estimate"
    assert summary["output_representative_theta_deg"] == m["representative_peak_theta_deg"]
    assert summary["representative_peak_separation_deg"] > .01
    assert summary["peak_is_not_claimed_unique"] is True


def test_headless_pointing_and_export(tmp_path):
    line = run_cli("--feed-network", "series", "--scan-deg", "-27", "--feed-phase", "scan", "--output-dir", str(tmp_path/"line"))
    assert line.returncode == 0, line.stderr
    r = json.loads(line.stdout)["feed_network"]
    assert r["electrical"]["pointing"]["output_cut_peak_deg"] == pytest.approx(-27, abs=1e-7)
    assert json.loads((tmp_path/"line/feed-network.json").read_text(encoding="utf-8")) == r
    planar = run_cli("--planar", "--nx", "2", "--ny", "2", "--scan-theta", "35", "--scan-phi", "140", "--feed-phase", "scan")
    assert planar.returncode == 0, planar.stderr
    summary = json.loads(planar.stdout)["planar"]["electrical"]["pointing"]
    assert summary["target_phi_deg"] == 140 and summary["representative_peak_separation_deg"] < 1e-7
