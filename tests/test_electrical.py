from dataclasses import asdict, replace
import csv
import io
import json
import math

import numpy as np
import pytest
from scipy.integrate import quad

from arraylab.core import ArrayParameters, calculate
from arraylab.feed import (FeedSettings, LEGACY_SETTINGS, design_feed, synthesize_impedance,
                           load_feed_configuration, save_feed_configuration, export_feed_directory)
from arraylab.excitation import target_amplitudes, read_excitation_csv, excitation_csv, wrap_phase
from arraylab.complex_pattern import pattern_analysis, field_at_cycles
from test_feed import independent_circuit
from test_cli import run_cli


@pytest.mark.parametrize("topology", ["corporate", "series", "symmetric_series"])
@pytest.mark.parametrize("mode,scan", [("scan", -35), ("scan", 60), ("equalize", 35)])
def test_compensated_network_abcd_and_uniform_baseline(topology, mode, scan):
    p = ArrayParameters(elements=8, scan_deg=scan)
    s = FeedSettings(topology=topology, phase_mode=mode, input_ohm=75, junction_ohm=33, load_ohm=100)
    report = design_feed(p, s)
    impedance, power, voltages = independent_circuit(report)
    assert impedance == pytest.approx(75)
    output = []
    for port in report["ports"]:
        # Independent matched terminal-line transfer after reconstructed tree.
        value = voltages[f"P{port['port']}"] / math.sqrt(s.load_ohm) * np.exp(-1j * np.deg2rad(port["applied_delay_deg"]))
        output.append(value)
        assert value == pytest.approx(complex(port["output_real"], port["output_imag"]), abs=1e-12)
        assert 0 <= port["matched_line_wavelengths"] < 1
    expected = calculate(replace(p, scan_deg=0 if mode == "equalize" else scan))
    actual = report["electrical"]["patterns"]["output"]
    assert actual["metrics"]["directivity_dbi"] == pytest.approx(expected.directivity_dbi, abs=1e-9)
    assert actual["metrics"]["peak_angles_deg"] == pytest.approx(expected.peaks_deg, abs=1e-7)
    if expected.beamwidth_deg is not None:
        assert actual["metrics"]["half_power_beamwidth_deg"] == pytest.approx(expected.beamwidth_deg, abs=1e-7)
    # Compare fields, including nulls, rather than dB at floating-point zeros.
    np.testing.assert_allclose(10**(np.array(actual["pattern_db"]) / 20), 10**(expected.pattern_db / 20), atol=1e-11)


@pytest.mark.parametrize("n,d,scan", [(1,.5,0),(5,.4,60),(8,1,0),(8,.5,90),(200,10,30),(8,1e-6,0)])
def test_complex_solver_degenerate_cases(n, d, scan):
    p = ArrayParameters(elements=n, spacing=d, scan_deg=scan, step_deg=1)
    report = design_feed(p, FeedSettings(phase_mode="scan"))
    actual = report["electrical"]["patterns"]["output"]
    expected = calculate(p)
    assert actual["metrics"]["directivity_dbi"] == pytest.approx(expected.directivity_dbi, abs=1e-8)
    assert actual["metrics"]["peak_angles_deg"] == pytest.approx(expected.peaks_deg, abs=1e-7)
    if expected.beamwidth_deg is None:
        assert actual["metrics"]["half_power_beamwidth_deg"] is None


def test_arbitrary_phase_directionality_against_independent_quadrature():
    p = ArrayParameters(elements=6, spacing=.43, step_deg=3)
    w = np.array([.2+.1j, .7-.2j, 1, -.3+.5j, -.4j, .6+.3j])
    result = pattern_analysis(p, w, sum(abs(w)))
    positions = (np.arange(6) - 2.5) * p.spacing_lambda
    def power(u):
        real = sum((weight * np.exp(2j * np.pi * x * u)).real for weight, x in zip(w, positions))
        imag = sum((weight * np.exp(2j * np.pi * x * u)).imag for weight, x in zip(w, positions))
        return real**2 + imag**2
    integral = quad(power, -1, 1, epsabs=1e-12, epsrel=1e-12)[0] / 2
    peak = result["metrics"]["main_lobe_angle_deg"]
    expected = 10 * math.log10(power(math.sin(math.radians(peak))) / integral)
    assert result["metrics"]["directivity_dbi"] == pytest.approx(expected, abs=1e-9)
    dense = max(power(u) for u in np.linspace(-1, 1, 10001))
    assert power(math.sin(math.radians(peak))) >= dense * (1 - 1e-9)
    fine = pattern_analysis(replace(p, step_deg=.1), w, sum(abs(w)))
    assert fine["metrics"] == result["metrics"]


def test_close_spaced_cancellation_integral_remains_finite():
    result = pattern_analysis(ArrayParameters(elements=2, spacing=1e-6), [1, -1], 2)
    assert result["metrics"]["directivity_dbi"] == pytest.approx(10 * math.log10(3), abs=1e-6)


@pytest.mark.parametrize("n,attenuation", [(8,45),(17,60),(32,100)])
def test_chebyshev_against_closed_form_array_factor(n, attenuation):
    p = ArrayParameters(elements=n)
    weights = target_amplitudes(p, FeedSettings(weighting="chebyshev", sidelobe_db=attenuation))
    degree = n - 1
    beta = np.cosh(np.arccosh(10**(attenuation / 20)) / degree)
    t = np.linspace(-.5, .5, 301)
    coefficients = np.zeros(n); coefficients[-1] = 1
    expected = np.polynomial.chebyshev.chebval(beta * np.cos(np.pi * t), coefficients) / np.polynomial.chebyshev.chebval(beta, coefficients)
    actual = field_at_cycles(t, weights).real / sum(weights)
    np.testing.assert_allclose(actual, expected, atol=2e-12)


def test_taylor_reference_product_formula():
    n, nbar, sll = 16, 4, 30
    A = math.acosh(10**(sll / 20)) / math.pi
    sigma2 = nbar**2 / (A*A + (nbar-.5)**2)
    coefficients = []
    for m in range(1, nbar):
        numerator = (-1)**(m+1) * math.prod(1 - m*m / (sigma2*(A*A+(k-.5)**2)) for k in range(1, nbar))
        denominator = 2 * math.prod(1-m*m/(k*k) for k in range(1,nbar) if k != m)
        coefficients.append(numerator/denominator)
    expected = np.array([1 + 2*sum(value*math.cos(2*math.pi*m*(i-(n-1)/2)/n) for m,value in enumerate(coefficients,1)) for i in range(n)])
    expected /= max(expected)
    actual = target_amplitudes(ArrayParameters(elements=n), FeedSettings(weighting="taylor", sidelobe_db=sll, taylor_nbar=nbar))
    np.testing.assert_allclose(actual, expected, atol=1e-14)


@pytest.mark.parametrize("topology", ["corporate", "series", "symmetric_series"])
@pytest.mark.parametrize("weighting", ["chebyshev", "taylor"])
def test_weighted_power_division_matches_abcd(topology, weighting):
    p = ArrayParameters(elements=8)
    s = FeedSettings(topology=topology, weighting=weighting, phase_mode="scan")
    report = design_feed(p, s)
    z, powers, _ = independent_circuit(report)
    target = target_amplitudes(p, s)**2
    target /= sum(target)
    assert z == pytest.approx(50)
    for i in range(8): assert powers[f"P{i+1}"] == pytest.approx(target[i], abs=1e-12)


def test_custom_csv_roundtrip_and_phase_offsets(tmp_path):
    a, phase = (1,2,3,2,1), (-30,20,0,15,-45)
    path = tmp_path / "中文 激励.csv"
    path.write_text(excitation_csv(a, phase), encoding="utf-8-sig")
    assert read_excitation_csv(path, 5) == (a, phase)
    s = FeedSettings(weighting="custom", phase_mode="scan", custom_amplitudes=a, custom_phases_deg=phase)
    p = ArrayParameters(elements=5, scan_deg=20)
    result = design_feed(p, s)
    export_feed_directory(tmp_path / "out", result)
    a2, phase2 = read_excitation_csv(tmp_path / "out/feed-target-excitation.csv", 5)
    np.testing.assert_allclose(a2, np.array(a)/3)
    assert phase2 == phase
    reloaded = design_feed(p, replace(s, custom_amplitudes=a2, custom_phases_deg=phase2))
    np.testing.assert_allclose(reloaded["electrical"]["patterns"]["output"]["pattern_db"], result["electrical"]["patterns"]["output"]["pattern_db"], atol=1e-8)
    assert result["electrical"]["metrics"]["max_active_phase_error_deg"] < 1e-10


@pytest.mark.parametrize("text", [
    "element,amplitude,phase_deg\n1,1,0\n", "element,amplitude,phase_deg\n2,1,0\n1,1,0\n",
    "element,amplitude,phase_deg\n1,0,0\n2,1,0\n", "element,amplitude,phase_deg\n1,nan,0\n2,1,0\n",
    "amplitude,phase_deg\n1,0\n1,0\n", "element,amplitude,phase_deg\n1,1,0,0\n2,1,0\n",
])
def test_bad_excitation_csv_rejected(text, tmp_path):
    path = tmp_path / "bad.csv"; path.write_text(text)
    with pytest.raises(ValueError): read_excitation_csv(path, 2)


@pytest.mark.parametrize("changes", [
    {"phase_mode":"bad"}, {"weighting":"bad"}, {"weighting":"chebyshev","sidelobe_db":30},
    {"weighting":"taylor","taylor_nbar":9}, {"phase_bits":True}, {"phase_mode":"none","phase_bits":3},
    {"amplitude_error_db":-1}, {"phase_error_deg":float("nan")}, {"seed":1.5},
    {"failed_ports":[1,1]}, {"failed_ports":[9]}, {"failed_ports":[[]]},
    {"weighting":"custom","custom_amplitudes":[1]*8,"custom_phases_deg":[0]*7},
])
def test_invalid_electrical_settings(changes):
    with pytest.raises(ValueError): design_feed(ArrayParameters(), FeedSettings(**changes))


def test_asymmetric_target_rejected_for_symmetric_topology():
    settings = FeedSettings(topology="symmetric_series", weighting="custom", custom_amplitudes=[1,2,3,4], custom_phases_deg=[0]*4)
    with pytest.raises(ValueError, match="镜像"): design_feed(ArrayParameters(elements=4), settings)


def test_errors_seed_and_zero_limit():
    p = ArrayParameters(scan_deg=25)
    ideal = design_feed(p, FeedSettings(phase_mode="scan"))
    s = FeedSettings(phase_mode="scan", amplitude_error_db=.5, phase_error_deg=3, seed=19, failed_ports=(2,))
    a, b = design_feed(p,s), design_feed(p,s)
    assert a == b
    c = design_feed(p,replace(s,seed=20))
    assert a["ports"] != c["ports"]
    assert a["ports"][1]["output_amplitude"] == 0
    zero = design_feed(p,replace(s,amplitude_error_db=0,phase_error_deg=0,failed_ports=()))
    assert zero["electrical"]["patterns"] == ideal["electrical"]["patterns"]


def test_quantization_and_all_failed_are_explicit():
    p = ArrayParameters(scan_deg=17)
    report = design_feed(p,FeedSettings(phase_mode="scan",phase_bits=3))
    for port in report["ports"]:
        assert port["applied_delay_deg"] % 45 == 0
        assert abs(wrap_phase(port["output_phase_deg"]-port["target_phase_deg"])) <= 22.5+1e-9
    failed = design_feed(p,FeedSettings(phase_mode="scan",failed_ports=tuple(range(1,9))))
    out = failed["electrical"]["patterns"]["output"]
    assert out["metrics"]["directivity_dbi"] is None and out["metrics"]["peak_angles_deg"] == []
    assert all(value == -300 for value in out["pattern_db"])
    json.dumps(failed,allow_nan=False)


def test_legacy_v1_read_as_uncompensated_and_v2_roundtrip(tmp_path):
    path = tmp_path / "legacy.json"
    defaults = asdict(FeedSettings())
    path.write_text(json.dumps({"format":"arraylab.feed_impedance","schema_version":1,
                               "array_parameters":asdict(ArrayParameters()),"settings":{k:defaults[k] for k in LEGACY_SETTINGS}}))
    p,s = load_feed_configuration(path)
    assert s.phase_mode == "none" and s.weighting == "uniform"
    s = replace(s,phase_mode="scan",weighting="taylor",phase_bits=5,failed_ports=(2,),seed=33)
    save_feed_configuration(path,p,s)
    assert json.loads(path.read_text())["schema_version"] == 2
    assert load_feed_configuration(path) == (p,s)


def test_cli_electrical_exports_and_agg_without_gui(tmp_path):
    out = tmp_path / "electrical"
    result = run_cli("--feed-network","symmetric_series","--feed-phase","scan","--scan-deg",30,
                     "--feed-weighting","taylor","--feed-sidelobe-db",30,"--feed-phase-bits",4,
                     "--feed-seed",42,"--output-dir",out,"--plot","both",guard=True)
    assert result.returncode == 0,result.stderr
    doc = json.loads(result.stdout)
    assert doc["feed_network"]["electrical"]["metrics"]["target_scan_deg"] == 30
    for name in ["feed-rectangular.png","feed-polar.png"]: assert (out/name).read_bytes().startswith(b"\x89PNG")
    data = np.loadtxt(out/"feed-patterns.csv",delimiter=",",skiprows=1)
    np.testing.assert_allclose(data[:,3],doc["feed_network"]["electrical"]["patterns"]["output"]["pattern_db"],atol=1e-8)


def test_cli_custom_file_and_orphan_options(tmp_path):
    file=tmp_path/"weights.csv";file.write_text(excitation_csv([1,2,2,1],[0,10,10,0]))
    good=run_cli("--feed-network","corporate","--elements",4,"--feed-phase","equalize","--feed-excitation",file,guard=True)
    assert good.returncode==0,good.stderr
    assert json.loads(good.stdout)["feed_network"]["settings"]["custom_amplitudes"]==[1,2,2,1]
    bad=run_cli("--feed-phase","scan",guard=True)
    assert bad.returncode==2 and not bad.stdout
