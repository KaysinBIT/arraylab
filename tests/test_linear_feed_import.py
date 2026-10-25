"""Saved linear results become clean, equivalent feed targets without solving."""
from dataclasses import asdict, replace
from types import SimpleNamespace

import numpy as np
import pytest

from arraylab.core import ArrayParameters, calculate
from arraylab.feed import FeedSettings, design_feed, settings_payload
from arraylab.linear_feed_import import prepare_linear_feed_import


def source(parameters):
    return SimpleNamespace(label="#9 原始天线", result=calculate(parameters))


@pytest.mark.parametrize("topology", ["corporate", "series", "symmetric_series"])
@pytest.mark.parametrize("scan", [-35, 0, 24])
def test_clean_handoff_retains_geometry_and_network_but_discards_old_excitation(topology, scan):
    data = source(ArrayParameters(elements=8, frequency_ghz=5, spacing=29.9792458,
                                  spacing_unit="mm", scan_deg=scan, step_deg=.5))
    old = FeedSettings(topology=topology, input_ohm=75, junction_ohm=60, load_ohm=80,
                       min_line_ohm=10, max_line_ohm=180, phase_mode="none", weighting="custom",
                       custom_amplitudes=(1, 2), custom_phases_deg=(33, 44), phase_bits=3,
                       amplitude_error_db=2, phase_error_deg=30, failed_ports=(1,), seed=87,
                       branch_delays_deg=(10,)*6, branch_delay_elements=4,
                       branch_delay_topology=topology)
    before = settings_payload(old)
    plan = prepare_linear_feed_import(data, old)
    assert plan.parameters == data.result.parameters and plan.parameters is not data.result.parameters
    assert plan.source_label == data.label and plan.notes == ()
    assert plan.settings == FeedSettings(topology=topology, input_ohm=75, junction_ohm=60,
                                         load_ohm=80, min_line_ohm=10, max_line_ohm=180)
    assert settings_payload(old) == before


@pytest.mark.parametrize("n", [1, 5, 9])
def test_odd_saved_array_falls_back_from_symmetric_to_corporate(n):
    plan = prepare_linear_feed_import(source(ArrayParameters(elements=n)),
                                    FeedSettings(topology="symmetric_series", input_ohm=75))
    assert plan.parameters.elements == n
    assert plan.settings.topology == "corporate" and plan.settings.input_ohm == 75
    assert "仅支持偶数" in plan.notes[0]


def test_external_curve_and_invalid_source_are_rejected():
    with pytest.raises(ValueError, match="外部曲线不能"):
        prepare_linear_feed_import(SimpleNamespace(label="measured.dat", result=None))
    data = source(ArrayParameters())
    data.result = replace(data.result, parameters=replace(data.result.parameters, spacing=-1))
    with pytest.raises(ValueError):
        prepare_linear_feed_import(data)


def test_retained_invalid_network_is_rejected_without_modifying_source():
    data = source(ArrayParameters(scan_deg=23))
    arrays = [data.result.pattern_db.copy(), data.result.excitation_phase_deg.copy()]
    with pytest.raises(ValueError):
        prepare_linear_feed_import(data, FeedSettings(input_ohm=-5))
    np.testing.assert_array_equal(data.result.pattern_db, arrays[0])
    np.testing.assert_array_equal(data.result.excitation_phase_deg, arrays[1])


def test_preparing_a_handoff_never_calls_the_design_engine(monkeypatch):
    data = source(ArrayParameters(elements=7, scan_deg=-19))
    def reject(*args, **kwargs):
        raise AssertionError("Import must not calculate a new result")
    monkeypatch.setattr("arraylab.feed.design_feed", reject)
    monkeypatch.setattr("arraylab.feed.synthesize_impedance", reject)
    monkeypatch.setattr("arraylab.core.calculate", reject)
    plan = prepare_linear_feed_import(data)
    assert asdict(plan.parameters) == asdict(data.result.parameters)


@pytest.mark.parametrize("topology", ["corporate", "series", "symmetric_series"])
def test_imported_target_and_ideal_output_match_saved_linear_power(topology):
    data = source(ArrayParameters(elements=8, spacing=.4, scan_deg=23, step_deg=.5))
    plan = prepare_linear_feed_import(data, FeedSettings(topology=topology))
    report = design_feed(plan.parameters, plan.settings)
    original_power = 10**(data.result.pattern_db/10)
    for key in ("target", "output"):
        curve = report["electrical"]["patterns"][key]
        np.testing.assert_allclose(curve["angles_deg"], data.result.angles_deg, atol=1e-12)
        np.testing.assert_allclose(10**(np.array(curve["pattern_db"])/10), original_power,
                                   atol=1e-12, rtol=1e-10)
