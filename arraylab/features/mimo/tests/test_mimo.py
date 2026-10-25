"""Independent radar-MIMO sum-array and digital beam response checks."""
from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from arraylab.features.linear.model import ArrayParameters, calculate
from arraylab.features.mimo.model import MIMOParameters, MAX_CHANNELS, calculate_mimo, mimo_cut_angles, mimo_field, mimo_geometry, mimo_parameters_from_dict
from arraylab.framework.parallel import ComputationCancelled


def test_ti_two_tx_four_rx_is_eight_halfwave_virtual_elements():
    p = MIMOParameters()
    report = calculate_mimo(p, workers=1)
    positions = np.array([[row['x_lambda'], row['y_lambda']] for row in report['geometry']['pairs']])
    np.testing.assert_array_equal(positions[:, 0], np.arange(8) / 2)
    assert report['counts']['tx_rx_channels'] == report['counts']['unique_virtual_positions'] == 8
    assert report['counts']['rf_chains'] is None
    reference = calculate(ArrayParameters(elements=8, spacing=.5, frequency_ghz=10, step_deg=.1))
    assert report['metrics']['cut']['half_power_beamwidth_deg'] == pytest.approx(reference.beamwidth_deg, abs=5e-5)
    assert report['metrics']['orthogonal_beamwidths_deg'][0] == pytest.approx(reference.beamwidth_deg, abs=1e-7)
    assert report['metrics']['orthogonal_beamwidths_deg'][1] is None
    angles = np.linspace(-88, 88, 279)
    u = np.sin(np.deg2rad(angles))
    independent = np.exp(2j * np.pi * u[:, None] * np.arange(8)[None, :] / 2).mean(axis=1)
    np.testing.assert_allclose(mimo_field(p, np.column_stack((u, np.zeros_like(u)))), independent, atol=1e-14)
    json.dumps(report, allow_nan=False)


def test_repeated_virtual_positions_retain_channel_multiplicity():
    p = MIMOParameters(tx_positions=((0, 0), (.5, 0)), rx_positions=((0, 0), (.5, 0)))
    report = calculate_mimo(p, workers=1)
    unique = report['geometry']['virtual_unique']
    assert [row['multiplicity'] for row in unique] == [1, 2, 1]
    assert [row['weight'] for row in unique] == [.25, .5, .25]
    assert report['counts']['redundant_channels'] == 1
    assert report['counts']['redundancy_ratio'] == .25
    assert [row['unique_index'] for row in report['geometry']['pairs']] == [1, 2, 2, 3]
    field = mimo_field(p, [[.5, 0]])[0]
    assert field == pytest.approx(.5j)
    assert abs(field - 1j / 3) > .1  # De-duplicating to three uniform elements is wrong.


def test_two_dimensional_cartesian_sum_array_and_independent_pair_field():
    p = MIMOParameters(dimension='planar', tx_positions=((0, 0), (1.3, .2), (.1, 1.7)),
                       rx_positions=((-.2, 0), (.7, .4)), tx_weights=(1, 2, 3), rx_weights=(3, 1),
                       theta_deg=27, phi_deg=31, cut_phi_deg=31, map_size=91)
    report = calculate_mimo(p, workers=1)
    pairs = report['geometry']['pairs']
    assert [(row['tx'], row['rx']) for row in pairs] == [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2)]
    xy = np.array([[tx[0] + rx[0], tx[1] + rx[1]] for tx in p.tx_positions for rx in p.rx_positions])
    weights = np.outer(np.array(p.tx_weights) / 6, np.array(p.rx_weights) / 4).ravel()
    rng = np.random.default_rng(771)
    uv = rng.uniform(-.65, .65, (205, 2))
    steering = math.sin(math.radians(27)) * np.array([math.cos(math.radians(31)), math.sin(math.radians(31))])
    expected = np.exp(2j * np.pi * ((uv - steering) @ xy.T)) @ weights
    np.testing.assert_allclose(mimo_field(p, uv), expected, atol=1e-14)
    np.testing.assert_allclose([[row['x_lambda'], row['y_lambda']] for row in pairs], xy)
    assert report['geometry']['virtual']['geometric_dimension'] == 2
    assert report['patterns']['map']['virtual_db'][0][0] is None
    assert report['patterns']['map']['virtual_db'][45][45] is not None
    assert report['metrics']['peak_method'] == 'theoretical_coherent_steering'
    assert report['metrics']['peak_uniqueness'] == 'not_established'
    assert report['metrics']['pointing_offset_deg'] == 0


def test_tx_rx_exchange_and_common_translations_preserve_magnitude():
    p = MIMOParameters(dimension='planar', tx_positions=((0, 0), (1.1, .2)),
                       rx_positions=((0, 0), (.6, .4), (.1, 1.2)), tx_weights=(1, 2), rx_weights=(3, 2, 1),
                       theta_deg=15, phi_deg=-25)
    swap = replace(p, tx_positions=p.rx_positions, rx_positions=p.tx_positions,
                   tx_weights=p.rx_weights, rx_weights=p.tx_weights)
    translated = replace(p, tx_positions=tuple((x + 15.75, y - 9.2) for x, y in p.tx_positions),
                          rx_positions=tuple((x - 12.3, y + 5.4) for x, y in p.rx_positions))
    uv = np.array([[0, 0], [.2, .4], [-.5, -.1], [.6, -.2]])
    np.testing.assert_allclose(mimo_field(p, uv), mimo_field(swap, uv), atol=1e-14)
    np.testing.assert_allclose(abs(mimo_field(p, uv)), abs(mimo_field(translated, uv)), atol=1e-13)


def test_coordinate_units_and_weight_scaling_preserve_response():
    p = MIMOParameters(theta_deg=-22, tx_weights=(1, 2), rx_weights=(1, 3, 2, 4))
    millimeters = replace(p, coordinate_unit='mm',
                          tx_positions=tuple(tuple(value * p.wavelength_mm for value in row) for row in p.tx_positions),
                          rx_positions=tuple(tuple(value * p.wavelength_mm for value in row) for row in p.rx_positions),
                          tx_weights=(1e200, 2e200), rx_weights=(1e-200, 3e-200, 2e-200, 4e-200))
    uv = np.column_stack((np.linspace(-1, 1, 235), np.zeros(235)))
    np.testing.assert_allclose(mimo_field(p, uv), mimo_field(millimeters, uv), atol=1e-14)
    first, second = mimo_geometry(p), mimo_geometry(millimeters)
    np.testing.assert_allclose(first['geometry']['virtual']['span_mm'], second['geometry']['virtual']['span_mm'])
    report = calculate_mimo(p, workers=1)
    assert report['metrics']['cut']['main_lobe_angle_deg'] == pytest.approx(-22)
    assert report['metrics']['representative_peak_theta_deg'] == -22
    assert report['metrics']['directivity_dbi'] is report['metrics']['gain_dbi'] is None


def test_numerical_coordinate_identity_does_not_round_positions_or_drop_pairs():
    p = MIMOParameters(tx_positions=((0, 0), (.1, 0)), rx_positions=((0, 0), (.2, 0), (.3, 0)))
    report = mimo_geometry(p)
    assert report['counts']['tx_rx_channels'] == 6
    assert report['counts']['unique_virtual_positions'] == 5
    rows = report['geometry']['pairs']
    assert rows[2]['unique_index'] == rows[4]['unique_index']
    assert rows[2]['x_lambda'] == .3 and rows[4]['x_lambda'] == .1 + .2
    assert report['aggregation']['coordinates_rounded'] is False
    arbitrary = replace(p, tx_positions=((.123456789123, 0), (1.1, 0)))
    point = mimo_geometry(arbitrary)['geometry']['virtual_unique'][0]
    assert point['x_lambda'] == .123456789123


def test_single_channel_has_no_direction_or_width_or_invented_gain():
    p = MIMOParameters(tx_positions=((3, 0),), rx_positions=((-2, 0),), theta_deg=35)
    report = calculate_mimo(p, workers=1)
    assert report['counts']['tx_rx_channels'] == 1
    assert report['geometry']['virtual']['min_spacing_lambda'] is None
    assert report['metrics']['cut']['status'] == 'flat_cut'
    assert report['metrics']['cut']['half_power_beamwidth_deg'] is None
    assert report['metrics']['pointing_offset_deg'] is None
    assert report['metrics']['orthogonal_beamwidths_deg'] == [None, None]
    assert report['metrics']['angular_resolution_deg'] is None
    assert report['metrics']['gain_dbi'] is None
    json.dumps(report, allow_nan=False)


def test_grating_lobes_and_endfire_do_not_get_false_unique_beamwidth():
    grating = calculate_mimo(MIMOParameters(tx_positions=((0, 0), (2, 0)), rx_positions=((0, 0),)), workers=1)
    metrics = grating['metrics']['cut']
    assert metrics['status'] == 'multiple_sampled_peaks'
    assert metrics['half_power_beamwidth_deg'] is None
    assert metrics['peak_sidelobe_db'] == pytest.approx(0)
    assert len(metrics['peak_angles_deg']) == 5
    endfire = calculate_mimo(MIMOParameters(theta_deg=90), workers=1)
    assert endfire['metrics']['cut']['half_power_beamwidth_deg'] is None


def test_parallel_and_serial_outputs_identical_and_cancel_checked():
    p = MIMOParameters(dimension='planar', tx_positions=((0, 0), (1.5, 0)),
                       rx_positions=((0, 0), (.5, 0), (0, .5), (.5, .5)), theta_deg=20, phi_deg=30, map_size=91)
    serial, parallel = calculate_mimo(p, workers=1), calculate_mimo(p, workers=2)
    assert serial['counts'] == parallel['counts']
    assert serial['patterns'] == parallel['patterns']
    assert serial['metrics'] == parallel['metrics']
    assert parallel['execution']['workers'] <= 2
    with pytest.raises(ComputationCancelled):
        calculate_mimo(p, cancelled=lambda: True)
    calls = []
    def stop_during_blocks():
        calls.append(1)
        return len(calls) > 2
    with pytest.raises(ComputationCancelled):
        mimo_field(p, np.zeros((10000, 2)), cancelled=stop_during_blocks)
    assert len(calls) == 3


@pytest.mark.parametrize('changes', [
    {'dimension': '3d'}, {'coordinate_unit': 'inch'}, {'tx_positions': ()},
    {'rx_positions': ((0, 0), (0, 0))}, {'tx_positions': ((0, 1),)},
    {'tx_positions': ((0,),)}, {'tx_positions': ((True, 0),)},
    {'tx_weights': (1,)}, {'tx_weights': (0, 1)}, {'tx_weights': (1, -1)},
    {'tx_weights': (1, 1e7)}, {'frequency_ghz': float('nan')},
    {'frequency_ghz': 2 ** 4096}, {'rx_positions': ((2 ** 4096, 0),)},
    {'frequency_ghz': True}, {'theta_deg': 91}, {'phi_deg': 1}, {'cut_phi_deg': 20},
    {'map_size': 80}, {'map_size': True}, {'map_size': 401.0}, {'step_deg': .001},
    {'tx_positions': tuple((i / 10, 0) for i in range(257))},
    {'tx_positions': ((0, 0), (81, 0))},
])
def test_invalid_inputs_fail_explicitly(changes):
    with pytest.raises(ValueError):
        replace(MIMOParameters(), **changes).validate()


def test_channel_bound_does_not_materialize_an_oversized_pair_array(monkeypatch):
    import arraylab.features.mimo.model as mimo
    p = MIMOParameters(tx_positions=tuple((i * .01, 0) for i in range(129)),
                       rx_positions=tuple((i * .01, 0) for i in range(128)))
    assert len(p.tx_positions) * len(p.rx_positions) > MAX_CHANNELS
    with pytest.raises(ValueError, match='通道数'):
        calculate_mimo(p)


def test_strict_dict_and_visible_direction_validation():
    assert mimo_parameters_from_dict(asdict(MIMOParameters())) == MIMOParameters()
    with pytest.raises(ValueError, match='未知'):
        mimo_parameters_from_dict({'not_a_parameter': 1})
    with pytest.raises(ValueError):
        mimo_parameters_from_dict([])
    for uv in ([1, 0], [[float('nan'), 0]], [[1, 1]], [[0, 0, 1]]):
        with pytest.raises(ValueError):
            mimo_field(MIMOParameters(), uv)


def test_numpy_scalar_parameters_and_tiny_common_weights_stay_json_safe():
    p = MIMOParameters(frequency_ghz=np.float64(10), theta_deg=np.float32(0),
                       tx_positions=((np.int64(0), 0), (np.float32(2), 0)),
                       tx_weights=(1e-320, 2e-320))
    report = calculate_mimo(p, workers=1)
    json.dumps(report, allow_nan=False)
    np.testing.assert_allclose(p.weights('tx'), [1 / 3, 2 / 3], atol=1e-15)


def test_projected_scan_angle_does_not_add_numerically_duplicate_sample():
    p = MIMOParameters(dimension='planar', theta_deg=20, phi_deg=30, cut_phi_deg=30)
    angles, preferred = mimo_cut_angles(p)
    assert len(angles) == 1801
    assert np.min(np.diff(angles)) > .099999999
    assert angles[np.argmin(abs(angles - preferred))] == preferred
    restored = mimo_parameters_from_dict(json.loads(json.dumps(asdict(p))))
    np.testing.assert_array_equal(mimo_cut_angles(restored)[0], angles)
    shifted = replace(p, theta_deg=20.12345)
    angles, preferred = mimo_cut_angles(shifted)
    assert len(angles) == 1802 and np.min(np.diff(angles)) > 1e-10
    assert preferred in angles
