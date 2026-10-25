from dataclasses import replace
import json
from pathlib import Path
import threading

import numpy as np
import pytest

from arraylab.framework.parallel import ComputationCancelled
from arraylab.features.subarray.service import SubarrayConfig, expanded_coordinates, initial_layout, layout_valid, load_config, normalized_field, optimize_subarrays, save_config, export_subarray
from arraylab.features.sparse.engine.scenarios import scenarios


def small(**kwargs):
    return replace(SubarrayConfig(), count=4, iterations=8, restarts=2, eval_grid=41, **kwargs)


@pytest.mark.parametrize('changes', [dict(count=True), dict(count=0), dict(nx=2.1), dict(spacing_x=float('nan')),
    dict(spacing_x='0.5'), dict(min_spacing=.6), dict(aperture_x=.4), dict(eval_grid=31),
    dict(scan_angles=123), dict(frequencies_ghz='1001'), dict(hole_shape='unknown'),
    dict(count=64, nx=8, ny=8), dict(exclusion_u=0), dict(exclusion_u=1, exclusion_v=1)])
def test_invalid_configuration(changes):
    with pytest.raises(ValueError):
        replace(SubarrayConfig(), **changes).validate()


def test_exact_template_factorization_and_scanning():
    config = small(theta=30, phi=40, frequencies_ghz='8,10,12')
    centers = np.array([[-2., -2.], [1., -2.], [-1., 2.], [2., 2.]])
    xy = expanded_coordinates(config, centers)
    uv = np.random.default_rng(8).uniform(-.6, .6, (71, 2))
    for case in scenarios(config):
        t, p = np.deg2rad([case['theta'], case['phi']])
        target = np.sin(t)*np.array([np.cos(p), np.sin(p)])
        explicit = np.exp(2j*np.pi*case['frequency_ratio']*((uv-target) @ xy.T)).mean(axis=1)
        np.testing.assert_allclose(normalized_field(config, centers, case, uv), explicit, atol=1e-14)
        assert normalized_field(config, centers, case, target[None, :])[0] == pytest.approx(1.)


def test_real_element_spacing_hole_and_aperture():
    config = small()
    assert layout_valid(config, np.array([[-2, -2], [-2, 2], [2, -2], [2, 2]]))
    assert not layout_valid(config, np.array([[0., 0.], [.6, 0.]]))  # centers separated; elements too close
    assert not layout_valid(config, np.array([[2.9, 0.]]))  # actual element outside aperture
    assert not layout_valid(replace(config, hole_shape='circle', hole_x=1.), np.array([[0., 0.]]))
    assert not layout_valid(replace(config, min_spacing=0.), np.array([[0., 0.], [.2, .2]]))  # overlapping templates
    assert not layout_valid(replace(config, min_spacing=0.), np.array([[0., 0.], [.5, .5]]))  # shared element
    assert not layout_valid(replace(config, nx=1, ny=4, min_spacing=.01), np.array([[0., 0.], [0., .7]]))


def test_deterministic_parallel_output_and_serial_equivalence():
    config = small()
    serial = optimize_subarrays(config, workers=1)
    threaded = optimize_subarrays(config, workers=2)
    assert serial['initial'] == threaded['initial']
    assert serial['best'] == threaded['best']
    assert serial['optimization'] == threaded['optimization']
    for report in (serial, threaded):
        assert report['preparation_execution']['workers'] == 1
        assert report['validation_execution']['workers'] <= 2
        assert report['preparation_execution']['native_blas_threads'] == 1
        assert report['validation_execution']['native_blas_threads'] == 1
    assert serial['best']['worst_psl_db'] <= serial['initial']['worst_psl_db']+1e-10
    assert layout_valid(config, serial['best']['centers'])
    assert len(serial['best']['elements']) == 16
    assert not [thread for thread in threading.enumerate() if thread.name.startswith('ArrayLab-calc')]
    # All output is plain finite JSON; no ndarray/complex/NaN values.
    json.dumps(serial, allow_nan=False)


@pytest.mark.parametrize('workers', [1, 2])
def test_all_numerical_phases_limit_and_restore_blas(workers, monkeypatch):
    import arraylab.features.subarray.service as module
    from threadpoolctl import threadpool_info, threadpool_limits
    original_preparation = module._objective_data
    original_evaluation = module._evaluate
    observed = {'preparation': [], 'evaluation': []}

    def counts():
        return [pool['num_threads'] for pool in threadpool_info() if pool['user_api'] == 'blas']

    def preparation(*args, **kwargs):
        observed['preparation'].append(counts())
        return original_preparation(*args, **kwargs)

    def evaluation(*args, **kwargs):
        observed['evaluation'].append(counts())
        return original_evaluation(*args, **kwargs)

    monkeypatch.setattr(module, '_objective_data', preparation)
    monkeypatch.setattr(module, '_evaluate', evaluation)
    previous = counts()
    with threadpool_limits(limits=3, user_api='blas'):
        assert counts() and all(value == 3 for value in counts())
        optimize_subarrays(small(), workers=workers)
        assert all(value == 3 for value in counts())
    assert counts() == previous
    assert len(observed['preparation']) == 1
    assert len(observed['evaluation']) == 2
    assert all(counts and all(value == 1 for value in counts)
               for phase in observed.values() for counts in phase)


def test_cancellation_cleans_workers_and_does_not_return_partial():
    calls = 0
    def cancelled():
        nonlocal calls
        calls += 1
        return calls > 12
    with pytest.raises(ComputationCancelled):
        optimize_subarrays(small(), workers=2, cancelled=cancelled)
    assert not [thread for thread in threading.enumerate() if thread.name.startswith('ArrayLab-calc')]


def test_single_isotropic_element_has_no_half_power_width():
    config = replace(small(), count=1, nx=1, ny=1)
    report = optimize_subarrays(config, workers=1)
    metrics = report['best']['cases'][0]['metrics']
    assert metrics['orthogonal_beamwidths_deg'] == [None, None]
    assert metrics['directivity_dbi'] == pytest.approx(0.)
    assert metrics['gain_dbi'] is None
    assert metrics['peak_sidelobe_db'] == pytest.approx(0.)


def test_export_roundtrip_and_actual_case_phases(tmp_path):
    config = small(theta=30, phi=20)
    path = tmp_path/'config.json'
    save_config(path, config)
    assert load_config(path) == config
    report = optimize_subarrays(config, workers=1)
    output = export_subarray(tmp_path/'out', report)
    assert json.loads((output/'report.json').read_text(encoding='utf-8')) == report
    assert len(list(output.glob('*-elements.csv'))) == 2
    assert (output/'case-metrics.csv').is_file()
    assert len((output/'best-case-0-sphere.csv').read_text(encoding='utf-8-sig').splitlines()) == 61*121+1
    case = report['best']['cases'][0]
    phases = np.array(case['element_phase_deg'])
    combined = (np.repeat(case['center_phase_deg'], config.nx*config.ny)
                + np.tile(case['internal_phase_deg'], config.count))
    np.testing.assert_allclose(np.exp(1j*np.deg2rad(phases)), np.exp(1j*np.deg2rad(combined)), atol=1e-13)
    with pytest.raises(FileExistsError):
        export_subarray(output, report)
    for text in ('{"count":4,"count":5}', '{"count":NaN}', '{"unknown":2}', '[]'):
        path.write_text(text, encoding='utf-8')
        with pytest.raises(ValueError):
            load_config(path)
    path.write_text('['*2000+']'*2000, encoding='utf-8')
    with pytest.raises(ValueError):
        load_config(path)


def test_export_failure_leaves_no_incomplete_result(tmp_path, monkeypatch):
    import arraylab.features.subarray.plotting as plots
    report = optimize_subarrays(small(), workers=1)
    def fail(*args):
        raise OSError('simulated plot write failure')
    monkeypatch.setattr(plots, 'render_subarray_plots', fail)
    with pytest.raises(OSError, match='simulated'):
        export_subarray(tmp_path/'failed', report, plot='all')
    assert not (tmp_path/'failed').exists()
    assert not list(tmp_path.glob('.arraylab-subarray-*'))


def test_headless_cli_with_gui_imports_blocked(tmp_path):
    from test_cli import run_cli
    config = tmp_path/'子阵设置.json'
    save_config(config, small())
    output = tmp_path/'子阵结果 with spaces'
    result = run_cli('subarray', '--config', config, '--output', output, '--plot', '3d', '--workers', '1', guard=True)
    assert result.returncode == 0, result.stderr + result.stdout
    assert json.loads(result.stdout)['ok'] is True
    assert (output/'subarray-3d-case-0.png').is_file()


def test_plot_all_views_with_figures_and_metrics(tmp_path):
    from arraylab.features.subarray.plotting import render_subarray_figure
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    report = optimize_subarrays(small(), workers=1)
    for view in ('layout', 'pattern', 'cuts', '3d'):
        figure = Figure(figsize=(9, 7)); canvas = FigureCanvasAgg(figure)
        render_subarray_figure(figure, report, view)
        canvas.draw()
        assert 'efficiency missing' in figure._arraylab_metric_footer.get_text()
        assert 'Sampled disk PSL' in figure._arraylab_metric_footer.get_text()
        if view == '3d':
            assert figure.axes[0].name == '3d'
        figure.savefig(tmp_path/f'{view}.png')
