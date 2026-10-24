"""Keep repeated plotting cheap without changing the saved sampling protocol."""
import copy
from dataclasses import asdict
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_info

from arraylab.sparse2d.core import Config, candidates, evaluate_result
from arraylab.sparse2d.elements import make_snapshot
from arraylab.sparse2d.scenarios import scenarios
from arraylab.sparse2d import pattern_metrics as metrics
from arraylab.sparse2d.plots import _plot_response


class SparsePlotCache(unittest.TestCase):
    def setUp(self):
        metrics._summary_cache.clear()
        metrics._metric_sources.clear()

    def bundle(self, **changes):
        config = Config(nx=4, ny=4, k=16, opt_grid=25, eval_grid=81, **changes)
        integer, xy, _ = candidates(config)
        result = {'xy': xy, 'integer': integer, 'indices': np.arange(len(xy)),
                  'weights': np.ones(len(xy))/len(xy), 'name': 'Dense reference'}
        evaluate_result(result, config)
        bundle = {'config': asdict(config), 'element_pattern': make_snapshot(config),
                  'scenarios': scenarios(config), 'results': [result]}
        return bundle, result

    def assert_same_metrics(self, actual, reference):
        for key in ('theta_deg', 'phi_deg', 'pointing_offset_deg', 'directivity_dbi', 'peak_sidelobe_db'):
            if reference[key] is None: self.assertIsNone(actual[key])
            else: self.assertAlmostEqual(actual[key], reference[key], places=8, msg=key)
        np.testing.assert_allclose(actual['orthogonal_beamwidths_deg'],
                                   reference['orthogonal_beamwidths_deg'], atol=1e-8, rtol=0)

    def test_saved_ideal_and_quantized_metrics_match_full_sampling(self):
        bundle, result = self.bundle(theta=27, phi=34, phase_bits=2)
        for mode in ('ideal', 'quantized'):
            reference = metrics._measure_summary(bundle, result, 0, 'total', mode, False)
            with patch.object(metrics, 'sample_domain', side_effect=AssertionError('saved report must avoid dense resampling')):
                actual = metrics.sparse_pattern_summary(bundle, result, phase_mode=mode)
            self.assert_same_metrics(actual, reference)

    def test_repeated_summary_is_cached_and_caller_cannot_mutate_cache(self):
        bundle, result = self.bundle()
        before = copy.deepcopy(bundle)
        with patch.object(metrics, 'spherical_beamwidths', wraps=metrics.spherical_beamwidths) as widths, \
             patch.object(metrics, 'integrated_power', wraps=metrics.integrated_power) as power:
            first = metrics.sparse_pattern_summary(bundle, result)
            expected = copy.deepcopy(first)
            first['orthogonal_beamwidths_deg'][0] = -1
            again = metrics.sparse_pattern_summary(bundle, result)
            copied = metrics.sparse_pattern_summary(copy.deepcopy(bundle), copy.deepcopy(result))
            self.assertEqual(again, expected)
            self.assertEqual(copied, expected)
            self.assertEqual(widths.call_count, 1)
            self.assertEqual(power.call_count, 1)
        self.assertEqual(bundle.keys(), before.keys())
        self.assertEqual(result.keys(), before['results'][0].keys())

    def test_case_phase_and_weights_invalidate_and_stale_saved_values_are_not_used(self):
        bundle, result = self.bundle(phase_bits=2, scan_angles='0,0;27,34')
        with patch.object(metrics, '_measure_summary', wraps=metrics._measure_summary) as measure:
            first = metrics.sparse_pattern_summary(bundle, result)
            second = metrics.sparse_pattern_summary(bundle, result, 1)
            quantized = metrics.sparse_pattern_summary(bundle, result, 1, phase_mode='quantized')
            self.assertEqual(measure.call_count, 3)
            self.assertNotEqual(first['theta_deg'], second['theta_deg'])
            self.assertNotEqual(second['phase_mode'], quantized['phase_mode'])
            result['weights'][:] = np.arange(1, len(result['weights'])+1)
            result['weights'] /= result['weights'].sum()
            changed = metrics.sparse_pattern_summary(bundle, result)
            self.assertEqual(measure.call_count, 4)
            reference = metrics._measure_summary(bundle, result, 0, 'total', 'ideal', False)
            self.assert_same_metrics(changed, reference)
            self.assertNotAlmostEqual(first['directivity_dbi'], changed['directivity_dbi'], places=6)

    def test_legacy_and_different_field_views_keep_original_sampling_grids(self):
        bundle, result = self.bundle(element_model='patch', theta=20, phi=30)
        with patch.object(metrics, 'sample_domain', wraps=metrics.sample_domain) as sample:
            metrics.sparse_pattern_summary(bundle, result, view_mode='array_factor')
            self.assertEqual([call.args[1] for call in sample.call_args_list], [81, 83])
        legacy = copy.deepcopy(result)
        del legacy['scenario_metrics'][0]['mainlobe_peak_uv']
        with patch.object(metrics, 'sample_domain', wraps=metrics.sample_domain) as sample:
            metrics.sparse_pattern_summary(bundle, legacy)
            self.assertEqual([call.args[1] for call in sample.call_args_list], [81, 83])
        with patch.object(metrics, 'sample_domain', wraps=metrics.sample_domain) as sample:
            metrics.sparse_pattern_summary(bundle, result, view_mode='element')
            self.assertEqual([call.args[1] for call in sample.call_args_list], [81, 83])

    def test_bounded_cache_and_blas_restoration(self):
        bundle, result = self.bundle()
        before = [(p['filepath'], p['num_threads']) for p in threadpool_info() if p['user_api'] == 'blas']
        def check_limit(*args, **kwargs):
            self.assertTrue(all(p['num_threads'] == 1 for p in threadpool_info() if p['user_api'] == 'blas'))
            return {'orthogonal_beamwidths_deg': [1., 1.]}
        with patch.object(metrics, 'spherical_beamwidths', side_effect=check_limit):
            for number in range(10):
                current = copy.deepcopy(result)
                current['weights'][0] += number * .001
                current['weights'] /= current['weights'].sum()
                metrics.sparse_pattern_summary(bundle, current)
        self.assertLessEqual(len(metrics._summary_cache), 8)
        self.assertLessEqual(len(metrics._metric_sources), 8)
        self.assertEqual(before, [(p['filepath'], p['num_threads']) for p in threadpool_info() if p['user_api'] == 'blas'])

    def test_plot_field_uses_serial_native_pool_and_restores_it(self):
        from arraylab.sparse2d import plots
        before = [(p['filepath'], p['num_threads']) for p in threadpool_info() if p['user_api'] == 'blas']
        def field(*args):
            self.assertTrue(all(p['num_threads'] == 1 for p in threadpool_info() if p['user_api'] == 'blas'))
            return np.array([1+0j])
        with patch.object(plots, 'response_field', side_effect=field):
            np.testing.assert_array_equal(_plot_response(None), [1+0j])
        self.assertEqual(before, [(p['filepath'], p['num_threads']) for p in threadpool_info() if p['user_api'] == 'blas'])


if __name__ == '__main__': unittest.main()
