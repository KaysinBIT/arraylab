from dataclasses import asdict, replace
import unittest
import numpy as np

from arraylab.sparse2d.core import Config, candidates
from arraylab.sparse2d.elements import make_snapshot
from arraylab.sparse2d.scenarios import scenarios
from arraylab.sparse2d.pattern_metrics import sparse_pattern_summary
from arraylab.planar_pattern import integrated_power


class SparseFigureMetrics(unittest.TestCase):
    def bundle(self, **changes):
        c = replace(Config(nx=4, ny=4, k=16, eval_grid=81, opt_grid=25), **changes)
        points = candidates(c)[1]
        result = {'xy': points, 'weights': np.ones(len(points))/len(points)}
        bundle = {'config': asdict(c), 'element_pattern': make_snapshot(c), 'scenarios': scenarios(c)}
        return c, bundle, result

    def test_isotropic_known_integral_and_symmetric_width(self):
        c, b, r = self.bundle()
        m = sparse_pattern_summary(b, r)
        expected = 10*np.log10(1/integrated_power(r['xy'], r['weights']))
        self.assertAlmostEqual(m['directivity_dbi'], expected, places=10)
        self.assertAlmostEqual(*m['orthogonal_beamwidths_deg'], places=8)
        self.assertIsNone(m['gain_dbi'])
        self.assertLess(m['peak_sidelobe_db'], 0)

    def test_front_only_element_does_not_invent_directivity(self):
        _, b, r = self.bundle(element_model='patch', theta=25, phi=30)
        m = sparse_pattern_summary(b, r)
        self.assertIsNone(m['directivity_dbi'])
        self.assertEqual(m['directivity_status'], 'unavailable_back_hemisphere')
        self.assertTrue(all(w is None or w > 0 for w in m['orthogonal_beamwidths_deg']))

    def test_array_factor_uses_own_response(self):
        _, b, r = self.bundle(element_model='patch', theta=25, phi=30)
        m = sparse_pattern_summary(b, r, view_mode='array_factor')
        self.assertIsNotNone(m['directivity_dbi'])
        self.assertAlmostEqual(m['pointing_offset_deg'], 0, places=9)

    def test_quantized_selection_changes_actual_metric(self):
        _, b, r = self.bundle(theta=27, phi=34, phase_bits=1)
        ideal = sparse_pattern_summary(b, r)
        actual = sparse_pattern_summary(b, r, phase_mode='quantized')
        self.assertNotAlmostEqual(ideal['directivity_dbi'], actual['directivity_dbi'], places=3)
        self.assertEqual(actual['phase_mode'], 'quantized')


if __name__ == '__main__': unittest.main()
