"""Created by nguigui on 6/8/20."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricAffine
import utils as dg


class TestDataGenerators(geomstats.tests.TestCase):
    def setUp(self):
        self.sphere = Hypersphere(2)
        self.spd = SPDMatrices(3)
        self.spd_metric = SPDMetricAffine(3)

    def test_random_orthonormal_sphere(self):
        point = self.sphere.random_uniform()
        tan_a, tan_b = dg.random_orthonormal_sphere(point)
        is_unit = gs.allclose(gs.linalg.norm([tan_a, tan_b], axis=1), 1.)
        self.assertTrue(is_unit)
        are_perpendicular = gs.isclose(gs.dot(tan_a, tan_b), 0.)
        self.assertTrue(are_perpendicular)

    def test_random_orthonormal_spd(self):
        point = self.spd.random_uniform()
        tan_a, tan_b = dg.random_orthonormal_spd(point)
        is_unit = gs.allclose(
            self.spd_metric.norm(gs.stack([tan_a, tan_b]), point), 1.)
        self.assertTrue(is_unit)
        are_perpendicular = gs.isclose(
            self.spd_metric.inner_product(tan_a, tan_b, point), 0.)
        self.assertTrue(are_perpendicular)
