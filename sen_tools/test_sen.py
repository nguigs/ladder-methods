"""Created by nguigui on 5/11/20."""

import geomstats.backend as gs
import geomstats.tests
from geomstats.geometry.matrices import Matrices

from sen_tools.sen import SenTools
from sen_tools.sen_cannonical_metric import CanonicalLeftInvariantMetric


class TestCanonicalMetric(geomstats.tests.TestCase):
    def setUp(self):
        n = 3
        self.n = n
        self.metric = CanonicalLeftInvariantMetric(self.n)
        self.tools = SenTools(n)

        self.space = self.metric.group
        n_samples = 2
        gs.random.seed(0)
        point = self.space.random_uniform(n_samples)

        tan_b = Matrices(n + 1, n + 1).random_uniform(n_samples)
        tan_b = self.tools.regularize(tan_b)

        # use a vector orthonormal to tan_b
        tan_a = Matrices(n + 1, n + 1).random_uniform(n_samples)
        tan_a = self.tools.regularize(tan_a)
        tan_a[..., 0, -1] -= gs.sum(tan_b * tan_a, axis=(-1, -2)) / tan_b[..., 0, -1]
        tan_b = gs.einsum('...ij,...->...ij', tan_b, 1. / self.metric.norm(tan_b, base_point=point))
        tan_a = gs.einsum('...ij,...->...ij', tan_a, 1. / self.metric.norm(tan_a, base_point=point))

        # normalize and move to base_point
        self.tan_b = self.space.compose(point, tan_b)
        self.tan_a = self.space.compose(point, tan_a)
        self.point = point
        self.n_samples = n_samples

    def test_inner_product(self):
        result = self.metric.inner_product(self.tan_a, self.tan_b, self.point)
        self.assertAllClose(result, 0.)

        result = self.metric.inner_product(self.tan_b, self.tan_b, self.point)
        self.assertAllClose(result, 1.)

    def test_inverse(self):
        point = self.space.random_uniform(1)
        inverse = self.tools.inverse(point)
        result = self.tools.compose(point, inverse)
        expected = self.space.identity
        self.assertAllClose(result, expected)

    def test_inverse_vectorization(self):
        point = self.space.random_uniform(2)
        inverse = self.tools.inverse(point)
        result = self.tools.compose(point, inverse)
        expected = self.space.identity
        self.assertTrue(gs.allclose(result, expected))

    def test_norm_anisotropic(self):
        self.tools.set_anisotropic_metric(2.)
        normalized_vec = gs.einsum(
            '...ij,...->...ij', self.tan_a,
            1. / self.tools.norm(self.tan_a, self.point))
        result = self.tools.norm(normalized_vec, self.point)
        expected = 1.
        self.assertAllClose(result, expected)

    def test_exp_and_belongs(self):
        exp = self.metric.exp(self.tan_b, self.point)
        self.assertTrue(gs.all(self.space.belongs(exp)))

    def test_log_and_is_tan(self):
        exp = self.metric.exp(self.tan_b, self.point)
        result = self.metric.log(exp, self.point)
        self.assertTrue(gs.all(self.space.is_tangent(result, self.point)))

    def test_exp_log(self):
        exp = self.metric.exp(self.tan_b, self.point)
        result = self.metric.log(exp, self.point)
        self.assertAllClose(result, self.tan_b)

    def test_parallel_transport(self):
        def is_isometry(tan_a, trans_a, basepoint, endpoint):
            is_tangent = self.space.is_tangent(trans_a, endpoint, atol=1e-6)
            is_equinormal = gs.isclose(
                self.metric.norm(trans_a, endpoint),
                self.metric.norm(tan_a, basepoint))
            return gs.logical_and(is_tangent, is_equinormal)

        transported, end_point = self.metric.parallel_transport(
            self.tan_a, self.tan_b, self.point)
        result = is_isometry(self.tan_a, transported, self.point, end_point)
        expected_end_point = self.metric.exp(self.tan_b, self.point)
        self.assertTrue(gs.all(result))
        self.assertAllClose(end_point, expected_end_point)

        new_tan_b = self.metric.log(self.point, end_point)
        result_vec, result_point = self.metric.parallel_transport(
            transported, new_tan_b, end_point)
        self.assertAllClose(result_vec, self.tan_a)
        self.assertAllClose(result_point, self.point)

    def test_sectional_curvature(self):
        result = self.tools.sectional_curvature(
            self.tools.basis[0], self.tools.basis[2])
        self.assertAllClose(result, 1. / 8.)

        result = self.tools.sectional_curvature(
            self.tools.basis[3], self.tools.basis[4])
        self.assertAllClose(result, 0.)

        result = self.tools.sectional_curvature(self.tan_a, self.tan_a)
        self.assertAllClose(result, gs.zeros(2))

        result = self.tools.sectional_curvature(self.tan_a, self.tan_b)
        self.assertAllClose(result.shape, (2,))

    def test_integrated_exp_at_id(self):
        tools = self.tools
        vecs = gs.random.rand(self.n_samples, len(tools.basis))
        vecs = gs.einsum('...j,jkl->...kl', vecs, gs.array(tools.basis))
        identity = self.space.identity
        result = tools.exp(vecs, identity, n_steps=100, step='rk4')
        expected = self.metric.exp(vecs, identity)
        self.assertAllClose(expected, result)

    def test_integrated_exp_and_log_at_id(self):
        tools = self.tools
        vecs = gs.random.rand(self.n_samples, len(tools.basis))
        vecs = gs.einsum('...j,jkl->...kl', vecs, gs.array(tools.basis))
        identity = self.space.identity
        exp = tools.exp(vecs, identity, n_steps=100, step='rk4')
        result = tools.log(
            exp, gs.stack([identity] * len(vecs)), n_steps=15, step='rk4')
        self.assertAllClose(vecs, result, atol=1e-5)

    def test_integrated_exp(self):
        tools = self.tools
        result = tools.exp(self.tan_a, self.point, n_steps=10, step='rk4')
        expected = self.metric.exp(self.tan_a, self.point)
        self.assertAllClose(expected, result)

    def test_integrated_exp_and_log(self):
        tools = self.tools
        exp = tools.exp(self.tan_a, self.point, n_steps=10, step='rk4')
        result = tools.log(exp, self.point, n_steps=15, step='rk4')
        self.assertAllClose(self.tan_a, result, atol=1e-5)

    def test_integrated_exp_and_belongs(self):
        exp = self.tools.exp(self.tan_b, self.point)
        self.assertTrue(gs.all(self.space.belongs(exp)))

    def test_integrated_log_and_is_tan(self):
        exp = self.tools.exp(self.tan_b, self.point)
        result = self.tools.log(exp, self.point)
        self.assertTrue(gs.all(self.space.is_tangent(result, self.point)))

    def test_one_step_schild_ladder(self):
        expected_dict = self.metric.ladder_parallel_transport(
            self.tan_a, self.tan_b, self.point, scheme='schild', n_ladders=1)
        result, result_point = self.tools.ladder_parallel_transport(
            self.tan_a, self.tan_b, self.point, n_steps=10, step='rk4',
            scheme='schild', n_rungs=1, tol=1e-10)
        expected = expected_dict['transported_tangent_vec']
        end_point = expected_dict['end_point']
        self.assertTrue(gs.all(self.space.is_tangent(result, result_point)))
        self.assertAllClose(end_point, result_point)
        self.assertAllClose(result, expected)

    def test_one_step_pole_ladder(self):
        expected_dict = self.metric.ladder_parallel_transport(
            self.tan_a, self.tan_b, self.point, scheme='pole', n_ladders=1)
        result, result_point = self.tools.ladder_parallel_transport(
            self.tan_a, self.tan_b, self.point, n_steps=10, step='rk4',
            scheme='pole', n_rungs=1, tol=1e-10)
        expected = expected_dict['transported_tangent_vec']
        end_point = expected_dict['end_point']
        self.assertTrue(gs.all(self.space.is_tangent(result, result_point)))
        self.assertAllClose(end_point, result_point)
        self.assertAllClose(result, expected)

    def test_ten_steps_pole_ladder(self):
        exp = self.metric.exp(self.tan_b, self.point)
        expected_dict = self.metric.ladder_parallel_transport(
            self.tan_a / 10, self.tan_b, self.point, scheme='pole',
            n_ladders=10)
        result, result_point = self.tools.ladder_parallel_transport(
            self.tan_a / 10, self.tan_b, self.point, n_steps=1, step='rk4',
            scheme='pole', n_rungs=10, tol=1e-14)
        result *= 10
        expected = expected_dict['transported_tangent_vec'] * 10
        end_point = expected_dict['end_point']
        self.assertTrue(gs.all(self.space.is_tangent(result, result_point)))
        self.assertAllClose(end_point, exp)
        self.assertAllClose(end_point, result_point)
        self.assertAllClose(result, expected)

    def test_ten_steps_schild_ladder(self):
        expected_dict = self.metric.ladder_parallel_transport(
            self.tan_a / 20, self.tan_b, self.point, scheme='schild',
            n_ladders=20)
        result, result_point = self.tools.ladder_parallel_transport(
            self.tan_a / 20, self.tan_b, self.point, n_steps=1, step='rk4',
            scheme='schild', n_rungs=20, tol=1e-14)
        result *= 20
        expected = expected_dict['transported_tangent_vec'] * 20
        end_point = expected_dict['end_point']
        self.assertTrue(gs.all(self.space.is_tangent(result, result_point)))
        self.assertAllClose(end_point, result_point)
        self.assertAllClose(result, expected)

    def test_set_anisotropic(self):
        beta = 3
        self.tools.set_anisotropic_metric(beta)
        expected = gs.ones((self.n + 1,) * 2)
        expected[0, self.n] = beta
        result = self.tools.metric_matrix
        self.assertAllClose(result, expected)

    def test_anisotropic_basis_is_orthonormal(self):
        self.tools.set_anisotropic_metric(3.)
        for i, x in enumerate(self.tools.basis):
            for j, y in enumerate(self.tools.basis[i:]):
                result = self.tools.metric(x, y)
                expected = 1. if j == 0 else 0.
                self.assertAllClose(result, expected)

    def test_anisotropic_metric_orthogonal_vec(self):
        beta = 3.
        tools = self.tools
        tools.set_anisotropic_metric(beta)
        tan_a = tools.compose(tools.inverse(self.point), self.tan_a)
        tan_b = tools.compose(tools.inverse(self.point), self.tan_b)
        tan_a[..., 0, -1] -= \
            tools.metric(tan_a, tan_b) / tan_b[..., 0, -1] / beta
        tan_a = tools.compose(self.point, tan_a)
        tan_b = tools.compose(self.point, tan_b)
        result = self.tools.inner_product(tan_a, tan_b, self.point)
        self.assertAllClose(result, 0.)

    def test_anisotropic_metric_coincides(self):
        beta = 1.
        self.tools.set_anisotropic_metric(beta)
        result = self.tools.inner_product(self.tan_a, self.tan_b, self.point)
        expected = self.metric.inner_product(self.tan_a, self.tan_b, self.point)
        self.assertAllClose(result, expected)

    def test_ten_steps_anisotropic_is_isometry(self):
        self.tools.set_anisotropic_metric(2.)
        exp = self.tools.exp(self.tan_b, self.point)
        result, result_point = self.tools.ladder_parallel_transport(
            self.tan_a / 10, self.tan_b, self.point, n_steps=1, step='rk4',
            scheme='pole', n_rungs=10, tol=1e-14)
        result *= 10
        is_equinormal = gs.isclose(
            self.tools.norm(result, result_point),
            self.tools.norm(self.tan_a, self.point), atol=1e-4)
        self.assertTrue(gs.all(self.space.is_tangent(result, result_point)))
        self.assertTrue(gs.all(is_equinormal))
        self.assertAllClose(result_point, exp)

    def test_parallel_transport_anisotropic(self):
        self.tools.set_anisotropic_metric(3.)
        result, result_point = self.tools.parallel_transport(
            self.tan_a, self.tan_b, self.point)
        expected_point = self.tools.exp(self.tan_b, self.point)
        self.assertAllClose(expected_point, result_point)
        is_equinormal = gs.isclose(
            self.tools.norm(result, result_point),
            self.tools.norm(self.tan_a, self.point), atol=1e-6)
        self.assertTrue(gs.all(self.space.is_tangent(result, result_point)))
        self.assertTrue(gs.all(is_equinormal))

        new_tan_b = self.tools.log(
            self.point, result_point, tol=1e-14, n_steps=20)
        result_vec, result_point = self.tools.parallel_transport(
            result, new_tan_b, result_point)
        self.assertAllClose(result_vec, self.tan_a, atol=1e-5)
        self.assertAllClose(result_point, self.point)

    def test_simplified_pole_ladder(self):
        n_lads = 10
        exp = self.tools.exp(self.tan_b, self.point, n_steps=2 * n_lads)
        result, result_point = self.tools.pole_ladder(
            self.tan_a / n_lads, self.tan_b, self.point, n_rungs=n_lads,
            n_steps=1, step='rk4', tol=1e-14)
        expected, expected_point = self.tools.ladder_parallel_transport(
            self.tan_a / n_lads, self.tan_b, self.point, n_steps=2, step='rk4',
            scheme='pole', n_rungs=n_lads,
            tol=1e-14)
        expected *= n_lads
        result *= n_lads
        self.assertAllClose(result_point, exp)
        self.assertAllClose(result_point, expected_point)
        self.assertTrue(gs.all(self.space.is_tangent(result, result_point)))
        self.assertAllClose(result, expected)
