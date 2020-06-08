"""Created by nguigui on 5/11/20."""

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear as gl
from geomstats.geometry.riemannian_metric import RiemannianMetric
from geomstats.geometry.special_euclidean import SpecialEuclidean

from sen_tools.sen import SenTools


class CanonicalLeftInvariantMetric(RiemannianMetric):
    def __init__(self, n):
        self.group = SpecialEuclidean(n)
        super(CanonicalLeftInvariantMetric, self).__init__(
            dim=self.group.dim, default_point_type='matrix')
        self.n = self.group.n
        self.tools = SenTools(n)

    @staticmethod
    def inner_product_at_identity(tangent_vec_a, tangent_vec_b):
        r"""
        :math: \sum_{i, j} g_{ij}x_{ij}y_{ij} where g is the metric matrix
        """
        is_vectorized = \
            (gs.ndim(gs.array(tangent_vec_a)) == 3) or (
                    gs.ndim(gs.array(tangent_vec_b)) == 3)
        axes = (2, 1) if is_vectorized else (0, 1)
        return gs.sum(tangent_vec_a * tangent_vec_b, axes)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        if base_point is None:
            base_point = self.group.identity
        tan_a_at_id = self.group.compose(
            self.group.inverse(base_point), tangent_vec_a)
        tan_b_at_id = self.group.compose(
            self.group.inverse(base_point), tangent_vec_b)
        return self.inner_product_at_identity(tan_a_at_id, tan_b_at_id)

    def sectional_curvature(
            self, tangent_vec_a, tangent_vec_b, base_point=None):
        if base_point is None:
            base_point = self.group.identity
        tan_a_at_id = self.group.compose(
            self.group.inverse(base_point), tangent_vec_a)
        tan_b_at_id = self.group.compose(
            self.group.inverse(base_point), tangent_vec_b)
        return self.tools.sectional_curvature(tan_a_at_id, tan_b_at_id)

    def exp(self, tangent_vec, base_point, **kwargs):
        exp = gs.zeros_like(tangent_vec)
        inf_rotation = tangent_vec[..., :self.n, :self.n]
        rotation = base_point[..., :self.n, :self.n]
        exp[..., :self.n, :self.n] = gl.compose(
            rotation,
            gl.exp(gl.compose(gl.inverse(rotation), inf_rotation)))
        exp[..., :, self.n] = tangent_vec[..., :, self.n] \
            + base_point[..., :, self.n]
        exp[..., self.n, self.n] = 1
        return exp

    def log(self, point, base_point, **kwargs):
        log = gs.zeros_like(point)
        rotation_bp = base_point[..., :self.n, :self.n]
        rotation_p = point[..., :self.n, :self.n]
        log[..., :self.n, :self.n] = gl.log(rotation_p, rotation_bp)
        log[..., :self.n, self.n] = point[..., :self.n, self.n] \
            - base_point[..., :self.n, self.n]
        return log

    def parallel_transport(self, tangent_vec_a, tangent_vec_b, base_point):
        point = self.exp(tangent_vec_a, base_point)
        midpoint = self.exp(1. / 2. * tangent_vec_b, base_point)
        next_point = self.exp(tangent_vec_b, base_point)
        first_sym = self.exp(- self.log(point, midpoint), midpoint)
        transported_vec = - self.log(first_sym, next_point)
        return transported_vec, next_point
