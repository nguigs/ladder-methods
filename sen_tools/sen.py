"""Tools for left-invariant metrics on Lie Groups.

Created on 09.02.2020
@author nicolas.guigui@inria.fr
"""

import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize

import geomstats.backend as gs
from geomstats.geometry.general_linear import GeneralLinear as gl


class SenTools(object):
    def __init__(self, n):
        self.n = n
        self.metric_matrix = gs.hstack([1 * gs.ones((n, n)), gs.ones((n, 1))])
        self.metric_matrix = gs.concatenate(
            [self.metric_matrix, gs.ones((1, n + 1))], axis=0)

        self.translation_mask = gs.hstack([
            gs.ones((n, n)), 2 * gs.ones((n, 1))])
        self.translation_mask = gs.concatenate(
            [self.translation_mask, gs.zeros((1, n + 1))], axis=0)
        self.rotations = []
        self.translations = []
        self.basis = []
        self.update_basis()
        self.identity = gs.eye(n + 1)

    def update_basis(self):
        self.rotations = []
        n = self.n
        for k in range(n):
            for l in range(k + 1, n):
                lower = np.fromfunction(lambda i, j: (i == l) * (j == k) * 1,
                                        shape=(n + 1, n + 1),
                                        dtype=int)
                self.rotations.append(
                    1 / gs.sqrt(2 * self.metric_matrix) * (lower - lower.T))

        self.translations = []
        for k in range(n):
            self.translations.append(np.fromfunction(
                lambda i, j: (i == k) * (j == n) * 1 / gs.sqrt(
                    self.metric_matrix),
                shape=(n + 1, n + 1),
                dtype=int))

        self.basis = self.rotations + self.translations
        return self

    def set_metric_matrix(self, matrix):
        self.metric_matrix = matrix
        self.update_basis()

    def set_anisotropic_metric(self, beta):
        _metric_matrix = gs.ones((self.n + 1, ) * 2)
        _metric_matrix[0, self.n] = beta
        self.set_metric_matrix(_metric_matrix)

    def metric(self, x, y):
        r"""
        :math: \sum_{i, j} g_{ij}x_{ij}y_{ij} where g is the metric matrix
        """
        is_vectorized = \
            (gs.ndim(gs.array(x)) == 3) or (gs.ndim(gs.array(y)) == 3)
        axes = (2, 1) if is_vectorized else (0, 1)
        return gs.sum(x * y * self.metric_matrix, axes)

    def inner_product(self, tangent_vec_a, tangent_vec_b, base_point=None):
        if base_point is None:
            return self.metric(tangent_vec_a, tangent_vec_b)
        tan_a_at_id = self.compose(self.inverse(base_point), tangent_vec_a)
        tan_b_at_id = self.compose(self.inverse(base_point), tangent_vec_b)
        return self.metric(tan_a_at_id, tan_b_at_id)

    def norm(self, tan_a, base_point=None):
        return gs.sqrt(self.inner_product(tan_a, tan_a, base_point))

    def inverse(self, point):
        translation_mask = self.translation_mask
        embedded_rotations = point * gs.where(
            translation_mask == 1, translation_mask, gs.eye(self.n + 1))
        transposed_rot = gl.transpose(embedded_rotations)
        translation = point[..., :, -1]
        translation = gs.einsum(
            '...ij,...j',
            transposed_rot,
            translation) * gs.where(
                translation_mask[:, -1] == 2,
                - translation_mask[:, -1] / 2,
                gs.ones(self.n + 1))
        translation = gs.to_ndarray(translation, to_ndim=2)
        if len(translation) > 1:
            translation = gs.to_ndarray(translation, to_ndim=3)
            translation = translation.reshape(-1, self.n + 1, 1)
        else:
            translation = translation.T
        return gs.concatenate(
            [transposed_rot[..., :, :-1], translation], axis=-1)

    @staticmethod
    def crochet(x, y):
        """
        xy - yx
        """
        return gs.matmul(x, y) - gs.matmul(y, x)

    def structure_constant(self, x, y, z):
        """
        <[x,y],z>
        """
        return self.metric(self.crochet(x, y), z)

    def crochet_star(self, x, y):
        """
        <ad(x)*y, z> = <[x,z], y > pour tout z
        """
        return - gs.einsum(
            'i...,ijk->...jk',
            gs.array([self.structure_constant(z, x, y) for z in self.basis]),
            gs.array(self.basis))

    def nabla(self, x, y):
        return 1. / 2 * (self.crochet(x, y) - self.crochet_star(
            x, y) - self.crochet_star(y, x))

    def curvature(self, x, y, z):
        """
        R(x, y)z
        """
        return self.nabla(self.crochet(x, y), z) - self.nabla(
            x, self.nabla(y, z)) + self.nabla(y, self.nabla(x, z))

    def sectional_curvature(self, x, y):
        """
        < R(x, y)x, y> for x, y orthogonal. This is compensated if not
        """
        num = self.metric(y, self.curvature(x, y, x))
        denom = self.metric(x, x) * self.metric(y, y) - self.metric(x, y) ** 2
        condition = gs.isclose(denom, 0.)
        return gs.divide(num, denom, where=~condition)

    def sectional_curvature_at_point(
            self, tangent_vec_a, tangent_vec_b, base_point=None):
        if base_point is None:
            return self.sectional_curvature(tangent_vec_a, tangent_vec_b)
        tan_a_at_id = self.compose(self.inverse(base_point), tangent_vec_a)
        tan_b_at_id = self.compose(self.inverse(base_point), tangent_vec_b)
        return self.sectional_curvature(tan_a_at_id, tan_b_at_id)

    def nabla_curvature(self, x, y, z, t):
        r"""
        (\nabla_x R)(y, z)t
        """
        return self.nabla(x, self.curvature(y, z, t)) - self.curvature(
            self.nabla(x, y), z, t) - self.curvature(y, self.nabla(x, z),
                                                     t) - self.curvature(
            y, z, self.nabla(x, t))

    def nabla_curvature_at_point(
            self, tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d,
            base_point=None):
        if base_point is None:
            return self.nabla_curvature(
                tangent_vec_a, tangent_vec_b, tangent_vec_c, tangent_vec_d)
        tan_a_at_id = self.compose(self.inverse(base_point), tangent_vec_a)
        tan_b_at_id = self.compose(self.inverse(base_point), tangent_vec_b)
        tan_c_at_id = self.compose(self.inverse(base_point), tangent_vec_c)
        tan_d_at_id = self.compose(self.inverse(base_point), tangent_vec_d)
        return self.compose(
            base_point, self.nabla_curvature(
                tan_a_at_id, tan_b_at_id, tan_c_at_id, tan_d_at_id))

    def pole_ladder_error(self, u, v):
        r1 = self.nabla_curvature(v, u, v, 5 * u - 2 * v)
        r2 = self.nabla_curvature(u, u, v, v - 2 * u)
        return 1. / 12 * (r1 + r2)

    def lie_acceleration(self, point, vector):
        """
        vector is the left-angular velocity, that is always in the Lie Algebra
        """
        x = vector
        return gs.einsum('i...,ijk->...jk',
                         gs.array([self.metric(
                             self.crochet(x, z), x) for z in self.basis]),
                         gs.array(self.basis))

    @staticmethod
    def compose(g, h):
        return gs.matmul(g, h)

    def regularize(self, tangent_vec):
        """
        makes skew-symmetric, only for vector(s) in Lie Algebra
        """
        is_vectorized = gs.ndim(gs.array(tangent_vec)) == 3
        axes = (0, 2, 1) if is_vectorized else (1, 0)
        tangent_vec = tangent_vec * gs.where(
            self.translation_mask != 0., 1., 0.)
        tangent_vec = 1 / 2 * (tangent_vec - gs.transpose(tangent_vec, axes))
        tangent_vec = tangent_vec * self.translation_mask
        return tangent_vec

    @classmethod
    def _rk4_step(cls, state, force, dt, k1=None):
        point, vector = state
        if k1 is None:
            k1 = cls.compose(point, vector)
        l1 = force(point, vector)
        k2 = cls.compose(point + dt / 2 * k1, vector + dt / 2 * l1)
        l2 = force(point + dt / 2 * k1, vector + dt / 2 * l1)
        k3 = cls.compose(point + dt / 2 * k2, vector + dt / 2 * l2)
        l3 = force(point + dt / 2 * k2, vector + dt / 2 * l2)
        k4 = cls.compose(point + dt * k3, vector + dt * l3)
        l4 = force(point + dt * k3, vector + dt * l3)
        point_new = point + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        vector_new = vector + dt / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        return point_new, vector_new

    @classmethod
    def _rk2_step(cls, state, force, dt, k1=None):
        point, vector = state
        if k1 is None:
            k1 = cls.compose(point, vector)
        l1 = force(point, vector)
        k2 = cls.compose(point + dt / 2 * k1, vector + dt / 2 * l1)
        l2 = force(point + dt / 2 * k1, vector + dt / 2 * l1)
        point_new = point + dt * k2
        vector_new = vector + dt * l2
        return point_new, vector_new

    @classmethod
    def integrate(cls, function, initial_state, end_time=1.0, n_steps=10,
                  step='rk4', **kwargs):
        dt = end_time / n_steps
        positions = [initial_state[0]]
        velocities = [initial_state[1]]
        current_state = (positions[0], velocities[0])
        step_function = cls._rk4_step if step=='rk4' else cls._rk2_step
        for _ in range(n_steps):
            current_state = step_function(current_state, function, dt)
            positions.append(current_state[0])
            velocities.append(current_state[1])
        return positions, velocities

    def exp(self, tangent_vec, base_point, n_steps=10, step='rk4', **kwargs):
        left_angular_vel = self.compose(self.inverse(base_point), tangent_vec)
        initial_state = (base_point, self.regularize(left_angular_vel))
        flow, _ = self.integrate(
            self.lie_acceleration, initial_state, n_steps=n_steps, step=step)
        return flow[-1]

    def log(
            self, point, base_point, n_steps=15, step='rk4', verbose=False,
            max_iter=25, tol=1e-10):

        def objective(velocity):
            """Define the objective function."""
            velocity = velocity.reshape(base_point.shape)
            delta = self.exp(velocity, base_point, n_steps, step) - point
            return gs.sum(delta ** 2)

        objective_with_grad = value_and_grad(objective)

        tangent_vec = gs.random.rand(base_point.size)
        res = minimize(
            objective_with_grad, tangent_vec, method='L-BFGS-B',
            jac=True, options={'disp': verbose, 'maxiter': max_iter}, tol=tol)

        tangent_vec = res.x
        tangent_vec = gs.reshape(tangent_vec, base_point.shape)
        tangent_vec = gl.compose(base_point, self.regularize(
            gl.compose(self.inverse(base_point), tangent_vec)))
        return tangent_vec

    def symmetry(self, point, base_point, **kwargs):
        return self.exp(- self.log(point, base_point, **kwargs), base_point,
                        **kwargs)

    def midpoint(self, point_1, point_2, **kwargs):
        return self.exp(
            1. / 2 * self.log(point_1, point_2, **kwargs), point_2, **kwargs)

    def _pole_ladder_step(self, base_point, next_point, base_shoot, **kwargs):
        midpoint_ = self.midpoint(next_point, base_point, **kwargs)
        first_sym = self.symmetry(base_shoot, midpoint_, **kwargs)
        transported_vec = - self.log(first_sym, next_point, **kwargs)
        next_shoot = self.exp(transported_vec, next_point, **kwargs)
        return transported_vec, next_shoot

    def _schild_ladder_step(self, base_point, next_point, base_shoot, **kwargs):
        midpoint_ = self.midpoint(next_point, base_shoot, **kwargs)
        first_sym = self.symmetry(base_point, midpoint_, **kwargs)
        transported_vec = self.log(first_sym, next_point, **kwargs)
        return transported_vec, first_sym

    def ladder_parallel_transport(
            self, tan_a, tan_b, base_point, n_rungs=1, scheme='pole',
            **kwargs):
        current_point = gs.copy(base_point)
        next_tangent_vec = gs.copy(tan_a)
        base_shoot = self.exp(
            base_point=current_point, tangent_vec=next_tangent_vec, **kwargs)
        single_step = self._schild_ladder_step if scheme == 'schild' \
            else self._pole_ladder_step
        current_speed = 1. / n_rungs * gs.copy(tan_b)

        for i_point in range(n_rungs):
            left_angular_vel = self.compose(
                gs.linalg.inv(current_point), current_speed)
            initial_state = (current_point, self.regularize(left_angular_vel))
            flow, vel = self.integrate(
                self.lie_acceleration, initial_state, **kwargs)
            next_point = flow[-1]
            current_speed = self.compose(next_point, vel[-1])
            next_tangent_vec, base_shoot = single_step(
                base_point=current_point,
                next_point=next_point,
                base_shoot=base_shoot,
                **kwargs)
            current_point = next_point
        transported_tangent_vec = next_tangent_vec

        return transported_tangent_vec, current_point

    def pole_ladder(self, tan_a, tan_b, base_point, n_rungs=1, **kwargs):
        current_shoot = self.exp(tan_a, base_point, **kwargs)
        left_angular_vel = self.compose(self.inverse(base_point), tan_b)
        initial_state = (base_point, self.regularize(left_angular_vel))
        flow, _ = self.integrate(
            self.lie_acceleration, initial_state, n_steps=2 * n_rungs,
            step='rk4')
        for i_point in range(n_rungs):
            mid_point = flow[2 * i_point + 1]
            current_shoot = self.symmetry(current_shoot, mid_point, **kwargs)
        end_point = flow[-1]
        transported = self.log(current_shoot, end_point, **kwargs)
        if n_rungs % 2 == 1:
            transported *= -1.
        return transported, end_point

    def parallel_transport(self, tangent_vec_a, tangent_vec_b, base_point):
        """
        The best approximation we can make, to serve as ground truth in simulations
        """
        n_rungs = 120
        beta = 1
        ladder, end_point = self.pole_ladder(
            tangent_vec_a / (n_rungs ** beta), tangent_vec_b, base_point,
            n_steps=1, step='rk4', n_rungs=n_rungs, tol=1e-14)
        transported = ladder * (n_rungs ** beta)
        return transported, end_point

    def fanning_scheme(
            self, tan_a, tan_b, base_point, n_rungs=1,two_perturbed=False,
            rk_order=2):
        step = self._rk4_step if rk_order == 4 else self._rk2_step
        dt = 1 / n_rungs
        left_angular_vel = self.compose(self.inverse(base_point), tan_b)
        next_state = (base_point, self.regularize(left_angular_vel))
        perturbed_vel_plus = self.compose(self.inverse(base_point), dt * tan_a)
        perturbed_state = (
            base_point, left_angular_vel + self.regularize(perturbed_vel_plus))
        if two_perturbed:
            perturbed_state_neg = (
                base_point,
                left_angular_vel - self.regularize(perturbed_vel_plus))
        transported = tan_a
        for i_point in range(n_rungs):
            next_state = step(
                force=self.lie_acceleration, state=next_state, dt=dt)
            perturbed_point, _ = step(
                state=perturbed_state, force=self.lie_acceleration, dt=dt)
            if two_perturbed:
                perturbed_neg, _ = step(
                    state=perturbed_state_neg,
                    force=self.lie_acceleration, dt=dt)
                transported = (perturbed_point - perturbed_neg) / 2 / dt
                perturbed_vel_neg = self.regularize(
                    next_state[1] - self.compose(
                        self.inverse(next_state[0]), transported))
                perturbed_state_neg = (
                    next_state[0], self.regularize(perturbed_vel_neg))
            else:
                transported = (perturbed_point - next_state[0]) / dt
            perturbed_vel_plus = self.regularize(next_state[1] + self.compose(
                self.inverse(next_state[0]), transported))
            perturbed_state = (
                next_state[0], self.regularize(perturbed_vel_plus))
        end_point = next_state[0]
        transported /= dt
        return transported, end_point
