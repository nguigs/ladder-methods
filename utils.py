"""Created by nguigui on 6/8/20."""

import matplotlib.pyplot as plt

import geomstats.backend as gs
from geomstats.geometry.hypersphere import Hypersphere
from geomstats.geometry.spd_matrices import SPDMatrices, SPDMetricAffine,\
    Matrices

SPHERE = Hypersphere(2)
SPD = SPDMatrices(3)
SPDMetric = SPDMetricAffine(3)


def random_orthonormal_sphere(point):
    tan_b = gs.random.rand(3)
    tan_b = SPHERE.to_tangent(tan_b, point)
    tan_b /= gs.linalg.norm(tan_b, axis=-1)

    tan_a = gs.random.rand(3)
    tan_a = SPHERE.to_tangent(tan_a, point)
    tan_a = tan_a - gs.dot(tan_a, tan_b) * tan_b
    tan_a /= gs.linalg.norm(tan_a, axis=-1)
    return tan_a, tan_b


def random_orthonormal_spd(point):
    point_sqrt = SPD.powerm(point, 1. / 2)

    tan_b = Matrices(3, 3).random_uniform()
    tan_b = Matrices.to_symmetric(tan_b)

    # generate vector orthonormal to tan_b
    tan_a = Matrices(3, 3).random_uniform()
    tan_a = Matrices.to_symmetric(tan_a)
    tan_a[0, 0] -= gs.sum(tan_b * tan_a, axis=(-1, -2)) / tan_b[0, 0]

    # normalize and move to base_point
    tan_b = Matrices.mul(point_sqrt, tan_b, point_sqrt)
    tan_a = Matrices.mul(point_sqrt, tan_a, point_sqrt)
    tan_b = gs.einsum(
        '...ij,...->...ij', tan_b, 1. / SPDMetric.norm(tan_b, point))
    tan_a = gs.einsum(
        '...ij,...->...ij', tan_a, 1. / SPDMetric.norm(tan_a, point))

    return tan_a, tan_b


def make_plot(
        ax, y_list, x_list, col='gray', title='Convergence error',
        xlabel='1 / n', **kwargs):
    ax.plot(
        x_list, y_list, marker='o', linewidth=1, c=col, linestyle='dashed',
        fillstyle='none', **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.legend(loc='best')
