"""Created by nguigui on 6/8/20."""

import matplotlib.pyplot as plt

import geomstats.backend as gs
import geomstats.visualization as visualization
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


def show_schild(tan_a, tan_b, point, n_rungs=4, n_points=15):
    metric = SPHERE.metric
    ladder = metric.ladder_parallel_transport(
        tan_a, tan_b, point, n_rungs=n_rungs, return_geodesics=True,
        scheme='schild')
    pole_ladder = ladder['transported_tangent_vec']
    trajectory = ladder['trajectory']
    first_geo = metric.geodesic(initial_point=point, initial_tangent_vec=tan_a)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    sphere_visu = visualization.Sphere(n_meridians=30)
    ax = sphere_visu.set_ax(ax=ax)

    t = gs.linspace(0, 1, n_points)
    t_diag = gs.linspace(0, 1, n_points * 4)
    sphere_visu.draw_points(ax, first_geo(t), marker='o', c='green', s=2)
    for points in trajectory:
        main_geodesic, diagonal, second_diagonal, final_geodesic = points
        sphere_visu.draw_points(
            ax, main_geodesic(t_diag), marker='o', c='blue', s=2)
        sphere_visu.draw_points(ax, diagonal(t), marker='o', c='r', s=2)
        sphere_visu.draw_points(ax, second_diagonal(t), marker='o', c='r', s=2)
        sphere_visu.draw_points(
            ax, final_geodesic(t), marker='o', c='green', s=2)

    tangent_vectors = gs.stack(
        [tan_b / n_rungs, tan_a, pole_ladder], axis=0)
    base_point = gs.to_ndarray(point, to_ndim=2)
    origin = gs.concatenate(
        [base_point, base_point, final_geodesic(0.)], axis=0)
    ax.quiver(
        origin[:, 0], origin[:, 1], origin[:, 2],
        tangent_vectors[:, 0], tangent_vectors[:, 1], tangent_vectors[:, 2],
        color=['black', 'black', 'black'],
        linewidth=2)
    sphere_visu.draw(ax, linewidth=1)
    return fig
