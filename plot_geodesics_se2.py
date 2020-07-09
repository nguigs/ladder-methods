import geomstats.backend as gs
import matplotlib.pyplot as plt
from numpy import linspace

from sen_tools.sen import SenTools

tools = SenTools(2)


def plot_one_param_sg():
    vecs = gs.array([gs.pi, 1, 1])
    tan_b = gs.einsum('i,ijk->jk', vecs, tools.basis)
    points = gs.linalg.expm(
        gs.einsum('i,jk->ijk', linspace(0, 1, 30), tan_b))
    translation = points[:, :2, 2]
    frame_1 = points[:, :2, 0]
    frame_2 = points[:, :2, 1]
    fig = plt.figure(figsize=(8, 8))
    plt.quiver(translation[:, 0], translation[:, 1], frame_1[:, 0],
               frame_1[:, 1], width=0.005, color='b')
    plt.quiver(translation[:, 0], translation[:, 1], frame_2[:, 0],
               frame_2[:, 1], width=0.005, color='r')
    plt.scatter(translation[:, 0], translation[:, 1], color='black')
    ax = fig.axes[0]
    ax.set_ylim([-1, 1])
    ax.set_xlim([-1, 1])
    plt.show()


def plot(vec, annotate=False, n_points=15):
    fig = plt.figure()
    maxs_x = []
    mins_y = []
    maxs = []
    for i, beta in enumerate([1, 3, 5]):
        ax = plt.subplot(f'{3}1{i + 1}')
        tools.set_anisotropic_metric(beta)
        tan_b = gs.einsum('i,ijk->jk', vec, tools.basis)
        points = tools.exp(
            gs.einsum('i,jk->ijk', linspace(0, 1, n_points), tan_b),
            tools.identity)
        translation = points[:, :2, 2]
        frame_1 = points[:, :2, 0]
        frame_2 = points[:, :2, 1]
        plt.quiver(translation[:, 0], translation[:, 1], frame_1[:, 0],
                   frame_1[:, 1], width=0.005, color='b')
        plt.quiver(translation[:, 0], translation[:, 1], frame_2[:, 0],
                   frame_2[:, 1], width=0.005, color='r')
        plt.scatter(translation[:, 0], translation[:, 1],
                    label=r'$\beta={}$'.format(beta), color='black')
        plt.legend()
        if annotate:
            for t, x, y in zip(linspace(0, 1, n_points), translation[:, 0],
                               translation[:, 1]):
                plt.annotate("{:.2f}".format(t), (x, y))
        mins_y.append(min(translation[:, 1]))
        maxs.append(max(translation[:, 1]))
        maxs_x.append(max(translation[:, 0]))
    for ax in fig.axes:
        y_lims = [min(mins_y) - 1, max(maxs) + 1]
        x_lim_inf, _ = plt.xlim()
        x_lims = [x_lim_inf, 1.1 * max(maxs_x)]
        ax.set_ylim(y_lims)
        ax.set_xlim(x_lims)
    plt.show()


if __name__ == '__main__':
    vec = 3 * gs.array([gs.pi / 4, 1, 1])
    plot(vec)
