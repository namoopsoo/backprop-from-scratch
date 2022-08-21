
import pylab
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
import numpy as np

from utils import utc_now, utc_ts


def plot_loss_vec(loss_vec):
    with plt.style.context("fivethirtyeight"):
        plt.plot(loss_vec)
        plt.xlabel("rounds")
        plt.ylabel("log loss")
        plt.title("log loss")
        out_loc = f"{utc_ts(utc_now())}.png"
        print("saving to", out_loc)
        pylab.savefig(out_loc, bbox_inches="tight")
        pylab.close()
    return out_loc

def surface_plot_type1(x, y, z, how_many=None):
    # reference:  https://www.geeksforgeeks.org/3d-surface-plotting-in-python-using-matplotlib/
    if how_many is None:
        how_many = x.shape[0]
    x = x[:how_many]
    y = y[:how_many]
    z = z[:how_many]

    # Creating figure
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')

    # Creating plot
    ax.plot_surface(x, y, z)
    out_loc = f"{n.utc_ts(n.utc_now())}-surface.png"
    pylab.savefig(out_loc, bbox_inches='tight')
    pylab.close()

def plot_type3(x, y, z, how_many=None):
    # reference: https://www.geeksforgeeks.org/3d-surface-plotting-in-python-using-matplotlib/

    if how_many is None:
        how_many = x.shape[0]

    x = x[:how_many]
    y = y[:how_many]
    z = z[:how_many]

    # Creating figure
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')

    # Creating color map
    my_cmap = plt.get_cmap('hot')

    # Creating plot
    surf = ax.plot_surface(x, y, z,
                        rstride = 8,
                        cstride = 8,
                        alpha = 0.8,
                        cmap = my_cmap)
    cset = ax.contourf(x, y, z,
                    zdir ='z',
                    offset = np.min(z),
                    cmap = my_cmap)
    cset = ax.contourf(x, y, z,
                    zdir ='x',
                    offset =-5,
                    cmap = my_cmap)
    cset = ax.contourf(x, y, z,
                    zdir ='y',
                    offset = 5,
                    cmap = my_cmap)
    fig.colorbar(surf, ax = ax,
                shrink = 0.5,
                aspect = 5)

    # Adding labels
    ax.set_xlabel('X-axis')
    ax.set_xlim(-5, 5)
    ax.set_ylabel('Y-axis')
    ax.set_ylim(-5, 5)
    ax.set_zlabel('Z-axis')
    ax.set_zlim(np.min(z), np.max(z))
    ax.set_title('3D surface having 2D contour plot projections')

    out_loc = f"{utc_ts(utc_now())}.png"
    print("saving to", out_loc)
    pylab.savefig(out_loc, bbox_inches="tight")
    pylab.close()



def scatter_plot_groups(X, Y):
    # reference, https://pythonspot.com/matplotlib-scatterplot/

    indexes_0 = [i for (i, y) in enumerate(Y) if y == 0]
    indexes_1 = [i for (i, y) in enumerate(Y) if y == 1]
    data = (X[indexes_0, :], X[indexes_1, :])
    colors = ("red", "blue")
    groups = ("0", "1")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # , axisbg="1.0")
    for data, color, group in zip(data, colors, groups):
        x, y = data[:, 0], data[:, 1]
        ax.scatter(x, y, alpha=0.8, c=color, edgecolors="none", s=30, label=group)

    plt.title("0s and 1s")
    plt.legend(loc=2)


    out_loc = f"{utc_ts(utc_now())}-scatter.png"
    print("saving to", out_loc)
    pylab.savefig(out_loc, bbox_inches="tight")
    pylab.close()

