
import pylab
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
import numpy as np

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
    # Creating figure
    fig = plt.figure(figsize =(14, 9))
    ax = plt.axes(projection ='3d')

    x = X[:how_many, 0]
    y = X[:how_many, 1]
    z = np.reshape(Y[:how_many], (how_many, 1))
    # Creating plot
    ax.plot_surface(x, y, z)
    out_loc = f"{n.utc_ts(n.utc_now())}-surface.png"
    pylab.savefig(out_loc, bbox_inches='tight')
    pylab.close()

def plot_type3(x, y, z, how_many=None):
    # reference: https://www.geeksforgeeks.org/3d-surface-plotting-in-python-using-matplotlib/

    if how_many is None:
        how_many = x.shape[0]
    # Creating dataset
    # x = np.outer(np.linspace(-3, 3, 32), np.ones(32))
    # y = x.copy().T # transpose
    # z = (np.sin(x **2) + np.cos(y **2) )

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
