
import pylab
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from mpl_toolkits import mplot3d
import numpy as np


# pip install colormap
# pip install easydev
from colormap import rgb2hex, rgb2hls, hls2rgb

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


def plot_train_and_validation_loss_vec(train_loss_vec, validation_loss_vec):
    with plt.style.context("fivethirtyeight"):
        plt.plot(train_loss_vec, label="train", color="blue")
        plt.plot(validation_loss_vec, label="validation", color="green")
        plt.xlabel("rounds")
        plt.ylabel("log loss")
        plt.title("log loss")
        plt.legend()
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
    """
    Args:
        X: (x, y) coordinates
        Y: Here, just expect Y has only 0s and 1s. 

    # reference, https://pythonspot.com/matplotlib-scatterplot/
    """


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


def scatter_plot_by_z(X, Y, scaled=False):
    """
    Args:
        X: (x, y) coordinates
        Y: float from 0 to 1
        scaled: apply MinMaxScaler to the data or not
            
    """
    # expecting that 
    x, y = X[:, 0], X[:, 1]

    if scaled:
        scaler = MinMaxScaler()
        Y = scaler.fit_transform(Y.reshape(-1, 1))
        
    colors = map_values_to_colors(Y)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1) # , axisbg="1.0")
    ax.scatter(x, y, alpha=0.8, c=colors, edgecolors="none", s=10)
    out_loc = f"{utc_ts(utc_now())}-scatter.png"
    print("saving to", out_loc)
    pylab.savefig(out_loc, bbox_inches="tight")
    pylab.close()
    return out_loc


def map_values_to_colors(Y):

    # base color is a light green
    r, g, b = 0, 253, 150

    colors = [
        darken_color(r, g, b, factor=y)
        for y in Y
    ]
    return colors


def hex_to_rgb(hex):
     hex = hex.lstrip('#')
     hlen = len(hex)
     return tuple(int(hex[i:i+hlen//3], 16) for i in range(0, hlen, hlen//3))

def adjust_color_lightness(r, g, b, factor):
    h, l, s = rgb2hls(r / 255.0, g / 255.0, b / 255.0)
    l = max(min(l * factor, 1.0), 0.0)
    r, g, b = hls2rgb(h, l, s)
    return rgb2hex(int(r * 255), int(g * 255), int(b * 255))

def darken_color(r, g, b, factor=0.1):
    return adjust_color_lightness(r, g, b, 1 - factor)


def plot_model_weights_across_rounds(model, artifacts):
    num_artifacts = len(artifacts)
    num_layers = len(artifacts["0"]["model"].layers)
    max_num_weights_per_layer = max([len(layer.weights.flatten()) for layer in artifacts["0"]["model"].layers])
    num_weights = sum([len(layer.weights.flatten()) for layer in artifacts["0"]["model"].layers])

    fig = plt.figure(figsize=(12,8))

    for i  in range(num_layers):
        for j, _ in enumerate(artifacts["0"]["model"].layers[i].weights.flatten()):
            ax = fig.add_subplot(num_layers, max_num_weights_per_layer, 1 + i*max_num_weights_per_layer + j)
            ax.plot([
              artifacts[str(k * 10)]["model"].layers[i].weights.flatten()[j] for k in range(num_artifacts)
            ])
            ax.set(title=f"layer={i}, weight={j}", xlabel="rounds")

    out_loc = f"{utc_ts(utc_now())}-weights.png"
    fig.tight_layout()  # nice tip from https://www.geeksforgeeks.org/how-to-set-the-spacing-between-subplots-in-matplotlib-in-python/ , just learned about this !
    pylab.savefig(out_loc, bbox_inches='tight')
    return out_loc



def plot_simple_historgram(Y_prob, label):
    fig = plt.figure(figsize=(4,3))
    ax = fig.add_subplot(111)
    out_loc = f"{utc_ts(utc_now())}-hist.png"
    ax.hist(Y_prob, bins=50)
    ax.set(title=f"{label} histogram")
    pylab.savefig(out_loc, bbox_inches='tight')
    return out_loc



def micro_batch_delta_loss_plot(metrics):
    deltas = [x["loss_after"] - x["loss_before"] for x in metrics["micro_batch_updates"]]
    with plt.style.context("fivethirtyeight"):
        fig = plt.figure(figsize =(20, 9))

        plt.plot(deltas, linewidth=0.7)
        plt.title("Microbatch loss_after - loss_before")
        out_loc = f"{utc_ts(utc_now())}-micro-batch-loss-deltas-over-steps.png"
        print("saving to", out_loc)
        pylab.savefig(out_loc, bbox_inches="tight")
        pylab.close()
        plt.close()
    return out_loc


def plot_grid(vec, side, title):

    with plt.style.context("fivethirtyeight"):
        fig = plt.figure(figsize=(20, 20))
        for i in range(side*side):
            values, subtitle = vec[i]
            ax = fig.add_subplot(side, side, i + 1)
            ax.hist(values, linewidth=0.7)
            ax.set(title=subtitle)

        out_loc = f"{utc_ts(utc_now())}-{title}.png"
        fig.tight_layout() 
        print("saving to", out_loc)
        pylab.savefig(out_loc, bbox_inches="tight")
        pylab.close()
        plt.close()
    return out_loc
