import numpy as np
import ipdb
from functools import partial
from sklearn.utils.extmath import softmax
from sklearn.metrics import log_loss
from collections import namedtuple
from tqdm import tqdm

from utils import utc_now, utc_ts

Layer = namedtuple("Layer", ["weights", "bias", "nodes"])

relu_scalar = partial(max, 0)
relu = np.vectorize(relu_scalar)

def derivative_of_relu(h_net):
    if h_net > 0:
        return 1
    else:
        return 0


def feed_forward(x, layers, verbose=False):
    # TODO make this take arbitrary number of inputs, i.e. , vectorize it.
    values = []
    H = x
    for (i, layer) in enumerate(layers):
        if verbose:
            print(i, layer)
        weights = concat_bias_weights(layer.weights)
        H = np.hstack([H, layer.bias])

        if verbose:
            print("sizes:", H.shape, weights.shape)

        net_H = np.matmul(H, weights)
        H = relu(net_H)

        if i == 0:
            layer.nodes["net_h1"] = net_H[0]
            layer.nodes["h1"] = H[0]
            layer.nodes["net_h2"] = net_H[1]
            layer.nodes["h2"] = H[1]
            layer.nodes["net_h3"] = net_H[2]
            layer.nodes["h3"] = H[2]
        elif i == 1:
            layer.nodes["net_h4"] = net_H[0]
            layer.nodes["h4"] = H[0]
            layer.nodes["net_h5"] = net_H[1]
            layer.nodes["h5"] = H[1]
        elif i == 2:
            layer.nodes["net_y_logit"] = net_H[0]
            layer.nodes["y_logit"] = H[0]

        if verbose:
            print("output of relu", H)

    y_logit = H[0]
    y_prob = logit_to_prob(y_logit) # aka sigmoid
    layers[-1].nodes["y_prob"] = y_prob


    assert isinstance(y_prob, np.int64) or isinstance(y_prob, np.float64)

    if np.isnan(y_prob):
        import ipdb; ipdb.set_trace()
    return y_prob

def logit_to_prob(y_logit):  # aka sigmoid
    # Well since this neural net is not returning a probability by default, 
    #   then we need something like softmax to do that. 
    #   Also this person , https://sebastiansauer.github.io/convert_logit2prob/ , 
    #   notes another possible option is 
    #
    #   given a logit, 
    #       odds = exp(logit)
    #       prob = odds / (1 + odds)
    # 
    # Hmm thing is I can't use softmax since I only have a single output logit.
    # so just using sigmoid..
    return 1 / (1 + np.exp(-y_logit))


def derivative_of_logit_to_prob_func(y_logit):
    return np.exp(y_logit) / ((1 + np.exp(y_logit))**2)


def loss(layers, X, Y):
    """Given the MLP and the dataset, find the loss.

    Args:
        layers: MLP layers
        X, Y: the dataset
    """

    Y_actual = []
    for i in tqdm(range(X.shape[0])):
        y = feed_forward(X[i], layers)
        Y_actual.append(y)

    Y_actual = np.array(Y_actual)

    total_loss = log_loss(Y, Y_actual, labels=[0, 1])
    return Y_actual, total_loss


def derivative_of_log_loss(y, y_prob):

    # Note, using y_hat and y_prob interchangeably

    if y_prob == 1:
        y_prob -= .00001
    elif y_prob == 0:
        y_prob -= .00001

    result = -y / y_prob - (1 - y) / (1 - y_prob)

    return result



def concat_bias_weights(x):
    num_cols = x.shape[1]
    return np.concatenate([x, np.ones((1, num_cols))])


def initialize_network_layers():
    layers = [
        Layer(
            weights=np.random.random((2, 3)), bias=np.array([1]),
            nodes={
                "net_h1": None, "h1": None,
                "net_h2": None, "h2": None,
                "net_h3": None, "h3": None,
            }
        ),
        Layer(
            weights=np.random.random((3, 2)), bias=np.array([1]),
            nodes={
                "net_h4": None, "h4": None,
                "net_h5": None, "h5": None,
            }
        ), 
        Layer(weights=np.random.random((2, 1)), bias=np.array([0]),
            nodes={
                "net_y_logit": None, "y_logit": None,
                "y_prob": None,
            }
        ), 
    ]
    return layers


def build_dataset_inside_outside_circle():
    # Create some data in a 20x20 box centered at origin.
    radius = 5
    X = np.random.random((10000, 2)) * 40 + -20
    f = (lambda a: int(np.sqrt(a[0]**2 + a[1]**2) <= radius))
    Y = np.array(list(map(f, X)))
    return X, Y




def train_network(X, Y, layers, log_loss_each_round=False):
    # sgd loop
    learning_rate = 0.5

    loss_vec = []

    num_examples = X.shape[0]
    for step in tqdm(range(60)):
        i = np.random.choice(range(num_examples))
        # sample minibatch , (x, y), 
        x, y = X[i], Y[i]

        # do the feed forward for (x, y) that. 
        y_prob = feed_forward(x, layers, verbose=False)

        # TODO actually oops, if we are recording the loss, want to do it for everything in a validation set. 
        if log_loss_each_round:
            _, total_loss = loss(layers, X, Y)
            loss_vec.append(total_loss)

        # for parameter in all_parameters:

        # calculate partial derivative at (x, y)
        pd_loss_wrt_w13 = calc_partial_derivative_of_loss_wrt_w13(layers, y, )

        pd_loss_wrt_w14 = calc_partial_derivative_of_loss_wrt_w14(layers, y, )
        # then update the parameter using the learning rate.
        #   storing in temporary values until later.


        # now finally update the actual parameters.
        layers[-1] = layers[-1]._replace(
            weights=layers[-1].weights + np.array(
                    [
                        [-1 * pd_loss_wrt_w13 * learning_rate],
                        [-1 * pd_loss_wrt_w14 * learning_rate]])
                )

        # next, 
        pd_loss_wrt_the_layer_1_weights = (
            calc_partial_derivative_of_loss_wrt_w_on_layer_1(
                layers, pd_loss_wrt_w13, pd_loss_wrt_w14,
            ))
        layers[1] = layers[1]._replace(
            weights=layers[1].weights + np.array(
                [
                    [pd_loss_wrt_the_layer_1_weights["w7"],
                        pd_loss_wrt_the_layer_1_weights["w8"], ],
                    [pd_loss_wrt_the_layer_1_weights["w9"],
                        pd_loss_wrt_the_layer_1_weights["w10"], ],
                    [pd_loss_wrt_the_layer_1_weights["w11"],
                        pd_loss_wrt_the_layer_1_weights["w12"], ],
                ]))




        
    _, total_loss = loss(layers, X, Y)
    loss_vec.append(total_loss)
    return loss_vec, layers

def calc_partial_derivative_of_loss_wrt_w_on_layer_1(
    layers, 
    pd_loss_wrt_w13, 
    pd_loss_wrt_w14, 
):
    # so layer 1 weights are w7, w9, w11, w8, w10, w12, 
    h1 = layers[0].nodes["h1"]
    h2 = layers[0].nodes["h2"]
    h3 = layers[0].nodes["h3"]
    net_h4 = layers[1].nodes["net_h4"]
    net_h5 = layers[1].nodes["net_h5"]
    pd_loss_wrt_weights = {
        # for h4 weights, 
        "w7": (pd_loss_wrt_w13
            * derivative_of_relu(net_h4)
            * h1),
        "w9": (pd_loss_wrt_w13
            * derivative_of_relu(net_h4)
            * h2),
        "w11": (pd_loss_wrt_w13
            * derivative_of_relu(net_h4)
            * h3),

        # for h5 weights, 
        "w8": (pd_loss_wrt_w14
            * derivative_of_relu(net_h5)
            * h1),
        "w10": (pd_loss_wrt_w14
            * derivative_of_relu(net_h5)
            * h2),
        "w12": (pd_loss_wrt_w14
            * derivative_of_relu(net_h5)
            * h3),
    }


def calc_partial_derivative_of_loss_wrt_w13(layers, y, ):

    # net_y = w13*h4 + w14*h5 
    # y_logit = relu(net_y)
    # y_prob = logit_to_prob(y_logit)
    # loss = log_loss(y_actual, y_prob)

    # by chain rule, 
    # derivative = pd_log_loss_wrt_prob * pd_prob_wrt_logit * pd_logit_wrt_net_y * pd_net_y_wrt_w13

    # y = y
    y_prob = layers[-1].nodes["y_prob"]
    y_logit = layers[-1].nodes["y_logit"]
    net_y_logit = layers[-1].nodes["net_y_logit"]   # TODO most likely I should just change the activation function here.
    h4 = layers[-2].nodes["h4"]

    g = (
        derivative_of_log_loss(y, y_prob)
        * derivative_of_logit_to_prob_func(y_logit)
        * derivative_of_relu(net_y_logit)
        * h4
    )
    return g


def calc_partial_derivative_of_loss_wrt_w14(layers, y, ):

    # net_y = w13*h4 + w14*h5 
    # y_logit = relu(net_y)
    # y_prob = logit_to_prob(y_logit)
    # loss = log_loss(y_actual, y_prob)

    # by chain rule, 
    # derivative = pd_log_loss_wrt_prob * pd_prob_wrt_logit * pd_logit_wrt_net_y * pd_net_y_wrt_w14

    # y = y
    y_prob = layers[-1].nodes["y_prob"]
    y_logit = layers[-1].nodes["y_logit"]
    net_y_logit = layers[-1].nodes["net_y_logit"]   # TODO most likely I should just change the activation function here.
    h5 = layers[-2].nodes["h5"]

    g = (
        derivative_of_log_loss(y, y_prob)
        * derivative_of_logit_to_prob_func(y_logit)
        * derivative_of_relu(net_y_logit)
        * h5
    )
    return g
