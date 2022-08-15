import numpy as np
from functools import partial
from sklearn.utils.extmath import softmax
from sklearn.metrics import log_loss
from collections import namedtuple

Layer = namedtuple("Layer", ["weights", "bias", "nodes"])

relu_scalar = partial(max, 0)
relu = np.vectorize(relu_scalar)

def derivative_of_relu(h_net):
    if h_net > 0:
        return 1
    else:
        return 0


def feed_forward(x, layers):
    values = []
    H = x
    for (i, layer) in enumerate(layers):
        print(i, layer)
        weights = concat_bias_weights(layer.weights)
        H = np.hstack([H, layer.bias])
        print("sizes:", H.shape, weights.shape)
        net_H = np.matmul(H, weights)
        H = relu(net_H)

        # TODO update the layer
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

        print("output of relu", H)

    y_logit = H[0]
    y_prob = logit_to_prob(y_logit)
    layers[-1].nodes["y_prob"] = y_prob


    assert isinstance(y_prob, np.int64) or isinstance(y_prob, np.float64)
    return y_prob

def logit_to_prob(y_logit):
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

    odds = np.exp(y_logit)
    prob = odds / (1 + odds)
    return prob


def derivative_of_logit_to_prob_func(y_logit):
    return np.exp(y_logit) / ((1 + np.exp(y_logit))**2)


def loss(y, y_hat):
    return log_loss(y, y_hat, labels=[0, 1])


def derivative_of_log_loss(y, y_hat):
    # using y_hat and y_prob interchangeably
    return -y / y_hat - (1 - y) / (1 - y_hat)



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
        Layer(weights=np.array([[1], [1]]), bias=np.array([0]),
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
X, Y = build_dataset_inside_outside_circle() 





def train_network():
    # sgd loop

    # sample minibatch , (x, y), 
    # do the feed forward for (x, y) that. 
    for step in range(10):

        for parameter in all_parameters:

            # calculate partial derivative at (x, y)

            # then update the parameter using the learning rate.
            #   storing in temporary values until later.
            ...


        # now finally update the actual parameters.



def calc_partial_derivative_of_loss_wrt_w13(layers, y):

    # net_y = w13*h4 + w14*h5 
    # y_logit = relu(net_y)
    # y_prob = logit_to_prob(y_logit)
    # loss = log_loss(y_actual, y_prob)

    # by chain rule, 
    # derivative = pd_log_loss_wrt_prob * pd_prob_wrt_logit * pd_logit_wrt_net_y * pd_net_y_wrt_w13

    # y = y
    y_hat = layers[-1].nodes["y_hat"]
    y_logit = layers[-1].nodes["y_logit"]
    net_y_logit = layers[-1].nodes["net_y_logit"]   # TODO most likely I should just change the activation function here.
    h4 = layers[-2].nodes["h4"]

    return (
        derivative_of_log_loss(y, y_hat)
        * derivative_of_logit_to_prob_func(y_logit)
        * derivative_of_relu(net_y_logit)
        * h4
    )

