import numpy as np
import pandas as pd
from functools import partial
from itertools import product
from sklearn.utils.extmath import softmax
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from collections import namedtuple, defaultdict
from copy import deepcopy
from tqdm import tqdm

from utils import utc_now, utc_ts

Layer = namedtuple("Layer", ["weights", "bias", "nodes"])
Model = namedtuple("Model", ["layers", "parameters", "min_max_scaler"])

relu_scalar = partial(max, 0)
relu = np.vectorize(relu_scalar)

def derivative_of_relu(h_net):
    if h_net > 0:
        return 1
    else:
        return 0


def feed_forward(x, model, verbose=False, scale_output=True):
    # TODO make this take arbitrary number of inputs, i.e. , vectorize it.
    values = []
    H = x
    for (i, layer) in enumerate(model.layers):
        if verbose:
            print(i, layer)
        weights = concat_bias_weights(layer.weights)
        H = np.hstack([H, layer.bias])

        if verbose:
            print("sizes:", H.shape, weights.shape)

        net_H = np.matmul(H, weights)

        if i == 0:
            # H = relu(net_H)
            H = logit_to_prob(net_H)

            layer.nodes["net_h1"] = net_H[0]
            layer.nodes["h1"] = H[0]
            layer.nodes["net_h2"] = net_H[1]
            layer.nodes["h2"] = H[1]
            layer.nodes["net_h3"] = net_H[2]
            layer.nodes["h3"] = H[2]
        elif i == 1:
            # H = relu(net_H)
            H = logit_to_prob(net_H)

            layer.nodes["net_h4"] = net_H[0]
            layer.nodes["h4"] = H[0]
            layer.nodes["net_h5"] = net_H[1]
            layer.nodes["h5"] = H[1]
        elif i == 2:
            H = net_H # TODO since this is the same now , remove one to avoid confusion.
            layer.nodes["net_y_logit"] = net_H[0]
            layer.nodes["y_logit"] = H[0]   

    y_logit = H[0]
    y_prob = logit_to_prob(y_logit) # aka sigmoid
    model.layers[-1].nodes["y_prob"] = y_prob

    assert isinstance(y_prob, np.int64) or isinstance(y_prob, np.float64)

    if scale_output:
        y_prob = model.min_max_scaler.transform(np.array([y_prob]).reshape(-1, 1))[0][0]
        y_prob = 1. if y_prob > 1 else 0. if y_prob < 0 else y_prob

    if np.isnan(y_prob):
        print("oops!")
        import ipdb; ipdb.set_trace()
    return y_prob


def logit_to_prob(y_logit):  # aka logistic func type sigmoid
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
    return np.exp(-1 * y_logit) / ((1 + np.exp(-1 * y_logit))**2)


def loss(model, X, Y):
    """Given the MLP and the dataset, find the loss.

    Args:
        model: MLP model
        X, Y: the dataset
    """

    Y_actual = []
    for i in tqdm(range(X.shape[0]), desc=" inner", position=1, leave=False):
        y = feed_forward(X[i], model)
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
            weights=-0.5 + np.random.random((2, 3)), bias=np.array([1]),
            nodes={
                "net_h1": None, "h1": None,
                "net_h2": None, "h2": None,
                "net_h3": None, "h3": None,
            }
        ),
        Layer(
            weights=-0.5 + np.random.random((3, 2)), bias=np.array([1]),
            nodes={
                "net_h4": None, "h4": None,
                "net_h5": None, "h5": None,
            }
        ),
        Layer(weights=-0.5 + np.random.random((2, 1)), bias=np.array([0]),
            nodes={
                "net_y_logit": None, "y_logit": None,
                "y_prob": None,
            }
        ),
    ]
    return layers


def initialize_model(parameters):
    model = Model(
        layers=initialize_network_layers(),
        parameters=parameters,
        min_max_scaler=None,
    )

    scaler = MinMaxScaler()
    outputs = np.array([
        feed_forward(np.array([x1, x2]), model, verbose=False, scale_output=False)
        for x1, x2 in product(np.arange(0, 1, .01), np.arange(0, 1, .01))])
    
    y_prob_scaled = scaler.fit_transform(outputs.reshape(-1, 1))
    model = model._replace(min_max_scaler=scaler)

    return model


def train_network(data, model, log_loss_every_k_steps=10, steps=60):
    # sgd loop
    learning_rate = model.parameters["learning_rate"]

    artifacts = {}

    metrics = {
        "train": {
            "loss_vec": [],
        },
        "validation": {
            "loss_vec": [],
        },
        "micro_batch_updates": [],
    }

    num_examples = data.X_train.shape[0]
    for step in tqdm(range(steps), desc=" outer", position=0):
        # i = np.random.choice(range(num_examples))
        # sample minibatch , (x, y),
        x, y = data.X_train[step], data.Y_train[step]

        y_prob = feed_forward(x, model, verbose=False)

        # Before update, 
        y_actual, micro_batch_loss = loss(model, x.reshape((1, -1)), y.reshape((1, 1)))
        micro_batch_dict = {"loss_before": micro_batch_loss, "y_actual_before": y_actual, 
                            "x": x, "y": y}

        if step % log_loss_every_k_steps == 0:
            _, total_loss = loss(model, data.X_validation, data.Y_validation)
            metrics["validation"]["loss_vec"].append(total_loss)

            _, total_loss = loss(model, data.X_train[:steps], data.Y_train[:steps])
            metrics["train"]["loss_vec"].append(total_loss)

            artifacts[str(step)] = {"model": deepcopy(model), "log_loss": total_loss}

        # calculate partial derivative at (x, y)
        pd_loss_wrt_w13 = calc_partial_derivative_of_loss_wrt_w13(model.layers, y, )

        pd_loss_wrt_w14 = calc_partial_derivative_of_loss_wrt_w14(model.layers, y, )
        # then update the parameter using the learning rate.
        #   storing in temporary values until later.
        # TODO , assert that after applying update that the loss for the single example should be reduced ! otherwise something is terribly wrong I think. 

        # now finally update the actual parameters.
        model.layers[-1] = model.layers[-1]._replace(
            weights=model.layers[-1].weights + np.array(
                    [
                        [-1 * pd_loss_wrt_w13 * learning_rate],
                        [-1 * pd_loss_wrt_w14 * learning_rate]])
                )

        # next,
        pd_loss_wrt_the_layer_1_weights = (
            calc_partial_derivative_of_loss_wrt_w_on_layer_1(
                model.layers, pd_loss_wrt_w13, pd_loss_wrt_w14,
            ))
        model.layers[1] = model.layers[1]._replace(
            weights=model.layers[1].weights + (-1) * learning_rate * np.array(
                [
                    [pd_loss_wrt_the_layer_1_weights["w7"],
                        pd_loss_wrt_the_layer_1_weights["w8"], ],
                    [pd_loss_wrt_the_layer_1_weights["w9"],
                        pd_loss_wrt_the_layer_1_weights["w10"], ],
                    [pd_loss_wrt_the_layer_1_weights["w11"],
                        pd_loss_wrt_the_layer_1_weights["w12"], ],
                ]))

        # and layer 0 too
        x1, x2 = x[0], x[1]
        pd_loss_wrt_the_layer_0_weights = (
            calc_partial_derivative_of_loss_wrt_w_on_layer_0(
                model.layers,
                pd_loss_wrt_the_layer_1_weights,
                x1, x2,))

        model.layers[0] = model.layers[0]._replace(
            weights=model.layers[0].weights + (-1) * learning_rate * np.array(
                [
                    [
                        pd_loss_wrt_the_layer_0_weights["w1"],
                        pd_loss_wrt_the_layer_0_weights["w2"],
                        pd_loss_wrt_the_layer_0_weights["w3"],
                        ],
                    [
                        pd_loss_wrt_the_layer_0_weights["w4"],
                        pd_loss_wrt_the_layer_0_weights["w5"],
                        pd_loss_wrt_the_layer_0_weights["w6"],
                        ],
                ]
            )
        )

        # micro batch loss after the gradient descent step with that data point.
        y_actual, micro_batch_loss = loss(model, x.reshape((1, -1)), y.reshape((1, 1)))
        micro_batch_dict.update({"loss_after": micro_batch_loss, "y_actual_after": y_actual,})
        metrics["micro_batch_updates"].append(micro_batch_dict)

    _, total_loss = loss(model, data.X_train[:steps], data.Y_train[:steps])
    metrics["train"]["loss_vec"].append(total_loss)

    Y_prob, total_loss = loss(model, data.X_validation, data.Y_validation)
    metrics["validation"]["loss_vec"].append(total_loss)

    return metrics, model, artifacts, Y_prob


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
            # * derivative_of_relu(net_h4)
            * derivative_of_logit_to_prob_func(net_h4)
            * h1),
        "w9": (pd_loss_wrt_w13
            # * derivative_of_relu(net_h4)
            * derivative_of_logit_to_prob_func(net_h4)
            * h2),
        "w11": (pd_loss_wrt_w13
            # * derivative_of_relu(net_h4)
            * derivative_of_logit_to_prob_func(net_h4)
            * h3),

        # for h5 weights,
        "w8": (pd_loss_wrt_w14
            # * derivative_of_relu(net_h5)
            * derivative_of_logit_to_prob_func(net_h5)
            * h1),
        "w10": (pd_loss_wrt_w14
            # * derivative_of_relu(net_h5)
            * derivative_of_logit_to_prob_func(net_h5)
            * h2),
        "w12": (pd_loss_wrt_w14
            # * derivative_of_relu(net_h5)
            * derivative_of_logit_to_prob_func(net_h5)
            * h3),
    }
    stop_for_nan(pd_loss_wrt_weights)
    return pd_loss_wrt_weights


def calc_partial_derivative_of_loss_wrt_w_on_layer_0(
    layers,
    pd_loss_wrt_the_layer_1_weights,
    x1, x2,
):
    net_h1, net_h2, net_h3 = (layers[0].nodes["net_h1"],
            layers[0].nodes["net_h2"],
            layers[0].nodes["net_h3"],
            )
    # x1, x2 = (layers xxx .... ) # TODO
    pd_loss_wrt_weights = {
        "w1": ((pd_loss_wrt_the_layer_1_weights["w7"]
                + pd_loss_wrt_the_layer_1_weights["w8"])
            # * derivative_of_relu(net_h1)
            * derivative_of_logit_to_prob_func(net_h1)
            * x1),
        "w2": ((pd_loss_wrt_the_layer_1_weights["w9"]
                + pd_loss_wrt_the_layer_1_weights["w10"])
            # * derivative_of_relu(net_h2)
            * derivative_of_logit_to_prob_func(net_h2)
            * x1),
        "w3": ((pd_loss_wrt_the_layer_1_weights["w11"]
                + pd_loss_wrt_the_layer_1_weights["w12"])
            # * derivative_of_relu(net_h3)
            * derivative_of_logit_to_prob_func(net_h3)
            * x1),

        "w4": ((pd_loss_wrt_the_layer_1_weights["w7"]
                + pd_loss_wrt_the_layer_1_weights["w8"])
            # * derivative_of_relu(net_h1)
            * derivative_of_logit_to_prob_func(net_h1)
            * x2),
        "w5": ((pd_loss_wrt_the_layer_1_weights["w9"]
                + pd_loss_wrt_the_layer_1_weights["w10"])
            # * derivative_of_relu(net_h2)
            * derivative_of_logit_to_prob_func(net_h2)
            * x2),
        "w6": ((pd_loss_wrt_the_layer_1_weights["w11"]
                + pd_loss_wrt_the_layer_1_weights["w12"])
            # * derivative_of_relu(net_h3)
            * derivative_of_logit_to_prob_func(net_h3)
            * x2),
    }
    stop_for_nan(pd_loss_wrt_weights)
    return pd_loss_wrt_weights


def calc_partial_derivative_of_loss_wrt_w13(layers, y, ):

    # net_y = w13*h4 + w14*h5
    # y_prob = logit_to_prob(y_logit)
    # loss = log_loss(y_actual, y_prob)

    # by chain rule,
    # derivative = pd_log_loss_wrt_prob * pd_prob_wrt_logit * pd_logit_wrt_net_y * pd_net_y_wrt_w13

    # y = y
    y_prob = layers[-1].nodes["y_prob"]
    y_logit = layers[-1].nodes["y_logit"]
    # net_y_logit = layers[-1].nodes["net_y_logit"]   # XXX no longer used.
    h4 = layers[-2].nodes["h4"]

    g = (
        derivative_of_log_loss(y, y_prob)
        * derivative_of_logit_to_prob_func(y_logit)
        # * derivative_of_relu(net_y_logit)
        * h4
    )

    stop_for_nan(g)

    return g

def stop_for_nan(x):
    if isinstance(x, dict):
        [stop_for_nan(thing) for thing in x.values()]
    elif isinstance(x, list):
        [stop_for_nan(thing) for thing in x]
    elif pd.isnull(x):
        import ipdb; ipdb.set_trace()
        ...
        ...


def calc_partial_derivative_of_loss_wrt_w14(layers, y, ):

    # net_y = w13*h4 + w14*h5
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
        # * derivative_of_relu(net_y_logit)
        * h5
    )
    stop_for_nan(g)
    return g



