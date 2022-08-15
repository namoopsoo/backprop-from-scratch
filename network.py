import numpy as np
from functools import partial
from sklearn.utils.extmath import softmax
from sklearn.metrics import log_loss
from collections import namedtuple

Layer = namedtuple("Layer", ["weights", "bias"])

relu_scalar = partial(max, 0)
relu = np.vectorize(relu_scalar)

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
        print("output of relu", H)

    y_hat = H[0]
    assert isinstance(y_hat, np.int64) or isinstance(y_hat, np.float64)
    return y_hat

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


def loss(y, y_hat):
    # log loss 
    ...


def concat_bias_weights(x):
    num_cols = x.shape[1]
    return np.concatenate([x, np.ones((1, num_cols))])

layers = [
    Layer(weights=np.random.random((2, 3)), bias=np.array([1])),
    Layer(weights=np.random.random((3, 2)), bias=np.array([1])), 
    Layer(weights=np.array([[1], [1]]), bias=np.array([0])), 
]

def build_dataset_inside_outside_circle():
    # Create some data in a 20x20 box centered at origin.
    radius = 5
    X = np.random.random((10000, 2)) * 40 + -20
    f = (lambda a: int(np.sqrt(a[0]**2 + a[1]**2) <= radius))
    Y = np.array(list(map(f, X)))
    return X, Y
X, Y = build_dataset_inside_outside_circle() 
