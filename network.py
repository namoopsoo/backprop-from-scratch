import numpy as np
from functools import partial
from sklearn.utils.extmath import softmax
from sklearn.metrics import log_loss


relu_scalar = partial(max, 0)
relu = np.vectorize(relu_scalar)

def feed_forward(x, layers):
    values = []
    H = x
    for (i, layer) in enumerate(layers):
        weights = concat_bias_weights(layer)
        net_H = np.matmul(H, weights)
        H = relu(net_H)

    y_hat = H[0][0]
    assert isinstance(y_hat, np.int64) or isinstance(y_hat, np.float64)
    return y_hat

def loss(y, y_hat):
    # log loss 
    ...

def concat_bias_weights(x):
    num_cols = x.shape[1]
    return np.concatenate([x, np.ones((1, num_cols))])

layers = [
    np.random.random((2, 3)),
    np.random.random((3, 2)),
    np.array([[1], [1]]),
]

def build_dataset_inside_outside_circle():
    # Create some data in a 20x20 box centered at origin.
    radius = 5
    X = np.random.random((10000, 2)) * 40 + -20
    f = (lambda a: int(np.sqrt(a[0]**2 + a[1]**2) <= radius))
    Y = np.array(list(map(f, X)))
    return X, Y
X, Y = build_dataset_inside_outside_circle() 
