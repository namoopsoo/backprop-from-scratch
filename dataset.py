
import numpy as np


def build_dataset_inside_outside_circle():
    # Create some data in a 20x20 box centered at origin.
    radius = 5
    X = np.random.random((10000, 2)) * 40 + -20
    f = (lambda a: int(np.sqrt(a[0]**2 + a[1]**2) <= radius))
    Y = np.array(list(map(f, X)))
    return X, Y


