import math
import numpy as np
from collections import Counter


def build_dataset_inside_outside_circle(balance=0.5):
    # Create some data in a 20x20 box centered at origin.
    num_samples = 10000
    radius = math.sqrt(40*40*balance/math.pi)
    X = np.random.random((num_samples, 2)) * 40 + -20
    f = (lambda a: int(np.sqrt(a[0]**2 + a[1]**2) <= radius))
    Y = np.array(list(map(f, X)))

    # Validate balance
    assert abs(Counter(Y)[1]/num_samples - balance) < 0.02
    return X, Y


