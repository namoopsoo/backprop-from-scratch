import math
import numpy as np
from collections import Counter, namedtuple
from sklearn.model_selection import train_test_split

Dataset = namedtuple("Dataset", [
    "X_train", "X_validation", "Y_train", "Y_validation"
    ])


def build_dataset_inside_outside_circle(balance=0.5):
    # Create some data in a 20x20 box centered at origin.
    num_samples = 10000
    radius = math.sqrt(40*40*balance/math.pi)
    X = np.random.random((num_samples, 2)) * 40 + -20
    f = (lambda a: int(np.sqrt(a[0]**2 + a[1]**2) <= radius))
    Y = np.array(list(map(f, X)))

    # Validate balance
    assert abs(Counter(Y)[1]/num_samples - balance) < 0.02
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, Y, test_size=0.1, random_state=42)

    return Dataset(X_train, X_validation, Y_train, Y_validation)
