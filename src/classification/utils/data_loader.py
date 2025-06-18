import numpy as np


def load_data(path, flatten=True):
    data = np.load(path)
    X = data["X"]
    y = data["y"]

    if flatten:
        X = X.reshape((X.shape[0], -1))
    return X, y
