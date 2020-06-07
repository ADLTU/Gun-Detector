import numpy as np


def load_weights(weights_file):
    weights = open(weights_file, "rb")
    header = np.fromfile(weights, dtype=np.int32, count=5)
    weights = np.fromfile(weights, dtype=np.float32)
    return weights
