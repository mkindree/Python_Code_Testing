import numpy as np


def togrid(var, I, J):
    return np.squeeze(np.reshape(var, (J, I, -1)))


def fromgrid(var_g, I, J):
    return np.squeeze(np.reshape(var_g, (J*I, -1)))
