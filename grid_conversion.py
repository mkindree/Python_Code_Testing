from __future__ import division
import numpy as np


def togrid(var, num_x, num_y):
    return np.squeeze(np.reshape(var, (num_x, num_y, -1)))


def fromgrid(var_g, num_x, num_y):
    print num_x, num_y
    return np.squeeze(np.reshape(var_g, (num_x*num_y, -1)))
