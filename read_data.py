from __future__ import division
import numpy as np
import os

import grid_conversion as gc


def read_data_from_DaVis(num_trials, num_snaps, RAWdirs, skiprows, usecols,
                         num_vects=False):
    if isinstance(RAWdirs, str):
        RAWdirs = [RAWdirs]
    if not isinstance(num_vects, int):
        I, J = read_IJ(RAWdirs[0])
        num_vects = I*J
    data = np.zeros([num_vects, np.size(usecols), num_trials*num_snaps])
    print data.shape
    trials = np.arange(num_trials)
    snaps = np.arange(num_snaps)
    for trial, RAWdir in zip(trials, RAWdirs):
        for snap in snaps:
            filename = 'B%05i.dat' % (snap + 1)
            filepath = RAWdir + '\\' + filename
            print 'Loading %s' % filepath
            trial_data = np.loadtxt(filepath, skiprows=skiprows,
                                    usecols=usecols)
            data[:, :, trial*num_snaps + snap] = trial_data
    return data


def read_IJ(RAWdir):
    filename = 'B00001.dat'
    filepath = RAWdir + '\\' + filename
    with open(filepath, 'r') as fID:
        fID.readline()
        fID.readline()
        header_str = fID.readline()
    J_ind = header_str.find('J')
    I_ind = header_str.find('I')
    K_ind = header_str.find('K')
    I = int(header_str[I_ind+2:J_ind-2])
    J = int(header_str[J_ind+2:K_ind-2])
    return I, J


def read_num_vec_downwards_flow(RAWdir):
    I, J = read_IJ(RAWdir)
    num_x = J
    num_y = I
    return num_x, num_y


def read_stereo_velocity(num_trials, num_snaps, num_vects, RAWdirs):
    skiprows = 3
    usecols = [3, 4, 5]
    data = read_data_from_DaVis(num_trials, num_snaps, RAWdirs,
                                skiprows, usecols, num_vects=num_vects)
    u0 = data[:, 0]
    u1 = data[:, 1]
    u2 = data[:, 2]
    return u0, u1, u2


def read_planar_geometry(num_trials, num_snaps, num_vects, RAWdirs):
    skiprows = 3
    usecols = [0, 1]
    data = read_data_from_DaVis(1, 1, RAWdirs[0], skiprows, usecols,
                                num_vects=num_vects)
    x0 = data[:, 0]
    x1 = data[:, 1]
    return x0, x1


def read_stereo_velocity_downwards_flow(num_trials, num_snaps, num_x, num_y,
                                        num_vects, RAWdirs):
    u0, u1, u2 = read_stereo_velocity(num_trials, num_snaps, num_vects,
                                      RAWdirs)
    u = u1
#    print np.size(u)
    u_g = -np.swapaxes(gc.togrid(u, num_x, num_y), 0, 1)
#    print u_g.shape()
    v = u0
    v_g = np.swapaxes(gc.togrid(v, num_x, num_y), 0, 1)
    w = u2
    w_g = np.swapaxes(gc.togrid(w, num_x, num_y), 0, 1)
    return u_g, v_g, w_g


def read_planar_geometry_downwards_flow(num_trials, num_snaps, RAWdirs):
    I, J = read_IJ(RAWdirs[0])
    num_x = J
    num_y = I
    num_vects = num_x*num_y
    x0, x1 = read_planar_geometry(num_trials, num_snaps, num_vects, RAWdirs)
    x = x1
    x_g = np.fliplr(gc.togrid(x, num_x, num_y).T)
    y = x0
    y_g = gc.togrid(y, num_x, num_y).T
    return x_g, y_g, num_x, num_y, num_vects


def read_geometry_and_velocity(num_trials, num_snaps, RAWdirs,
                               flowdir='downward'):
    if flowdir == 'downward':
        x_g, y_g, num_x, num_y, num_vects \
            = read_planar_geometry_downwards_flow(num_trials, num_snaps,
                                                  RAWdirs)
        u_g, v_g, w_g = read_stereo_velocity_downwards_flow(
                num_trials, num_snaps, num_x, num_y, num_vects, RAWdirs)
    elif flowdir == 'right':
        print 'add functionality here'
    return x_g, y_g, num_x, num_y, u_g, v_g, w_g


def get_trial_info(RAWdir):
    trial_paths = os.listdir(RAWdir)
    num_trials = np.size(trial_paths, axis=0)
    RAWdirs = np.array([])
    for trial_path in trial_paths:
        RAWdirs = np.append(RAWdirs, RAWdir + '\\' + trial_path)
    snapshot_files = os.listdir(RAWdirs[0])
    num_snaps = np.size(snapshot_files, axis=0)
    return RAWdirs, num_trials, num_snaps
