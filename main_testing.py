from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import h5py

CODEdir = r'D:\6 - Circle_AR4_LaminarBL\PIV\Test_1_2_3\PIV\Test_1_2_3_testing\Python Codes'
if CODEdir not in sys.path:
    sys.path.append(CODEdir)

import grid_conversion as gc
import read_data_from_DaVis as rd
import data_input_output as io


###############################################################################


RAWdir = r'D:\6 - Circle_AR4_LaminarBL\PIV\Test_1_2_3\PIV\Test_1_2_3_testing\RAW DATA'
RAWdirs, num_trials, num_snaps = rd.get_trial_info(RAWdir)
#print num_trials, num_snapshots, RAWdirs

num_trials = 3
num_snaps = 20

x_g, y_g, num_x, num_y, u_g, v_g, w_g = rd.read_geometry_and_velocity(
        num_trials, num_snaps, RAWdirs, flowdir='downward')


main_folder_path = r'D:\6 - Circle_AR4_LaminarBL\PIV\Test_1_2_3\PIV\Test_1_2_3_testing\Variables'
file_name = 'Test_python'

folder_path = 'raw_data/geometry'
io.save_2D_geometry(x_g, y_g, num_x, num_y, main_folder_path, folder_path,
                    file_name=file_name)
folder_path = 'raw_data/velocity'
io.save_3C_velocity_snapshots(u_g, v_g, w_g, num_x, num_y, main_folder_path,
                              folder_path, file_name=file_name)
