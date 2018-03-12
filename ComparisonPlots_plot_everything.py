from __future__ import division
import sys

CODEdir = r'D:\6 - Circle_AR4_LaminarBL\PIV\Test_1_2_3\PIV\Test_1_2_3_testing\Python Codes'
if CODEdir not in sys.path:
    sys.path.append(CODEdir)

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import copy as copy
import scipy.interpolate as sinterp
import scipy.linalg as slinalg

import plotting_functions as pltf
import grid_conversion as gc
import spectrum_fft_v2 as spect
import load_from_matlab_workspace as load_mat


#%% User-Defined Variables

mainfolder = r'D:\7 - SquareAR4_LaminarBL'# The main working directory
PUBdir =  r'D:\EXF Paper\Figures Page V2_full_data\figs'# Where the .eps figures go

WRKSPCdirs = (r'D:\Workspaces', r'D:\Workspaces')# Where the workspaces are located
WRKSPCfilenames_both = [[r'circle_z_1.5mm', r'circle_z_3mm', 
                         r'circle_z_4.5mm', r'circle_z_6mm', 
                         r'circle_z_7.5mm', r'circle_z_9mm', 
                         r'circle_z_10.5mm', r'circle_z_12mm', 
                         r'circle_z_15mm', r'circle_z_18mm', 
                         r'circle_z_21mm', r'circle_z_24mm', 
                         r'circle_z_27mm', r'circle_z_30mm', 
                         r'circle_z_33mm', r'circle_z_36mm', 
                         r'circle_z_39mm', r'circle_z_42mm', 
                         r'circle_z_45mm', r'circle_z_46.5mm', 
                         r'circle_z_48mm', r'circle_z_49.5mm', 
                         r'circle_z_51mm', r'circle_z_52.5mm', 
                         r'circle_z_54mm', r'circle_z_55.5mm',  
                         r'circle_z_57mm'], 
                        [r'square_z_1.5mm', r'square_z_3mm', 
                         r'square_z_4.5mm', r'square_z_6mm', 
                         r'square_z_7.5mm', r'square_z_9mm', 
                         r'square_z_10.5mm', r'square2_z_12mm', 
                         r'square_z_15mm', r'square_z_18mm', 
                         r'square_z_21mm', r'square_z_24mm', 
                         r'square_z_27mm', r'square_z_30mm', 
                         r'square_z_33mm', r'square_z_36mm', 
                         r'square_z_39mm', r'square_z_42mm', 
                         r'square_z_45mm', r'square_z_46.5mm', 
                         r'square_z_48mm', r'square_z_49.5mm', 
                         r'square_z_51mm', r'square_z_52.5mm', 
                         r'square_z_54mm', r'square_z_55.5mm',  
                         r'square_z_57mm']]# Name of the MATLAB workspaces

#save_names = ['z_1.5mm',
#              'z_4.5mm',
#              'z_7.5mm'
#              ]

zs = [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39,
      42, 45, 46.5, 48, 49.5, 51, 52.5, 54, 55.5, 57]
num_planes = len(zs)
plane_inds = np.arange(num_planes, dtype=int)

save_names = ['z_%1.1fmm' %z for z in zs]

num_geos = 2
obstacles = ['circle', 'square']
obstacle_AR = np.array([4.01, 3.84])
geo_inds = np.arange(num_geos, dtype=int)

#%%

num_modes = 10
num_modes_total = 200
modes = np.arange(num_modes)
num_trials = 6
trials = np.arange(num_trials)

#%% Load variables

[B, B1, f_capture, d, t, I, J, num_vect, dx, dy, f_shed, St_shed, U_inf, Re,
 x_g, y_g, z_g] = \
    load_mat.load_general(WRKSPCdirs, WRKSPCfilenames_both, num_geos,
                          num_planes, zs)

[M_U_g, M_V_g, M_W_g,
 M_ufuf_g, M_ufvf_g, M_ufwf_g, M_vfvf_g, M_vfwf_g, M_wfwf_g,
 M_tke_g,
 M_vfvf_ufuf_g, M_wfwf_ufuf_g, M_wfwf_vfvf_g] = \
    load_mat.load_mean_field(WRKSPCdirs, WRKSPCfilenames_both, I, J, num_geos,
                             num_planes)

#[akf, akf_sy, akf_as, akf_sy_gaus, akf_sy_harm,
# lambdaf, lambdaf_sy, lambdaf_as, lambdaf_sy_gaus, lambdaf_sy_harm,
# Psi_uf_g, Psi_uf_sy_g, Psi_uf_as_g, Psi_uf_sy_gaus_g, Psi_uf_sy_harm_g,
# Psi_vf_g, Psi_vf_sy_g, Psi_vf_as_g, Psi_vf_sy_gaus_g, Psi_vf_sy_harm_g,
# Psi_wf_g, Psi_wf_sy_g, Psi_wf_as_g, Psi_wf_sy_gaus_g, Psi_wf_sy_harm_g] = \
#    load_mat.load_POD(WRKSPCdirs, WRKSPCfilenames_both, I, J, B, num_geos,
#                      num_planes, num_modes)

num_snaps = 16368

#[uf_g, vf_g, wf_g] = \
#    load_mat.load_snapshots(WRKSPCdirs, WRKSPCfilenames_both, I, J, B,
#                            num_geos, num_planes, num_snaps)

#%% Load variables

#variable_names1 = ['B', 'B1', 'f_capture', 'f_shed', 'd']
#B = np.zeros(num_geos, dtype=int)
#B1 = np.zeros(num_geos, dtype=int)
#f_capture = np.zeros(num_geos)
#d = np.zeros(num_geos)
#for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
#                                           WRKSPCfilenames_both):
#    os.chdir(WRKSPCdir)
#    matvariables = sio.loadmat(WRKSPCfilenames[0],
#                               variable_names=variable_names1)
#    B[geo] = int(matvariables['B'])
#    B1[geo] = int(matvariables['B1'])
#    f_capture[geo] = float(matvariables['f_capture'])
#    d[geo] = float(matvariables['d'])
#    del matvariables
#
#variable_names2 = ['t']
#t = (np.zeros(B[0]), np.zeros(B[1]))
#for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
#                                           WRKSPCfilenames_both):
#    os.chdir(WRKSPCdir)
#    matvariables = sio.loadmat(WRKSPCfilenames[0],
#                               variable_names=variable_names2)
#    t[geo][:] = np.squeeze(matvariables['t'])
#
#variable_names3 = ['I', 'J', 'dx', 'dy', 'f_shed', 'St_shed', 'U_inf', 'Re']
#I = [np.zeros(num_planes, dtype=int) for _ in geo_inds]
#J = [np.zeros(num_planes, dtype=int) for _ in geo_inds]
#num_vect = [np.zeros(num_planes, dtype=int) for _ in geo_inds]
#dx = [np.zeros(num_planes) for _ in geo_inds]
#dy =[np.zeros(num_planes) for _ in geo_inds]
#f_shed = [np.zeros(num_planes) for _ in geo_inds]
#St_shed = [np.zeros(num_planes) for _ in geo_inds]
#U_inf = [np.zeros(num_planes) for _ in geo_inds]
#Re = [np.zeros(num_planes) for _ in geo_inds]
#for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
#                                           WRKSPCfilenames_both):
#    os.chdir(WRKSPCdir)
#    for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
#        matvariables = sio.loadmat(WRKSPCfilename,
#                                   variable_names=variable_names3)
#        I[geo][plane] = int(matvariables['I'])
#        J[geo][plane] = int(matvariables['J'])
#        num_vect[geo][plane]= I[geo][plane]*J[geo][plane]
#        dx[geo][plane] = float(matvariables['dx'])
#        dy[geo][plane] = float(matvariables['dy'])
#        f_shed[geo][plane] = float(matvariables['f_shed'])
#        St_shed[geo][plane] = float(matvariables['St_shed'])
#        U_inf[geo][plane] = float(matvariables['U_inf'])
#        Re[geo][plane] = float(matvariables['Re'])
#        del matvariables
#
#variable_names4 = ['x_g', 'y_g', 
#                   'M_U_g', 'M_V_g', 'M_W_g',
#                   'M_ufuf_g', 'M_ufvf_g', 'M_ufwf_g',
#                   'M_vfvf_g', 'M_vfwf_g', 'M_wfwf_g'
#                   ]
#x_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes)) for geo in geo_inds]
#y_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes)) for geo in geo_inds]
#z_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes)) for geo in geo_inds]
#M_U_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes)) 
#         for geo in geo_inds]
#M_V_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#         for geo in geo_inds]
#M_W_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#         for geo in geo_inds]
#M_ufuf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#            for geo in geo_inds]
#M_ufvf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#            for geo in geo_inds]
#M_ufwf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#            for geo in geo_inds]
#M_vfvf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#            for geo in geo_inds]
#M_vfwf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#            for geo in geo_inds]
#M_wfwf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#            for geo in geo_inds]
#M_tke_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#            for geo in geo_inds]
#M_vfvf_ufuf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#                 for geo in geo_inds]
#M_wfwf_ufuf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#                 for geo in geo_inds]
#M_wfwf_vfvf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
#                 for geo in geo_inds]
#for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
#                                           WRKSPCfilenames_both):
#    os.chdir(WRKSPCdir)
#    for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
#        matvariables = sio.loadmat(WRKSPCfilename,
#                                   variable_names=variable_names4)
#        x_g_temp = np.squeeze(matvariables['x_g'])
#        y_g_temp = np.squeeze(matvariables['y_g'])
#        M_U_g_temp = np.squeeze(matvariables['M_U_g'])
#        M_V_g_temp = np.squeeze(matvariables['M_V_g'])
#        M_W_g_temp = np.squeeze(matvariables['M_W_g'])
#        M_ufuf_g_temp = np.squeeze(matvariables['M_ufuf_g'])
#        M_ufvf_g_temp = np.squeeze(matvariables['M_ufvf_g'])
#        M_ufwf_g_temp = np.squeeze(matvariables['M_ufwf_g'])
#        M_vfvf_g_temp = np.squeeze(matvariables['M_vfvf_g'])
#        M_vfwf_g_temp = np.squeeze(matvariables['M_vfwf_g'])
#        M_wfwf_g_temp = np.squeeze(matvariables['M_wfwf_g'])
#
#        cut_row = (J[geo][plane] - J[geo].min())//2
#        cut_col = (I[geo][plane] - I[geo].min())//2
#        cut_row_f = cut_row
#        cut_row_b = cut_row
#        cut_col_f = cut_col
#        cut_col_b = cut_col
#
#        if cut_row_b == 0:
#            cut_row_b = -J[geo][plane]
#        if cut_col_b == 0:
#            cut_col_b = -I[geo][plane]
#
#        x_g[geo][:, :, plane] \
#            = x_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        y_g[geo][:, :, plane] \
#            = y_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        M_U_g[geo][:, :, plane] \
#            = M_U_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        M_V_g[geo][:, :, plane] \
#            = M_V_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        M_W_g[geo][:, :, plane] \
#            = M_W_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        M_ufuf_g[geo][:, :, plane] \
#            = M_ufuf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        M_ufvf_g[geo][:, :, plane] \
#            = M_ufvf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        M_ufwf_g[geo][:, :, plane] \
#            = M_ufwf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        M_vfvf_g[geo][:, :, plane] \
#            = M_vfvf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        M_vfwf_g[geo][:, :, plane] \
#            = M_vfwf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        M_wfwf_g[geo][:, :, plane] \
#            = M_wfwf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
#        del matvariables
#
#        z_g[geo][:, :, plane] = np.full((J[geo].min(), I[geo].min()),
#                                        zs[plane])/(d[geo]*1000)
#
#        M_tke_g[geo][:, :, plane] = (M_ufuf_g[geo][:, :, plane] +
#                                     M_vfvf_g[geo][:, :, plane] +
#                                     M_wfwf_g[geo][:, :, plane])/2
#
#        M_vfvf_ufuf_g[geo][:, :, plane] = M_vfvf_g[geo][:, :, plane] / \
#                                          M_ufuf_g[geo][:, :, plane]
#        M_wfwf_ufuf_g[geo][:, :, plane] = M_wfwf_g[geo][:, :, plane] / \
#                                          M_ufuf_g[geo][:, :, plane]
#        M_wfwf_vfvf_g[geo][:, :, plane] = M_wfwf_g[geo][:, :, plane] / \
#                                          M_vfvf_g[geo][:, :, plane]
#
#variable_names5 = ['akf', 'lambdaf',
#                   'akf_sy', 'lambdaf_sy',
#                   'akf_as', 'lambdaf_as',
#                   'akf_sy_gaus', 'lambdaf_sy_gaus',
#                   'akp_sy', 'lambdap_sy'
#                   ]
#akf = [[np.zeros((B[geo], num_modes))
#        for plane in plane_inds] for geo in geo_inds]
#lambdaf = [[np.zeros((B[geo]))
#            for plane in plane_inds] for geo in geo_inds]
#akf_sy = [[np.zeros((B[geo], num_modes))
#           for plane in plane_inds] for geo in geo_inds]
#lambdaf_sy = [[np.zeros((B[geo]))
#               for plane in plane_inds] for geo in geo_inds]
#akf_as = [[np.zeros((B[geo], num_modes))
#           for plane in plane_inds] for geo in geo_inds]
#lambdaf_as = [[np.zeros((B[geo]))
#               for plane in plane_inds] for geo in geo_inds]
#akf_sy_gaus = [[np.zeros((B[geo], num_modes))
#                for plane in plane_inds] for geo in geo_inds]
#lambdaf_sy_gaus = [[np.zeros((B[geo]))
#                    for plane in plane_inds] for geo in geo_inds]
#akf_sy_harm = [[np.zeros((B[geo], num_modes))
#                for plane in plane_inds] for geo in geo_inds]
#lambdaf_sy_harm = [[np.zeros((B[geo]))
#                    for plane in plane_inds] for geo in geo_inds]
#for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
#                                           WRKSPCfilenames_both):
#    os.chdir(WRKSPCdir)
#    for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
#        matvariables = sio.loadmat(WRKSPCfilename,
#                                   variable_names=variable_names5)
#        akf_temp = np.squeeze(matvariables['akf'])
#        akf_sy_temp = np.squeeze(matvariables['akf_sy'])
#        akf_as_temp = np.squeeze(matvariables['akf_as'])
#        akf_sy_gaus_temp = np.squeeze(matvariables['akf_sy_gaus'])
#        akf_sy_harm_temp = np.squeeze(matvariables['akp_sy'])
#
#        akf[geo][plane] = akf_temp[:, 0:num_modes]
#        akf_sy[geo][plane] = akf_sy_temp[:, 0:num_modes]
#        akf_as[geo][plane] = akf_as_temp[:, 0:num_modes]
#        akf_sy_gaus[geo][plane] = akf_sy_gaus_temp[:, 0:num_modes]
#        akf_sy_harm[geo][plane] = akf_sy_harm_temp[:, 0:num_modes]
#
#        lambdaf[geo][plane] = np.squeeze(matvariables['lambdaf'])
#        lambdaf_sy[geo][plane] = np.squeeze(matvariables['lambdaf_sy'])
#        lambdaf_as[geo][plane] = np.squeeze(matvariables['lambdaf_as'])
#        lambdaf_sy_gaus[geo][plane] = np.squeeze(matvariables['lambdaf_sy_gaus'])
#        lambdaf_sy_harm[geo][plane] = np.squeeze(matvariables['lambdap_sy'])
#
#variable_names6 = ['Psi_uf', 'Psi_vf', 'Psi_wf',
#                   'Psi_uf_sy', 'Psi_vf_sy', 'Psi_wf_sy',
#                   'Psi_uf_as', 'Psi_vf_as', 'Psi_wf_as',
#                   'Psi_uf_sy_gaus', 'Psi_vf_sy_gaus', 'Psi_wf_sy_gaus',
#                   'Psi_up_sy', 'Psi_vp_sy', 'Psi_wp_sy'
#                   ]
#
#Psi_uf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#             for plane in plane_inds] for geo in geo_inds]
#Psi_vf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#             for plane in plane_inds] for geo in geo_inds]
#Psi_wf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#             for plane in plane_inds] for geo in geo_inds]
#Psi_uf_sy_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                for plane in plane_inds] for geo in geo_inds]
#Psi_vf_sy_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                for plane in plane_inds] for geo in geo_inds]
#Psi_wf_sy_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                for plane in plane_inds] for geo in geo_inds]
#Psi_uf_as_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                for plane in plane_inds] for geo in geo_inds]
#Psi_vf_as_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                for plane in plane_inds] for geo in geo_inds]
#Psi_wf_as_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                for plane in plane_inds] for geo in geo_inds]
#Psi_uf_sy_gaus_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                     for plane in plane_inds] for geo in geo_inds]
#Psi_vf_sy_gaus_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                     for plane in plane_inds] for geo in geo_inds]
#Psi_wf_sy_gaus_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                     for plane in plane_inds] for geo in geo_inds]
#Psi_uf_sy_harm_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                     for plane in plane_inds] for geo in geo_inds]
#Psi_vf_sy_harm_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                     for plane in plane_inds] for geo in geo_inds]
#Psi_wf_sy_harm_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                     for plane in plane_inds] for geo in geo_inds]
#for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
#                                           WRKSPCfilenames_both):
#    os.chdir(WRKSPCdir)
#    for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
#        matvariables = sio.loadmat(WRKSPCfilename,
#                                   variable_names=variable_names6)
#        Psi_uf_temp = np.squeeze(matvariables['Psi_uf'])
#        Psi_vf_temp = np.squeeze(matvariables['Psi_vf'])
#        Psi_wf_temp = np.squeeze(matvariables['Psi_wf'])
#        Psi_uf_sy_temp = np.squeeze(matvariables['Psi_uf_sy'])
#        Psi_vf_sy_temp = np.squeeze(matvariables['Psi_vf_sy'])
#        Psi_wf_sy_temp = np.squeeze(matvariables['Psi_wf_sy'])
#        Psi_uf_as_temp = np.squeeze(matvariables['Psi_uf_as'])
#        Psi_vf_as_temp = np.squeeze(matvariables['Psi_vf_as'])
#        Psi_wf_as_temp = np.squeeze(matvariables['Psi_wf_as'])
#        Psi_uf_sy_gaus_temp = np.squeeze(matvariables['Psi_uf_sy_gaus'])
#        Psi_vf_sy_gaus_temp = np.squeeze(matvariables['Psi_vf_sy_gaus'])
#        Psi_wf_sy_gaus_temp = np.squeeze(matvariables['Psi_wf_sy_gaus'])
#        Psi_uf_sy_harm_temp = np.squeeze(matvariables['Psi_up_sy'])
#        Psi_vf_sy_harm_temp = np.squeeze(matvariables['Psi_vp_sy'])
#        Psi_wf_sy_harm_temp = np.squeeze(matvariables['Psi_wp_sy'])
#        
#        Psi_uf_g_temp = gc.togrid(Psi_uf_temp, J[geo][plane], I[geo][plane])
#        Psi_vf_g_temp = gc.togrid(Psi_vf_temp, J[geo][plane], I[geo][plane])
#        Psi_wf_g_temp = gc.togrid(Psi_wf_temp, J[geo][plane], I[geo][plane])
#        Psi_uf_sy_g_temp = gc.togrid(Psi_uf_sy_temp, J[geo][plane],
#                                     I[geo][plane])
#        Psi_vf_sy_g_temp = gc.togrid(Psi_vf_sy_temp, J[geo][plane],
#                                     I[geo][plane])
#        Psi_wf_sy_g_temp = gc.togrid(Psi_wf_sy_temp, J[geo][plane],
#                                     I[geo][plane])
#        Psi_uf_as_g_temp = gc.togrid(Psi_uf_as_temp, J[geo][plane],
#                                     I[geo][plane])
#        Psi_vf_as_g_temp = gc.togrid(Psi_vf_as_temp, J[geo][plane],
#                                     I[geo][plane])
#        Psi_wf_as_g_temp = gc.togrid(Psi_wf_as_temp, J[geo][plane],
#                                     I[geo][plane])
#        Psi_uf_sy_gaus_g_temp = gc.togrid(Psi_uf_sy_gaus_temp, J[geo][plane],
#                                          I[geo][plane])
#        Psi_vf_sy_gaus_g_temp = gc.togrid(Psi_vf_sy_gaus_temp, J[geo][plane],
#                                          I[geo][plane])
#        Psi_wf_sy_gaus_g_temp = gc.togrid(Psi_wf_sy_gaus_temp, J[geo][plane],
#                                          I[geo][plane])
#        Psi_uf_sy_harm_g_temp = gc.togrid(Psi_uf_sy_harm_temp, J[geo][plane],
#                                          I[geo][plane])
#        Psi_vf_sy_harm_g_temp = gc.togrid(Psi_vf_sy_harm_temp, J[geo][plane],
#                                          I[geo][plane])
#        Psi_wf_sy_harm_g_temp = gc.togrid(Psi_wf_sy_harm_temp, J[geo][plane],
#                                          I[geo][plane])
#
#        cut_row = (J[geo][plane] - J[geo].min())//2
#        cut_col = (I[geo][plane] - I[geo].min())//2
#        cut_row_f = cut_row
#        cut_row_b = cut_row
#        cut_col_f = cut_col
#        cut_col_b = cut_col
#
#        if cut_row_b == 0:
#            cut_row_b = -J[geo][plane]
#        if cut_col_b == 0:
#            cut_col_b = -I[geo][plane]
#
#        Psi_uf_g[geo][plane][:, :, :] \
#            = Psi_uf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_vf_g[geo][plane][:, :, :] \
#            = Psi_vf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_wf_g[geo][plane][:, :, :] \
#            = Psi_wf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_uf_sy_g[geo][plane][:, :, :] \
#            = Psi_uf_sy_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_vf_sy_g[geo][plane][:, :, :] \
#            = Psi_vf_sy_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_wf_sy_g[geo][plane][:, :, :] \
#            = Psi_wf_sy_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_uf_as_g[geo][plane][:, :, :] \
#            = Psi_uf_as_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_vf_as_g[geo][plane][:, :, :] \
#            = Psi_vf_as_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_wf_as_g[geo][plane][:, :, :] \
#            = Psi_wf_as_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_uf_sy_gaus_g[geo][plane][:, :, :] \
#            = Psi_uf_sy_gaus_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_vf_sy_gaus_g[geo][plane][:, :, :] \
#            = Psi_vf_sy_gaus_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_wf_sy_gaus_g[geo][plane][:, :, :] \
#            = Psi_wf_sy_gaus_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_uf_sy_harm_g[geo][plane][:, :, :] \
#            = Psi_uf_sy_harm_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_vf_sy_harm_g[geo][plane][:, :, :] \
#            = Psi_vf_sy_harm_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#        Psi_wf_sy_harm_g[geo][plane][:, :, :] \
#            = Psi_wf_sy_harm_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#
#variable_names7 = ['uf_g', 'vf_g', 'wf_g']
#
#uf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
#         for plane in plane_inds] for geo in geo_inds]
#vf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
#         for plane in plane_inds] for geo in geo_inds]
#wf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
#         for plane in plane_inds] for geo in geo_inds]
#for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
#                                           WRKSPCfilenames_both):
#    os.chdir(WRKSPCdir)
#    for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
#        matvariables = sio.loadmat(WRKSPCfilename,
#                                   variable_names=variable_names7)
#        print 'loading %s snapshots' % WRKSPCfilename
#        uf_g_temp = np.squeeze(matvariables['uf_g'])
#        vf_g_temp = np.squeeze(matvariables['vf_g'])
#        wf_g_temp = np.squeeze(matvariables['wf_g'])
#        
#        cut_row = (J[geo][plane] - J[geo].min())//2
#        cut_col = (I[geo][plane] - I[geo].min())//2
#        cut_row_f = cut_row
#        cut_row_b = cut_row
#        cut_col_f = cut_col
#        cut_col_b = cut_col
#
#        if cut_row_b == 0:
#            cut_row_b = -J[geo][plane]
#        if cut_col_b == 0:
#            cut_col_b = -I[geo][plane]
#
#        uf_g[geo][plane][:, :, :] \
#            = uf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
#        vf_g[geo][plane][:, :, :] \
#            = vf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
#        wf_g[geo][plane][:, :, :] \
#            = wf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]



#%% Spectral Analysis of POD temporal coefficients

akf_PSD = [[np.zeros((int(np.floor(B1[geo]/2)), num_modes))
            for plane in plane_inds] for geo in geo_inds]
akf_sy_PSD = [[np.zeros((int(np.floor(B1[geo]/2)), num_modes))
               for plane in plane_inds] for geo in geo_inds]
akf_as_PSD = [[np.zeros((int(np.floor(B1[geo]/2)), num_modes))
               for plane in plane_inds] for geo in geo_inds]
akf_sy_gaus_PSD = [[np.zeros((int(np.floor(B1[geo]/2)), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
akf_sy_harm_PSD = [[np.zeros((int(np.floor(B1[geo]/2)), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
f = [np.zeros((int(np.floor(B1[geo]/2)))) for geo in geo_inds]
for geo in geo_inds:
    for plane in plane_inds:
        for mode in modes:
            if mode % 10 == 0:
                print 'geo: %1.0f/%1.0f,  plane: %2.0f/%2.0f,  mode: %4.0f/%4.0f' % \
            (geo, num_geos - 1, plane, num_planes - 1, mode, num_modes - 1)
            akfs = [akf, akf_sy, akf_as, akf_sy_gaus, akf_sy_harm]
            num_POD = np.size(akfs, axis=0)
            POD_inds = np.arange(num_POD)
            akf_PSDs = [np.zeros((int(np.floor(B1[geo]/2)), 1))
                        for _ in POD_inds]
            for (akf_temp, POD) in zip(akfs, POD_inds):
                akf_PSD_temp = np.zeros((int(np.floor(B1[geo]/2)),
                                         num_trials))
                for trial in trials:
                    [f_temp, PSD_temp] = \
                        spect.spectrumFFT_v2(np.squeeze(
                            akf_temp[geo][plane][trial*B1[geo]:
                                                 trial*B1[geo] + B1[geo], mode]
                                                        ),
                                             f_capture[geo])
                    akf_PSD_temp[:, trial] = PSD_temp
                akf_PSDs[POD] = np.squeeze(np.mean(akf_PSD_temp, 1))
            akf_PSD[geo][plane][:, mode] = np.squeeze(akf_PSDs[0])
            akf_sy_PSD[geo][plane][:, mode] = np.squeeze(akf_PSDs[1])
            akf_as_PSD[geo][plane][:, mode] = np.squeeze(akf_PSDs[2])
            akf_sy_gaus_PSD[geo][plane][:, mode] = np.squeeze(akf_PSDs[3])
            akf_sy_harm_PSD[geo][plane][:, mode] = np.squeeze(akf_PSDs[4])
    f[geo] = f_temp
#del akfs, akf_PSDs, f_temp, PSD_temp, akf_PSD_temp


#%% Correct plane height

z_g_raw_uncorr = copy.deepcopy(z_g)
correction = 55.5/57
for geo in geo_inds:
    z_g[geo][:, :, :] = z_g_raw_uncorr[geo][:, :, :]*correction

zs_raw_uncorr = copy.deepcopy(zs)
zs = [zs_raw_uncorr[plane_temp]*correction for plane_temp in plane_inds]


#%% Correct square x location

#undo adhoc correction
x_g_raw_uncorr = copy.deepcopy(x_g)
geo = 1
x_g[geo][:, :, :] = x_g_raw_uncorr[geo][:, :, :] + 0.5/1000/d[geo]

# correct properly
ruler_angle = 2.5 #deg
recirc_lengths = [3.188, 2.746, 2.655, 2.645, 2.620, 2.711, 2.699, 2.752,
                  2.768, 2.742, 2.743, 2.627, 2.545, 2.367, 2.157, 2.031,
                  1.901, 1.640, 1.438, 1.267, 1.073, 0.0834, 1.073, 1.073,
                  1.073, 1.073, 1.073];
for plane in plane_inds:
    ruler_angle_corr = (1 - np.cos(ruler_angle*np.pi/180))*recirc_lengths[plane]
    x_g[geo][:, :, plane] = x_g[geo][:, :, plane] - ruler_angle_corr
    print ruler_angle_corr
#measure_edge_offset = 15.5/1000/d[geo]
#measure_edge_correction = np.tan(ruler_angle*np.pi/180)*measure_edge_offset
#print measure_edge_correction
#x_g[geo][:, :, :] = x_g[geo][:, :, :] + measure_edge_correction
ruler_offset = 1.4/1000/d[geo]
print ruler_offset
x_g[geo][:, :, :] = x_g[geo][:, :, :] + ruler_offset


#%% 
geo = 1
plane = 9
x_g[geo][:, :, plane] = x_g[geo][:, :, plane] +0.054
geo = 1
plane = 7
x_g[geo][:, :, plane] = x_g[geo][:, :, plane] -0.07

#%% Save a copy of raw data before interpolating

x_raw_g = copy.deepcopy(x_g)
y_raw_g = copy.deepcopy(y_g)
z_raw_g = copy.deepcopy(z_g)

M_U_raw_g = copy.deepcopy(M_U_g)
M_V_raw_g = copy.deepcopy(M_V_g)
M_W_raw_g = copy.deepcopy(M_W_g)

M_ufuf_raw_g = copy.deepcopy(M_ufuf_g)
M_vfvf_raw_g = copy.deepcopy(M_vfvf_g)
M_wfwf_raw_g = copy.deepcopy(M_wfwf_g)
M_ufvf_raw_g = copy.deepcopy(M_ufvf_g)
M_ufwf_raw_g = copy.deepcopy(M_ufwf_g)
M_vfwf_raw_g = copy.deepcopy(M_vfwf_g)

M_tke_raw_g = copy.deepcopy(M_tke_g)

M_vfvf_ufuf_raw_g = copy.deepcopy(M_vfvf_ufuf_g)
M_wfwf_ufuf_raw_g = copy.deepcopy(M_wfwf_ufuf_g)
M_wfwf_vfvf_raw_g = copy.deepcopy(M_wfwf_vfvf_g)

#%% Interpolate on to a rectangular non-uniform grid

J = [J[geo].min() for geo in geo_inds]
I = [I[geo].min() for geo in geo_inds]
K = [num_planes]*num_geos

dx =np.zeros(num_geos)
dy =np.zeros(num_geos)
dz =np.zeros(num_geos)
x_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
y_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
z_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_U_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_V_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_W_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_ufuf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_ufvf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_ufwf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_vfvf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_vfwf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_wfwf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_tke_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_vfvf_ufuf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_wfwf_ufuf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_wfwf_vfvf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
for geo in geo_inds:
    for plane in plane_inds:

        x_vect_raw = x_raw_g[geo][0, :, plane]
        y_vect_raw = y_raw_g[geo][:, 0, plane]

        x_vect = np.linspace(np.max(x_raw_g[geo][0, 0, :]),
                             np.min(x_raw_g[geo][0, -1, :]), num=I[geo])
        y_vect = np.linspace(np.max(y_raw_g[geo][0, 0, :]),
                             np.min(y_raw_g[geo][-1, 0, :]), num=J[geo])
        z_vect = z_raw_g[geo][0, 0, :]

        x_g_temp, y_g_temp, z_g_temp = np.meshgrid(x_vect, y_vect, z_vect)
        x_g[geo][:, :, :] = x_g_temp
        y_g[geo][:, :, :] = y_g_temp
        z_g[geo][:, :, :] = z_g_temp

        dx[geo] = x_g[geo][0, 1, 0] - x_g[geo][0, 0, 0]
        dy[geo] = y_g[geo][1, 0, 0] - y_g[geo][0, 0, 0]
        dz[geo] = z_g[geo][0, 0, 1] - z_g[geo][0, 0, 0]

        interp_M_U = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                      M_U_raw_g[geo][:, :, plane])
        M_U_g[geo][:, :, plane] = interp_M_U(x_vect, y_vect)
        interp_M_V = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                      M_V_raw_g[geo][:, :, plane])
        M_V_g[geo][:, :, plane] = interp_M_V(x_vect, y_vect)
        interp_M_W = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                      M_W_raw_g[geo][:, :, plane])
        M_W_g[geo][:, :, plane] = interp_M_W(x_vect, y_vect)

        interp_M_ufuf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                         M_ufuf_raw_g[geo][:, :, plane])
        M_ufuf_g[geo][:, :, plane] = interp_M_ufuf(x_vect, y_vect)
        interp_M_ufvf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                         M_ufvf_raw_g[geo][:, :, plane])
        M_ufvf_g[geo][:, :, plane] = interp_M_ufvf(x_vect, y_vect)
        interp_M_ufwf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                         M_ufwf_raw_g[geo][:, :, plane])
        M_ufwf_g[geo][:, :, plane] = interp_M_ufwf(x_vect, y_vect)
        interp_M_vfvf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                         M_vfvf_raw_g[geo][:, :, plane])
        M_vfvf_g[geo][:, :, plane] = interp_M_vfvf(x_vect, y_vect)
        interp_M_vfwf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                         M_vfwf_raw_g[geo][:, :, plane])
        M_vfwf_g[geo][:, :, plane] = interp_M_vfwf(x_vect, y_vect)
        interp_M_wfwf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                         M_wfwf_raw_g[geo][:, :, plane])
        M_wfwf_g[geo][:, :, plane] = interp_M_wfwf(x_vect, y_vect)

        interp_M_tke = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                        M_tke_raw_g[geo][:, :, plane])
        M_tke_g[geo][:, :, plane] = interp_M_tke(x_vect, y_vect)

        interp_M_vfvf_ufuf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                              M_vfvf_ufuf_raw_g[geo][:, :, plane])
        M_vfvf_ufuf_g[geo][:, :, plane] = interp_M_vfvf_ufuf(x_vect, y_vect)
        interp_M_wfwf_ufuf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                              M_wfwf_ufuf_raw_g[geo][:, :, plane])
        M_wfwf_ufuf_g[geo][:, :, plane] = interp_M_wfwf_ufuf(x_vect, y_vect)
        interp_M_wfwf_vfvf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                              M_wfwf_vfvf_raw_g[geo][:, :, plane])
        M_wfwf_vfvf_g[geo][:, :, plane] = interp_M_wfwf_vfvf(x_vect, y_vect)

#%% Save a copy of raw snapshots data before interpolating

uf_raw_g = copy.deepcopy(uf_g)
vf_raw_g = copy.deepcopy(vf_g)
wf_raw_g = copy.deepcopy(wf_g)

#%% Interpolate on to a rectangular non-uniform grid

#uf_g = [np.zeros((J[geo], I[geo], K[geo], num_snaps)) for geo in geo_inds]
#vf_g = [np.zeros((J[geo], I[geo], K[geo], num_snaps)) for geo in geo_inds]
#wf_g = [np.zeros((J[geo], I[geo], K[geo], num_snaps)) for geo in geo_inds]
for geo in geo_inds:
    for plane in plane_inds:
        print geo, plane
        for snap in np.arange(num_snaps):

            x_vect_raw = x_raw_g[geo][0, :, plane]
            y_vect_raw = y_raw_g[geo][:, 0, plane]
    
            x_vect = np.linspace(np.max(x_raw_g[geo][0, 0, :]),
                                 np.min(x_raw_g[geo][0, -1, :]), num=I[geo])
            y_vect = np.linspace(np.max(y_raw_g[geo][0, 0, :]),
                                 np.min(y_raw_g[geo][-1, 0, :]), num=J[geo])
            z_vect = z_raw_g[geo][0, 0, :]
    
            x_g_temp, y_g_temp, z_g_temp = np.meshgrid(x_vect, y_vect, z_vect)
            x_g[geo][:, :, :] = x_g_temp
            y_g[geo][:, :, :] = y_g_temp
            z_g[geo][:, :, :] = z_g_temp
    
            dx[geo] = x_g[geo][0, 1, 0] - x_g[geo][0, 0, 0]
            dy[geo] = y_g[geo][1, 0, 0] - y_g[geo][0, 0, 0]
            dz[geo] = z_g[geo][0, 0, 1] - z_g[geo][0, 0, 0]
    
            interp_uf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                         uf_g[geo][plane][:, :, snap])
            uf_g[geo][:, :, plane, snap] = interp_uf(x_vect, y_vect)
            interp_vf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                         vf_g[geo][plane][:, :, snap])
            vf_g[geo][:, :, plane, snap] = interp_vf(x_vect, y_vect)
            interp_wf = sinterp.interp2d(x_vect_raw, y_vect_raw,
                                         wf_g[geo][plane][:, :, snap])
            wf_g[geo][:, :, plane, snap] = interp_wf(x_vect, y_vect)


#%% Save a copy of rectangular grid data before interpolating

x_rect_g = copy.deepcopy(x_g)
y_rect_g = copy.deepcopy(y_g)
z_rect_g = copy.deepcopy(z_g)

M_U_rect_g = copy.deepcopy(M_U_g)
M_V_rect_g = copy.deepcopy(M_V_g)
M_W_rect_g = copy.deepcopy(M_W_g)

M_ufuf_rect_g = copy.deepcopy(M_ufuf_g)
M_vfvf_rect_g = copy.deepcopy(M_vfvf_g)
M_wfwf_rect_g = copy.deepcopy(M_wfwf_g)
M_ufvf_rect_g = copy.deepcopy(M_ufvf_g)
M_ufwf_rect_g = copy.deepcopy(M_ufwf_g)
M_vfwf_rect_g = copy.deepcopy(M_vfwf_g)

M_tke_rect_g = copy.deepcopy(M_tke_g)

M_vfvf_ufuf_rect_g = copy.deepcopy(M_vfvf_ufuf_g)
M_wfwf_ufuf_rect_g = copy.deepcopy(M_wfwf_ufuf_g)
M_wfwf_vfvf_rect_g = copy.deepcopy(M_wfwf_vfvf_g)

#%% Interpolate on to a regular uniform grid

K = [38]*num_geos

dx =np.zeros(num_geos)
dy =np.zeros(num_geos)
dz =np.zeros(num_geos)
x_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
y_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
z_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_U_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_V_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_W_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_ufuf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_ufvf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_ufwf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_vfvf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_vfwf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_wfwf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_tke_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_vfvf_ufuf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_wfwf_ufuf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_wfwf_vfvf_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
for geo in geo_inds:
    for i in np.arange(I[geo]):
        for j in np.arange(J[geo]):

            z_vect_raw = z_rect_g[geo][0, 0, :]

            x_vect = x_rect_g[geo][0, :, 0]
            y_vect = y_rect_g[geo][:, 0, 0]
            z_vect = np.linspace(z_rect_g[geo][0, 0, 0],
                                 z_rect_g[geo][0, 0, -1], num=K[geo])

            x_g_temp, y_g_temp, z_g_temp = np.meshgrid(x_vect, y_vect, z_vect)
            x_g[geo][:, :, :] = x_g_temp
            y_g[geo][:, :, :] = y_g_temp
            z_g[geo][:, :, :] = z_g_temp

            dx[geo] = x_g[geo][0, 1, 0] - x_g[geo][0, 0, 0]
            dy[geo] = y_g[geo][1, 0, 0] - y_g[geo][0, 0, 0]
            dz[geo] = z_g[geo][0, 0, 1] - z_g[geo][0, 0, 0]

            interp_M_U = sinterp.interp1d(z_vect_raw, M_U_rect_g[geo][j, i, :])
            M_U_g[geo][j, i, :] = interp_M_U(z_vect)
            interp_M_V = sinterp.interp1d(z_vect_raw, M_V_rect_g[geo][j, i, :])
            M_V_g[geo][j, i, :] = interp_M_V(z_vect)
            interp_M_W = sinterp.interp1d(z_vect_raw, M_W_rect_g[geo][j, i, :])
            M_W_g[geo][j, i, :] = interp_M_W(z_vect)

            interp_M_ufuf = sinterp.interp1d(z_vect_raw,
                                             M_ufuf_rect_g[geo][j, i, :])
            M_ufuf_g[geo][j, i, :] = interp_M_ufuf(z_vect)
            interp_M_ufvf = sinterp.interp1d(z_vect_raw,
                                             M_ufvf_rect_g[geo][j, i, :])
            M_ufvf_g[geo][j, i, :] = interp_M_ufvf(z_vect)
            interp_M_ufwf = sinterp.interp1d(z_vect_raw,
                                             M_ufwf_rect_g[geo][j, i, :])
            M_ufwf_g[geo][j, i, :] = interp_M_ufwf(z_vect)
            interp_M_vfvf = sinterp.interp1d(z_vect_raw,
                                             M_vfvf_rect_g[geo][j, i, :])
            M_vfvf_g[geo][j, i, :] = interp_M_vfvf(z_vect)
            interp_M_vfwf = sinterp.interp1d(z_vect_raw,
                                             M_vfwf_rect_g[geo][j, i, :])
            M_vfwf_g[geo][j, i, :] = interp_M_vfwf(z_vect)
            interp_M_wfwf = sinterp.interp1d(z_vect_raw,
                                             M_wfwf_rect_g[geo][j, i, :])
            M_wfwf_g[geo][j, i, :] = interp_M_wfwf(z_vect)

            interp_M_tke = sinterp.interp1d(z_vect_raw,
                                            M_tke_rect_g[geo][j, i, :])
            M_tke_g[geo][j, i, :] = interp_M_tke(z_vect)

            interp_M_vfvf_ufuf = sinterp.interp1d(z_vect_raw,
                                                  M_vfvf_ufuf_rect_g[geo][j, i, :])
            M_vfvf_ufuf_g[geo][j, i, :] = interp_M_vfvf_ufuf(z_vect)
            interp_M_wfwf_ufuf = sinterp.interp1d(z_vect_raw,
                                                  M_wfwf_ufuf_rect_g[geo][j, i, :])
            M_wfwf_ufuf_g[geo][j, i, :] = interp_M_wfwf_ufuf(z_vect)
            interp_M_wfwf_vfvf = sinterp.interp1d(z_vect_raw,
                                                  M_wfwf_vfvf_rect_g[geo][j, i, :])
            M_wfwf_vfvf_g[geo][j, i, :] = interp_M_wfwf_vfvf(z_vect)

#%% Calculate mean gradients

M_dU_dx_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_dU_dy_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_dU_dz_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_dV_dx_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_dV_dy_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_dV_dz_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_dW_dx_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_dW_dy_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_dW_dz_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
for geo in geo_inds:
    [M_dU_dy_temp, M_dU_dx_temp, M_dU_dz_temp] \
        = np.gradient(M_U_g[geo], y_g[geo][:, 0, 0], x_g[geo][0, :, 0], z_g[geo][0, 0, :])
    [M_dV_dy_temp, M_dV_dx_temp, M_dV_dz_temp] \
        = np.gradient(M_V_g[geo], y_g[geo][:, 0, 0], x_g[geo][0, :, 0], z_g[geo][0, 0, :])
    [M_dW_dy_temp, M_dW_dx_temp, M_dW_dz_temp] \
        = np.gradient(M_W_g[geo], y_g[geo][:, 0, 0], x_g[geo][0, :, 0], z_g[geo][0, 0, :])
    M_dU_dx_g[geo][:, :, :] = M_dU_dx_temp
    M_dU_dy_g[geo][:, :, :] = M_dU_dy_temp
    M_dU_dz_g[geo][:, :, :] = M_dU_dz_temp
    M_dV_dx_g[geo][:, :, :] = M_dV_dx_temp
    M_dV_dy_g[geo][:, :, :] = M_dV_dy_temp
    M_dV_dz_g[geo][:, :, :] = M_dV_dz_temp
    M_dW_dx_g[geo][:, :, :] = M_dW_dx_temp
    M_dW_dy_g[geo][:, :, :] = M_dW_dy_temp
    M_dW_dz_g[geo][:, :, :] = M_dW_dz_temp


#%% Calculate mean vorticity

M_Omega_x_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_Omega_y_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_Omega_z_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_Omega_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_Omega_x_g2 = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_Omega_y_g2 = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_Omega_z_g2 = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_Circ_x_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_Circ_y_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_Circ_z_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
for geo in geo_inds:
    M_Omega_x_g[geo][:, :, :] = M_dW_dy_g[geo] - M_dV_dz_g[geo]
    M_Omega_y_g[geo][:, :, :] = M_dU_dz_g[geo] - M_dW_dx_g[geo]
    M_Omega_z_g[geo][:, :, :] = M_dV_dx_g[geo] - M_dU_dy_g[geo]
    M_Omega_g[geo][:, :, :] = np.sqrt(M_Omega_x_g[geo]**2 +
                                      M_Omega_y_g[geo]**2 +
                                      M_Omega_z_g[geo]**2)
    
#    x_vect_temp = np.linspace(x_g[geo][0, :, 0].min()-dx[geo], 
#                              x_g[geo][0, :, 0].max()+dx[geo], I[geo]+2)
#    y_vect_temp = np.linspace(y_g[geo][:, 0, 0].min()-dy[geo], 
#                              y_g[geo][:, 0, 0].max()+dy[geo], J[geo]+2)
#    z_vect_temp = np.linspace(z_g[geo][0, 0, :].min()-dz[geo], 
#                              z_g[geo][0, 0, :].max()+dz[geo], K[geo]+2)
#    x_g_temp, y_g_temp, z_g_temp = np.meshgrid(x_vect_temp, y_vect_temp, z_vect_temp)
#    interp_M_U = sinterp.RegularGridInterpolator((y_g[geo][:, 0, 0],
#                                                  x_g[geo][0, :, 0], 
#                                                  z_g[geo][0, 0, :]),
#                                                 M_U_g[geo], bounds_error=True,
#                                                 fill_value=None)
#    M_U_g_temp = interp_M_U([x_g_temp[:, 0, 0], y_g_temp[0, :, 0], z_g_temp[0, 0, :]])
#    M_U_g_temp = sinterp.interpn((y_g[geo], x_g[geo], z_g[geo]), M_U_g[geo],
#                                 (y_vect_temp, x_vect_temp, z_vect_temp))
#    M_V_g_temp = sinterp.interpn((y_vect_temp, x_vect_temp, z_vect_temp), M_V_g[geo],
#                               (y_g[geo], x_g[geo], z_g[geo]), fill_value=0)
#    M_W_g_temp = sinterp.interpn((y_vect_temp, x_vect_temp, z_vect_temp), M_W_g[geo],
#                               (y_g[geo], x_g[geo], z_g[geo]), fill_value=0)
#    for k in np.arange(K[geo]) + 1:
#        for i in np.arange(I[geo]) + 1:
#            for j in np.arange(J[geo]) + 1:
#                M_Circ_z_temp = 0.5*dx[geo]*(M_U_g_temp[j-1, i-1, k] +
#                                             M_U_g_temp[j-1, i, k] +
#                                             M_U_g_temp[j-1, i+1, k]) \
#                                      + 0.5*dy[geo]*(M_V_g_temp[j-1, i+1, k] +
#                                                     M_V_g_temp[j, i+1, k] +
#                                                     M_V_g_temp[j+1, i+1, k]) \
#                                      - 0.5*dx[geo]*(M_U_g_temp[j+1, i+1, k] +
#                                                     M_U_g_temp[j+1, i, k] +
#                                                     M_U_g_temp[j+1, i-1, k]) + \
#                                      - 0.5*dy[geo]*(M_V_g_temp[j+1, i-1, k] +
#                                                     M_V_g_temp[j, i-1, k] +
#                                                     M_V_g_temp[j-1, i-1, k])
#                M_Circ_x_temp = 0.5*dy[geo]*(M_V_g_temp[j-1, i, k-1] +
#                                             M_V_g_temp[j, i, k-1] +
#                                             M_V_g_temp[j+1, i, k-1]) \
#                                      + 0.5*dz[geo]*(M_W_g_temp[j+1, i, k-1] +
#                                                     M_W_g_temp[j+1, i, k] +
#                                                     M_W_g_temp[j+1, i, k+1]) \
#                                      - 0.5*dy[geo]*(M_V_g_temp[j+1, i, k+1] +
#                                                     M_V_g_temp[j, i, k+1] +
#                                                     M_V_g_temp[j-1, i, k+1]) + \
#                                      - 0.5*dz[geo]*(M_W_g_temp[j-1, i, k+1] +
#                                                     M_W_g_temp[j-1, i, k] +
#                                                     M_W_g_temp[j-1, i, k-1])
#                M_Circ_y_temp = 0.5*dx[geo]*(M_U_g_temp[j, i+1, k-1] +
#                                             M_U_g_temp[j, i, k-1] +
#                                             M_U_g_temp[j, i-1, k-1]) \
#                                      + 0.5*dz[geo]*(M_W_g_temp[j, i-1, k-1] +
#                                                     M_W_g_temp[j, i-1, k] +
#                                                     M_W_g_temp[j, i-1, k+1]) \
#                                      - 0.5*dx[geo]*(M_U_g_temp[j, i-1, k+1] +
#                                                     M_U_g_temp[j, i, k+1] +
#                                                     M_U_g_temp[j, i+1, k+1]) + \
#                                      - 0.5*dz[geo]*(M_W_g_temp[j, i+1, k+1] +
#                                                     M_W_g_temp[j, i+1, k] +
#                                                     M_W_g_temp[j, i+1, k-1])
#                M_Circ_x_g[geo][j-1, i-1, k-1] = M_Circ_x_temp
#                M_Circ_y_g[geo][j-1, i-1, k-1] = M_Circ_y_temp
#                M_Circ_z_g[geo][j-1, i-1, k-1] = M_Circ_z_temp
#                M_Omega_z_g2[geo][j-1, i-1, k-1] = M_Circ_z_temp/(4*dx[geo]*dy[geo])
#                M_Omega_x_g2[geo][j-1, i-1, k-1] = M_Circ_x_temp/(4*dy[geo]*dz[geo])
#                M_Omega_y_g2[geo][j-1, i-1, k-1] = M_Circ_y_temp/(4*dx[geo]*dz[geo])   

#%% Calculate Q-criterion and Lambda2-criterion

M_Q_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_Div_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_lambda2_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
#M_lambda2_Q_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
for geo in geo_inds:
    for k in np.arange(K[geo]):
        for i in np.arange(I[geo]):
            for j in np.arange(J[geo]):
                M_dU_dx_tens = np.array([
                                         [
                                          M_dU_dx_g[geo][j, i, k],
                                          M_dV_dx_g[geo][j, i, k],
                                          M_dW_dx_g[geo][j, i, k]
                                          ],
                                         [
                                          M_dU_dy_g[geo][j, i, k],
                                          M_dV_dy_g[geo][j, i, k],
                                          M_dW_dy_g[geo][j, i, k]
                                          ],
                                         [
                                          M_dU_dz_g[geo][j, i, k],
                                          M_dV_dz_g[geo][j, i, k],
                                          M_dW_dz_g[geo][j, i, k]
                                          ]
                                         ])
                M_Omega_tens = 0.5*(M_dU_dx_tens - M_dU_dx_tens.T)
                M_S_tens = 0.5*(M_dU_dx_tens + M_dU_dx_tens.T)

                Q = 0.5*(np.trace(M_dU_dx_tens)**2 -
                         np.trace(M_dU_dx_tens.dot(M_dU_dx_tens)))
                M_Q_g[geo][j, i, k] = Q

                M_Div_g[geo][j, i, k] = np.trace(M_dU_dx_tens)

                M_S2_Omega2_tens = M_S_tens.dot(M_S_tens) +\
                    M_Omega_tens.dot(M_Omega_tens)
                eig_val, eig_vect = slinalg.eig(M_S2_Omega2_tens)
                eig_val_sorted = np.sort(np.real(eig_val))
                M_lambda2_g[geo][j, i, k] = eig_val_sorted[1]
#                M_lambda2_Q_g[geo][j, i, k] = 0.5*(np.trace(M_dU_dx_tens)**2 -
#                                                   np.sum(eig_val))

#%% Calculate Turbulent Production 

M_G_k_g = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
for geo in geo_inds:
    M_G_k_g[geo] = -(M_ufuf_g[geo]*M_dU_dx_g[geo] +
                     M_vfvf_g[geo]*M_dV_dy_g[geo] +
                     M_wfwf_g[geo]*M_dW_dz_g[geo] +
                     M_ufvf_g[geo]*(M_dV_dx_g[geo] + M_dU_dy_g[geo]) +
                     M_ufwf_g[geo]*(M_dU_dz_g[geo] + M_dW_dx_g[geo]) +
                     M_vfwf_g[geo]*(M_dV_dz_g[geo] + M_dW_dy_g[geo]))

#%%










#%% Figures of everything v1

#%% Find indices of xy-planes

z_locs = np.array([0, 1, 2, 3, 4])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[0][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[0][0, 0, :] - z_loc)))
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


#%% Plot Mean Velocity Fields

field_g_list = [M_U_g, M_V_g, M_W_g]
field_labels = [
                r'$\displaystyle \frac{\overline{U}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{V}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{W}}{U_\infty}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels, 
                                             xy_plane_titles, 'xy', 
                                             xy_plane_inds, num_geos, obstacles,
                                             PUBdir, xy_save_names, 'mean_vel',
                                             cb_label_1=True, )
plt.close('all')


#%% Plot Reynolds Normal Stress Fields

field_g_list = (M_ufuf_g, M_vfvf_g, M_wfwf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels,
                                             xy_plane_titles, 'xy', 
                                             xy_plane_inds, num_geos, obstacles,
                                             PUBdir, xy_save_names, 'rs_norm',
                                             cb_tick_format='%.3f')
plt.close('all')


#%% Plot Reynolds Shear Stress Fields

field_g_list = (M_ufvf_g, M_ufwf_g, M_vfwf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels,
                                             xy_plane_titles, 'xy', 
                                             xy_plane_inds, num_geos, obstacles,
                                             PUBdir, xy_save_names, 'rs_shear',
                                             cb_tick_format='%.3f')
plt.close('all')


#%% Plot Turbulent Kinetic Energy Fields

field_g_list = [M_tke_g]
field_labels = [r'$\displaystyle \frac{\overline{k}}{U_\infty^2}$']
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels,
                                             xy_plane_titles, 'xy', 
                                             xy_plane_inds, num_geos, obstacles,
                                             PUBdir, xy_save_names, 'tke',
                                             cb_tick_format='%.3f')
plt.close('all')


#%% Plot Turbulent Kinetic Energy Production Fields

field_g_list = [M_G_k_g]
field_labels = [r'$\displaystyle \frac{\overline{u_i\textquotesingle u_j\textquotesingle}}{U_\infty^2}\frac{\partial\frac{U_i}{U_\infty}}{\partial\frac{x_j}{d}}$']
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels,
                                             xy_plane_titles, 'xy', 
                                             xy_plane_inds, num_geos, obstacles,
                                             PUBdir, xy_save_names, 'tke_prod',
                                             cb_tick_format='%.3f')
plt.close('all')


#%% Plot Mean Vorticity Fields

field_g_list = [M_Omega_x_g, M_Omega_y_g, M_Omega_z_g]
field_labels = [
                r'$\displaystyle \frac{\overline{\Omega_x}}{U_\infty/d}$',
                r'$\displaystyle \frac{\overline{\Omega_y}}{U_\infty/d}$',
                r'$\displaystyle \frac{\overline{\Omega_z}}{U_\infty/d}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels, 
                                             xy_plane_titles, 'xy', 
                                             xy_plane_inds, num_geos, obstacles,
                                             PUBdir, xy_save_names, 'mean_vort',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Mean Vorticity Magnitude Fields

field_g_list = [M_Omega_g]
field_labels = [r'$\displaystyle \frac{\left|\overline{\Omega}\right|}{U_\infty/d}$']
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels, 
                                             xy_plane_titles, 'xy', 
                                             xy_plane_inds, num_geos, obstacles,
                                             PUBdir, xy_save_names, 'mean_vort_mag',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Shape Factor Fields

field_g_list = (M_vfvf_ufuf_g, M_wfwf_ufuf_g, M_wfwf_vfvf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{v\textquotesingle^2}}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels,
                                             xy_plane_titles, 'xy', 
                                             xy_plane_inds, num_geos, obstacles,
                                             PUBdir, xy_save_names, 'rs_shape_fact')
plt.close('all')


#%% Find indices of xz-planes

y_locs = np.array([0, 0.25, 0.5])
num_xz_planes = np.size(y_locs, axis=0)

xz_plane_inds = [np.zeros(num_xz_planes, dtype=int) for _ in geo_inds]
xz_plane_titles = [['' for _ in np.arange(num_xz_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (y_loc, y_loc_ind) in zip(y_locs, np.arange(num_xz_planes)):
        ind = np.where(np.abs(y_g[geo][:, 0, 0] - y_loc) ==
                       np.min(np.abs(y_g[geo][:, 0, 0] - y_loc)))
        xz_plane_inds[geo][y_loc_ind] = ind[0][0]
        xz_plane_titles[geo][y_loc_ind] = 'y/d = %1.2f' \
            % np.abs(y_g[geo][ind[0][0], 0, 0])
xz_save_names_temp = ['y_%1.1fd' % y_locs[ind]
                      for ind in np.arange(num_xz_planes)]
xz_save_names = [xz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_xz_planes)]


#%% Plot Mean Velocity Fields

field_g_list = [M_U_g, M_V_g, M_W_g]
field_labels = [
                r'$\displaystyle \frac{\overline{U}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{V}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{W}}{U_\infty}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, z_g, field_g_list, M_U_g,
                                             M_W_g, field_labels, 
                                             xz_plane_titles, 'xz', 
                                             xz_plane_inds, num_geos, obstacles,
                                             PUBdir, xz_save_names, 'mean_vel',
                                             obstacle_AR=obstacle_AR,
                                             cb_label_1=True)
plt.close('all')


#%% Plot Reynolds Normal Stress Fields

field_g_list = (M_ufuf_g, M_vfvf_g, M_wfwf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, z_g, field_g_list, M_U_g,
                                             M_W_g, field_labels,
                                             xz_plane_titles, 'xz', 
                                             xz_plane_inds, num_geos, obstacles,
                                             PUBdir, xz_save_names, 'rs_norm',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Reynolds Shear Stress Fields

field_g_list = (M_ufvf_g, M_ufwf_g, M_vfwf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, z_g, field_g_list, M_U_g,
                                             M_W_g, field_labels,
                                             xz_plane_titles, 'xz', 
                                             xz_plane_inds, num_geos, obstacles,
                                             PUBdir, xz_save_names, 'rs_shear',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Turbulent Kinetic Energy Fields

field_g_list = [M_tke_g]
field_labels = [r'$\displaystyle \frac{\overline{k}}{U_\infty^2}$']
pltf.plot_normalized_planar_field_comparison(x_g, z_g, field_g_list, M_U_g,
                                             M_W_g, field_labels,
                                             xz_plane_titles, 'xz', 
                                             xz_plane_inds, num_geos, obstacles,
                                             PUBdir, xz_save_names, 'tke',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Turbulent Kinetic Energy Production Fields

field_g_list = [M_G_k_g]
field_labels = [r'$\displaystyle \frac{\overline{u_i\textquotesingle u_j\textquotesingle}}{U_\infty^2}\frac{\partial\frac{U_i}{U_\infty}}{\partial\frac{x_j}{d}}$']
pltf.plot_normalized_planar_field_comparison(x_g, z_g, field_g_list, M_U_g,
                                             M_W_g, field_labels,
                                             xz_plane_titles, 'xz', 
                                             xz_plane_inds, num_geos, obstacles,
                                             PUBdir, xz_save_names, 'tke_prod',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Mean Vorticity Fields

field_g_list = [M_Omega_x_g, M_Omega_y_g, M_Omega_z_g]
field_labels = [
                r'$\displaystyle \frac{\overline{\Omega_x}}{U_\infty/d}$',
                r'$\displaystyle \frac{\overline{\Omega_y}}{U_\infty/d}$',
                r'$\displaystyle \frac{\overline{\Omega_z}}{U_\infty/d}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, z_g, field_g_list, M_U_g,
                                             M_W_g, field_labels, 
                                             xz_plane_titles, 'xz', 
                                             xz_plane_inds, num_geos, obstacles,
                                             PUBdir, xz_save_names, 'mean_vort',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Mean Vorticity Magnitude Fields

field_g_list = [M_Omega_g]
field_labels = [r'$\displaystyle \frac{\left|\overline{\Omega}\right|}{U_\infty/d}$']
pltf.plot_normalized_planar_field_comparison(x_g, z_g, field_g_list, M_U_g,
                                             M_W_g, field_labels, 
                                             xz_plane_titles, 'xz', 
                                             xz_plane_inds, num_geos, obstacles,
                                             PUBdir, xz_save_names, 'mean_vort_mag',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Shape Factor Fields

field_g_list = (M_vfvf_ufuf_g, M_wfwf_ufuf_g, M_wfwf_vfvf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{v\textquotesingle^2}}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, z_g, field_g_list, M_U_g,
                                             M_W_g, field_labels,
                                             xz_plane_titles, 'xz', 
                                             xz_plane_inds, num_geos, obstacles,
                                             PUBdir, xz_save_names, 'rs_shape_fact',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Find indices of yz-planes

x_locs = np.array([0, 1, 2, 3, 4, 5, 5.5])
num_yz_planes = np.size(x_locs, axis=0)

yz_plane_inds = [np.zeros(num_yz_planes, dtype=int) for _ in geo_inds]
yz_plane_titles = [['' for _ in np.arange(num_yz_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (x_loc, x_loc_ind) in zip(x_locs, np.arange(num_yz_planes)):
        ind = np.where(np.abs(x_g[geo][0, :, 0] - x_loc) ==
                       np.min(np.abs(x_g[geo][0, :, 0] - x_loc)))
        yz_plane_inds[geo][x_loc_ind] = ind[0][0]
        yz_plane_titles[geo][x_loc_ind] = 'x/d = %1.2f' \
            % np.abs(x_g[geo][0, ind[0][0], 0])
yz_save_names_temp = ['x_%1.1fd' % x_locs[ind]
                      for ind in np.arange(num_yz_planes)]
yz_save_names = [yz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_yz_planes)]


#%% Plot Mean Velocity Fields

os.chdir(PUBdir)

field_g_list = [M_U_g, M_V_g, M_W_g]
field_labels = [
                r'$\displaystyle \frac{\overline{U}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{V}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{W}}{U_\infty}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels, 
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'mean_vel',
                                             obstacle_AR=obstacle_AR,
                                             cb_label_1=True)
plt.close('all')


#%% Plot Reynolds Normal Stress Fields

os.chdir(PUBdir)

field_g_list = (M_ufuf_g, M_vfvf_g, M_wfwf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'rs_norm',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Reynolds Shear Stress Fields

os.chdir(PUBdir)

field_g_list = (M_ufvf_g, M_ufwf_g, M_vfwf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'rs_shear',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Turbulent Kinetic Energy Fields

os.chdir(PUBdir)

field_g_list = [M_tke_g]
field_labels = [r'$\displaystyle \frac{\overline{k}}{U_\infty^2}$']
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'tke',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Turbulent Kinetic Energy Production Fields

field_g_list = [M_G_k_g]
field_labels = [r'$\displaystyle \frac{\overline{u_i\textquotesingle u_j\textquotesingle}}{U_\infty^2}\frac{\partial\frac{U_i}{U_\infty}}{\partial\frac{x_j}{d}}$']
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'tke_prod',
                                             cb_tick_format='%.3f', cb_label_pad=25,
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Mean Vorticity Fields

os.chdir(PUBdir)

field_g_list = [M_Omega_x_g, M_Omega_y_g, M_Omega_z_g]
field_labels = [
                r'$\displaystyle \frac{\overline{\Omega_x}}{U_\infty/d}$',
                r'$\displaystyle \frac{\overline{\Omega_y}}{U_\infty/d}$',
                r'$\displaystyle \frac{\overline{\Omega_z}}{U_\infty/d}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels, 
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'mean_vort',
                                             obstacle_AR=obstacle_AR)
#plt.close('all')


#%% Plot Mean Vorticity Magnitude Fields

field_g_list = [M_Omega_g]
field_labels = [r'$\displaystyle \frac{\left|\overline{\Omega}\right|}{U_\infty/d}$']
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels, 
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'mean_vort_mag',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Shape Factor Fields

os.chdir(PUBdir)

field_g_list = (M_vfvf_ufuf_g, M_wfwf_ufuf_g, M_wfwf_vfvf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{v\textquotesingle^2}}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'rs_shape_fact',
                                             obstacle_AR=obstacle_AR)
plt.close('all')

#%%









#%% Figures of everything v2

#%% Find indices of xy-planes

z_locs = np.array([0, 1, 2, 3, 4])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[0][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[0][0, 0, :] - z_loc)))
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


#%% Plot xy-fields

field_g_list = [
                M_U_rect_g,
                M_V_rect_g,
                M_W_rect_g,
                M_ufuf_rect_g,
                M_vfvf_rect_g,
                M_wfwf_rect_g,
                M_ufvf_rect_g,
                M_ufwf_rect_g,
                M_vfwf_rect_g,
                M_tke_rect_g,
                M_G_k_g,
                M_Omega_x_g,
                M_Omega_y_g,
                M_Omega_z_g,
                M_vfvf_ufuf_g,
                M_wfwf_ufuf_g,
                M_wfwf_vfvf_g
                ]
field_label_list = [
                    r'$\displaystyle \frac{\overline{U}}{U_\infty}$',
                    r'$\displaystyle \frac{\overline{V}}{U_\infty}$',
                    r'$\displaystyle \frac{\overline{W}}{U_\infty}$',
                    r'$\displaystyle \frac{\overline{u\textquotesingle^2}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{k}}{U_\infty^2}$',
                    r'$\displaystyle \frac{G_k d}{U_\infty^3}$',
                    r'$\displaystyle \frac{\overline{\Omega_x}}{U_\infty/d}$',
                    r'$\displaystyle \frac{\overline{\Omega_y}}{U_\infty/d}$',
                    r'$\displaystyle \frac{\overline{\Omega_z}}{U_\infty/d}$',
                    r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                    r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                    r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{v\textquotesingle^2}}$'
                    ]
save_name_list = [
                  'M_U',
                  'M_V',
                  'M_W',
                  'M_ufuf',
                  'M_vfvf',
                  'M_wfwf',
                  'M_ufvf',
                  'M_ufwf',
                  'M_vfwf',
                  'M_tke',
                  'M_G_k',
                  'M_Omega_x',
                  'M_Omega_y',
                  'M_Omega_z',
                  'M_vfvf_ufuf',
                  'M_wfwf_ufuf',
                  'M_wfwf_vfvf'
                  ]
save_name_suffix = '_xy_planes'
cont_lvl_field_g = [M_lambda2_g]*np.size(field_g_list, axis=0)
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_label_list,
                                             xy_plane_titles, 'xy',
                                             xy_plane_inds, num_geos,
                                             obstacles, PUBdir, save_name_list,
                                             save_name_suffix,
                                             cb_tick_format='%.3f',
                                             cb_label_1=True,
                                             comparison_type='plane',
                                             plot_contour_lvls=True,
                                             cont_lvl_field_g_list=cont_lvl_field_g,
                                             cont_lvls=[0], mark_max=True,
                                             mark_min=True)

#%% Find indices of xz-planes

y_locs = np.array([0, 0.25, 0.5, 0.75])
num_xz_planes = np.size(y_locs, axis=0)

xz_plane_inds = [np.zeros(num_xz_planes, dtype=int) for _ in geo_inds]
xz_plane_titles = [['' for _ in np.arange(num_xz_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (y_loc, y_loc_ind) in zip(y_locs, np.arange(num_xz_planes)):
        ind = np.where(np.abs(y_g[geo][:, 0, 0] - y_loc) ==
                       np.min(np.abs(y_g[geo][:, 0, 0] - y_loc)))
        xz_plane_inds[geo][y_loc_ind] = ind[0][0]
        xz_plane_titles[geo][y_loc_ind] = 'y/d = %1.2f' \
            % np.abs(y_g[geo][ind[0][0], 0, 0])
xz_save_names_temp = ['y_%1.1fd' % y_locs[ind]
                      for ind in np.arange(num_xz_planes)]
xz_save_names = [xz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_xz_planes)]


#%% Plot xz-fields

field_g_list = [
                M_U_rect_g,
                M_V_rect_g,
                M_W_rect_g,
                M_ufuf_rect_g,
                M_vfvf_rect_g,
                M_wfwf_rect_g,
                M_ufvf_rect_g,
                M_ufwf_rect_g,
                M_vfwf_rect_g,
                M_tke_rect_g,
                M_G_k_g,
                M_Omega_x_g,
                M_Omega_y_g,
                M_Omega_z_g,
                M_vfvf_ufuf_g,
                M_wfwf_ufuf_g,
                M_wfwf_vfvf_g
                ]
field_label_list = [
                    r'$\displaystyle \frac{\overline{U}}{U_\infty}$',
                    r'$\displaystyle \frac{\overline{V}}{U_\infty}$',
                    r'$\displaystyle \frac{\overline{W}}{U_\infty}$',
                    r'$\displaystyle \frac{\overline{u\textquotesingle^2}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{k}}{U_\infty^2}$',
                    r'$\displaystyle \frac{G_k d}{U_\infty^3}$',
                    r'$\displaystyle \frac{\overline{\Omega_x}}{U_\infty/d}$',
                    r'$\displaystyle \frac{\overline{\Omega_y}}{U_\infty/d}$',
                    r'$\displaystyle \frac{\overline{\Omega_z}}{U_\infty/d}$',
                    r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                    r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                    r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{v\textquotesingle^2}}$'
                    ]
save_name_list = [
                  'M_U',
                  'M_V',
                  'M_W',
                  'M_ufuf',
                  'M_vfvf',
                  'M_wfwf',
                  'M_ufvf',
                  'M_ufwf',
                  'M_vfwf',
                  'M_tke',
                  'M_G_k',
                  'M_Omega_x',
                  'M_Omega_y',
                  'M_Omega_z',
                  'M_vfvf_ufuf',
                  'M_wfwf_ufuf',
                  'M_wfwf_vfvf'
                  ]
save_name_suffix = '_xz_planes'
cont_lvl_field_g = [M_lambda2_g]*np.size(field_g_list, axis=0)
pltf.plot_normalized_planar_field_comparison(x_g, z_g, field_g_list, M_U_g,
                                             M_W_g, field_label_list,
                                             xz_plane_titles, 'xz',
                                             xz_plane_inds, num_geos,
                                             obstacles, PUBdir, save_name_list,
                                             save_name_suffix,
                                             obstacle_AR=obstacle_AR,
                                             cb_tick_format='%.3f',
                                             cb_label_1=True,
                                             comparison_type='plane',
                                             plot_contour_lvls=True,
                                             cont_lvl_field_g_list=cont_lvl_field_g,
                                             cont_lvls=[0], mark_max=True,
                                             mark_min=True)


#%% Find indices of yz-planes

x_locs = np.array([0, 1, 2, 3, 4, 5])
num_yz_planes = np.size(x_locs, axis=0)

yz_plane_inds = [np.zeros(num_yz_planes, dtype=int) for _ in geo_inds]
yz_plane_titles = [['' for _ in np.arange(num_yz_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (x_loc, x_loc_ind) in zip(x_locs, np.arange(num_yz_planes)):
        ind = np.where(np.abs(x_g[geo][0, :, 0] - x_loc) ==
                       np.min(np.abs(x_g[geo][0, :, 0] - x_loc)))
        yz_plane_inds[geo][x_loc_ind] = ind[0][0]
        yz_plane_titles[geo][x_loc_ind] = 'x/d = %1.2f' \
            % np.abs(x_g[geo][0, ind[0][0], 0])
yz_save_names_temp = ['x_%1.1fd' % x_locs[ind]
                      for ind in np.arange(num_yz_planes)]
yz_save_names = [yz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_yz_planes)]


#%% Plot yz-fields

field_g_list = [
                M_U_rect_g,
                M_V_rect_g,
                M_W_rect_g,
                M_ufuf_rect_g,
                M_vfvf_rect_g,
                M_wfwf_rect_g,
                M_ufvf_rect_g,
                M_ufwf_rect_g,
                M_vfwf_rect_g,
                M_tke_rect_g,
                M_G_k_g,
                M_Omega_x_g,
                M_Omega_y_g,
                M_Omega_z_g,
                M_vfvf_ufuf_g,
                M_wfwf_ufuf_g,
                M_wfwf_vfvf_g
                ]
field_label_list = [
                    r'$\displaystyle \frac{\overline{U}}{U_\infty}$',
                    r'$\displaystyle \frac{\overline{V}}{U_\infty}$',
                    r'$\displaystyle \frac{\overline{W}}{U_\infty}$',
                    r'$\displaystyle \frac{\overline{u\textquotesingle^2}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                    r'$\displaystyle \frac{\overline{k}}{U_\infty^2}$',
                    r'$\displaystyle \frac{G_k d}{U_\infty^3}$',
                    r'$\displaystyle \frac{\overline{\Omega_x}}{U_\infty/d}$',
                    r'$\displaystyle \frac{\overline{\Omega_y}}{U_\infty/d}$',
                    r'$\displaystyle \frac{\overline{\Omega_z}}{U_\infty/d}$',
                    r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                    r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                    r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{v\textquotesingle^2}}$'
                    ]
save_name_list = [
                  'M_U',
                  'M_V',
                  'M_W',
                  'M_ufuf',
                  'M_vfvf',
                  'M_wfwf',
                  'M_ufvf',
                  'M_ufwf',
                  'M_vfwf',
                  'M_tke',
                  'M_G_k',
                  'M_Omega_x',
                  'M_Omega_y',
                  'M_Omega_z',
                  'M_vfvf_ufuf',
                  'M_wfwf_ufuf',
                  'M_wfwf_vfvf'
                  ]
save_name_suffix = '_yz_planes'
cont_lvl_field_g = [M_lambda2_g]*np.size(field_g_list, axis=0)
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_label_list,
                                             yz_plane_titles, 'yz',
                                             yz_plane_inds, num_geos,
                                             obstacles, PUBdir, save_name_list,
                                             save_name_suffix,
                                             obstacle_AR=obstacle_AR,
                                             cb_tick_format='%.3f',
                                             cb_label_1=True,
                                             comparison_type='plane',
                                             plot_contour_lvls=True,
                                             cont_lvl_field_g_list=cont_lvl_field_g,
                                             cont_lvls=[0], mark_max=True,
                                             mark_min=True)

#%% Plot Mean Velocity Fields

os.chdir(PUBdir)

field_g_list = [M_U_g, M_V_g, M_W_g]
field_labels = [
                r'$\displaystyle \frac{\overline{U}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{V}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{W}}{U_\infty}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels, 
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'mean_vel',
                                             obstacle_AR=obstacle_AR,
                                             cb_label_1=True)
plt.close('all')


#%% Plot Reynolds Normal Stress Fields

os.chdir(PUBdir)

field_g_list = (M_ufuf_g, M_vfvf_g, M_wfwf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'rs_norm',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Reynolds Shear Stress Fields

os.chdir(PUBdir)

field_g_list = (M_ufvf_g, M_ufwf_g, M_vfwf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'rs_shear',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Turbulent Kinetic Energy Fields

os.chdir(PUBdir)

field_g_list = [M_tke_g]
field_labels = [r'$\displaystyle \frac{\overline{k}}{U_\infty^2}$']
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'tke',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Turbulent Kinetic Energy Production Fields

field_g_list = [M_G_k_g]
field_labels = [r'$\displaystyle \frac{\overline{u_i\textquotesingle u_j\textquotesingle}}{U_\infty^2}\frac{\partial\frac{U_i}{U_\infty}}{\partial\frac{x_j}{d}}$']
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'tke_prod',
                                             cb_tick_format='%.3f', cb_label_pad=25,
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Mean Vorticity Fields

os.chdir(PUBdir)

field_g_list = [M_Omega_x_g, M_Omega_y_g, M_Omega_z_g]
field_labels = [
                r'$\displaystyle \frac{\overline{\Omega_x}}{U_\infty/d}$',
                r'$\displaystyle \frac{\overline{\Omega_y}}{U_\infty/d}$',
                r'$\displaystyle \frac{\overline{\Omega_z}}{U_\infty/d}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels, 
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'mean_vort',
                                             obstacle_AR=obstacle_AR)
#plt.close('all')


#%% Plot Mean Vorticity Magnitude Fields

field_g_list = [M_Omega_g]
field_labels = [r'$\displaystyle \frac{\left|\overline{\Omega}\right|}{U_\infty/d}$']
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels, 
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'mean_vort_mag',
                                             obstacle_AR=obstacle_AR)
plt.close('all')


#%% Plot Shape Factor Fields

os.chdir(PUBdir)

field_g_list = (M_vfvf_ufuf_g, M_wfwf_ufuf_g, M_wfwf_vfvf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{v\textquotesingle^2}}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'rs_shape_fact',
                                             obstacle_AR=obstacle_AR)
plt.close('all')

#%%










#%% Playing

#%% Find indices of yz-planes

x_locs = np.array([5.5])
num_yz_planes = np.size(x_locs, axis=0)

yz_plane_inds = [np.zeros(num_yz_planes, dtype=int) for _ in geo_inds]
yz_plane_titles = [['' for _ in np.arange(num_yz_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (x_loc, x_loc_ind) in zip(x_locs, np.arange(num_yz_planes)):
        ind = np.where(np.abs(x_g[geo][0, :, 0] - x_loc) ==
                       np.min(np.abs(x_g[geo][0, :, 0] - x_loc)))
        yz_plane_inds[geo][x_loc_ind] = ind[0][0]
        yz_plane_titles[geo][x_loc_ind] = 'x/d = %1.2f' \
            % np.abs(x_g[geo][0, ind[0][0], 0])
yz_save_names_temp = ['x_%1.1fd' % x_locs[ind]
                      for ind in np.arange(num_yz_planes)]
yz_save_names = [yz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_yz_planes)]

#%% Plot Reynolds Shear Stress Fields

os.chdir(PUBdir)

field_g_list = (M_Omega_x_g, M_Q_g, M_lambda2_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(y_g, z_g, field_g_list, M_V_g,
                                             M_W_g, field_labels,
                                             yz_plane_titles, 'yz', 
                                             yz_plane_inds, num_geos, obstacles,
                                             PUBdir, yz_save_names, 'rs_shear',
                                             cb_tick_format='%.3f',
                                             obstacle_AR=obstacle_AR, save_fig=False,
                                             plot_contour_lvls=True, cont_field_g_list=cont_field_g_list,
                                             cont_lvls=[0], mark_max=True, mark_min=True)
#plt.close('all')

#%% Find indices of xy-planes

z_locs = np.array([1.25])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[0][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[0][0, 0, :] - z_loc)))
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]

#%% Plot Reynolds Shear Stress Fields

field_g_list = (M_Omega_z_g, M_Q_g, M_lambda2_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$'
                ]
cont_field_g_list = [M_lambda2_g]*3
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels,
                                             xy_plane_titles, 'xy', 
                                             xy_plane_inds, num_geos, obstacles,
                                             PUBdir, xy_save_names, 'rs_shear',
                                             cb_tick_format='%.3f', save_fig=False,
                                             plot_contour_lvls=True, cont_field_g_list=cont_field_g_list,
                                             cont_lvls=[0], cont_colour='g', cont_lw=1.5)
#plt.close('all')

#%% Find indices of xz-planes

y_locs = np.array([0])
num_xz_planes = np.size(y_locs, axis=0)

xz_plane_inds = [np.zeros(num_xz_planes, dtype=int) for _ in geo_inds]
xz_plane_titles = [['' for _ in np.arange(num_xz_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (y_loc, y_loc_ind) in zip(y_locs, np.arange(num_xz_planes)):
        ind = np.where(np.abs(y_g[geo][:, 0, 0] - y_loc) ==
                       np.min(np.abs(y_g[geo][:, 0, 0] - y_loc)))
        xz_plane_inds[geo][y_loc_ind] = ind[0][0]
        xz_plane_titles[geo][y_loc_ind] = 'y/d = %1.2f' \
            % np.abs(y_g[geo][ind[0][0], 0, 0])
xz_save_names_temp = ['y_%1.1fd' % y_locs[ind]
                      for ind in np.arange(num_xz_planes)]
xz_save_names = [xz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_xz_planes)]

#%% Plot Mean Vorticity Fields

field_g_list = [M_Omega_y_g, M_Q_g, M_lambda2_g]
field_labels = [
                r'$\displaystyle \frac{\overline{\Omega_x}}{U_\infty/d}$',
                r'$\displaystyle \frac{\overline{\Omega_y}}{U_\infty/d}$',
                r'$\displaystyle \frac{\overline{\Omega_z}}{U_\infty/d}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, z_g, field_g_list, M_U_g,
                                             M_W_g, field_labels, 
                                             xz_plane_titles, 'xz', 
                                             xz_plane_inds, num_geos, obstacles,
                                             PUBdir, xz_save_names, 'mean_vort',
                                             obstacle_AR=obstacle_AR, save_fig=False,
                                             plot_contour_lvls=True, cont_field_g_list=cont_field_g_list,
                                             cont_lvls=[0], cont_colour='g', cont_lw=1.5)
#plt.close('all')

#%%










#%% mask out areas  of M_lambda_2

M_lambda2_2_g = copy.deepcopy(M_lambda2_g)
for geo in geo_inds:
    for i in np.arange(I[geo]):
        for j in np.arange(J[geo]):
            for k in np.arange(K[geo]):
                if z_g[geo][j, i, k] >= 3.5 and x_g[geo][j, i, k] >= 3:
                    M_lambda2_2_g[geo][j, i, k] = M_lambda2_2_g[geo].max()
                if z_g[geo][j, i, k] >= 3.5 and np.abs(y_g[geo][j, i, k]) >= 2:
                    M_lambda2_2_g[geo][j, i, k] = M_lambda2_2_g[geo].max()                

#%% Save data for reloading becasue that makes so much sense. 

SAVEdir= r'D:\Workspaces'
SAVEnames = obstacles
for geo in geo_inds:
    SAVEpath = SAVEdir + '\\' + SAVEnames[geo] + '_mean_field.csv'
    savedata = np.array([x_g[geo].flatten(), 
                         y_g[geo].flatten(), 
                         z_g[geo].flatten(), 
                         M_U_g[geo].flatten(), 
                         M_V_g[geo].flatten(), 
                         M_W_g[geo].flatten(),
                         M_Q_g[geo].flatten(),
                         M_Omega_x_g[geo].flatten(),
                         M_Omega_y_g[geo].flatten(),
                         M_Omega_z_g[geo].flatten(),
                         M_lambda2_2_g[geo].flatten(),
                         M_ufuf_g[geo].flatten(),
                         M_vfvf_g[geo].flatten(),
                         M_wfwf_g[geo].flatten(),
                         M_ufvf_g[geo].flatten(),
                         M_ufwf_g[geo].flatten(),
                         M_vfwf_g[geo].flatten(),
                         M_tke_g[geo].flatten(),
                         M_G_k_g[geo].flatten(),
                         ]).T
    savedataheader = 'x,y,z,M_U,M_V,M_W,Q,M_Omega_x,M_Omega_y,M_Omega_z,M_lambda2,M_ufuf,M_vfvf,M_wfwf,M_ufvf,M_ufwf,M_vfwf,M_k,M_G_k'
    np.savetxt(SAVEpath, savedata, fmt='%1.6e', delimiter = ',', header=savedataheader, comments='')


#%% Plot Profiles

os.chdir(PUBdir)

y_label_strs = [
                r'$\displaystyle \frac{\overline{u''v''}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{u''w''}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v''w''}}{U_\infty^2}$'
                ]

fig = plt.figure(figsize=(6.7, 9))

num_geo = 2
num_comp = 3
for geo in np.arange(num_geo):
    for comp in np.arange(num_comp):
        for loc in loc_inds:
            plt.plot(x_g_list[geo][ind, :], M_U_g_list[geo][ind, :, comp])
