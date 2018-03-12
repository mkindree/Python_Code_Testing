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
import string
import matplotlib.text as mtext
import matplotlib.patches as mpatch

import plotting_functions as pltf
import grid_conversion as gc
import spectrum_fft_v2 as spect
import load_from_matlab_workspace as load_mat


#%% User-Defined Variables

mainfolder = r'D:\7 - SquareAR4_LaminarBL'# The main working directory
PUBdir =  r'D:\EXF Paper\EXF Paper V4\figs'# Where the .eps figures go

WRKSPCdirs = (r'D:\Workspaces', r'D:\Workspaces')# Where the workspaces are located
#WRKSPCdirs = (r'D:\Okanagan Conference\Workspaces\old tests',
#              r'D:\Okanagan Conference\Workspaces\old tests')# Where the workspaces are located
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
                         r'square_z_10.5mm', r'square_z_12mm', 
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
#WRKSPCfilenames_both = [[r'circle_z_1.5mm', r'circle_z_7.5mm', 
#                         r'circle_z_10.5mm', r'circle_z_27mm', 
#                         r'circle_z_45mm'], 
#                        [r'square_z_1.5mm', r'square_z_7.5mm', 
#                         r'square_z_10.5mm', r'square_z_27mm',
#                         r'square_z_45mm', 
#                         ]]# Name of the MATLAB workspaces
#WRKSPCfilenames_both = [[r'circle_z_1.5mm', r'circle_z_12mm', 
#                         r'circle_z_27mm', r'circle_z_39mm'], 
#                        [r'square_z_1.5mm', r'square_z_12mm', 
#                         r'square_z_27mm', r'square_z_39mm']]# Name of the MATLAB workspaces
#                                                             # for POD
WRKSPCfilenames_both = [[r'circle_z_10.5mm'], 
                        [r'square_z_10.5mm']]# Name of the MATLAB workspaces   
    
#WRKSPCfilenames_both = [[r'circle_z_52mm_POD2'], 
#                        [r'square_z_52mm_POD2']]# Name of the MATLAB workspaces
    
zs = [1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39,
      42, 45, 46.5, 48, 49.5, 51, 52.5, 54, 55.5, 57]
#zs = [1.5, 7.5, 10.5, 27, 45]
#zs = [1.5, 12, 27, 39] #POD
zs = [10.5]
#zs = [52]
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

num_snaps = 16368

#%% Load variables

[B, B1, f_capture, d, t, I, J, num_vect, dx, dy, f_shed, St_shed, U_inf, Re,
 x_g, y_g, z_g] = \
    load_mat.load_general(WRKSPCdirs, WRKSPCfilenames_both, num_geos,
                          num_planes, zs)

#[B, B1, f_capture, d, I, J, num_vect, f_shed, St_shed, U_inf,
# x_g, y_g, z_g] = \
#    load_mat.load_general_old_version(WRKSPCdirs, WRKSPCfilenames_both,
#                                      num_geos, num_planes, zs)
#d = [0.0127, 0.0133]
#z_g = [z_g[geo]/1000/d[geo] for geo in geo_inds]

[M_U_g, M_V_g, M_W_g,
 M_ufuf_g, M_ufvf_g, M_ufwf_g, M_vfvf_g, M_vfwf_g, M_wfwf_g,
 M_tke_g,
 M_vfvf_ufuf_g, M_wfwf_ufuf_g, M_wfwf_vfvf_g] = \
    load_mat.load_mean_field(WRKSPCdirs, WRKSPCfilenames_both, I, J, num_geos,
                             num_planes)

#[M_U_g, M_V_g, 
# M_ufuf_g, M_ufvf_g, M_vfvf_g,
# M_tke_g,
# M_vfvf_ufuf_g] = \
#    load_mat.load_mean_field_2D(WRKSPCdirs, WRKSPCfilenames_both, I, J, num_geos,
#                             num_planes)

#[akf, akf_sy, akf_as, akf_sy_gaus, akf_sy_harm,
# lambdaf, lambdaf_sy, lambdaf_as, lambdaf_sy_gaus, lambdaf_sy_harm,
# Psi_uf_g, Psi_uf_sy_g, Psi_uf_as_g, Psi_uf_sy_gaus_g, Psi_uf_sy_harm_g,
# Psi_vf_g, Psi_vf_sy_g, Psi_vf_as_g, Psi_vf_sy_gaus_g, Psi_vf_sy_harm_g,
# Psi_wf_g, Psi_wf_sy_g, Psi_wf_as_g, Psi_wf_sy_gaus_g, Psi_wf_sy_harm_g] = \
#    load_mat.load_POD(WRKSPCdirs, WRKSPCfilenames_both, I, J, B, num_geos,
#                      num_planes, num_modes)

#[akf, akf_sy, akf_as, akf_sy_gaus, akf_sy_harm,
# lambdaf, lambdaf_sy, lambdaf_as, lambdaf_sy_gaus, lambdaf_sy_harm,
# Psi_uf_g, Psi_uf_sy_g, Psi_uf_as_g, Psi_uf_sy_gaus_g, Psi_uf_sy_harm_g,
# Psi_vf_g, Psi_vf_sy_g, Psi_vf_as_g, Psi_vf_sy_gaus_g, Psi_vf_sy_harm_g] = \
#    load_mat.load_POD_2D(WRKSPCdirs, WRKSPCfilenames_both, I, J, B, num_geos,
#                      num_planes, num_modes)

[uf_g, vf_g, wf_g] = \
    load_mat.load_fluctuating_snapshots(WRKSPCdirs, WRKSPCfilenames_both, 
                                        I, J, B, num_geos, num_planes,
                                        num_snaps)

#[u_g, v_g, w_g] = \
#    load_mat.load_snapshots(WRKSPCdirs, WRKSPCfilenames_both, 
#                                        I, J, B, num_geos, num_planes,
#                                        num_snaps)

#[uf_g, vf_g] = \
#    load_mat.load_fluctuating_snapshots_2D(WRKSPCdirs, WRKSPCfilenames_both, 
#                                        I, J, B, num_geos, num_planes,
#                                        num_snaps)

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

# correct edge of ruler not touching obstacle
ruler_offset = 1.4/1000/d[geo]
x_g[geo][:, :, :] = x_g[geo][:, :, :] + ruler_offset


#%% Correct diameters

d_old = copy.deepcopy(d)
d = [0.01290, 0.01305]
x_g = [x_g[geo]*d_old[geo]/d[geo] for geo in geo_inds]
y_g = [y_g[geo]*d_old[geo]/d[geo] for geo in geo_inds]
z_g = [z_g[geo]*d_old[geo]/d[geo] for geo in geo_inds]
obstacle_AR = [obstacle_AR[geo]*d_old[geo]/d[geo] for geo in geo_inds]


#%% 
#geo = 1
#plane = 9
#x_g[geo][:, :, plane] = x_g[geo][:, :, plane] + 0.054
#geo = 1
#plane = 7
#x_g[geo][:, :, plane] = x_g[geo][:, :, plane] - 0.07

#%% Correct St_shed
St_shed_raw = copy.deepcopy(St_shed)

low_threshold = 0.09
up_threshold = 0.21
correct_planes = [np.full([num_planes], False)]*num_geos
St_shed_mean = [0 for geo in geo_inds]
for geo in geo_inds:
    correct_planes[geo] = np.logical_or(St_shed_raw[geo] < low_threshold,  St_shed_raw[geo] > up_threshold)
    St_shed_mean[geo] = np.mean(St_shed_raw[geo][np.invert(correct_planes[geo])])
    for plane in plane_inds:
        if correct_planes[geo][plane]:
            St_shed[geo][plane] = St_shed_mean[geo]
    
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

uf_g = [np.zeros((J[geo], I[geo], K[geo], num_snaps)) for geo in geo_inds]
vf_g = [np.zeros((J[geo], I[geo], K[geo], num_snaps)) for geo in geo_inds]
wf_g = [np.zeros((J[geo], I[geo], K[geo], num_snaps)) for geo in geo_inds]
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
        = np.gradient(M_U_g[geo], y_g[geo][:, 0, 0], x_g[geo][0, :, 0],
                      z_g[geo][0, 0, :])
    [M_dV_dy_temp, M_dV_dx_temp, M_dV_dz_temp] \
        = np.gradient(M_V_g[geo], y_g[geo][:, 0, 0], x_g[geo][0, :, 0],
                      z_g[geo][0, 0, :])
    [M_dW_dy_temp, M_dW_dx_temp, M_dW_dz_temp] \
        = np.gradient(M_W_g[geo], y_g[geo][:, 0, 0], x_g[geo][0, :, 0],
                      z_g[geo][0, 0, :])
    M_dU_dx_g[geo][:, :, :] = M_dU_dx_temp
    M_dU_dy_g[geo][:, :, :] = M_dU_dy_temp
    M_dU_dz_g[geo][:, :, :] = M_dU_dz_temp
    M_dV_dx_g[geo][:, :, :] = M_dV_dx_temp
    M_dV_dy_g[geo][:, :, :] = M_dV_dy_temp
    M_dV_dz_g[geo][:, :, :] = M_dV_dz_temp
    M_dW_dx_g[geo][:, :, :] = M_dW_dx_temp
    M_dW_dy_g[geo][:, :, :] = M_dW_dy_temp
    M_dW_dz_g[geo][:, :, :] = M_dW_dz_temp


#%% Calculate mean gradients rectangular grid

M_dU_dx_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_dU_dy_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_dU_dz_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_dV_dx_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_dV_dy_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_dV_dz_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_dW_dx_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_dW_dy_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_dW_dz_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
for geo in geo_inds:
    [M_dU_dy_temp, M_dU_dx_temp, M_dU_dz_temp] \
        = np.gradient(M_U_rect_g[geo], y_rect_g[geo][:, 0, 0],
                      x_rect_g[geo][0, :, 0], z_rect_g[geo][0, 0, :])
    [M_dV_dy_temp, M_dV_dx_temp, M_dV_dz_temp] \
        = np.gradient(M_V_rect_g[geo], y_rect_g[geo][:, 0, 0],
                      x_rect_g[geo][0, :, 0], z_rect_g[geo][0, 0, :])
    [M_dW_dy_temp, M_dW_dx_temp, M_dW_dz_temp] \
        = np.gradient(M_W_rect_g[geo], y_rect_g[geo][:, 0, 0],
                      x_rect_g[geo][0, :, 0], z_rect_g[geo][0, 0, :])
    M_dU_dx_rect_g[geo][:, :, :] = M_dU_dx_temp
    M_dU_dy_rect_g[geo][:, :, :] = M_dU_dy_temp
    M_dU_dz_rect_g[geo][:, :, :] = M_dU_dz_temp
    M_dV_dx_rect_g[geo][:, :, :] = M_dV_dx_temp
    M_dV_dy_rect_g[geo][:, :, :] = M_dV_dy_temp
    M_dV_dz_rect_g[geo][:, :, :] = M_dV_dz_temp
    M_dW_dx_rect_g[geo][:, :, :] = M_dW_dx_temp
    M_dW_dy_rect_g[geo][:, :, :] = M_dW_dy_temp
    M_dW_dz_rect_g[geo][:, :, :] = M_dW_dz_temp


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

#%% Calculate mean vorticity rectangular grid

M_Omega_x_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_Omega_y_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_Omega_z_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
M_Omega_rect_g = [np.zeros((J[geo], I[geo], num_planes)) for geo in geo_inds]
for geo in geo_inds:
    M_Omega_x_rect_g[geo][:, :, :] = M_dW_dy_rect_g[geo] - M_dV_dz_rect_g[geo]
    M_Omega_y_rect_g[geo][:, :, :] = M_dU_dz_rect_g[geo] - M_dW_dx_rect_g[geo]
    M_Omega_z_rect_g[geo][:, :, :] = M_dV_dx_rect_g[geo] - M_dU_dy_rect_g[geo]
    M_Omega_rect_g[geo][:, :, :] = np.sqrt(M_Omega_x_rect_g[geo]**2 +
                                           M_Omega_y_rect_g[geo]**2 +
                                           M_Omega_z_rect_g[geo]**2)  


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











#%% Figures for Paper

#%% Plot Mean Vorticity Fields - y/d=0

y_locs = np.array([0])
num_xz_planes = np.size(y_locs, axis=0)

xz_plane_inds = [np.zeros(num_xz_planes, dtype=int) for _ in geo_inds]
xz_plane_titles = [['' for _ in np.arange(num_xz_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (y_loc, y_loc_ind) in zip(y_locs, np.arange(num_xz_planes)):
        ind = np.where(np.abs(y_g[geo][:, 0, 0] - y_loc) ==
                       np.min(np.abs(y_g[geo][:, 0, 0] - y_loc)))
        xz_plane_inds[geo][y_loc_ind] = ind[0][0]
        xz_plane_titles[geo][y_loc_ind] = '$y = %1.2f$' \
            % np.abs(y_g[geo][ind[0][0], 0, 0])
xz_save_names_temp = ['y_%1.1fd' % y_locs[ind]
                      for ind in np.arange(num_xz_planes)]
xz_save_names = [xz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_xz_planes)]

field_g_list = [M_vfvf_g]

field_labels = [r'$\overline{v\textquotesingle^2}$']
#field_labels = [r'$\displaystyle \frac{\overline{v\textquotesingle v\textquotesingle}}{U_\infty^2}$']
bifurc_seed_pt_xy = [[[2, 0.18], [3, 0.3], [5, 0.2]], [None]] # 3mm plane has almost zero W so all of the streamlines stop there, adding a few extra
fig, axs = pltf.plot_normalized_planar_field_comparison(
               x_g, z_g, field_g_list, M_U_g, M_W_g, field_labels, 
               xz_plane_titles, 'xz', xz_plane_inds, num_geos, obstacles, PUBdir,
               [''], 'mean_rs_y_planes', strm_x_g=x_g, strm_y_g=z_g,
               obstacle_AR=obstacle_AR, cb_tick_format='%.2f', y_subplot_label=0.96, x_subplot_label=0.03,
               comparison_type='plane', cb_label_0=False, num_extra_cticks=1, strm_dens=0.6, y_min=-1.1, y_max=4.8,
               plot_bifurcation_lines=[True, False], bifurc_seed_pt_xy=bifurc_seed_pt_xy,
               bifurc_colour='k', bifurc_lw=0.5, smooth_colourplot=False, colour_subplot_label=False,
               plot_arrow=False, halve_cb=True, figsize=(6.8, 2.5), close_fig=False, save_fig=False)
#plt.close('all')
#figsize=(5.5, 2)

# Annotating

ax_ind = 0
text = r'$\mathbf{F_{s1}}$'
x_coord = 0.9
y_coord = 3.19
t_x_coord = 0.7#0.4#1.2
t_y_coord = 4.05#3.19#2.9
a_x_coord = 0.6
a_y_coord = 4.05
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='right', va='bottom', ws=1.2, hs=1.35, lsf=-0.1,
                   bsf=0.07, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 0
text = r'$\mathbf{F_{s2}}$'
x_coord = 0.75
y_coord = 0.1
t_x_coord = 0.5#1
t_y_coord = -0.6#-0.2#0.3
a_x_coord = 0.8
a_y_coord = -0.4
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='left', va='top', ws=1.25, hs=1.35, lsf=-0.1,
                   bsf=0.07, a_alpha=1, a_shrink=0.12, a_h_width=6, a_width=2)

ax_ind = 0
text = r'$\mathbf{S\textquotesingle_{s1}}$'
x_coord = 0.5
y_coord = 0.65
t_x_coord = -0.55#0.3#1
t_y_coord = -0.6#0.65
a_x_coord = -0.3
a_y_coord = -0.4
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='left', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0.1, a_alpha=1, a_shrink=0.12, a_h_width=7, a_width=3)

ax_ind = 0
text = r'$\mathbf{S\textquotesingle_{s2}}$'
x_coord = 1
y_coord = 0
t_x_coord = 1.9#0.3#1
t_y_coord = -0.6#0.65
a_x_coord = 1.95
a_y_coord = -0.4
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0.1, a_alpha=1, a_shrink=0.14, a_h_width=6, a_width=2)

ax_ind = 0
text = r'$\mathbf{S\textquotesingle_{s3}}$'
x_coord = 3.82
y_coord = 0
t_x_coord = 3.82#4.2
t_y_coord = -0.6#0.3
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, ha='center', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0.1)

ax_ind = 1
text = r'$\mathbf{F_{s1}}$'
x_coord = 0.66
y_coord = 3.7
t_x_coord = 0.7#0.4#1
t_y_coord = 4.0#3.55#3.5
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, ha='right', va='bottom', ws=1.2, hs=1.35, lsf=-0.1,
                   bsf=0.07)

ax_ind = 1
text = r'$\mathbf{F_{s2}}$'
x_coord = 1.15
y_coord = 0.1
t_x_coord = 1.2#1.4
t_y_coord = -0.6#-0.2#0.27
a_x_coord = 0.9
a_y_coord = -0.4
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='right', va='top', ws=1.25, hs=1.35, lsf=-0.1,
                   bsf=0.07, a_alpha=1, a_shrink=0.12, a_h_width=6, a_width=2)

ax_ind = 1
text = r'$\mathbf{S\textquotesingle_{s1}}$'
x_coord = 0.5
y_coord = 0.59
t_x_coord = 0#0.3#1.05
t_y_coord = -0.6#0.59#1
a_x_coord = 0
a_y_coord = -0.4
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0.1, a_alpha=1, a_shrink=0.12, a_h_width=7, a_width=3)

ax_ind = 1
text = r'$\mathbf{S\textquotesingle_{s2}}$'
x_coord = 1.5
y_coord = 0
t_x_coord = 1.5#0.3#1.05
t_y_coord = -0.6#0.59#1
a_x_coord = 1.9
a_y_coord = -0.4
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='left', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0.1, a_alpha=1, a_shrink=0.12, a_h_width=6, a_width=2)

ax_ind = 1
text = r'$\mathbf{S\textquotesingle_{s3}}$'
x_coord = 3.3
y_coord = 0
t_x_coord = 3.45#3.8
t_y_coord = -0.6#0.3
a_x_coord = 3.85
a_y_coord = -0.4
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='left', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0.1, a_alpha=1, a_shrink=0.12, a_h_width=6, a_width=2)

ax_ind = 1
text = r'$\mathbf{S_{s4}}$'
x_coord = 2.94
y_coord = 0.86
t_x_coord = 4.9#3.44
t_y_coord = 3.2#1.28
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, ha='left', va='bottom', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0.1, a_alpha=1, a_shrink=0.1, a_h_width=10, a_width=4)

ax_ind = 1
text = r'$\mathbf{N_{s1}}$'
x_coord = 3.1
y_coord = 0.1
t_x_coord = 3.15#3.8
t_y_coord = -0.6#-0.2#0.3
a_x_coord = 2.85
a_y_coord = -0.4
pltf.annotate_plot(axs[ax_ind], fig[0], text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='right', va='top', ws=1.2, hs=1.3, lsf=-0.1,
                   bsf=0.05, a_alpha=1, a_shrink=0.12, a_h_width=6, a_width=2)

#plt.sca(axs[0])
#transform = axs[0].transAxes
#plt.text(4.3, 0.3, r'\textbf{S}$\textquotesingle$', ha='left',
#         va='bottom', bbox=bbox_props, size=10)



#plt.sca(axs[1])
#transform = axs[0].transAxes
#plt.plot(0.9, 3.19, 'og', ms=4.5)
#plt.annotate(r'$\mathbf{F_{s1}}$', xy=(0.6, 3.7), xytext=(1.2, 3.5),
#             ha='right', va='center', size=10, bbox=bbox_props,
#             arrowprops=arrow_props)
#
#plt.sca(axs[1])
#transform = axs[0].transAxes
#plt.text(2.5, 1.2, r'\textbf{S}', ha='right',
#         va='bottom', bbox=bbox_props, size=10)

#plt.sca(axs[1])
#transform = axs[0].transAxes
#plt.text(3.5, 0.2, r'\textbf{N}$\textquotesingle$', ha='left',
#         va='bottom', bbox=bbox_props, size=10)

#save_path = PUBdir + '\\' + 'mean_rs_y_planes' + '.eps'
#plt.savefig(save_path, bbox_inches='tight')

#%% Plot Mean Velocity Fields - z=const planes WHich plane to use

z_locs = np.array([0, 1, 2, 3, 4])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = '$z = %1.2f \quad (z_h = %1.2f)$' \
            % (z_g[geo][0, 0, ind[0][0]], z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]

field_g_list = [M_W_g]
field_labels = [r'$W$']
pltf.plot_normalized_planar_field_comparison(
        x_g, y_g, field_g_list, M_U_g, M_V_g, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''], 'mean_vel_z_planes_choices',
        cb_tick_format='%.2f', comparison_type='plane', plot_arrow=False,
        strm_max_len=2, figsize=(5.25, 8), close_fig=False)
#plt.close('all')


#%% Plot Mean Velocity Fields - z=const planes

z_locs = np.array([0, 1, 3.5])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = '$z = %1.2f \quad (z_h = %1.2f)$' \
            % (z_g[geo][0, 0, ind[0][0]], z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]

field_g_list = [M_W_g]
field_labels = [r'$W$']
pltf.plot_normalized_planar_field_comparison(
        x_g, y_g, field_g_list, M_U_g, M_V_g, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''], 'mean_vel_z_planes',
        cb_tick_format='%.2f', comparison_type='plane', plot_arrow=False,
        strm_max_len=2, figsize=(5.25, 5), close_fig=False, extension='.eps')
#plt.close('all')


#%% Plot Mean Vorticity Fields - x=const planes WHich plane to use

x_locs = np.array([2, 2.5, 3, 4, 5, 5.5])

num_yz_planes = np.size(x_locs, axis=0)

yz_plane_inds = [np.zeros(num_yz_planes, dtype=int) for _ in geo_inds]
yz_plane_titles = [['' for _ in np.arange(num_yz_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (x_loc, x_loc_ind) in zip(x_locs, np.arange(num_yz_planes)):
        ind = np.where(np.abs(x_g[geo][0, :, 0] - x_loc) ==
                       np.min(np.abs(x_g[geo][0, :, 0] - x_loc)))
        yz_plane_inds[geo][x_loc_ind] = ind[0][0]
        yz_plane_titles[geo][x_loc_ind] = '$x/d = %1.2f$' \
            % np.abs(x_g[geo][0, ind[0][0], 0])
yz_save_names_temp = ['x_%1.1fd' % x_locs[ind]
                      for ind in np.arange(num_yz_planes)]
yz_save_names = [yz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_yz_planes)]

field_g_list = [M_Omega_x_g]
field_labels = [r'$\displaystyle \frac{\overline{\Omega_x}d}{U_\infty}$']
pltf.plot_normalized_planar_field_comparison(
        y_g, z_g, field_g_list, M_V_g, M_W_g, field_labels, yz_plane_titles,
        'yz', yz_plane_inds, num_geos, obstacles, PUBdir, [''], 'mean_vort_x_planes_choices',
        strm_x_g=y_g, strm_y_g=z_g, obstacle_AR=obstacle_AR,
        cb_tick_format='%.2f', comparison_type='plane', figsize=(5.25, 9),
        close_fig=False, save_fig=False)
#plt.close('all')


#%% Plot Mean Vorticity Fields - x=const planes

x_locs = np.array([0.8, 2.75, 5.75])
num_yz_planes = np.size(x_locs, axis=0)

yz_plane_inds = [np.zeros(num_yz_planes, dtype=int) for _ in geo_inds]
yz_plane_titles = [['' for _ in np.arange(num_yz_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (x_loc, x_loc_ind) in zip(x_locs, np.arange(num_yz_planes)):
        ind = np.where(np.abs(x_g[geo][0, :, 0] - x_loc) ==
                       np.min(np.abs(x_g[geo][0, :, 0] - x_loc)))
        yz_plane_inds[geo][x_loc_ind] = ind[0][0]
        yz_plane_titles[geo][x_loc_ind] = '$x = %1.2f$' \
            % np.abs(x_g[geo][0, ind[0][0], 0])
yz_save_names_temp = ['x_%1.1fd' % x_locs[ind]
                      for ind in np.arange(num_yz_planes)]
yz_save_names = [yz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_yz_planes)]

field_g_list = [M_Omega_x_g]
field_labels = [r'$\Omega_x$']
pltf.plot_normalized_planar_field_comparison(
        y_g, z_g, field_g_list, M_V_g, M_W_g, field_labels, yz_plane_titles,
        'yz', yz_plane_inds, num_geos, obstacles, PUBdir, [''], 'mean_vort_x_planes',
        strm_x_g=y_g, strm_y_g=z_g, plot_streamlines=False, obstacle_AR=obstacle_AR,
        cb_tick_format='%.2f', comparison_type='plane', figsize=(5.25, 6.0),
        x_subplot_label=0, y_subplot_label=1.11,
        plot_filled_contours=True, num_cont_lvls=12, cont_lw=0.5,
        cont_colour='k', plot_contour_lvls=True, alpha_cmap=True, alpha_scale=3.5,
        close_fig=False, save_fig=False, extension='.eps')
#plt.close('all')
num_cont_lvls=8

fig=plt.gcf()
axs = fig.axes

ax_ind = 0
text = r'$\mathbf{T^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -1.3
t_y_coord = 3.6
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 0
text = r'$\mathbf{T^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 1.3
t_y_coord = 3.6
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 0
text = r'$\mathbf{D^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -1.3
t_y_coord = 2.4
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 0
text = r'$\mathbf{D^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 1.3
t_y_coord = 2.4
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 2
text = r'$\mathbf{T_1^-}$'
x_coord = -0.6
y_coord = 4.15
t_x_coord = -1.7
t_y_coord = 3.5
a_x_coord = -1.6
a_y_coord = 3.9
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 2
text = r'$\mathbf{T_1^+}$'
x_coord = 0.6
y_coord = 4.15
t_x_coord = 1.8
t_y_coord = 3.5
a_x_coord = 1.7
a_y_coord = 3.9
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 2
text = r'$\mathbf{T_2^-}$'
x_coord = -0.8
y_coord = 3.6
t_x_coord = -1.9#-1.7
t_y_coord = 2.65#3.2#3.5
a_x_coord = -1.8
a_y_coord = 3.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 2
text = r'$\mathbf{T_2^+}$'
x_coord = 0.8
y_coord = 3.6
t_x_coord = 2.0#1.7
t_y_coord = 2.65#3.2#3.5
a_x_coord = 1.9
a_y_coord = 3.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 2
text = r'$\mathbf{D^-}$'
x_coord = -0.8
y_coord = 2.9
t_x_coord = -1.6
t_y_coord = 1.9
a_x_coord = -1.4
a_y_coord = 2.3
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 2
text = r'$\mathbf{D^+}$'
x_coord = 0.8
y_coord = 2.9
t_x_coord = 1.7
t_y_coord = 1.9
a_x_coord = 1.5
a_y_coord = 2.3
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 4
text = r'$\mathbf{T^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -0.9
t_y_coord = 3.4
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 4
text = r'$\mathbf{T^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 0.9
t_y_coord = 3.4
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 4
text = r'$\mathbf{D^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -1.3
t_y_coord = 2.0
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 4
text = r'$\mathbf{D^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 1.3
t_y_coord = 2.0
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 6
text = r'$\mathbf{T_2^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -1.0
t_y_coord = 3.3
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 6
text = r'$\mathbf{T_2^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 1.0
t_y_coord = 3.3
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 6
text = r'$\mathbf{D^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -1.4
t_y_coord = 2.5
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 6
text = r'$\mathbf{D^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 1.4
t_y_coord = 2.5
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 6
text = r'$\mathbf{DV^-}$'
x_coord = -2.1
y_coord = 1.3
t_x_coord = -1.6#-2.0
t_y_coord = 1.2
a_x_coord = -2.1
a_y_coord = 1.3
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.2, hs=1.35, lsf=-0.1,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 6
text = r'$\mathbf{DV^+}$'
x_coord = 2.1
y_coord = 1.3
t_x_coord = 1.65#2.0
t_y_coord = 1.2
a_x_coord = 2.1
a_y_coord = 1.3
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.2, hs=1.35, lsf=-0.1,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 8
text = r'$\mathbf{T^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -1.0
t_y_coord = 3.1
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 8
text = r'$\mathbf{T^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 1.0
t_y_coord = 3.1
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 8
text = r'$\mathbf{D^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -1.7
t_y_coord = 1.9
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 8
text = r'$\mathbf{D^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 1.7
t_y_coord = 1.9
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 8
text = r'$\mathbf{P^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -1.8
t_y_coord = -0.55
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='top', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 8
text = r'$\mathbf{P^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 1.9
t_y_coord = -0.55
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='top', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 10
text = r'$\mathbf{D^-}$'
x_coord = -1.0
y_coord = 2.9
t_x_coord = -0.7#-1.0
t_y_coord = 3.4
a_x_coord = -1.0
a_y_coord = 3.3
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=5, a_width=2.25)

ax_ind = 10
text = r'$\mathbf{D^+}$'
x_coord = 1.0
y_coord = 2.9
t_x_coord = 0.8#1.0
t_y_coord = 3.4
a_x_coord = 1.0
a_y_coord = 3.3
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=5, a_width=2.25)

ax_ind = 10
text = r'$\mathbf{DV^-}$'
x_coord = -2.1
y_coord = 1.3
t_x_coord = -1.55#-2.0
t_y_coord = 2.9
a_x_coord = -2.3
a_y_coord = 2.8
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='bottom', ws=1.2, hs=1.35, lsf=-0.1,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 10
text = r'$\mathbf{DV^+}$'
x_coord = 2.1
y_coord = 1.3
t_x_coord = 1.6#2.0
t_y_coord = 2.9
a_x_coord = 2.3
a_y_coord = 2.8
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='bottom', ws=1.2, hs=1.35, lsf=-0.1,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 10
text = r'$\mathbf{P^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -1.9
t_y_coord = -0.55
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='top', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 10
text = r'$\mathbf{P^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 2.0
t_y_coord = -0.55
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='top', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 10
text = r'$\mathbf{FW^-}$'
x_coord = -0.2
y_coord = 4.1
t_x_coord = -0.2
t_y_coord = -0.55
a_x_coord = -0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='right', va='top', ws=1.3, hs=1.35, lsf=-0.2,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ax_ind = 10
text = r'$\mathbf{FW^+}$'
x_coord = 0.2
y_coord = 4.1
t_x_coord = 0.3
t_y_coord = -0.55
a_x_coord = 0.2
a_y_coord = 4.1
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord, ms=0,
                   ha='left', va='top', ws=1.3, hs=1.35, lsf=-0.15,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=6, a_width=2.5)

ylim = axs[8].get_ylim()
axs[8].set_ylim([ylim[0] - 1, ylim[1]])

ylim = axs[10].get_ylim()
axs[10].set_ylim([ylim[0] - 1, ylim[1]])


fig=plt.gcf()
axs = fig.axes
for ax in axs:
    txts = ax.findobj(plt.Text)
#    xtxts = ax.xaxis.findobj(plt.Text)
#    ytxts = ax.yaxis.findobj(plt.Text)
    txts.extend(ax.xaxis.findobj(plt.Text))
    txts.extend(ax.yaxis.findobj(plt.Text))
    for txt in txts:
        txt.set_zorder(11)
    ax.set_rasterization_zorder(1.4)

save_path = PUBdir + '\\' + 'mean_vort_x_planes' + '.eps'
#plt.savefig(save_path, bbox_inches='tight')
plt.savefig(save_path, rasterized=True, dpi=600, bbox_inches='tight')


#%% Plot Reynolds Shear Stress Fields - z=const planes

z_locs = np.array([0, 2])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
#        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
#            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        xy_plane_titles[geo][z_loc_ind] = '$z = %1.2f \quad (z_h = %1.2f)$' \
            % (z_g[geo][0, 0, ind[0][0]], z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]

field_g_list = [M_ufvf_g, M_ufwf_g, M_vfwf_g]
#field_labels = [r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
#                r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
#                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$']
field_labels = [r'$\overline{u\textquotesingle v\textquotesingle}$',
                r'$\overline{u\textquotesingle w\textquotesingle}$',
                r'$\overline{v\textquotesingle w\textquotesingle}$']
pltf.plot_normalized_planar_field_comparison(
        x_g, y_g, field_g_list, M_U_g, M_V_g, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''],
        're_shear_z_planes', cb_tick_format='%.3f', plot_arrow=False,
        strm_max_len=2, strm_dens=[0.25, 0.5],
        comparison_type='field and plane', figsize=(5.25, 9), close_fig=False)
plt.annotate('', xy=[0, 0.675], xytext=[1, 0.675], xycoords='figure fraction',
             arrowprops=dict(arrowstyle='-'))
plt.annotate('', xy=[0, 0.36], xytext=[1, 0.36], xycoords='figure fraction',
             arrowprops=dict(arrowstyle='-'))
save_path = PUBdir + '\\' + 're_shear_z_planes' + '.eps'
plt.savefig(save_path, bbox_inches='tight')
#plt.close('all')


#%% Plot Turbulent Kinetic Energy Fields - y/d=0.75

y_locs = np.array([0.8])
num_xz_planes = np.size(y_locs, axis=0)

xz_plane_inds = [np.zeros(num_xz_planes, dtype=int) for _ in geo_inds]
xz_plane_titles = [['' for _ in np.arange(num_xz_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (y_loc, y_loc_ind) in zip(y_locs, np.arange(num_xz_planes)):
        ind = np.where(np.abs(y_g[geo][:, 0, 0] - y_loc) ==
                       np.min(np.abs(y_g[geo][:, 0, 0] - y_loc)))
        xz_plane_inds[geo][y_loc_ind] = ind[0][0]
        xz_plane_titles[geo][y_loc_ind] = 'y = %1.2f' \
            % np.abs(y_g[geo][ind[0][0], 0, 0])
xz_save_names_temp = ['y_%1.1fd' % y_locs[ind]
                      for ind in np.arange(num_xz_planes)]
xz_save_names = [xz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_xz_planes)]

field_g_list = [M_tke_g]
cont_lvl_field_g=[M_lambda2_2_g]
#field_labels = [r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$']
field_labels = [r'$k$']
pltf.plot_normalized_planar_field_comparison(
        x_g, z_g, field_g_list, M_U_g, M_W_g, field_labels, 
        xz_plane_titles, 'xz', xz_plane_inds, num_geos, obstacles, PUBdir,
        [''], 'tke_y_plane', strm_x_g=x_g, strm_y_g=z_g, plot_streamlines=False,
        obstacle_AR=obstacle_AR, cb_tick_format='%.2f', plot_contour_lvls=True,
        cont_lvl_field_g=cont_lvl_field_g, cont_lvls=[0], plot_arrow=False,
        comparison_type='plane', cb_label_0=True, halve_cb=True, cont_colour='k',
        figsize=(5.5, 2), close_fig=False, save_fig=True, extension='.eps')
#plt.close('all')


#%%


#%% Reynolds stress plots from before

#%% Plot Reynolds Normal Stress Fields - y/d=0

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

field_g_list = [M_ufuf_rect_g, M_vfvf_rect_g, M_wfwf_rect_g]
field_labels = [r'$\displaystyle \frac{\overline{u\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{U_\infty^2}$']
bifurc_seed_pt_xy = [[[2, 0.18], [3, 0.3], [5, 0.2]], [None]] # 3mm plane has almost zero W so all of the streamlines stop there, adding a few extra
pltf.plot_normalized_planar_field_comparison(
        x_rect_g, z_rect_g, field_g_list, M_U_g, M_W_g, field_labels, 
        xz_plane_titles, 'xz', xz_plane_inds, num_geos, obstacles, PUBdir,
        [''], 're_norm_y_planes', strm_x_g=x_g, strm_y_g=z_g,
        obstacle_AR=obstacle_AR, cb_tick_format='%.2f',
        comparison_type='field', cb_label_0=False, halve_cb=False, strm_dens=0.5,
        plot_streamlines=True, plot_bifurcation_lines=[False]*num_geos, plot_arrow=False,
        bifurc_seed_pt_xy=bifurc_seed_pt_xy, cmap_name='viridis',
        figsize=(5.5, 4.5), close_fig=False)
#plt.close('all')


#%% Plot Reynold's Normal Stress Fields - z=const planes

z_locs = np.array([2])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]

field_g_list = [M_ufuf_g, M_vfvf_g, M_wfwf_g]
field_labels = [r'$\displaystyle \frac{\overline{u\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{U_\infty^2}$']
pltf.plot_normalized_planar_field_comparison(
        x_g, y_g, field_g_list, M_U_g, M_V_g, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''],
        're_norm_z_planes', cb_tick_format='%.2f', halve_cb=True, plot_arrow=False,
        comparison_type='fields', figsize=(5.25, 5), close_fig=False)
#plt.close('all')


#%% Plot Reynolds Shear Stress Fields - y/d=0

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

field_g_list = [M_ufwf_rect_g]
field_labels = [r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$']
pltf.plot_normalized_planar_field_comparison(
        x_rect_g, z_rect_g, field_g_list, M_U_g, M_W_g, field_labels, 
        xz_plane_titles, 'xz', xz_plane_inds, num_geos, obstacles, PUBdir,
        [''], 're_shear_y_planes', strm_x_g=x_g, strm_y_g=z_g,
        obstacle_AR=obstacle_AR, cb_tick_format='%.3f',
        comparison_type='plane', cb_label_0=True, halve_cb=True, strm_dens=0.5,
        plot_streamlines=True, plot_bifurcation_lines=[False]*num_geos, plot_arrow=False,
        bifurc_seed_pt_xy=[[[2.0, -0.1], [2.4, -0.1]], [[1.35, 0.55], [1.4, 0.6]]],
        figsize=(5.5, 1.75), close_fig=False)
#plt.close('all')


#%% Plot Reynolds Shear Stress Fields - z=const planes

z_locs = np.array([0, 2])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
#        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
#            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        xy_plane_titles[geo][z_loc_ind] = '$z = %1.2f \quad (z_h = %1.2f)$' \
            % (z_g[geo][0, 0, ind[0][0]], z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]

field_g_list = [M_ufvf_g, M_ufwf_g, M_vfwf_g]
#field_labels = [r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
#                r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
#                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$']
field_labels = [r'$\overline{u\textquotesingle v\textquotesingle}$',
                r'$\overline{u\textquotesingle w\textquotesingle}$',
                r'$\overline{v\textquotesingle w\textquotesingle}$']
pltf.plot_normalized_planar_field_comparison(
        x_g, y_g, field_g_list, M_U_g, M_V_g, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''],
        're_shear_z_planes', cb_tick_format='%.3f', plot_arrow=False,
        strm_max_len=2, strm_dens=[0.25, 0.5],
        comparison_type='field and plane', figsize=(5.25, 9), close_fig=False)
plt.annotate('', xy=[0, 0.675], xytext=[1, 0.675], xycoords='figure fraction',
             arrowprops=dict(arrowstyle='-'))
plt.annotate('', xy=[0, 0.36], xytext=[1, 0.36], xycoords='figure fraction',
             arrowprops=dict(arrowstyle='-'))
save_path = PUBdir + '\\' + 're_shear_z_planes' + '.eps'
plt.savefig(save_path, bbox_inches='tight')
#plt.close('all')


#%% SINGLES Plot Reynolds Shear Stress Fields - z=const planes

z_locs = np.array([0, 2])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]

field_g_list = [M_ufvf_g]
field_labels = [r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$']
pltf.plot_normalized_planar_field_comparison(
        x_g, y_g, field_g_list, M_U_g, M_V_g, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''],
        're_shear_uv_z_plane', cb_tick_format='%.3f',
        comparison_type='field and plane', figsize=(5.5, 3), close_fig=False)
#plt.close('all')


#%%


#%% 2D plots

#%% Calculate Omega_z

M_dU_dx_g = [np.zeros((J[geo][0], I[geo][0], num_planes)) for geo in geo_inds]
M_dU_dy_g = [np.zeros((J[geo][0], I[geo][0], num_planes)) for geo in geo_inds]
M_dV_dx_g = [np.zeros((J[geo][0], I[geo][0], num_planes)) for geo in geo_inds]
M_dV_dy_g = [np.zeros((J[geo][0], I[geo][0], num_planes)) for geo in geo_inds]
M_dW_dz_g = [np.zeros((J[geo][0], I[geo][0], num_planes)) for geo in geo_inds]
for geo in geo_inds:
    [M_dU_dy_temp, M_dU_dx_temp] \
        = np.gradient(M_U_g[geo][:, :, 0], y_g[geo][:, 0, 0], x_g[geo][0, :, 0])
    [M_dV_dy_temp, M_dV_dx_temp] \
        = np.gradient(M_V_g[geo][:, :, 0], y_g[geo][:, 0, 0], x_g[geo][0, :, 0])
    M_dU_dx_g[geo][:, :, 0] = M_dU_dx_temp
    M_dU_dy_g[geo][:, :, 0] = M_dU_dy_temp
    M_dV_dx_g[geo][:, :, 0] = M_dV_dx_temp
    M_dV_dy_g[geo][:, :, 0] = M_dV_dy_temp
    M_dW_dz_g[geo][:, :, 0] = -M_dV_dy_temp - M_dU_dx_temp

M_Omega_z_g = [np.zeros((J[geo][0], I[geo][0], num_planes)) for geo in geo_inds]
for geo in geo_inds:
    M_Omega_z_g[geo][:, :, :] = M_dV_dx_g[geo] - M_dU_dy_g[geo]


#%% 2D planes WHich to choose

obstacles = ['hollow circle', 'hollow square']

z_locs = np.array([4])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = '$z = %1.2f \quad (z_h = %1.2f)$' \
            % (z_g[geo][0, 0, ind[0][0]], z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]


y_max = 1
x_min = np.max([np.min(x_g[geo]) for geo in geo_inds])
x_max = 1.5
y_max_ind = [np.where(np.abs(y_g[geo][:, 0, 0] - y_max) == \
                     np.min(np.abs(y_g[geo][:, 0, 0] - y_max)))[0][0]
             for geo in geo_inds]
y_min_ind = [np.where(np.abs(y_g[geo][:, 0, 0] + y_max) == \
                     np.min(np.abs(y_g[geo][:, 0, 0] + y_max)))[0][0]
             for geo in geo_inds]
x_max_ind = [np.where(np.abs(x_g[geo][0, :, 0] - x_max) == \
                     np.min(np.abs(x_g[geo][0, :, 0] - x_max)))[0][0]
             for geo in geo_inds]
x_min_ind = [np.where(np.abs(x_g[geo][0, :, 0] - x_min) == \
                     np.min(np.abs(x_g[geo][0, :, 0] - x_min)))[0][0]
             for geo in geo_inds]


field_g_list = [
                [M_U_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
                 for geo in geo_inds],
                [M_V_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
                 for geo in geo_inds],
                [M_Omega_z_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
                 for geo in geo_inds],
                [M_dW_dz_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
                 for geo in geo_inds]
                ]
x_g_temp = [x_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
            for geo in geo_inds]
y_g_temp = [y_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
            for geo in geo_inds]
M_U_g_temp = [M_U_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
              for geo in geo_inds]
M_V_g_temp = [M_V_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
              for geo in geo_inds]
field_labels = [r'$U$', r'$V$', r'$\Omega_z$',
                r'$\displaystyle \frac{\mathrm{d}\! W}{\mathrm{d}\! z}$']
pltf.plot_normalized_planar_field_comparison(
        x_g_temp, y_g_temp, field_g_list, M_U_g_temp, M_V_g_temp, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''], 'mean_vel_top_plane',
        cb_tick_format='%.2f', comparison_type='field', plot_arrow=False, tick_space=0.5,
        strm_max_len=2, strm_dens=[0.75, 0.65], figsize=(5.5, 9), close_fig=False, save_fig=True)
#plt.close('all')


#%% 2D planes WHich to choose part 2

obstacles = ['hollow circle', 'hollow square']

z_locs = np.array([4])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = '$z = %1.2f \quad (z_h = %1.2f)$' \
            % (z_g[geo][0, 0, ind[0][0]], z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]


y_max = 1
x_min = np.max([np.min(x_g[geo]) for geo in geo_inds])
x_max = 1.5
y_max_ind = [np.where(np.abs(y_g[geo][:, 0, 0] - y_max) == \
                     np.min(np.abs(y_g[geo][:, 0, 0] - y_max)))[0][0]
             for geo in geo_inds]
y_min_ind = [np.where(np.abs(y_g[geo][:, 0, 0] + y_max) == \
                     np.min(np.abs(y_g[geo][:, 0, 0] + y_max)))[0][0]
             for geo in geo_inds]
x_max_ind = [np.where(np.abs(x_g[geo][0, :, 0] - x_max) == \
                     np.min(np.abs(x_g[geo][0, :, 0] - x_max)))[0][0]
             for geo in geo_inds]
x_min_ind = [np.where(np.abs(x_g[geo][0, :, 0] - x_min) == \
                     np.min(np.abs(x_g[geo][0, :, 0] - x_min)))[0][0]
             for geo in geo_inds]


field_g_list = [
                [M_ufuf_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
                 for geo in geo_inds],
                [M_vfvf_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
                 for geo in geo_inds],
                [M_ufvf_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
                 for geo in geo_inds]
                ]
x_g_temp = [x_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
            for geo in geo_inds]
y_g_temp = [y_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
            for geo in geo_inds]
M_U_g_temp = [M_U_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
              for geo in geo_inds]
M_V_g_temp = [M_V_g[geo][y_min_ind[geo]:y_max_ind[geo], x_min_ind[geo]:x_max_ind[geo], :]
              for geo in geo_inds]
field_labels = [r'$\overline{u\textquotesingle^2}$',
                r'$\overline{v\textquotesingle^2}$',
                r'$\overline{u\textquotesingle v\textquotesingle}$']
pltf.plot_normalized_planar_field_comparison(
        x_g_temp, y_g_temp, field_g_list, M_U_g_temp, M_V_g_temp, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''], 're_stress_top_plane',
        cb_tick_format='%.2f', comparison_type='field', plot_arrow=False, tick_space=0.5,
        strm_max_len=2, strm_dens=[0.75, 0.65], figsize=(5.5, 9), close_fig=False, save_fig=True)
#plt.close('all')


#%% 2D planes 

obstacles = ['hollow circle', 'hollow square']

z_locs = np.array([4])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[geo][0, 0, :] - z_loc)))
#        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = '$z = %1.2f \quad (z_h = %1.2f)$' \
            % (z_g[geo][0, 0, ind[0][0]], z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]


y_max = 1
x_min = np.min([np.min(x_g[geo]) for geo in geo_inds])
x_max = 1.5
y_max_ind = [np.where(np.abs(y_g[geo][:, 0, 0] - y_max) == \
                     np.min(np.abs(y_g[geo][:, 0, 0] - y_max)))[0][0]
             for geo in geo_inds]
y_min_ind = [np.where(np.abs(y_g[geo][:, 0, 0] + y_max) == \
                     np.min(np.abs(y_g[geo][:, 0, 0] + y_max)))[0][0]
             for geo in geo_inds]
x_max_ind = [np.where(np.abs(x_g[geo][0, :, 0] - x_max) == \
                     np.min(np.abs(x_g[geo][0, :, 0] - x_max)))[0][0]
             for geo in geo_inds]
x_min_ind = [np.where(np.abs(x_g[geo][0, :, 0] - x_min) == \
                     np.min(np.abs(x_g[geo][0, :, 0] - x_min)))[0][0]
             for geo in geo_inds]


field_g_list = [
                [M_U_g[geo][y_min_ind[0]:y_max_ind[0], x_min_ind[0]:x_max_ind[0], :]
                 for geo in geo_inds],
                [M_Omega_z_g[geo][y_min_ind[0]:y_max_ind[0], x_min_ind[0]:x_max_ind[0], :]
                 for geo in geo_inds],
                [M_ufvf_g[geo][y_min_ind[0]:y_max_ind[geo], x_min_ind[0]:x_max_ind[0], :]
                 for geo in geo_inds]
                ]
x_g_temp = [x_g[geo][y_min_ind[0]:y_max_ind[0], x_min_ind[0]:x_max_ind[0], :]
            for geo in geo_inds]
y_g_temp = [y_g[geo][y_min_ind[0]:y_max_ind[0], x_min_ind[0]:x_max_ind[0], :]
            for geo in geo_inds]
M_U_g_temp = [M_U_g[geo][y_min_ind[0]:y_max_ind[geo], x_min_ind[0]:x_max_ind[0], :]
              for geo in geo_inds]
M_V_g_temp = [M_V_g[geo][y_min_ind[0]:y_max_ind[geo], x_min_ind[0]:x_max_ind[0], :]
              for geo in geo_inds]
field_labels = [r'$U$',
                r'$\Omega_z$',
                r'$\overline{u\textquotesingle v\textquotesingle}$']
pltf.plot_normalized_planar_field_comparison(
        x_g_temp, y_g_temp, field_g_list, M_U_g_temp, M_V_g_temp, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''], 'mean_top_plane',
        cb_tick_format='%.2f', comparison_type='field', plot_arrow=False, tick_space=0.5,
        strm_max_len=2, strm_dens=[0.75, 0.65], strm_num_arrow_heads=4,
        cb_label_0=[[False, True, True] for _ in geo_inds],
        figsize=(5.5, 5.25), close_fig=False, save_fig=True, extension='.eps')
#plt.close('all')


#%%








#%% POD plots

#%% Plot POD Energy Bar plot

num_modes_plt = 10
extension = '.png'
fontsize = 10

mode_nums = np.arange(1,num_modes_plt+1)


z_locs = np.array([1, 2, 3])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
#        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
#            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])

        xy_plane_titles[geo][z_loc_ind] = '$z = %1.2f \quad (z_h = %1.2f)$' \
            % (z_raw_g[geo][0, 0, ind[0][0]], z_raw_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_raw_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]


save_name = 'POD_energies'

fig = plt.figure(figsize=(5.5, 6))
axs = []

bar_width = 0.3
align_sign_list = np.array([-1, 1])
hatch_list = [None, '///']
ec_list = ['k', 'k']
fc_list = ['k', 'w']
lambdaf_list = [lambdaf_as, lambdaf_sy]
lambdaf_labels = ['Anti-symmetric Modes', 'Symmetric Modes']
#energy_str = '$\sum_ \lambda_i$'
tke_str = '$tke$'
num_lambdafs = np.size(lambdaf_list, axis=0)
for geo, align_sign, hatch, ec, fc in \
        zip(geo_inds, align_sign_list, hatch_list, ec_list, fc_list):
    for plane, plane_ind, title in \
            zip(xy_plane_inds[geo], np.arange(num_xy_planes),
                xy_plane_titles[geo]):
        for lambdaf_temp, lambdaf_ind, lambdaf_label in \
                zip(lambdaf_list, np.arange(num_lambdafs), lambdaf_labels):
            subplot_ind = plane_ind*num_lambdafs + lambdaf_ind + 1
            subplot_label = plane_ind*num_lambdafs + lambdaf_ind + 1
            if geo == 0:
                ax = plt.subplot(num_xy_planes, num_lambdafs, subplot_ind)
                axs.append(ax)
            else:
                ax = axs[subplot_ind - 1]
                plt.sca(ax)
            mode_energy = lambdaf_temp[geo][plane][0:num_modes_plt]/np.sum(lambdaf[geo][plane])*100
#            mode_energies = [lambdaf_temp[geo][plane][0:num_modes_plt]/np.sum(lambdaf[geo][plane])*100 for geo in geo_inds]
            mode_nums_list = np.repeat(np.expand_dims(mode_nums, 1), num_lambdafs, axis=1).T
            plt.bar(mode_nums, mode_energy, align_sign*bar_width, align='edge',
                    hatch=hatch, ec=[ec]*num_modes, fc=fc) # the list of edge color may not be eeded in future releases
            plt.title(title, fontsize=fontsize)
            plt.ylabel('\% of ' + tke_str)
            plt.xticks(mode_nums)
            if plane_ind == num_xy_planes - 1:
                plt.xlabel(lambdaf_label)
    #            else:
    #                plt.tick_params(labelbottom=False)
            plt.text(-0, 1.13, #0.15, 0.95
                 '(' + string.ascii_lowercase[subplot_label - 1] + ')',
                 ha='left', va='top', bbox={'pad' : 0.1, 'fc' : 'w',
                                            'ec' : 'w', 'alpha' : 0},
                 transform=ax.transAxes)
            plt.legend([obstacles[geo1] + ' (' + tke_str + 
                        ' = %1.2f' % np.sum(lambdaf[geo1][plane]) + ')' 
                        for geo1 in geo_inds], handlelength=1, borderpad=0.2,
                        labelspacing=0.2, handletextpad=0.5)
plt.tight_layout()
save_path = PUBdir + '\\' + save_name + extension
plt.savefig(save_path, bbox_inches='tight')


#%% Plot First Harmonic POD Modes

z_locs = np.array([1, 2, 3])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


plt_modes = [[[0, 1], [0, 1], [0, 1]], [[0, 1], [0, 1], [0, 1]]]
correct_sign = [[[1, 1], [1, 1], [1, 1]], [[1, 1], [1, 1], [1, 1]]]
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
save_names = ['first_harm_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Anti-symmetric' + ' \n'

pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_as_g, Psi_vf_as_g, Psi_wf_as_g,
        akf_as_PSD, lambdaf_as, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, num_geos, PUBdir, save_names, correct_sign,
        POD_type, '', harm_freqs=harm_freqs, figsize=(6.7, 5), close_fig=False, save_fig=True, extension='.png')

#%% Circ Plot First Harmonic POD Modes

z_locs = np.array([1, 2, 3])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


plt_modes = [[[0, 1], [0, 1], [0, 1]], [[], [], []]]
correct_sign = [[[1, 1], [1, 1], [1, 1]], [[], [], []]]
harm_freqs = [[0.25, 1, 2, 3], []]
save_names = ['circle_first_harm_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Anti-symmetric' + ' \n'

pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_as_g, Psi_vf_as_g, Psi_wf_as_g,
        akf_as_PSD, lambdaf_as, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, 1, PUBdir, save_names, correct_sign,
        POD_type, '', harm_freqs=harm_freqs, #figsize=(6.7, 5),
        close_fig=False, save_fig=True, extension='.png')

#%% Square Plot First Harmonic POD Modes

z_locs = np.array([1, 2, 3])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


plt_modes = [[[0, 1], [0, 1], [0, 1]], [[], [], []]]
correct_sign = [[[1, 1], [1, 1], [1, 1]], [[], [], []]]
harm_freqs = [[0.1, 2, 3], []]

save_names = ['square_first_harm_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Anti-symmetric' + ' \n'

pltf.plot_POD_modes(
        [x_raw_g[1]], [y_raw_g[1]], [Psi_uf_as_g[1]], [Psi_vf_as_g[1]], [Psi_wf_as_g[1]],
        [akf_as_PSD[1]], [lambdaf_as[1]], [lambdaf[1]], [f[1]], [d[1]], [U_inf[1]], [St_shed[1]], plt_modes,
        [obstacles[1]], [xy_plane_inds[1]], 1, PUBdir, save_names, correct_sign,
        POD_type, '', harm_freqs=harm_freqs, #figsize=(6.7, 5),
        close_fig=False, save_fig=True, extension='.png')


#%% Comparison Plot First Harmonic POD Modes

z_locs = np.array([1, 2, 3])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


plt_modes = [[[0], [0], [0]], [[0], [0], [0]]]
correct_sign = [[[1], [1], [1]], [[1], [1], [1]]]
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
save_names = ['comp_first_harm_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Anti-symmetric' + ' \n'

pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_as_g, Psi_vf_as_g, Psi_wf_as_g,
        akf_as_PSD, lambdaf_as, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, num_geos, PUBdir, save_names, correct_sign,
        POD_type, '', harm_freqs=harm_freqs, figsize=(1.5*6.7, 1.5*5/4*2), close_fig=False, save_fig=True, extension='.png')


#%% Plot Slowly Varying POD Modes

z_locs = np.array([1])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


plt_modes = [[[0]], [[0]]]
correct_sign = [[[1]], [[1]]]
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
save_names = ['slowly_varying_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Symmetric' + ' \n'

pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_sy_g, Psi_vf_sy_g, Psi_wf_sy_g,
        akf_sy_PSD, lambdaf_sy, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, num_geos, PUBdir, save_names, correct_sign,
        POD_type, '', harm_freqs=harm_freqs, figsize=(6.7, 2.5),
        close_fig=False, extension='.png')

z_locs = np.array([2, 3])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


plt_modes = [[[0, 1], [0, 1]], [[0, 1], [0, 2]]]
correct_sign = [[[1, 1], [1, 1]], [[1, 1], [1, 1]]]
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
save_names = ['slowly_varying_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Symmetric' + ' \n'

pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_sy_g, Psi_vf_sy_g, Psi_wf_sy_g,
        akf_sy_PSD, lambdaf_sy, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, num_geos, PUBdir, save_names, correct_sign,
        POD_type, '', harm_freqs=harm_freqs, figsize=(6.7, 5), close_fig=False, extension='.png')

#%% Circ Plot Slowly Varying POD Modes

z_locs = np.array([1])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


plt_modes = [[[0, 0]], [[]]]
correct_sign = [[[1, 1]], [[]]]
harm_freqs = [[0.25, 1, 2, 3], []]
save_names = ['circ_slowly_varying_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Symmetric' + ' \n'

pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_sy_g, Psi_vf_sy_g, Psi_wf_sy_g,
        akf_sy_PSD, lambdaf_sy, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, 1, PUBdir, save_names, correct_sign,
        POD_type, '', harm_freqs=harm_freqs, #figsize=(6.7, 2.5),
        close_fig=False, extension='.png')

z_locs = np.array([2, 3])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


plt_modes = [[[0, 1], [0, 1]], [[], []]]
correct_sign = [[[1, 1], [1, 1]], [[], []]]
harm_freqs = [[0.25, 1, 2, 3], []]
save_names = ['circ_slowly_varying_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Symmetric' + ' \n'

pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_sy_g, Psi_vf_sy_g, Psi_wf_sy_g,
        akf_sy_PSD, lambdaf_sy, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, 1, PUBdir, save_names, correct_sign,
        POD_type, '', harm_freqs=harm_freqs,# figsize=(6.7, 5),
        close_fig=False, extension='.png')


#%% Square Plot Slowly Varying POD Modes

z_locs = np.array([1])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


plt_modes = [[[0, 0]], [[]]]
correct_sign = [[[1, 1]], [[]]]
harm_freqs = [[0.1, 1, 2, 3], []]
save_names = ['square_slowly_varying_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Symmetric' + ' \n'

pltf.plot_POD_modes(
        [x_raw_g[1]], [y_raw_g[1]], [Psi_uf_sy_g[1]], [Psi_vf_sy_g[1]], [Psi_wf_sy_g[1]],
        [akf_sy_PSD[1]], [lambdaf_sy[1]], [lambdaf[1]], [f[1]], [d[1]], [U_inf[1]], [St_shed[1]], plt_modes,
        [obstacles[1]], [xy_plane_inds[1]], 1, PUBdir, save_names, correct_sign,
        POD_type, '', harm_freqs=harm_freqs, #figsize=(6.7, 2.5),
        close_fig=False, extension='.png')

z_locs = np.array([2, 3])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_raw_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_raw_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd' % z_locs[ind] for ind in np.arange(num_xy_planes)]


plt_modes = [[[0, 1], [0, 2]], [[], []]]
correct_sign = [[[1, 1], [1, 1]], [[], []]]
harm_freqs = [[0.1, 1, 2, 3], []]
save_names = ['square_slowly_varying_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Symmetric' + ' \n'

pltf.plot_POD_modes(
        [x_raw_g[1]], [y_raw_g[1]], [Psi_uf_sy_g[1]], [Psi_vf_sy_g[1]], [Psi_wf_sy_g[1]],
        [akf_sy_PSD[1]], [lambdaf_sy[1]], [lambdaf[1]], [f[1]], [d[1]], [U_inf[1]], [St_shed[1]], plt_modes,
        [obstacles[1]], [xy_plane_inds[1]], 1, PUBdir, save_names, correct_sign,
        POD_type, '', harm_freqs=harm_freqs,# figsize=(6.7, 5),
        close_fig=False, extension='.png')

#%% Plot POD Modes OLD I think

plt_plane=4
plt_modes = [1, 1]
save_name = 'slow_drift_39'
POD_type = 'Symmetric' + ' \n'
correct_sign = [1, 1]
geo_inds_temp = [1, 0]

print zs_raw_uncorr[plt_plane]

os.chdir(PUBdir)
pltf.plot_POD_mode_comparison(x_raw_g, y_raw_g, 
                              Psi_uf_as_g, Psi_vf_as_g, Psi_wf_as_g,
                              akf_as_PSD, lambdaf_as, lambdaf, 
                              f, d, U_inf, St_shed, POD_type, correct_sign, 
                              plt_modes, plt_plane, geo_inds_temp,
                              obstacles, PUBdir, save_name, figsize=(9, 5), extension='.jpg')
pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_as_g, Psi_vf_as_g, Psi_wf_as_g,
        akf_as_PSD, lambdaf_as, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, num_geos, PUBdir, save_names, correct_sign, POD_type, '',
        additional_relative_freqs=additional_rel_freqs,
        figsize=(6.7, 5), close_fig=False)

#%%








#%% Find good Spectra

# rms fields
for geo in geo_inds:
    plt.figure()
    
    print 'u', np.where(M_ufuf_raw_g[geo][:, :, plane] == np.max(M_ufuf_raw_g[geo][:, :, plane]))
    plt.subplot(3, 1, 1, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane],
                                      M_ufuf_raw_g[geo][:, :, plane],
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)
    print 'v', np.where(M_vfvf_raw_g[geo][:, :, plane] == np.max(M_vfvf_raw_g[geo][:, :, plane]))
    plt.subplot(3, 1, 2, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane],
                                      M_vfvf_raw_g[geo][:, :, plane],
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)
    print 'w', np.where(M_wfwf_raw_g[geo][:, :, plane] == np.max(M_wfwf_raw_g[geo][:, :, plane]))
    plt.subplot(3, 1, 3, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane],
                                      M_wfwf_raw_g[geo][:, :, plane],
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)

#%% Find good Spectra

# f_shed
mode = 0
akf_PSD_temp = akf_as_PSD

spectra_temp = [[akf_PSD_temp[geo][p][:, mode] for p in plane_inds]
                for geo in geo_inds]
spectra_labels = [['%2.1f' % z for z in zs] for geo in geo_inds]
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
pltf.plot_spectra_comparison(f, spectra_temp, num_geos, PUBdir, 'test',
                             spectra_labels=spectra_labels, label_spectra=True, label_axes=False, x_label='f (Hz)',
                             harm_freqs=harm_freqs, f_shed = [160, 120],
                             tight=True, label_y_ticks=False,
                             save_fig=False, close_fig=False, figsize=(4, 9))

plane = 0
mode1 = 0
mode2 = 1
plt_plane_inds = [[plane], [plane]]
plt_modes = [[[mode1, mode2]], [[mode1, mode2]]]
correct_sign = [[[1, 1]], [[1, 1]]]
pltf.plot_POD_modes(x_raw_g, y_raw_g, Psi_uf_as_g, Psi_vf_as_g, Psi_wf_as_g,
                    akf_as_PSD, lambdaf_as, lambdaf, f, d, U_inf, St_shed,
                    plt_modes, obstacles, plt_plane_inds, num_geos, PUBdir,
                    'test', correct_sign, '', '', harm_freqs=harm_freqs,
                    close_fig=False, save_fig=False)

for geo in geo_inds:
    plt.figure()
    
    Psi_temp = np.abs(Psi_uf_as_g[geo][plane][:, :, mode1])/np.max(Psi_uf_as_g[geo][plane][:, :, mode1]) + \
               np.abs(Psi_uf_as_g[geo][plane][:, :, mode2])/np.max(Psi_uf_as_g[geo][plane][:, :, mode2])
    print 'u', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 1, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)
    Psi_temp = np.abs(Psi_vf_as_g[geo][plane][:, :, mode1])/np.max(Psi_vf_as_g[geo][plane][:, :, mode1]) + \
               np.abs(Psi_vf_as_g[geo][plane][:, :, mode2])/np.max(Psi_vf_as_g[geo][plane][:, :, mode2])
    print 'v', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 2, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)
    Psi_temp = np.abs(Psi_wf_as_g[geo][plane][:, :, mode1])/np.max(Psi_wf_as_g[geo][plane][:, :, mode1]) + \
               np.abs(Psi_wf_as_g[geo][plane][:, :, mode2])/np.max(Psi_wf_as_g[geo][plane][:, :, mode2])
    print 'w', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 3, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)

#%% Find good Spectra

# 1/4 f_shed
mode = 0
akf_PSD_temp = akf_sy_PSD

spectra_temp = [[akf_PSD_temp[geo][p][:, mode] for p in plane_inds]
                for geo in geo_inds]
spectra_labels = [['%2.1f' % z for z in zs] for geo in geo_inds]
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
pltf.plot_spectra_comparison(f, spectra_temp, num_geos, PUBdir, 'test',
                             spectra_labels=spectra_labels, label_spectra=True, label_axes=False, x_label='f (Hz)',
                             harm_freqs=harm_freqs, f_shed = [160, 120],
                             tight=True, label_y_ticks=False,
                             save_fig=False, close_fig=False, figsize=(4, 9))

plane = 18
mode1 = 0
mode2 = 1
plt_plane_inds = [[plane], [plane]]
plt_modes = [[[mode1, mode2]], [[mode1, mode2]]]
correct_sign = [[[1, 1]], [[1, 1]]]
pltf.plot_POD_modes(x_raw_g, y_raw_g, Psi_uf_sy_g, Psi_vf_sy_g, Psi_wf_sy_g,
                    akf_sy_PSD, lambdaf_sy, lambdaf, f, d, U_inf, St_shed,
                    plt_modes, obstacles, plt_plane_inds, num_geos, PUBdir,
                    'test', correct_sign, '', '', harm_freqs=harm_freqs,
                    close_fig=False, save_fig=False)

for geo in geo_inds:
    plt.figure()
    
    Psi_temp = np.abs(Psi_uf_sy_g[geo][plane][:, :, mode1])/np.max(Psi_uf_sy_g[geo][plane][:, :, mode1]) + \
               np.abs(Psi_uf_sy_g[geo][plane][:, :, mode2])/np.max(Psi_uf_sy_g[geo][plane][:, :, mode2])
    print 'u', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 1, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)
    Psi_temp = np.abs(Psi_vf_sy_g[geo][plane][:, :, mode1])/np.max(Psi_vf_sy_g[geo][plane][:, :, mode1]) + \
               np.abs(Psi_vf_sy_g[geo][plane][:, :, mode2])/np.max(Psi_vf_sy_g[geo][plane][:, :, mode2])
    print 'v', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 2, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)
    Psi_temp = np.abs(Psi_wf_sy_g[geo][plane][:, :, mode1])/np.max(Psi_wf_sy_g[geo][plane][:, :, mode1]) + \
               np.abs(Psi_wf_sy_g[geo][plane][:, :, mode2])/np.max(Psi_wf_sy_g[geo][plane][:, :, mode2])
    print 'w', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 3, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)

#%% Find good Spectra

# 2 f_shed
mode = 3
akf_PSD_temp = akf_sy_PSD

spectra_temp = [[akf_PSD_temp[geo][p][:, mode] for p in plane_inds]
                for geo in geo_inds]
spectra_labels = [['%2.1f' % z for z in zs] for geo in geo_inds]
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
pltf.plot_spectra_comparison(f, spectra_temp, num_geos, PUBdir, 'test',
                             spectra_labels=spectra_labels, label_spectra=True, label_axes=False, x_label='f (Hz)',
                             harm_freqs=harm_freqs, f_shed = [160, 120],
                             tight=True, label_y_ticks=False,
                             save_fig=False, close_fig=False, figsize=(4, 9))

plane = 6
mode1 = 2
mode2 = 3
plt_plane_inds = [[plane], [plane]]
plt_modes = [[[mode1, mode2]], [[mode1, mode2]]]
correct_sign = [[[1, 1]], [[1, 1]]]
pltf.plot_POD_modes(x_raw_g, y_raw_g, Psi_uf_sy_g, Psi_vf_sy_g, Psi_wf_sy_g,
                    akf_sy_PSD, lambdaf_sy, lambdaf, f, d, U_inf, St_shed,
                    plt_modes, obstacles, plt_plane_inds, num_geos, PUBdir,
                    'test', correct_sign, '', '', harm_freqs=harm_freqs,
                    close_fig=False, save_fig=False)

for geo in geo_inds:
    plt.figure()
    
    Psi_temp = np.abs(Psi_uf_sy_g[geo][plane][:, :, mode1])/np.max(Psi_uf_sy_g[geo][plane][:, :, mode1]) + \
               np.abs(Psi_uf_sy_g[geo][plane][:, :, mode2])/np.max(Psi_uf_sy_g[geo][plane][:, :, mode2])
    print 'u', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 1, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)
    Psi_temp = np.abs(Psi_vf_sy_g[geo][plane][:, :, mode1])/np.max(Psi_vf_sy_g[geo][plane][:, :, mode1]) + \
               np.abs(Psi_vf_sy_g[geo][plane][:, :, mode2])/np.max(Psi_vf_sy_g[geo][plane][:, :, mode2])
    print 'v', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 2, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)
    Psi_temp = np.abs(Psi_wf_sy_g[geo][plane][:, :, mode1])/np.max(Psi_wf_sy_g[geo][plane][:, :, mode1]) + \
               np.abs(Psi_wf_sy_g[geo][plane][:, :, mode2])/np.max(Psi_wf_sy_g[geo][plane][:, :, mode2])
    print 'w', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 3, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)

#%% Find good Spectra

# 3 f_shed
mode = 4
akf_PSD_temp = akf_as_PSD

spectra_temp = [[akf_PSD_temp[geo][p][:, mode] for p in plane_inds]
                for geo in geo_inds]
spectra_labels = [['%2.1f' % z for z in zs] for geo in geo_inds]
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
pltf.plot_spectra_comparison(f, spectra_temp, num_geos, PUBdir, 'test',
                             spectra_labels=spectra_labels, label_spectra=True, label_axes=False, x_label='f (Hz)',
                             harm_freqs=harm_freqs, f_shed = [160, 120],
                             tight=True, label_y_ticks=False,
                             save_fig=False, close_fig=False, figsize=(4, 9))

plane = 6
mode1 = 3
mode2 = 4
plt_plane_inds = [[plane], [plane]]
plt_modes = [[[mode1, mode2]], [[mode1, mode2]]]
correct_sign = [[[1, 1]], [[1, 1]]]
pltf.plot_POD_modes(x_raw_g, y_raw_g, Psi_uf_as_g, Psi_vf_as_g, Psi_wf_as_g,
                    akf_as_PSD, lambdaf_as, lambdaf, f, d, U_inf, St_shed,
                    plt_modes, obstacles, plt_plane_inds, num_geos, PUBdir,
                    'test', correct_sign, '', '', harm_freqs=harm_freqs,
                    close_fig=False, save_fig=False)

for geo in geo_inds:
    plt.figure()
    
#    Psi_temp = np.abs(Psi_uf_as_g[geo][plane][:, :, mode1])/np.max(Psi_uf_as_g[geo][plane][:, :, mode1]) + \
    Psi_temp =           np.abs(Psi_uf_as_g[geo][plane][:, :, mode2])/np.max(Psi_uf_as_g[geo][plane][:, :, mode2])
    print 'u', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 1, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)
#    Psi_temp = np.abs(Psi_vf_as_g[geo][plane][:, :, mode1])/np.max(Psi_vf_as_g[geo][plane][:, :, mode1]) + \
    Psi_temp =           np.abs(Psi_vf_as_g[geo][plane][:, :, mode2])/np.max(Psi_vf_as_g[geo][plane][:, :, mode2])
    print 'v', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 2, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)
#    Psi_temp = np.abs(Psi_wf_as_g[geo][plane][:, :, mode1])/np.max(Psi_wf_as_g[geo][plane][:, :, mode1]) + \
    Psi_temp =           np.abs(Psi_wf_as_g[geo][plane][:, :, mode2])/np.max(Psi_wf_as_g[geo][plane][:, :, mode2])
    print 'w', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 3, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)


#%%
    
plane = 12
mode1 = 0
mode2 = 1
mode3 = 0
mode4 = 1

for geo in geo_inds:
    plt.figure()
    
    Psi_temp = np.abs(Psi_uf_as_g[geo][plane][:, :, mode1])/np.max(Psi_uf_as_g[geo][plane][:, :, mode1]) + \
               np.abs(Psi_uf_as_g[geo][plane][:, :, mode2])/np.max(Psi_uf_as_g[geo][plane][:, :, mode2]) + \
               np.abs(Psi_uf_sy_g[geo][plane][:, :, mode3])/np.max(Psi_uf_sy_g[geo][plane][:, :, mode3]) + \
               np.abs(Psi_uf_sy_g[geo][plane][:, :, mode4])/np.max(Psi_uf_sy_g[geo][plane][:, :, mode4])
    print 'u', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 1, aspect='equal')
    pltf.plot_normalized_planar_field(x_raw_g[geo][:, :, plane],
                                      y_raw_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)

#%% Caluclate spectra

uf_PSD_g = [[np.zeros((J[geo][0], I[geo][0], int(np.floor(B1[geo]/2))))
             for _ in plane_inds] for geo in geo_inds]
vf_PSD_g = [[np.zeros((J[geo][0], I[geo][0], int(np.floor(B1[geo]/2))))
             for _ in plane_inds] for geo in geo_inds]
wf_PSD_g = [[np.zeros((J[geo][0], I[geo][0], int(np.floor(B1[geo]/2))))
             for _ in plane_inds] for geo in geo_inds]
f = [np.zeros((int(np.floor(B1[geo]/2)))) for geo in geo_inds]
for geo in geo_inds:
    for k in np.arange(num_planes): #np.arange(K[geo]):
        ufs = [uf_g[geo][k], vf_g[geo][k], wf_g[geo][k]]
        num_ufs = np.size(ufs, axis=0)
        uf_inds = np.arange(num_ufs)
        uf_PSDs = [np.zeros((int(np.floor(B1[geo]/2)), 1))
                   for _ in uf_inds]
        for j in np.arange(J[geo]):
            for i in np.arange(I[geo]):
                point = j*I[geo] + i
                if point % 1000 == 0:
                    print 'geo: %1.0f/%1.0f,  plane: %2.0f/%2.0f,  point: %4.0f/%4.0f' % \
                          (geo, num_geos - 1, k, num_planes - 1, point, I[geo]*J[geo] - 1)
#                          (geo, num_geos - 1, k, K[geo] - 1, point, I[geo]*J[geo] - 1)
                for (uf_temp, uf_ind) in zip(ufs, uf_inds):
                    uf_PSD_temp = np.zeros((int(np.floor(B1[geo]/2)),
                                             num_trials))
                    for trial in trials:
                        start_ind = 0 + trial*B1[geo]
                        end_ind = B1[geo] + trial*B1[geo]
                        [f_temp, PSD_temp] = \
                            spect.spectrumFFT_v2(
                                                 uf_temp[j, i, start_ind:end_ind], 
                                                 f_capture[geo])
                        uf_PSD_temp[:, trial] = PSD_temp
                    uf_PSDs[uf_ind] = np.mean(uf_PSD_temp, 1)
                uf_PSD_g[geo][k][j, i, :] = uf_PSDs[0]
                vf_PSD_g[geo][k][j, i, :] = uf_PSDs[1]
                wf_PSD_g[geo][k][j, i, :] = uf_PSDs[2]
    f[geo] = f_temp
#del akfs, akf_PSDs, f_temp, PSD_temp, akf_PSD_temp


#%% Plot Spectra

#plot_x_coords = [[1.0, 2.7, 2.1, 3.9, 4.1, 5.5] for _ in geo_inds]
#plot_y_coords = [[0.05, 0.8, 0.05, 0.6, 0.05, 0.8] for _ in geo_inds]
#plot_z_coords = [[3.4, 2.0, 0.1, 0.6, 0.8, 0.8] for _ in geo_inds]

plot_x_coords = [[1.0, 2.7, 2.1, 5.5] for _ in geo_inds]
plot_y_coords = [[0.05, 0.8, 0.05, 0.8] for _ in geo_inds]
plot_z_coords = [[3.4, 2.0, 0.1, 0.8] for _ in geo_inds]

uf_PSDs_temp = [wf_PSD_g, uf_PSD_g, vf_PSD_g, vf_PSD_g, vf_PSD_g, vf_PSD_g]
comp_labels = [r'$w\textquotesingle$', r'$u\textquotesingle$',
               r'$v\textquotesingle$', r'$v\textquotesingle$',
               r'$v\textquotesingle$', r'$v\textquotesingle$']
#u
#plot_x_coords = [[1.715, 3.535, 3.625, 5.514], [1.740, 1.619, 2.865, 5.514]]
#plot_y_coords = [[0.450, 0.720, 0.900, 1.753], [0, 0.707, 1.222, 1.753]]


#u rms
#plot_x_coords = [[3.085, 2.709, 2.612, 1.714], [1.884, 1.739, 1.730, 1.307]]
#plot_y_coords = [[0.630, 0.723, 0.719, 0.451], [0.618, 0.784, 0.786, 0.701]]
#plot_z_coords = [[0.113, 0.566, 0.792, 3.397] for geo in geo_inds]

#v
#plot_x_coords = [[1.444, 3.355, 3.877, 4.111], [1.216, 2.149, 3.389, 4.111]]
#plot_y_coords = [[0.452, 0, 0.629, 0], [0.436, 0, 0.873, 0]]

#v rms
#plot_x_coords = [[3.445, 3.522, 3.517, 2.255], [2.237, 2.436, 2.516, 1.657]]
#plot_y_coords = [[0, 0.452, 0.540, 0.271], [0, 0, 0, 0.438]]

#w
#plot_x_coords = [[0.992, 3.625, 6.125, 5.514], [1.129, 4.092, 3.127, 5.514]]
#plot_y_coords = [[0.181, 0.720, 0, 0.789], [0, 0.353, 0, 0.789]]

#w rms
#plot_x_coords = [[2.185, 3.703, 3.967, 2.075], [2.149, 2.436, 2.428, 1.22]]
#plot_y_coords = [[0.630, 0.633, 0.6295, 0.270], [0, 0, 0.611, 0]]



num_spectras = np.size(plot_x_coords, axis=-1)
spectra_inds = np.arange(num_spectras)
end_freq = [f[geo][-1] for geo in geo_inds]
end_freq_ind = [int(np.where(f[geo] == end_freq[geo])[0])
                for geo in geo_inds]
plot_z_inds = [[np.where(np.abs(z_raw_g[geo][0, 0, :] - plot_z_coords[geo][ind]) ==
                         np.min(np.abs(z_raw_g[geo][0, 0, :] - plot_z_coords[geo][ind])))[0][0]
                for ind in np.arange(num_spectras)] for geo in geo_inds]
plot_x_inds = [[np.where(np.abs(x_raw_g[geo][0, :, plot_z_inds[geo][ind]] - plot_x_coords[geo][ind]) ==
                         np.min(np.abs(x_raw_g[geo][0, :, plot_z_inds[geo][ind]] - plot_x_coords[geo][ind])))[0][0]
                for ind in np.arange(num_spectras)] for geo in geo_inds]
plot_y_inds = [[np.where(np.abs(y_raw_g[geo][:, 0, plot_z_inds[geo][ind]] - plot_y_coords[geo][ind]) ==
                         np.min(np.abs(y_raw_g[geo][:, 0, plot_z_inds[geo][ind]] - plot_y_coords[geo][ind])))[0][0]
                for ind in np.arange(num_spectras)] for geo in geo_inds]
plot_xs = [[x_raw_g[geo][plot_y_ind, plot_x_ind, plot_z_ind]
            for plot_x_ind, plot_y_ind, plot_z_ind in
            zip(plot_x_inds[geo], plot_y_inds[geo], plot_z_inds[geo])]
           for geo in geo_inds]
plot_ys = [[y_raw_g[geo][plot_y_ind, plot_x_ind, plot_z_ind]
            for plot_x_ind, plot_y_ind, plot_z_ind in
            zip(plot_x_inds[geo], plot_y_inds[geo], plot_z_inds[geo])]
           for geo in geo_inds]
plot_zs = [[z_raw_g[geo][plot_y_ind, plot_x_ind, plot_z_ind]
            for plot_x_ind, plot_y_ind, plot_z_ind in
            zip(plot_x_inds[geo], plot_y_inds[geo], plot_z_inds[geo])]
           for geo in geo_inds]

spectras_temp = [[uf_PSDs_temp[ind][geo][plot_z_ind][plot_y_ind, plot_x_ind, :end_freq_ind[geo]]
                  for ind, plot_x_ind, plot_y_ind, plot_z_ind
                  in zip(spectra_inds, plot_x_inds[geo], plot_y_inds[geo], plot_z_inds[geo])]
                 for geo in geo_inds]
fd_U_inf_temp = [[f[geo][:end_freq_ind[geo]]*d[geo]/U_inf[geo][plane]
                    for plane in plot_z_inds[geo]] for geo in geo_inds]
spectra_labels = [[comp_labels[ind] + r''' at
                   $\left(%1.1f, %1.1f, %1.1f\right)$'''
                   % (plot_x, plot_y, plot_z)
                   for ind, plot_x, plot_y, plot_z 
                   in zip(spectra_inds, plot_xs[geo], plot_ys[geo], plot_zs[geo])]
                  for geo in geo_inds]
x_label = r'$\displaystyle \frac{fd}{U_\infty}$'
x_label = r'$f$'
y_label = 'PSD'
save_name = 'vel_spect'
harm_freqs = [[0.25, 1, 2, 3, 4], [0.1, 1, 2, 3, 4]]
label_y_fracs = [[0.8, 0.8, 0.8, 0.8], [0.6, 0.65, 0.75, 0.9]]
St_shed_temp = [np.mean(St_shed[geo]) for geo in geo_inds]
pltf.plot_spectra_comparison(fd_U_inf_temp, spectras_temp, num_geos, PUBdir,
                             save_name, label_spectra=True, spectra_labels=spectra_labels,
                             x_label=x_label, y_label=y_label,
                             harm_freqs=harm_freqs, f_shed=St_shed_temp,
                             tight=True, label_y_ticks=False, close_fig=False,
                             axes_label_y=0.985, axes_label_x=0.04,
                             label_x_frac=1/32, label_y_fracs=label_y_fracs,
                             colour_spectra_label=False,
                             x_ticks=[10**-3, 10**-2, 10**-1, 10**0],
                             y_ticks_single=[10**-6, 10**-4, 10**-2],
                             y_lims=[10**-7, 10**9.5],
                             figsize=(3.3, 4), save_fig=False, extension='.eps')

#%%



#%% 2D spectra

#%% Find good Spectra

# rms fields
for geo in geo_inds:
    plt.figure()
    
    print 'u', np.where(M_ufuf_g[geo][:, :, plane] == np.max(M_ufuf_g[geo][:, :, plane]))
    plt.subplot(3, 1, 1, aspect='equal')
    pltf.plot_normalized_planar_field(x_g[geo][:, :, plane],
                                      y_g[geo][:, :, plane],
                                      M_ufuf_g[geo][:, :, plane],
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)
    print 'v', np.where(M_vfvf_g[geo][:, :, plane] == np.max(M_vfvf_g[geo][:, :, plane]))
    plt.subplot(3, 1, 2, aspect='equal')
    pltf.plot_normalized_planar_field(x_g[geo][:, :, plane],
                                      y_g[geo][:, :, plane],
                                      M_vfvf_g[geo][:, :, plane],
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)

#%% Find good Spectra

# f_shed
mode = 0
akf_PSD_temp = akf_as_PSD

spectra_temp = [[akf_PSD_temp[geo][p][:, mode] for p in plane_inds]
                for geo in geo_inds]
spectra_labels = [['%2.1f' % z for z in zs] for geo in geo_inds]
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
pltf.plot_spectra_comparison(f, spectra_temp, num_geos, PUBdir, 'test',
                             spectra_labels=spectra_labels, label_spectra=True, label_axes=False, x_label='f (Hz)',
                             harm_freqs=harm_freqs, f_shed = [160, 120],
                             tight=True, label_y_ticks=False,
                             save_fig=False, close_fig=False, figsize=(4, 3))

plane = 0
mode1 = [0, 0]
mode2 = [3, 1]
plt_plane_inds = [[plane], [plane]]
plt_modes = [[[mode1[geo], mode2[geo]]] for geo in geo_inds]
correct_sign = [[[1, 1]], [[1, 1]]]
pltf.plot_POD_modes(x_g, y_g, Psi_uf_as_g, Psi_vf_as_g, Psi_vf_as_g,
                    akf_as_PSD, lambdaf_as, lambdaf, f, d, U_inf, St_shed,
                    plt_modes, obstacles, plt_plane_inds, num_geos, PUBdir,
                    'test', correct_sign, '', '', harm_freqs=harm_freqs,
                    close_fig=False, save_fig=False)

for geo in geo_inds:
    plt.figure()
    
    Psi_temp = np.abs(Psi_uf_as_g[geo][plane][:, :, mode1[geo]])/np.max(Psi_uf_as_g[geo][plane][:, :, mode1[geo]]) + \
               np.abs(Psi_uf_as_g[geo][plane][:, :, mode2[geo]])/np.max(Psi_uf_as_g[geo][plane][:, :, mode2[geo]])
    print 'u', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 1, aspect='equal')
    pltf.plot_normalized_planar_field(x_g[geo][:, :, plane],
                                      y_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)
    Psi_temp = np.abs(Psi_vf_as_g[geo][plane][:, :, mode1[geo]])/np.max(Psi_vf_as_g[geo][plane][:, :, mode1[geo]]) + \
               np.abs(Psi_vf_as_g[geo][plane][:, :, mode2[geo]])/np.max(Psi_vf_as_g[geo][plane][:, :, mode2[geo]])
    print 'v', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 2, aspect='equal')
    pltf.plot_normalized_planar_field(x_g[geo][:, :, plane],
                                      y_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)

#%% Find good Spectra

# 1/2 f_shed
mode = 0
akf_PSD_temp = akf_sy_PSD

spectra_temp = [[akf_PSD_temp[geo][p][:, mode] for p in plane_inds]
                for geo in geo_inds]
spectra_labels = [['%2.1f' % z for z in zs] for geo in geo_inds]
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
pltf.plot_spectra_comparison(f, spectra_temp, num_geos, PUBdir, 'test',
                             spectra_labels=spectra_labels, label_spectra=True, label_axes=False, x_label='f (Hz)',
                             harm_freqs=harm_freqs, f_shed = [160, 120],
                             tight=True, label_y_ticks=False,
                             save_fig=False, close_fig=False, figsize=(4, 3))

plane = 0
mode1 = [4, 2]
mode2 = [7, 3]
plt_plane_inds = [[plane], [plane]]
plt_modes = [[[mode1[geo], mode2[geo]]] for geo in geo_inds]
correct_sign = [[[1, 1]], [[1, 1]]]
pltf.plot_POD_modes(x_g, y_g, Psi_uf_sy_g, Psi_vf_sy_g, Psi_vf_sy_g,
                    akf_sy_PSD, lambdaf_sy, lambdaf, f, d, U_inf, St_shed,
                    plt_modes, obstacles, plt_plane_inds, num_geos, PUBdir,
                    'test', correct_sign, '', '', harm_freqs=harm_freqs,
                    close_fig=False, save_fig=False)

for geo in geo_inds:
    plt.figure()
    
    Psi_temp = np.abs(Psi_uf_sy_g[geo][plane][:, :, mode1[geo]])/np.max(Psi_uf_sy_g[geo][plane][:, :, mode1[geo]]) + \
               np.abs(Psi_uf_sy_g[geo][plane][:, :, mode2[geo]])/np.max(Psi_uf_sy_g[geo][plane][:, :, mode2[geo]])
    print 'u', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 1, aspect='equal')
    pltf.plot_normalized_planar_field(x_g[geo][:, :, plane],
                                      y_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)
    Psi_temp = np.abs(Psi_vf_sy_g[geo][plane][:, :, mode1[geo]])/np.max(Psi_vf_sy_g[geo][plane][:, :, mode1[geo]]) + \
               np.abs(Psi_vf_sy_g[geo][plane][:, :, mode2[geo]])/np.max(Psi_vf_sy_g[geo][plane][:, :, mode2[geo]])
    print 'v', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 2, aspect='equal')
    pltf.plot_normalized_planar_field(x_g[geo][:, :, plane],
                                      y_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)

#%% Find good Spectra

# 1/4 f_shed

plane = 0
mode1 = [1, 1]
plt_plane_inds = [[plane], [plane]]
plt_modes = [[[mode1[geo]]] for geo in geo_inds]
correct_sign = [[[1]], [[1]]]
pltf.plot_POD_modes(x_g, y_g, Psi_uf_as_g, Psi_vf_as_g, Psi_vf_as_g,
                    akf_as_PSD, lambdaf_as, lambdaf, f, d, U_inf, St_shed,
                    plt_modes, obstacles, plt_plane_inds, num_geos, PUBdir,
                    'test', correct_sign, '', '', harm_freqs=harm_freqs,
                    close_fig=False, save_fig=False)

for geo in geo_inds:
    plt.figure()
    
    Psi_temp = np.abs(Psi_uf_as_g[geo][plane][:, :, mode1[geo]])/np.max(Psi_uf_as_g[geo][plane][:, :, mode1[geo]])
    print 'u', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 1, aspect='equal')
    pltf.plot_normalized_planar_field(x_g[geo][:, :, plane],
                                      y_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='u',
                                      plot_arrow=False)
    Psi_temp = np.abs(Psi_vf_as_g[geo][plane][:, :, mode1[geo]])/np.max(Psi_vf_as_g[geo][plane][:, :, mode1[geo]])
    print 'v', np.where(Psi_temp == np.max(Psi_temp))
    plt.subplot(3, 1, 2, aspect='equal')
    pltf.plot_normalized_planar_field(x_g[geo][:, :, plane],
                                      y_g[geo][:, :, plane], Psi_temp,
                                      obstacle=obstacles[geo], cb_label='v',
                                      plot_arrow=False)


#%% Caluclate spectra

uf_PSD_g = [[np.zeros((J[geo][0], I[geo][0], int(np.floor(B1[geo]/2))))
             for _ in plane_inds] for geo in geo_inds]
vf_PSD_g = [[np.zeros((J[geo][0], I[geo][0], int(np.floor(B1[geo]/2))))
             for _ in plane_inds] for geo in geo_inds]
f = [np.zeros((int(np.floor(B1[geo]/2)))) for geo in geo_inds]
for geo in geo_inds:
    for k in plane_inds:
        ufs = [uf_g[geo][k], vf_g[geo][k]]
        num_ufs = np.size(ufs, axis=0)
        uf_inds = np.arange(num_ufs)
        uf_PSDs = [np.zeros((int(np.floor(B1[geo]/2)), 1))
                   for _ in uf_inds]
        for j in np.arange(J[geo]):
            for i in np.arange(I[geo]):
                point = j*I[geo] + i
                if point % 1000 == 0:
                    print 'geo: %1.0f/%1.0f,  plane: %2.0f/%2.0f,  point: %4.0f/%4.0f' % \
                          (geo, num_geos - 1, k, num_planes - 1, point, I[geo]*J[geo] - 1)
                for (uf_temp, uf_ind) in zip(ufs, uf_inds):
                    uf_PSD_temp = np.zeros((int(np.floor(B1[geo]/2)),
                                             num_trials))
                    for trial in trials:
                        start_ind = 0 + trial*B1[geo]
                        end_ind = B1[geo] + trial*B1[geo]
                        [f_temp, PSD_temp] = \
                            spect.spectrumFFT_v2(
                                                 uf_temp[j, i, start_ind:end_ind], 
                                                 f_capture[geo])
                        uf_PSD_temp[:, trial] = PSD_temp
                    uf_PSDs[uf_ind] = np.mean(uf_PSD_temp, 1)
                uf_PSD_g[geo][k][j, i, :] = uf_PSDs[0]
                vf_PSD_g[geo][k][j, i, :] = uf_PSDs[1]
    f[geo] = f_temp
#del akfs, akf_PSDs, f_temp, PSD_temp, akf_PSD_temp


#%% Plot Spectra

plot_x_coords = [[-0.3, -0.3, 0.4, 1.2], [-0.2, -0.2, 0.6, 1.2]]
plot_y_coords = [[0, 0, 0.45, 1.0], [0, 0, 0.3, 1.0]]
plot_z_coords = [[4, 4, 4, 4] for _ in geo_inds]

#plot_x_coords = [[-0.29, -0.29, 0.77], [-0.16, 1.22, 0.27]]
#plot_y_coords = [[0, 0, 0.11], [0.32, 0.11, 0.45]]

uf_PSDs_temp = [vf_PSD_g, uf_PSD_g, vf_PSD_g, vf_PSD_g]
comp_labels = [r'$v\textquotesingle$', r'$u\textquotesingle$',
               r'$v\textquotesingle$', r'$v\textquotesingle$']
#u
#plot_x_coords = [[0.104, 0.772, 0.271], [0.207, 0.955, 0]]
#plot_y_coords = [[0.445, 0.445, 0.445], [0.694, 0.534, 0]]


#u rms
#plot_x_coords = [[0.339], [0.528]]
#plot_y_coords = [[0.451], [0.640]]

#v
#plot_x_coords = [[-0.285, 0.772, 0.772], [1.222, -0.166, 0]]
#plot_y_coords = [[0, 0.445, 0.111], [0.107, 0.320, 0]]

#v rms
#plot_x_coords = [[0.285], [1.008]]
#plot_y_coords = [[0.231], [0.427]]



num_spectras = np.size(plot_x_coords, axis=-1)
spectra_inds = np.arange(num_spectras)
end_freq = [f[geo][-1] for geo in geo_inds]
end_freq_ind = [int(np.where(f[geo] == end_freq[geo])[0])
                for geo in geo_inds]
plot_z_inds = [[np.where(np.abs(z_g[geo][0, 0, :] - plot_z_coords[geo][ind]) ==
                         np.min(np.abs(z_g[geo][0, 0, :] - plot_z_coords[geo][ind])))[0][0]
                for ind in np.arange(num_spectras)] for geo in geo_inds]
plot_x_inds = [[np.where(np.abs(x_g[geo][0, :, plot_z_inds[geo][ind]] - plot_x_coords[geo][ind]) ==
                         np.min(np.abs(x_g[geo][0, :, plot_z_inds[geo][ind]] - plot_x_coords[geo][ind])))[0][0]
                for ind in np.arange(num_spectras)] for geo in geo_inds]
plot_y_inds = [[np.where(np.abs(y_g[geo][:, 0, plot_z_inds[geo][ind]] - plot_y_coords[geo][ind]) ==
                         np.min(np.abs(y_g[geo][:, 0, plot_z_inds[geo][ind]] - plot_y_coords[geo][ind])))[0][0]
                for ind in np.arange(num_spectras)] for geo in geo_inds]
plot_xs = [[x_g[geo][plot_y_ind, plot_x_ind, plot_z_ind]
            for plot_x_ind, plot_y_ind, plot_z_ind in
            zip(plot_x_inds[geo], plot_y_inds[geo], plot_z_inds[geo])]
           for geo in geo_inds]
plot_ys = [[y_g[geo][plot_y_ind, plot_x_ind, plot_z_ind]
            for plot_x_ind, plot_y_ind, plot_z_ind in
            zip(plot_x_inds[geo], plot_y_inds[geo], plot_z_inds[geo])]
           for geo in geo_inds]
plot_zs = [[z_g[geo][plot_y_ind, plot_x_ind, plot_z_ind]
            for plot_x_ind, plot_y_ind, plot_z_ind in
            zip(plot_x_inds[geo], plot_y_inds[geo], plot_z_inds[geo])]
           for geo in geo_inds]

spectras_temp = [[uf_PSDs_temp[ind][geo][plot_z_ind][plot_y_ind, plot_x_ind, :end_freq_ind[geo]]#/np.std(uf_PSDs_temp[ind][geo][plot_z_ind])
                  for ind, plot_x_ind, plot_y_ind, plot_z_ind
                  in zip(spectra_inds, plot_x_inds[geo], plot_y_inds[geo], plot_z_inds[geo])]
                 for geo in geo_inds]
fd_U_inf_temp = [[f[geo][:end_freq_ind[geo]]*d[geo]/U_inf[geo][plane]
                    for plane in plot_z_inds[geo]] for geo in geo_inds]
spectra_labels = [[comp_labels[ind] + r''' at
                   $\left(%1.1f, %1.1f, %1.1f\right)$'''
                   % (plot_x, plot_y, plot_z)
                   for ind, plot_x, plot_y, plot_z 
                   in zip(spectra_inds, plot_xs[geo], plot_ys[geo], plot_zs[geo])]
                  for geo in geo_inds]
x_label = r'$\displaystyle \frac{fd}{U_\infty}$'
x_label = r'$f$'
y_label = 'PSD'
save_name = '2D_vel_spect'
harm_freqs = [[0.25, 1, 2], [0.1, 1, 2]]
#label_y_fracs = [[0.85, 0.7, 0.65, 0.9], [0.85, 0.75, 0.45, 0.8]]
label_y_fracs = [[0.8, 0.6, 0.55, 0.85], [0.8, 0.7, 0.35, 0.65]]
St_shed_temp = [np.mean(St_shed[geo]) for geo in geo_inds]
pltf.plot_spectra_comparison(fd_U_inf_temp, spectras_temp, num_geos, PUBdir,
                             save_name, label_spectra=True, spectra_labels=spectra_labels,
                             x_label=x_label, y_label=y_label,
                             harm_freqs=harm_freqs, f_shed=St_shed_temp,
                             tight=True, label_y_ticks=False, close_fig=False,
                             axes_label_y=0.985, axes_label_x=0.04,
                             label_x_frac=1/32, label_y_fracs=label_y_fracs,
                             colour_spectra_label=False,
                             x_ticks=[10**-3, 10**-2, 10**-1, 10**0],
                             y_ticks_single=[10**-7, 10**-6, 10**-5, 10**-4],
                             y_lims=[10**-8.5, 10**5.5],
                             figsize=(3.3, 4), save_fig=True, extension='.eps')
#y_ticks_single=[10**-7, 10**-6, 10**-5, 10**-4],
#y_lims=[10**-8.5, 10**5.5]
#font size
#%%









#%% mask out areas  of M_lambda_2

M_lambda2_2_g = copy.deepcopy(M_lambda2_g)
for geo in geo_inds:
    for i in np.arange(I[geo]):
        for j in np.arange(J[geo]):
            for k in np.arange(K[geo]):
                if z_g[geo][j, i, k] >= 3.5 and x_g[geo][j, i, k] >= 2.75:
                    M_lambda2_2_g[geo][j, i, k] = M_lambda2_2_g[geo].max()
                if x_g[geo][j, i, k] <= 3.5 and np.abs(y_g[geo][j, i, k]) >= 1.75 \
                        and z_g[geo][j, i, k]>=0.5:
                    M_lambda2_2_g[geo][j, i, k] = M_lambda2_2_g[geo].max() 
                if x_g[geo][j, i, k] >= 3.5 and np.abs(y_g[geo][j, i, k]) >= 2.6 \
                        and z_g[geo][j, i, k] >= 0.5:
                    M_lambda2_2_g[geo][j, i, k] = M_lambda2_2_g[geo].max() 
                if geo == 0:
                    if y_g[geo][j, i, k] >= 2.6 and z_g[geo][j, i, k] >= 0.45:
                        M_lambda2_2_g[geo][j, i, k] = M_lambda2_2_g[geo].max() 
                    if x_g[geo][j, i, k] >= 6 and z_g[geo][j, i, k] >= 2.5:
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
                         M_lambda2_g[geo].flatten()
                         ]).T
    savedataheader = 'x,y,z,M_U,M_V,M_W,Q,M_Omega_x,M_Omega_y,M_Omega_z,M_lambda2,M_ufuf,M_vfvf,M_wfwf,M_ufvf,M_ufwf,M_vfwf,M_k,M_G_k,M_lambda2_raw'
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

#%% Plot Mean Velocity Fields - z=const planes

z_locs = np.array([0])
num_xy_planes = np.size(z_locs, axis=0)

xy_plane_inds = [np.zeros(num_xy_planes, dtype=int) for _ in geo_inds]
xy_plane_titles = [['' for _ in np.arange(num_xy_planes)] for _ in geo_inds]
for geo in geo_inds:
    for (z_loc, z_loc_ind) in zip(z_locs, np.arange(num_xy_planes)):
        ind = np.where(np.abs(z_g[geo][0, 0, :] - z_loc) ==
                       np.min(np.abs(z_g[geo][0, 0, :] - z_loc)))
        print ind[0][0]
        xy_plane_inds[geo][z_loc_ind] = ind[0][0]
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]

field_g_list = [M_V_g]
field_labels = [r'$\displaystyle \frac{\overline{W}}{U_\infty}$']
pltf.plot_normalized_planar_field_comparison(
        x_g, y_g, field_g_list, M_U_g, M_V_g, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''], 'mean_vel_z_planes',
        cb_tick_format='%.2f', comparison_type='plane', plot_arrow=False,
        figsize=(18, 9), close_fig=False, plot_field=True, strm_dens=3, strm_lw=1.5)
#plt.close('all')