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

import plotting_functions as pltf
import grid_conversion as gc
import spectrum_fft_v2 as spect
import load_from_matlab_workspace as load_mat


#%% User-Defined Variables

mainfolder = r'D:\7 - SquareAR4_LaminarBL'# The main working directory
PUBdir =  r'D:\EXF Paper\EXF Paper V3\figs'# Where the .eps figures go

WRKSPCdirs = (r'D:\Workspaces', r'D:\Workspaces')# Where the workspaces are located
WRKSPCfilenames_both = [[r'circle_z_1.5mm', r'circle_z_4.5mm', 
                         r'circle_z_7.5mm', r'circle_z_10.5mm', 
                         r'circle_z_15mm', r'circle_z_21mm', 
                         r'circle_z_27mm', r'circle_z_33mm', 
                         r'circle_z_39mm', r'circle_z_45mm',
                         r'circle_z_48mm', r'circle_z_51mm',
                         r'circle_z_54mm', r'circle_z_57mm'], 
                        [r'square_z_1.5mm', r'square_z_4.5mm', 
                         r'square_z_7.5mm', r'square_z_10.5mm', 
                         r'square_z_15mm', r'square_z_21mm', 
                         r'square_z_27mm', r'square_z_33mm', 
                         r'square_z_39mm', r'square_z_45mm',
                         r'square_z_48mm', r'square_z_51mm',
                         r'square_z_54mm', r'square_z_57mm']]# Name of the MATLAB workspaces

#save_names = ['z_1.5mm',
#              'z_4.5mm',
#              'z_7.5mm'
#              ]

zs = [1.5, 4.5, 7.5, 10.5, 15, 21, 27, 33, 39, 45, 48, 51, 54, 57]
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

[akf, akf_sy, akf_as, akf_sy_gaus, akf_sy_harm,
 lambdaf, lambdaf_sy, lambdaf_as, lambdaf_sy_gaus, lambdaf_sy_harm,
 Psi_uf_g, Psi_uf_sy_g, Psi_uf_as_g, Psi_uf_sy_gaus_g, Psi_uf_sy_harm_g,
 Psi_vf_g, Psi_vf_sy_g, Psi_vf_as_g, Psi_vf_sy_gaus_g, Psi_vf_sy_harm_g,
 Psi_wf_g, Psi_wf_sy_g, Psi_wf_as_g, Psi_wf_sy_gaus_g, Psi_wf_sy_harm_g] = \
    load_mat.load_POD(WRKSPCdirs, WRKSPCfilenames_both, I, J, B, num_geos,
                      num_planes, num_modes)

num_snaps = 16368

#[uf_g, vf_g, wf_g] = \
#    load_mat.load_snapshots(WRKSPCdirs, WRKSPCfilenames_both, I, J, B,
#                            num_geos, num_planes, num_snaps)

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
        print plane, correct_planes[geo][plane]
        if correct_planes[geo][plane]:
            print plane
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

K = [19]*num_geos

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


#%% Calculate mean gradients reectangular grid

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
        xz_plane_titles[geo][y_loc_ind] = 'y/d = %1.2f' \
            % np.abs(y_g[geo][ind[0][0], 0, 0])
xz_save_names_temp = ['y_%1.1fd' % y_locs[ind]
                      for ind in np.arange(num_xz_planes)]
xz_save_names = [xz_save_names_temp[ind].replace('.', '_')
                 for ind in np.arange(num_xz_planes)]

field_g_list = [M_Omega_y_rect_g]
field_labels = [r'$\displaystyle \frac{\overline{\Omega_y}d}{U_\infty}$']
pltf.plot_normalized_planar_field_comparison(
        x_rect_g, z_rect_g, field_g_list, M_U_g, M_W_g, field_labels, 
        xz_plane_titles, 'xz', xz_plane_inds, num_geos, obstacles, PUBdir,
        [''], 'mean_vort_y_planes', strm_x_g=x_g, strm_y_g=z_g,
        obstacle_AR=obstacle_AR, cb_tick_format='%.2f',
        comparison_type='plane', cb_label_0=False, strm_dens=1,
        plot_bifurcation_lines=True, plot_arrow=False,
        bifurc_seed_pt_xy=[[[2.0, -0.1], [2.4, -0.1]], [[1.35, 0.55], [1.4, 0.6]]],
        figsize=(5.5, 2), close_fig=False)
#plt.close('all')


#%% Plot Mean Velocity Fields - z=const planes

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
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]

field_g_list = [M_W_g]
field_labels = [r'$\displaystyle \frac{\overline{W}}{U_\infty}$']
pltf.plot_normalized_planar_field_comparison(
        x_g, y_g, field_g_list, M_U_g, M_V_g, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''], 'mean_vel_z_planes',
        cb_tick_format='%.2f', comparison_type='plane', plot_arrow=False,
        figsize=(5.25, 8), close_fig=False)
#plt.close('all')


#%% Plot Mean Vorticity Fields - x=const planes

x_locs = np.array([0.85, 1, 2.75, 5])
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

field_g_list = [M_Omega_z_rect_g]
field_labels = [r'$\displaystyle \frac{\overline{\Omega_z}d}{U_\infty}$']
pltf.plot_normalized_planar_field_comparison(
        y_rect_g, z_rect_g, field_g_list, M_V_g, M_W_g, field_labels, yz_plane_titles,
        'yz', yz_plane_inds, num_geos, obstacles, PUBdir, [''], 'mean_vort_x_planes',
        strm_x_g=y_g, strm_y_g=z_g, obstacle_AR=obstacle_AR,
        cb_tick_format='%.2f', comparison_type='plane', figsize=(5.5, 6.5),
        close_fig=False)
#plt.close('all')

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
pltf.plot_normalized_planar_field_comparison(
        x_rect_g, z_rect_g, field_g_list, M_U_g, M_W_g, field_labels, 
        xz_plane_titles, 'xz', xz_plane_inds, num_geos, obstacles, PUBdir,
        [''], 're_norm_y_planes', strm_x_g=x_g, strm_y_g=z_g,
        obstacle_AR=obstacle_AR, cb_tick_format='%.2f',
        comparison_type='field', cb_label_0=False, halve_cb=True, strm_dens=0.5,
        plot_streamlines=True, plot_bifurcation_lines=True, plot_arrow=False,
        bifurc_seed_pt_xy=[[[2.0, -0.1], [2.4, -0.1]], [[1.35, 0.55], [1.4, 0.6]]],
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
        plot_streamlines=True, plot_bifurcation_lines=True, plot_arrow=False,
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
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]

field_g_list = [M_ufvf_g, M_ufwf_g, M_vfwf_g]
field_labels = [r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$']
pltf.plot_normalized_planar_field_comparison(
        x_g, y_g, field_g_list, M_U_g, M_V_g, field_labels, xy_plane_titles,
        'xy', xy_plane_inds, num_geos, obstacles, PUBdir, [''],
        're_shear_z_planes', cb_tick_format='%.3f', plot_arrow=False,
        comparison_type='field and plane', figsize=(5.25, 9), close_fig=False)
plt.annotate('', xy=[0, 0.68], xytext=[1, 0.68], xycoords='figure fraction',
             arrowprops=dict(arrowstyle='-'))
plt.annotate('', xy=[0, 0.37], xytext=[1, 0.37], xycoords='figure fraction',
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








#%% POD plots

#%% Plot POD Energy Bar plot

num_modes_plt = 10
extension = '.eps'


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
        xy_plane_titles[geo][z_loc_ind] = 'z/h = %1.2f' \
            % (z_g[geo][0, 0, ind[0][0]]/obstacle_AR[geo])
        save_name = 'z_%1.1f' % np.abs(z_g[geo][0, 0, ind[0][0]])
xy_save_names = ['z_%1.0fd_' % z_locs[ind] for ind in np.arange(num_xy_planes)]


save_name = 'POD_energies'

fig = plt.figure(figsize=(5.5, 6))

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
            ax = plt.subplot(num_xy_planes, num_lambdafs, subplot_ind)
            mode_energy = lambdaf_temp[geo][plane][0:num_modes_plt]/np.sum(lambdaf[geo][plane])*100
#            mode_energies = [lambdaf_temp[geo][plane][0:num_modes_plt]/np.sum(lambdaf[geo][plane])*100 for geo in geo_inds]
            mode_nums_list = np.repeat(np.expand_dims(mode_nums, 1), num_lambdafs, axis=1).T
            plt.bar(mode_nums, mode_energy, align_sign*bar_width, align='edge',
                    hatch=hatch, ec=ec, fc=fc)
            plt.title(title, fontsize=10)
            plt.ylabel('\% of ' + tke_str)
            plt.xticks(mode_nums)
            if plane_ind == num_xy_planes - 1:
                plt.xlabel(lambdaf_label)
    #            else:
    #                plt.tick_params(labelbottom=False)
            plt.text(-0.05, 1.15, #0.15, 0.95
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
additional_rel_freqs = [[0.25], [0.1]]
save_names = ['first_harm_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Anti-symmetric' + ' \n'

pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_as_g, Psi_vf_as_g, Psi_wf_as_g,
        akf_as_PSD, lambdaf_as, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, num_geos, PUBdir, save_names, correct_sign, POD_type, '',
        additional_relative_freqs=additional_rel_freqs,
        figsize=(6.7, 5), close_fig=False)

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
additional_rel_freqs = [[0.25], [0.1]]
save_names = ['slowly_varying_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Symmetric' + ' \n'

pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_sy_g, Psi_vf_sy_g, Psi_wf_sy_g,
        akf_sy_PSD, lambdaf_sy, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, num_geos, PUBdir, save_names, correct_sign, POD_type, '',
        additional_relative_freqs=additional_rel_freqs,
        figsize=(6.7, 2.5), close_fig=False)

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
additional_rel_freqs = [[0.25], [0.1]]
save_names = ['slowly_varying_' + xy_save_names[ind] for ind in np.arange(num_xy_planes)]
POD_type = 'Symmetric' + ' \n'

pltf.plot_POD_modes(
        x_raw_g, y_raw_g, Psi_uf_sy_g, Psi_vf_sy_g, Psi_wf_sy_g,
        akf_sy_PSD, lambdaf_sy, lambdaf, f, d, U_inf, St_shed, plt_modes,
        obstacles, xy_plane_inds, num_geos, PUBdir, save_names, correct_sign, POD_type, '',
        additional_relative_freqs=additional_rel_freqs,
        figsize=(6.7, 5), close_fig=False)

#%% Plot POD Modes

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