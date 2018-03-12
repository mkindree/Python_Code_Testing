from __future__ import division
import sys

CODEdir = r'D:\6 - Circle_AR4_LaminarBL\PIV\Test_1_2_3\PIV\Test_1_2_3_testing\Python Codes'
if CODEdir not in sys.path:
    sys.path.append(CODEdir)

import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import scipy.io as sio

import plotting_functions as pltf

#%% User-Defined Variables

mainfolder = r'D:\7 - SquareAR4_LaminarBL'# The main working directory
PUBdir = r''# Where the .eps figures go

WRKSPCdirs = (r'D:\Workspaces', r'D:\Workspaces')# Where the workspaces are located
WRKSPCfilenames_both = ([r'circle_z_1.5mm', r'circle_z_4.5mm', r'circle_z_7.5mm'], 
                        [r'circle_z_1.5mm', r'circle_z_4.5mm', r'circle_z_7.5mm'])# Name of the MATLAB workspaces

#save_names = ['z_1.5mm',
#              'z_4.5mm',
#              'z_7.5mm'
#              ]

zs = [1.5, 4.5, 7.5]
num_planes = len(zs)
plane_inds = np.arange(num_planes, dtype=int)

num_geos = 2
obstacles = ('circle', 'square')
geo_inds = np.arange(num_geos, dtype=int)

#WRKSPCdirs_both = (WRKSPCdir_circ, WRKSPCdir_square)
#WRKSPCfilenames_both = (WRKSPCfilenames_circ, WRKSPCfilenames_square)



#%% Load variables

os.chdir(WRKSPCdir_circ)


variable_names = ['I', 'J', 'B', 'B1', 'f_capture',
                  'f_shed', 'St_shed', 'U_inf', 'd', 'Re'
                  'x_g', 'y_g', 'dx', 'dy', 't',
                  'M_U_g', 'M_V_g', 'M_W_g',
                  'M_ufuf_g', 'M_ufvf_g', 'M_ufwf_g',
                  'M_vfvf_g', 'M_vfwf_g', 'M_wfwf_g'
                  ]

variable_names1 = ['I', 'J', 'B', 'B1', 'f_capture', 'f_shed', 'd', 'dx', 'dy']

I = (np.zeros(1), np.zeros(1))
J = (np.zeros(1), np.zeros(1))
B = (np.zeros(1), np.zeros(1))
B1 = (np.zeros(1), np.zeros(1))
f_capture = (np.zeros(1), np.zeros(1))
d = (np.zeros(1), np.zeros(1))
num_vect = (np.zeros(1), np.zeros(1))
dx = (np.zeros(1), np.zeros(1))
dy =(np.zeros(1), np.zeros(1))
for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                           WRKSPCfilenames_both):
    os.chdir(WRKSPCdir)
    matvariables = sio.loadmat(WRKSPCfilenames[0],
                               variable_names=variable_names1)
    I[geo][0] = int(matvariables['I'])
    J[geo][0] = int(matvariables['J'])
    B[geo][0] = int(matvariables['B'])
    B1[geo][0] = int(matvariables['B1'])
    f_capture[geo][0] = float(matvariables['f_capture'])
    d[geo][0] = float(matvariables['d'])
    num_vect[geo][0]= I*J
    t[geo][0] = float(matvariables['t'])
    dx[geo][0] = float(matvariables['dx'])
    dy[geo][0] = float(matvariables['dy'])
    del matvariables

variable_names2 = ['t']

t = (np.zeros(B[0]), np.zeros(B[1]))
for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                           WRKSPCfilenames_both):
    os.chdir(WRKSPCdir)
    matvariables = sio.loadmat(WRKSPCfilenames[0],
                               variable_names=variable_names2)
    t[geo][:] = float(matvariables['t'])

variable_names3 = ['f_shed', 'St_shed', 'U_inf', 'Re']

f_shed = (np.zeros(num_planes), np.zeros(num_planes))
St_shed = (np.zeros(num_planes), np.zeros(num_planes))
U_inf = (np.zeros(num_planes), np.zeros(num_planes))
Re = (np.zeros(num_planes), np.zeros(num_planes))
for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                           WRKSPCfilenames_both):
    os.chdir(WRKSPCdir)
    for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
        matvariables = sio.loadmat(WRKSPCfilename,
                                   variable_names=variable_names3)
        f_shed[geo][plane] = float(matvariables['f_shed'])
        St_shed[geo][plane] = float(matvariables['St_shed'])
        U_inf[geo][plane] = float(matvariables['U_inf'])
        Re[geo][plane] = float(matvariables['Re'])
        del matvariables
#f_shed_circ = St_shed_circ*U_inf_circ/d_circ
#num_vect[geo, plane] = I*J
#dx[geo, plane] = x_g[0, 1] - x_g[0, 0]
#dy[geo, plane] = y_g[1, 0] - y_g[0, 0]

#I_maxs = np.zeros(num_geos, dtype=int)
#J_maxs = np.zeros(num_geos, dtype=int)
#for geo in geo_inds:
#    I_maxs[geo] = np.max(I[geo])
#    J_maxs[geo] = np.max(J[geo])

variable_names4 = ['M_U_g', 'M_V_g', 'M_W_g',
                   'M_ufuf_g', 'M_ufvf_g', 'M_ufwf_g',
                   'M_vfvf_g', 'M_vfwf_g', 'M_wfwf_g'
                   ]

x_g = (np.zeros((J[0], I[0], num_planes)),
       np.zeros((J[1], I[1], num_planes)))
y_g = (np.zeros((J[0], I[0], num_planes)),
       np.zeros((J[1], I[1], num_planes)))
M_U_g = (np.zeros((J[0], I[0], num_planes)),
         np.zeros((J[1], I[1], num_planes)))
M_V_g = (np.zeros((J[0], I[0], num_planes)),
         np.zeros((J[1], I[1], num_planes)))
M_W_g = (np.zeros((J[0], I[0], num_planes)),
         np.zeros((J[1], I[1], num_planes)))
M_ufuf_g = (np.zeros((J[0], I[0], num_planes)),
            np.zeros((J[1], I[1], num_planes)))
M_ufvf_g = (np.zeros((J[0], I[0], num_planes)),
            np.zeros((J[1], I[1], num_planes)))
M_ufwf_g = (np.zeros((J[0], I[0], num_planes)),
            np.zeros((J[1], I[1], num_planes)))
M_vfvf_g = (np.zeros((J[0], I[0], num_planes)),
            np.zeros((J[1], I[1], num_planes)))
M_vfwf_g = (np.zeros((J[0], I[0], num_planes)),
            np.zeros((J[1], I[1], num_planes)))
M_wfwf_g = (np.zeros((J[0], I[0], num_planes)),
            np.zeros((J[1], I[1], num_planes)))
M_vfvf_ufuf_g = (np.zeros((J[0], I[0], num_planes)),
                 np.zeros((J[1], I[1], num_planes)))
M_wfwf_ufuf_g = (np.zeros((J[0], I[0], num_planes)),
                 np.zeros((J[1], I[1], num_planes)))
M_wfwf_vfvf_g = (np.zeros((J[0], I[0], num_planes)),
                 np.zeros((J[1], I[1], num_planes)))
for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                           WRKSPCfilenames_both):
    os.chdir(WRKSPCdir)
    for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
        matvariables = sio.loadmat(WRKSPCfilename,
                                   variable_names=variable_names4)
        x_g[geo][:, :, plane] = np.squeeze(matvariables['x_g'])
        y_g[geo][:, :, plane] = np.squeeze(matvariables['y_g'])
        M_U_g[geo][:, :, plane] = np.squeeze(matvariables['M_U_g'])
        M_V_g[geo][:, :, plane] = np.squeeze(matvariables['M_V_g'])
        M_W_g[geo][:, :, plane] = np.squeeze(matvariables['M_W_g'])
        M_ufuf_g[geo][:, :, plane] = np.squeeze(matvariables['M_ufuf_g'])
        M_ufvf_g[geo][:, :, plane] = np.squeeze(matvariables['M_ufvf_g'])
        M_ufwf_g[geo][:, :, plane] = np.squeeze(matvariables['M_ufwf_g'])
        M_vfvf_g[geo][:, :, plane] = np.squeeze(matvariables['M_vfvf_g'])
        M_vfwf_g[geo][:, :, plane] = np.squeeze(matvariables['M_vfwf_g'])
        M_wfwf_g[geo][:, :, plane] = np.squeeze(matvariables['M_wfwf_g'])
        del matvariables
        
        M_vfvf_ufuf_g[geo][:, :, plane] = M_vfvf_g[geo][:, :, plane] / M_ufuf_g[geo][:, :, plane]
        M_wfwf_ufuf_g[geo][:, :, plane] = M_wfwf_g[geo][:, :, plane] / M_ufuf_g[geo][:, :, plane]
        M_wfwf_vfvf_g[geo][:, :, plane] = M_wfwf_g[geo][:, :, plane] / M_vfvf_g[geo][:, :, plane]


#%%

def plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                            M_V_g, field_labels, num_planes,
                                            num_geos, obstacles, PUBdir,
                                            save_names, save_suffix,
                                            figsize=(6.7, 9)):
    label_0= [False, True]
    for plane in np.arange(num_planes):
        fig = plt.figure(figsize=figsize)

        num_ind = np.size(field_g_list, axis=0)
        for geo in np.arange(num_geos):
            for ind in np.arange(num_ind):
                climmax_geo = np.zeros(num_geos)
                for i in np.arange(num_geos):
                    climmax_geo[i] = np.max([
                            np.abs(np.max(field_g_list[ind][i][:, :, plane])),
                            np.abs(np.min(field_g_list[ind][i][:, :, plane]))
                            ])
                climmax = np.max(climmax_geo)
                subplot_ind = num_geos*ind + geo + 1
                ax = plt.subplot(num_ind, num_geos, subplot_ind,
                                 aspect='equal')
                pltf.plot_normalized_planar_field(x_g[geo][:, :, plane],
                                             y_g[geo][:, :, plane],
                                             field_g_list[ind][geo][:, :, plane],
                                             cb_label=field_labels[ind],
                                             subplot_ind=subplot_ind,
                                             clim=climmax,
                                             obstacle=obstacles[geo],
                                             plot_streamlines=True,
                                             U_g=M_U_g[geo][:, :, plane],
                                             V_g=M_V_g[geo][:, :, plane],
                                             plot_obstacle=True, strm_dens=1.75,
                                             arrow_xlims=[-1.25, -0.75], cb_ar=0.05,
                                             y_subplot_label=0.6, x_subplot_label=-1.0,
                                             cb_label_0=label_0[ind], cb_label_1=False)
        plt.tight_layout()
        save_path = PUBdir + '\\' + save_names[plane] + '_' + save_suffix \
            + '.jpg'
        plt.savefig(save_path, bbox_inches='tight')

#%% Plot 2d tip Mean Field

os.chdir(PUBdir)

obstacles = ('hollow circle', 'hollow square')

row_start = [14, 13]
row_stop = [-14, -13]
col_start = [6, 0]
col_stop = [-24, -28]

field_g_list = ((M_U_g[0][row_start[0]:row_stop[0],col_start[0]:col_stop[0],:], M_U_g[1][row_start[1]:row_stop[1],col_start[1]:col_stop[1],:]),
                (M_V_g[0][row_start[0]:row_stop[0],col_start[0]:col_stop[0],:], M_V_g[1][row_start[1]:row_stop[1],col_start[1]:col_stop[1],:]))
x_g_temp = (x_g[0][row_start[0]:row_stop[0],col_start[0]:col_stop[0],:], x_g[1][row_start[1]:row_stop[1],col_start[1]:col_stop[1],:])
y_g_temp = (y_g[0][row_start[0]:row_stop[0],col_start[0]:col_stop[0],:], y_g[1][row_start[1]:row_stop[1],col_start[1]:col_stop[1],:])
M_U_g_temp = (M_U_g[0][row_start[0]:row_stop[0],col_start[0]:col_stop[0],:], M_U_g[1][row_start[1]:row_stop[1],col_start[1]:col_stop[1],:])
M_V_g_temp = (M_V_g[0][row_start[0]:row_stop[0],col_start[0]:col_stop[0],:], M_V_g[1][row_start[1]:row_stop[1],col_start[1]:col_stop[1],:])

field_labels = [
                r'$\displaystyle \frac{\overline{U}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{V}}{U_\infty}$'
                ]
plot_normalized_planar_field_comparison(x_g_temp, y_g_temp, field_g_list, M_U_g_temp,
                                             M_V_g_temp, field_labels, num_planes,
                                             num_geos, obstacles, PUBdir,
                                             save_names, '2Dmean', figsize=(5.5, 3.7))

#%% Plot Mean Field

os.chdir(PUBdir)

field_g_list = (M_U_g, M_V_g, M_W_g)
field_labels = [
                r'$\displaystyle \frac{\overline{U}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{V}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{W}}{U_\infty}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels, num_planes,
                                             num_geos, obstacles, PUBdir,
                                             save_names, 'mean')


#%% Plot Reynolds Normal Stress Field

os.chdir(PUBdir)

field_g_list = (M_ufuf_g, M_vfvf_g, M_wfwf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels, num_planes,
                                             num_geos, obstacles, PUBdir,
                                             save_names, 'rs_norm',
                                             cb_tick_format='%.3f')


#%% Plot Reynolds Shear Stress  Field

os.chdir(PUBdir)

field_g_list = (M_ufvf_g, M_ufwf_g, M_vfwf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{u\textquotesingle v\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{u\textquotesingle w\textquotesingle}}{U_\infty^2}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels, num_planes,
                                             num_geos, obstacles, PUBdir,
                                             save_names, 'rs_shear',
                                             cb_tick_format='%.3f')


#%% Plot Reynolds Shear Stress  Field

os.chdir(PUBdir)

field_g_list = (M_W_g)
field_labels = [
                r'$\displaystyle \frac{\overline{W}}{U_\infty}$',
                r'$\displaystyle \frac{\overline{v\textquotesingle w\textquotesingle}}{U_\infty^2}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels, num_planes,
                                             num_geos, obstacles, PUBdir,
                                             save_names, 'w_comp',
                                             cb_format='%.3f', figsize=(6.7, 4.5), extension='.jpg')

#%% Plot Shape Factor Fields

os.chdir(PUBdir)

field_g_list = (M_vfvf_ufuf_g, M_wfwf_ufuf_g, M_wfwf_vfvf_g)
field_labels = [
                r'$\displaystyle \frac{\overline{v\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{u\textquotesingle^2}}$',
                r'$\displaystyle \frac{\overline{w\textquotesingle^2}}{\overline{v\textquotesingle^2}}$'
                ]
pltf.plot_normalized_planar_field_comparison(x_g, y_g, field_g_list, M_U_g,
                                             M_V_g, field_labels, num_planes,
                                             num_geos, obstacles, PUBdir,
                                             save_names, 'shape_fact')


#%%


row, col = np.where(y_g[0][:,:,0] == 0)
if row.size == 0:
    


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
