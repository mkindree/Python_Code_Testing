#%% load packages and initialize python

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
import string

# These options make the figure text match the default LaTex font
font_size = 8.5
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Times'], 'size': font_size})

#%% load raw image

folder_path = r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\Reattachment_close_up\test_131_video_image1_C001H001S0001'
extension = '.tif'

slash_ind = folder_path[::-1].index('\\')
file_name = folder_path[-slash_ind:] + '000001'

file_path = folder_path + '\\' + file_name + extension
im = Image.open(file_path)
imarray = np.array(im)
imarray = np.fliplr(imarray).T
width = np.size(imarray, axis=0)
height = np.size(imarray, axis=1)
left_raw = -width/2
right_raw = left_raw + width
top_raw = height/2
bottom_raw = top_raw - height
extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
x_min_raw = np.fix(left_raw/100)*100
x_max_raw = np.fix(right_raw/100)*100
x_ticks_raw = np.arange(x_min_raw, x_max_raw + 100, 100)
y_min_raw = np.fix(bottom_raw/100)*100
y_max_raw = np.fix(top_raw/100)*100
y_ticks_raw = np.arange(y_min_raw, y_max_raw + 100, 100)

#%% plot raw image and image histogram to help choose colormap limits

plt.figure()
plt.subplot(111)
plt.hist(imarray.ravel(), bins=256, fc='k', ec='k')
plt.xlabel('pixel count')
plt.ylabel('number of pixels')

plt.figure()
plt.subplot(111)
plt.imshow(imarray, extent=extent_raw, cmap='binary_r', aspect='equal')
plt.xticks(x_ticks_raw)
plt.yticks(y_ticks_raw)
plt.xlabel(r'$x$ [px]')
plt.ylabel(r'$y$ [px]')
plt.grid('on')
plt.tight_layout()

#%% Adjust colormap limits, shift origin, crop image

vmin = 50
vmax = 140
shift_x = 1515 # negative moves field left
shift_y = 24 # negative moves field down
c_u = 0
c_d = 0
r_t = 100
r_b = 60

left = left_raw + shift_x + c_u
right = left + width - c_u - c_d
top = top_raw + shift_y - r_t
bottom = top - height + r_t + r_b
extent = [left, right, bottom, top]

x_min = np.fix(left/100)*100
x_max = np.fix(right/100)*100
x_ticks = np.arange(x_min, x_max + 100, 100)
y_min = np.fix(bottom/100)*100
y_max = np.fix(top/100)*100
y_ticks = np.arange(y_min, y_max + 100, 100)

plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent, cmap='binary_r', aspect='equal')
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.xlabel(r'$x$ [px]')
plt.ylabel(r'$y$ [px]', rotation='horizontal')
plt.grid('on')
plt.tight_layout()

#%% Scale image to physical nondimensional units

scale = 353 # px per diameter

left_d = left/scale
right_d = right/scale
top_d = top/scale
bottom_d = bottom/scale
extent_d = [left_d, right_d, bottom_d, top_d]

step = 0.5
x_min_d = np.fix(left_d/step)*step
x_min_d = np.ceil(left_d/step)*step #*****
x_max_d = np.fix(right_d/step)*step
x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#x_ticks_d_plus = np.hstack([x_ticks_d, -0.5, 0.5])
#x_ticks_d_plus = np.sort(x_ticks_d_plus)
y_min_d = np.fix(bottom_d/step)*step
y_max_d = np.fix(top_d/step)*step
y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#y_ticks_d_plus = np.hstack([y_ticks_d, -0.5, 0.5])
#y_ticks_d_plus = np.sort(y_ticks_d_plus)

plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
plt.xticks(x_ticks_d)
plt.yticks(y_ticks_d)
plt.xlabel(r'$\displaystyle \frac{x}{d}$')
plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
plt.grid('on')
plt.tight_layout()

#%% Create final figure

plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
plt.xticks(x_ticks_d)
plt.yticks(y_ticks_d)
plt.tick_params(top='on', right='on')
plt.xlabel(r'$\displaystyle \frac{x}{d}$')
plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
plt.tight_layout()

C_ind = file_name[::-1].index('C')
save_name = file_name[:-C_ind-2]
save_extension = '.png'

save_path = folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')

#%% Save variables

save_param_name = save_name + '_parameters2'
save_param_path = folder_path + '\\' + save_param_name + '.txt'

data = np.hstack([vmin, vmax, shift_x, shift_y, c_u, c_d, r_t, r_b, scale])
header = 'vmin, vmax, shift_x, shift_y, c_u, c_d, r_t, r_b, scale'
np.savetxt(save_param_path, data[None], delimiter=', ', fmt='%.0f',
           header=header)

#%%


#%% Overlays

obstacle_str = 'circle'
save_extension = '.png'
alpha = 0.5
i = 0
g = 0


field_g = M_U_g
cb_label = r'$U$'
save_str = 'U'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_V_g
cb_label = r'$V$'
save_str = 'V'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_W_g
cb_label = r'$W$'
save_str = 'W'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_dU_dx_g
cb_label = r'$\displaystyle \frac{\mathrm{d}U}{\mathrm{d}x}$'
save_str = 'ddx_U'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_dU_dy_g
cb_label = r'$\displaystyle \frac{\mathrm{d}U}{\mathrm{d}y}$'
save_str = 'ddy_U'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_dU_dz_g
cb_label = r'$\displaystyle \frac{\mathrm{d}U}{\mathrm{d}z}$'
save_str = 'ddz_U'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_dV_dx_g
cb_label = r'$\displaystyle \frac{\mathrm{d}V}{\mathrm{d}x}$'
save_str = 'ddx_V'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_dV_dy_g
cb_label = r'$\displaystyle \frac{\mathrm{d}V}{\mathrm{d}y}$'
save_str = 'ddy_V'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_dV_dz_g
cb_label = r'$\displaystyle \frac{\mathrm{d}V}{\mathrm{d}z}$'
save_str = 'ddz_V'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_dW_dx_g
cb_label = r'$\displaystyle \frac{\mathrm{d}W}{\mathrm{d}x}$'
save_str = 'ddx_W'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_dW_dy_g
cb_label = r'$\displaystyle \frac{\mathrm{d}W}{\mathrm{d}y}$'
save_str = 'ddy_W'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

field_g = M_dW_dz_g
cb_label = r'$\displaystyle \frac{\mathrm{d}W}{\mathrm{d}z}$'
save_str = 'ddz_W'
plt.figure()
plt.subplot(111, aspect='equal')
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  field_g[g][:, :, i], cb_label=cb_label,
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=True, num_cont_lvls=10,
                                  alpha=alpha, label_subplots=False)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

save_str = 'strm'
plt.figure()
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  M_U_g[g][:, :, i],
                                  strm_U_g=M_U_g[g][:, :, i],
                                  strm_V_g=M_V_g[g][:, :, i],
                                  obstacle=obstacles[g], plane_type='xy',
                                  obstacle_AR=4, plot_arrow=False,
                                  plot_filled_contours=False,
                                  label_subplots=False, plot_field=False,
                                  plot_streamlines=True)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

#%%
field_g = M_lambda2_g
title = r'$\lambda_2$'
save_str = 'lambda_2'
plt.figure()
plt.subplot(111, aspect='equal')
plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin, vmax=vmax,
           extent=extent_d, cmap='binary_r', aspect='equal')
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  M_lambda2_g[g][:, :, i],
                                  title=r'$\lambda_2$',
                                  obstacle=obstacles[g],
                                  plane_type='xy', obstacle_AR=4,
                                  plot_arrow=False, plot_filled_contours=False,
                                  num_cont_lvls=3, alpha=alpha,
                                  label_subplots=False, cmap_name='binary_r',
                                  plot_contour_lvls=True, add_cb=False,
                                  plot_field=False, cont_lvls=0)
save_path = folder_path + '\\' + obstacle_str + '_' + save_str + save_extension
plt.savefig(save_path, bbox_inches='tight')

#%%


#%% Copy variables for first image
imarray1 = copy.deepcopy(imarray)
r_t1 = copy.deepcopy(r_t)
height1 = copy.deepcopy(height)
r_b1 = copy.deepcopy(r_b)
c_u1 = copy.deepcopy(c_u)
width1 = copy.deepcopy(width)
c_d1 = copy.deepcopy(c_d)
vmin1 = copy.deepcopy(vmin)
vmax1 = copy.deepcopy(vmax)
extent_d1 = copy.deepcopy(extent_d)
x_ticks_d1 = copy.deepcopy(x_ticks_d)
y_ticks_d1 = copy.deepcopy(y_ticks_d)

#%% Copy variables for second image
imarray2 = copy.deepcopy(imarray)
r_t2 = copy.deepcopy(r_t)
height2 = copy.deepcopy(height)
r_b2 = copy.deepcopy(r_b)
c_u2 = copy.deepcopy(c_u)
width2 = copy.deepcopy(width)
c_d2 = copy.deepcopy(c_d)
vmin2 = copy.deepcopy(vmin)
vmax2 = copy.deepcopy(vmax)
extent_d2 = copy.deepcopy(extent_d)
x_ticks_d2 = copy.deepcopy(x_ticks_d)
y_ticks_d2 = copy.deepcopy(y_ticks_d)

#%% Create figure for paper

import string

plt.figure(figsize=(6.7, 3.1))

ax = plt.subplot(1, 2, 1, aspect='equal')
plt.imshow(imarray1[r_t1:height1-r_b1, c_u1:width1-c_d1], vmin=vmin1, vmax=vmax1,
           extent=extent_d1, cmap='binary_r', aspect='equal')
plt.xticks(x_ticks_d1)
plt.yticks(y_ticks_d1)
plt.tick_params(top='on', right='on')
#plt.xlabel(r'$\displaystyle \frac{x}{d}$')
plt.xlabel(r'$x$')
#plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
plt.ylabel(r'$y$', rotation='horizontal')

bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
transform = ax.transAxes
plt.text(0.02, 0.97,
         '(' + string.ascii_lowercase[1 - 1] + ')',
         ha='left', va='top', bbox=bbox_props,
         transform=transform)

ax = plt.subplot(1, 2, 2, aspect='equal')
plt.imshow(imarray2[r_t2:height2-r_b2, c_u2:width2-c_d2], vmin=vmin2, vmax=vmax2,
           extent=extent_d2, cmap='binary_r', aspect='equal')
plt.xticks(x_ticks_d2)
plt.yticks(y_ticks_d2)
plt.tick_params(top='on', right='on')
#plt.xlabel(r'$\displaystyle \frac{x}{d}$')
plt.xlabel(r'$x$')
#plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
plt.ylabel(r'$y$', rotation='horizontal')

bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
transform = ax.transAxes
plt.text(0.02, 0.98,
         '(' + string.ascii_lowercase[2 - 1] + ')',
         ha='left', va='top', bbox=bbox_props,
         transform=transform)


plt.tight_layout()


save_name = r'test'
save_folder_path = r'D:\EXF Paper\EXF Paper V3\figs'
save_extension = '.eps'

#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')

#%%


