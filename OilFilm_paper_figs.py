#%% load packages and initialize python

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import copy
import string
import matplotlib.gridspec as gridspec

# These options make the figure text match the default LaTex font
font_size = 8.5
plt.rc('text', usetex=True)
plt.rc('axes', **{'titlesize': 'medium'})
plt.rc('font', **{'family': 'serif', 'sans-serif': ['helvetica'],
                  'serif': ['times'], 'size': font_size})

save_folder_path = r'D:\EXF Paper\EXF Paper V4\figs'


#%%

def plot_images(axs, folder_paths, vmins, vmaxs, shift_xs, shift_ys, r_ts, r_bs,
                c_us, c_ds, scales, sub_x_label_fs, sub_y_label_fs, step=1,
                extension='.tif', fig_size=(6.7, 3.1)):
#    plt.figure(figsize=fig_size)

    ax_inds = np.arange(2)

    for (ax_ind, folder_path, vmin, vmax, shift_x, shift_y, c_u, c_d, r_t, r_b,
            scale, sub_x_label_f, sub_y_label_f) in zip(ax_inds, folder_paths,
            vmins, vmaxs, shift_xs, shift_ys, c_us, c_ds, r_ts, r_bs, scales,
            sub_x_label_fs, sub_y_label_fs):
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

        left = left_raw + shift_x + c_u
        right = left + width - c_u - c_d
        top = top_raw + shift_y - r_t
        bottom = top - height + r_t + r_b
        extent = [left, right, bottom, top]

        left_d = left/scale
        right_d = right/scale
        top_d = top/scale
        bottom_d = bottom/scale
        extent_d = [left_d, right_d, bottom_d, top_d]

        x_min_d = np.fix(left_d/step)*step
        x_min_d = np.ceil(left_d/step)*step #*****
        x_max_d = np.fix(right_d/step)*step
        x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
        y_min_d = np.fix(bottom_d/step)*step
        y_max_d = np.fix(top_d/step)*step
        y_ticks_d = np.arange(y_min_d, y_max_d + step, step)

#        ax = plt.subplot(1, 2, ax_ind + 1, aspect='equal')
        ax = axs[ax_ind]
        plt.sca(ax)
        plt.imshow(imarray[r_t:height-r_b, c_u:width-c_d], vmin=vmin,
                   vmax=vmax, extent=extent_d, cmap='binary_r', aspect='equal')
        plt.xticks(x_ticks_d)
        plt.yticks(y_ticks_d)
        plt.tick_params(top='on', right='on')
        #plt.xlabel(r'$\displaystyle \frac{x}{d}$')
        plt.xlabel(r'$x$')
        #plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
        plt.ylabel(r'$y$', rotation='horizontal')

        bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
        transform = ax.transAxes
        plt.text(sub_x_label_f, sub_y_label_f,
                 '(' + string.ascii_lowercase[ax_ind] + ')',
                 ha='left', va='top', bbox=bbox_props,
                 transform=transform)
        

#%% Plate overview tests 122, 41

folder_paths = [r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\test_122_video_image3_C001H001S0001',
                r'H:\Oil_Film\Square_plate_naturalBL_14m_s\test_41_video_image2_C001H001S0001']
vmins = [35, 20]
vmaxs = [175, 120]
shift_xs = [-8, 17] # negative moves field left
shift_ys = [-16, -11] # negative moves field down
c_us = [200, 140]
c_ds = [20, 0]
r_ts = [130, 100]
r_bs = [170, 125]
scales = [78, 86] # px per diameter
sub_x_label_fs = [0.02, 0.02]
sub_y_label_fs = [0.98, 0.98]
step = 1
fig_size=(6.7, 3.1)


plt.figure(figsize=fig_size)
ax1 = plt.subplot(1, 2, 1, aspect='equal')
ax2 = plt.subplot(1, 2, 2, aspect='equal')
axs = [ax1, ax2]
plot_images(axs, folder_paths, vmins, vmaxs, shift_xs, shift_ys, r_ts, r_bs,
            c_us, c_ds, scales, sub_x_label_fs, sub_y_label_fs, step=step,
            fig_size=fig_size)  

plt.tight_layout()

save_name = r'oil_film_plate1'
save_extension = '.eps'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')
save_extension = '.png'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')


#%% Plate overview tests 122, 41 with lambda2 overlay

folder_paths = [r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\test_122_video_image3_C001H001S0001',
                r'H:\Oil_Film\Square_plate_naturalBL_14m_s\test_41_video_image2_C001H001S0001']
vmins = [35, 20]
vmaxs = [175, 120]
shift_xs = [-8, 17] # negative moves field left
shift_ys = [-16, -11] # negative moves field down
c_us = [200, 140]
c_ds = [20, 0]
r_ts = [130, 100]
r_bs = [170, 125]
scales = [78, 86] # px per diameter
sub_x_label_fs = [0.025, 0.025]
sub_y_label_fs = [0.975, 0.975]
step = 1
fig_size=(6.7, 3.1)

plt.figure(figsize=fig_size)
ax1 = plt.subplot(1, 2, 1, aspect='equal')
ax2 = plt.subplot(1, 2, 2, aspect='equal')
axs = [ax1, ax2]

plot_images(axs, folder_paths, vmins, vmaxs, shift_xs, shift_ys, r_ts, r_bs,
            c_us, c_ds, scales, sub_x_label_fs, sub_y_label_fs, step=step,
            fig_size=fig_size)  

plt.sca(ax1)
g = 0
i = 0
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  M_lambda2_g[g][:, :, i],
                                  obstacle=obstacles[g],
                                  plane_type='xy', obstacle_AR=4,
                                  plot_arrow=False, plot_filled_contours=True,
                                  label_subplots=False, cmap_name='RdBu',
                                  plot_contour_lvls=True, add_cb=False,
                                  plot_field=False, cont_lvl_at_zero=True,
                                  num_cont_lvls=2, alpha=0.2,
                                  cont_colour=([1.0, 0.3, 0.3], [1.0, 0, 0]))

folder_path = folder_paths[0]
shift_x = shift_xs[0]
shift_y = shift_ys[0]
scale = scales[0]
r_t = r_ts[0]
r_b = r_bs[0]
c_u = c_us[0]
c_d = c_ds[0]

slash_ind = folder_path[::-1].index('\\')
file_name = folder_path[-slash_ind:] + '000001'
file_path = folder_path + '\\' + file_name + '.tif'
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
left = left_raw + shift_x + c_u
right = left + width - c_u - c_d
top = top_raw + shift_y - r_t
bottom = top - height + r_t + r_b
extent = [left, right, bottom, top]
left_d = left/scale
right_d = right/scale
top_d = top/scale
bottom_d = bottom/scale
extent_d = [left_d, right_d, bottom_d, top_d]
x_min_d = np.fix(left_d/step)*step
x_min_d = np.ceil(left_d/step)*step #*****
x_max_d = np.fix(right_d/step)*step
x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
y_min_d = np.fix(bottom_d/step)*step
y_max_d = np.fix(top_d/step)*step
y_ticks_d = np.arange(y_min_d, y_max_d + step, step)

plt.xticks(x_ticks_d)
plt.yticks(y_ticks_d)


plt.sca(ax2)
g = 1
i = 0
pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
                                  M_lambda2_g[g][:, :, i],
                                  obstacle=obstacles[g],
                                  plane_type='xy', obstacle_AR=4,
                                  plot_arrow=False, plot_filled_contours=True,
                                  label_subplots=False, cmap_name='RdBu',
                                  plot_contour_lvls=True, add_cb=False,
                                  plot_field=False, cont_lvl_at_zero=True,
                                  num_cont_lvls=2, alpha=0.3,
                                  cont_colour=([1.0, 0.3, 0.3], [1.0, 0, 0]))

folder_path = folder_paths[0]
shift_x = shift_xs[0]
shift_y = shift_ys[0]
scale = scales[0]
r_t = r_ts[0]
r_b = r_bs[0]
c_u = c_us[0]
c_d = c_ds[0]

slash_ind = folder_path[::-1].index('\\')
file_name = folder_path[-slash_ind:] + '000001'
file_path = folder_path + '\\' + file_name + '.tif'
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
left = left_raw + shift_x + c_u
right = left + width - c_u - c_d
top = top_raw + shift_y - r_t
bottom = top - height + r_t + r_b
extent = [left, right, bottom, top]
left_d = left/scale
right_d = right/scale
top_d = top/scale
bottom_d = bottom/scale
extent_d = [left_d, right_d, bottom_d, top_d]
x_min_d = np.fix(left_d/step)*step
x_min_d = np.ceil(left_d/step)*step #*****
x_max_d = np.fix(right_d/step)*step
x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
y_min_d = np.fix(bottom_d/step)*step
y_max_d = np.fix(top_d/step)*step
y_ticks_d = np.arange(y_min_d, y_max_d + step, step)

plt.xticks(x_ticks_d)
plt.yticks(y_ticks_d)


plt.tight_layout()



save_name = r'oil_film_plate_lambda2'
save_extension = '.eps'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')
save_extension = '.png'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')








#%% Plate overview 2 tests 9, 28

folder_paths = [r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\test_9_C001H001S0001',
               r'H:\Oil_Film\Square_plate_naturalBL_14m_s\test_28_C001H001S0001']
vmins = [50, 75]
vmaxs = [240, 256]
shift_xs = [9, 30] # negative moves field left
shift_ys = [-2, -11] # negative moves field down
c_us = [150, 125]
c_ds = [0, 20]
r_ts = [120, 105]
r_bs = [120, 130]
scales = [87, 88] # px per diameter
sub_x_label_fs = [0.025, 0.025]
sub_y_label_fs = [0.975, 0.975]
step = 1
fig_size=(6.7, 3.1)

fig = plt.figure(figsize=fig_size)
ax1 = plt.subplot(1, 2, 1, aspect='equal')
ax2 = plt.subplot(1, 2, 2, aspect='equal')
axs = [ax1, ax2]
plot_images(axs, folder_paths, vmins, vmaxs, shift_xs, shift_ys, r_ts, r_bs,
            c_us, c_ds, scales, sub_x_label_fs, sub_y_label_fs, step=step,
            fig_size=fig_size) 

plt.tight_layout()


ax_ind = 0
text = r'$\mathbf{S_{p1}}$'
x_coord = -3
y_coord = 0
t_x_coord = -3
t_y_coord = 1.4
a_x_coord = -3.3
a_y_coord = 1.4
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='right', va='bottom', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.07, a_h_width=8, a_width=3)

ax_ind = 0
text = r'$\mathbf{F_{p1}}$'
x_coord = 1.8
y_coord = 0.3
t_x_coord = 2
t_y_coord = 1.3
a_x_coord = 2
a_y_coord = 1.3
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='bottom', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.07, a_h_width=8, a_width=3)

ax_ind = 0
text = r'$\mathbf{F_{p2}}$'
x_coord = 1.8
y_coord = -0.3
t_x_coord = 2
t_y_coord = -1.3
a_x_coord = 2
a_y_coord = -1.2
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.12, a_h_width=8, a_width=3)

ax_ind = 0
text = r'$\mathbf{S_{p2}}$'
x_coord = 0.89
y_coord = 0
t_x_coord = 0.85
t_y_coord = 1
a_x_coord = 0.85
a_y_coord = 0.7
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='center', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=7, a_width=3)

ax_ind = 0
text = r'$\mathbf{S_{p3}}$'
x_coord = 4.28
y_coord = 0.1
t_x_coord = 4.28
t_y_coord = 1.4
a_x_coord = 4.28
a_y_coord = 1.4
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='bottom', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=7, a_width=3)

ax_ind = 0
text = r'$\mathbf{A}$'
x_coord = -1.65
y_coord = 0
t_x_coord = -1.9
t_y_coord = 0.2
a_x_coord = -1.9
a_y_coord = 0.2
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='right', va='bottom', ws=1.3, hs=1.3, lsf=-0.2,
                   bsf=0, a_alpha=0, a_shrink=0.07, a_h_width=8, a_width=3)


ax_ind = 1
text = r'$\mathbf{S_{p1}}$'
x_coord = -3.67
y_coord = 0
t_x_coord = -3.5
t_y_coord = 2.4
a_x_coord = -3.5
a_y_coord = 2.4
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='bottom', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.07, a_h_width=8, a_width=3)

ax_ind = 1
text = r'$\mathbf{A}$'
x_coord = -2.1
y_coord = 0
t_x_coord = -2.5
t_y_coord = 0.3
a_x_coord = -2.5
a_y_coord = 0.3
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='bottom', ws=1.3, hs=1.3, lsf=-0.15,
                   bsf=0, a_alpha=0, a_shrink=0.07, a_h_width=8, a_width=3)

ax_ind = 1
text = r'$\mathbf{B}$'
x_coord = 1.22
y_coord = 0
t_x_coord = 1.22
t_y_coord = 1.5
a_x_coord = 1.22
a_y_coord = 1.5
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='bottom', ws=1.4, hs=1.3, lsf=-0.2,
                   bsf=0, a_alpha=1.0, a_shrink=0.07, a_h_width=8, a_width=3)

ax_ind = 1
text = r'$\mathbf{N_{p1}}$'
x_coord = 3.84
y_coord = 0
t_x_coord = 4.1
t_y_coord = 1.5
a_x_coord = 4.5
a_y_coord = 1.5
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='left', va='bottom', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=7, a_width=3)

ax_ind = 1
text = r'$\mathbf{S_{p2}}$'
x_coord = 3.55
y_coord = 0.43
t_x_coord = 3.4
t_y_coord = 1.7
a_x_coord = 3.4
a_y_coord = 1.7
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='bottom', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=7, a_width=3)

ax_ind = 1
text = r'$\mathbf{S_{p3}}$'
x_coord = 3.55
y_coord = -0.43
t_x_coord = 3.4
t_y_coord = -1.7
a_x_coord = 3.4
a_y_coord = -1.7
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.1, a_h_width=7, a_width=3)


save_name = r'oil_film_plate2'
save_extension = '.eps'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')
save_extension = '.png'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')


#%% Obstacle close up tests 114, 58

folder_paths = [r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\Obstacle_close_up\test_114_video_image1_C001H001S0001',
               r'H:\Oil_Film\Square_plate_naturalBL_14m_s\Obstacle_close_up\test_58_video_image2_C001H001S0001']
vmins = [20, 8]
vmaxs = [85, 59]
shift_xs = [-10, -20] # negative moves field left
shift_ys = [0, 22] # negative moves field down
c_us = [400, 300]
c_ds = [130, 0]
r_ts = [350, 210]
r_bs = [350, 160]
scales = [132, 262] # px per diameter
sub_x_label_fs = [0.025, 0.03]
sub_y_label_fs = [0.96, 0.96]
step = 0.5
fig_size=(5.25, 2.2)

gs = gridspec.GridSpec(1, 2, width_ratios=[1.35, 1])
fig = plt.figure(figsize=fig_size)
ax1 = plt.subplot(gs[0], aspect='equal')
ax2 = plt.subplot(gs[1], aspect='equal')
#ax1 = plt.subplot(1, 2, 1, aspect='equal')
#ax2 = plt.subplot(1, 2, 2, aspect='equal')
axs = [ax1, ax2]
plot_images(axs, folder_paths, vmins, vmaxs, shift_xs, shift_ys, r_ts, r_bs,
            c_us, c_ds, scales, sub_x_label_fs, sub_y_label_fs, step=step,
            fig_size=fig_size)  

plt.tight_layout()


ax_ind = 0
text = r'$\mathbf{F_{p1}}$'
x_coord = 1.8
y_coord = 0.3
t_x_coord = 1.75
t_y_coord = 0.89#0.15
a_x_coord = 1.65
a_y_coord = 0.89
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='right', va='center', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=0, a_shrink=0.07, a_h_width=8, a_width=3)

ax_ind = 0
text = r'$\mathbf{F_{p2}}$'
x_coord = 1.8
y_coord = -0.3
t_x_coord = 1.75
t_y_coord = -0.89#-0.23
a_x_coord = 1.65
a_y_coord = -0.89
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='right', va='center', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=0, a_shrink=0.07, a_h_width=8, a_width=3)

ax_ind = 0
text = r'$\mathbf{S_{p2}}$'
x_coord = 0.89
y_coord = 0
t_x_coord = 0.75
t_y_coord = -0.79
a_x_coord = 0.63
a_y_coord = -0.79
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='right', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.15, a_h_width=7, a_width=3)

#ax_ind = 0
#text = r'$\mathbf{N\textquotesingle_{p1}}$'
#x_coord = -0.5
#y_coord = 0
#t_x_coord = -0.35#-0.25#-0
#t_y_coord = -0.79#0.79#0.35
#a_x_coord = -0.55#0#-0.25
#a_y_coord = -0.79#0.79#0.25
#pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
#                   t_y_coord, a_x_coord, a_y_coord,
#                   ha='right', va='top', ws=1.3, hs=1.3, lsf=-0.13,
#                   bsf=0, a_alpha=1, a_shrink=0.15, a_h_width=7, a_width=3)
#
#ax_ind = 0
#text = r'$\mathbf{N\textquotesingle_{p2}}$'
#x_coord = 0.5
#y_coord = 0
#t_x_coord = 0.25#0
#t_y_coord = -0.79#-0.35
#a_x_coord = 0.05#0.25
#a_y_coord = -0.79#-0.25
#pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
#                   t_y_coord, a_x_coord, a_y_coord,
#                   ha='right', va='top', ws=1.3, hs=1.3, lsf=-0.13,
#                   bsf=0, a_alpha=1, a_shrink=0.15, a_h_width=7, a_width=3)


ax_ind = 1
text = r'$\mathbf{B}$'
x_coord = 1.55
y_coord = 0
t_x_coord = 1.55#-0.25#-0
t_y_coord = 0.65#0.79#0.35
a_x_coord = 1.55#0#-0.25
a_y_coord = 0.65#0.79#0.25
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='top', ws=1.4, hs=1.3, lsf=-0.2,
                   bsf=0, a_alpha=0, a_shrink=0.15, a_h_width=7, a_width=3)

#ax_ind = 1
#text = r'$\mathbf{N\textquotesingle_{p1}}$'
#x_coord = -0.5
#y_coord = 0
#t_x_coord = -0.4
#t_y_coord = 0.79
#a_x_coord = -0.2
#a_y_coord = 0.79
#pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
#                   t_y_coord, a_x_coord, a_y_coord,
#                   ha='left', va='bottom', ws=1.3, hs=1.3, lsf=-0.13,
#                   bsf=0, a_alpha=1, a_shrink=0.15, a_h_width=7, a_width=3)
#
#ax_ind = 1
#text = r'$\mathbf{N\textquotesingle_{p2}}$'
#x_coord = 0.5
#y_coord = 0
#t_x_coord = 0.55
#t_y_coord = 0.79
#a_x_coord = 0.35
#a_y_coord = 0.79
#pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
#                   t_y_coord, a_x_coord, a_y_coord,
#                   ha='right', va='bottom', ws=1.3, hs=1.3, lsf=-0.13,
#                   bsf=0, a_alpha=1, a_shrink=0.15, a_h_width=7, a_width=3)

save_name = r'oil_film_obstacle_close_up1'
save_extension = '.eps'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')
save_extension = '.png'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')


#%% Plate close up tests 131, 53

folder_paths = [r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\Reattachment_close_up\test_131_video_image1_C001H001S0001',
               r'H:\Oil_Film\Square_plate_naturalBL_14m_s\Reattachment_close_up\test_53_video_image1_C001H001S0001']
vmins = [50, 8]
vmaxs = [140, 56]
shift_xs = [1515, 1262] # negative moves field left
shift_ys = [24, 12] # negative moves field down
c_us = [0, 0]
c_ds = [0, 0]
r_ts = [100, 85]
r_bs = [60, 50]
scales = [353, 356] # px per diameter
sub_x_label_fs = [0.03, 0.03]
sub_y_label_fs = [0.97, 0.97]
step = 0.5
fig_size=(5.25, 2.5)

fig = plt.figure(figsize=fig_size)
ax1 = plt.subplot(1, 2, 1, aspect='equal')
ax2 = plt.subplot(1, 2, 2, aspect='equal')
axs = [ax1, ax2]
plot_images(axs, folder_paths, vmins, vmaxs, shift_xs, shift_ys, r_ts, r_bs,
            c_us, c_ds, scales, sub_x_label_fs, sub_y_label_fs, step=step,
            fig_size=fig_size)   

plt.tight_layout()

ax_ind = 0
text = r'$\mathbf{S_{p3}}$'
x_coord = 4.28
y_coord = 0
t_x_coord = 4.4
t_y_coord = -0.7
a_x_coord = 4.4
a_y_coord = -0.7
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.15, a_h_width=7, a_width=3)


ax_ind = 1
text = r'$\mathbf{S_{p2}}$'
x_coord = 3.55
y_coord = 0.43
t_x_coord = 3.6
t_y_coord = 0.9
a_x_coord = 3.6
a_y_coord = 0.9
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='bottom', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.15, a_h_width=7, a_width=3)

ax_ind = 1
text = r'$\mathbf{S_{p3}}$'
x_coord = 3.55
y_coord = -0.43
t_x_coord = 3.45
t_y_coord = -0.9
a_x_coord = 3.45
a_y_coord = -0.9
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='center', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.15, a_h_width=7, a_width=3)

ax_ind = 1
text = r'$\mathbf{N_{p1}}$'
x_coord = 3.84
y_coord = 0
t_x_coord = 4.25
t_y_coord = -0.9
a_x_coord = 4.1
a_y_coord = -0.9
pltf.annotate_plot(axs[ax_ind], fig, text, x_coord, y_coord, t_x_coord,
                   t_y_coord, a_x_coord, a_y_coord,
                   ha='right', va='top', ws=1.3, hs=1.3, lsf=-0.13,
                   bsf=0, a_alpha=1, a_shrink=0.15, a_h_width=7, a_width=3)


save_name = r'oil_film_plate_close_up1'
save_extension = '.eps'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')
save_extension = '.png'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')


#%% Plate close up tests 2 131, 42

folder_paths = [r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\Reattachment_close_up\test_131_video_image1_C001H001S0001',
               r'H:\Oil_Film\Square_plate_naturalBL_14m_s\Reattachment_close_up\test_42_video_image1_C001H001S0001']
vmins = [50, 0]
vmaxs = [140, 256]
shift_xs = [1515, 550] # negative moves field left
shift_ys = [24, -28] # negative moves field down
c_us = [0, 330]
c_ds = [0, 270]
r_ts = [100, 310]
r_bs = [60, 365]
scales = [353, 140] # px per diameter
step = 0.5
sub_x_label_fs = [0.02, 0.02]
sub_y_label_fs = [0.98, 0.98]
fig_size=(6.7, 3.1)

plt.figure(figsize=fig_size)
ax1 = plt.subplot(1, 2, 1, aspect='equal')
ax2 = plt.subplot(1, 2, 2, aspect='equal')
axs = [ax1, ax2]
plot_images(axs, folder_paths, vmins, vmaxs, shift_xs, shift_ys, r_ts, r_bs,
            c_us, c_ds, scales, sub_x_label_fs, sub_y_label_fs, step=step,
            fig_size=fig_size) 

plt.tight_layout()



save_name = r'oil_film_plate_close_up2'
save_extension = '.eps'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')
save_extension = '.png'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')


#%% Free end tests 2 108, 73

folder_paths = [r'H:\Oil_Film\Circle_free_end_naturalBL_14m_s\test_108_video_image1_C001H001S0001',
               r'H:\Oil_Film\Square_free_end_naturalBL_14m_s\test_73_video_image3_C001H001S0001']
vmins = [20, 20]
vmaxs = [100, 256]
shift_xs = [-8,  -17]# negative moves field left
shift_ys = [67, 28] # negative moves field down
c_us = [340, 345]
c_ds = [325, 311]
r_ts = [400, 356]
r_bs = [266, 300]
scales = [345, 352] # px per diameter
step = 0.25
sub_x_label_fs = [0.02, 0.02]
sub_y_label_fs = [0.98, 0.98]
fig_size=(5.25, 2.5)

plt.figure(figsize=fig_size)
ax1 = plt.subplot(1, 2, 1, aspect='equal')
ax2 = plt.subplot(1, 2, 2, aspect='equal')
axs = [ax1, ax2]
plot_images(axs, folder_paths, vmins, vmaxs, shift_xs, shift_ys, r_ts, r_bs,
            c_us, c_ds, scales, sub_x_label_fs, sub_y_label_fs, step=step,
            fig_size=fig_size) 

plt.tight_layout()



save_name = r'oil_film_free_end1'
save_extension = '.eps'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')
save_extension = '.png'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')


#%% Free end tests 2 108, 76

folder_paths = [r'H:\Oil_Film\Circle_free_end_naturalBL_14m_s\test_108_video_image1_C001H001S0001',
               r'H:\Oil_Film\Square_free_end_naturalBL_14m_s\test_76_video_image4_C001H001S0001']
vmins = [20, 20]
vmaxs = [100, 256]
shift_xs = [-8, -17] # negative moves field left
shift_ys = [67, 24] # negative moves field down
c_us = [340, 345]
c_ds = [325, 311]
r_ts = [400, 352]
r_bs = [266, 305]
scales = [345, 352] # px per diameter
step = 0.25
sub_x_label_fs = [0.02, 0.02]
sub_y_label_fs = [0.98, 0.98]
fig_size=(6.7, 3.1)

plt.figure(figsize=fig_size)
ax1 = plt.subplot(1, 2, 1, aspect='equal')
ax2 = plt.subplot(1, 2, 2, aspect='equal')
axs = [ax1, ax2]
plot_images(axs, folder_paths, vmins, vmaxs, shift_xs, shift_ys, r_ts, r_bs,
            c_us, c_ds, scales, sub_x_label_fs, sub_y_label_fs, step=step,
            fig_size=fig_size) 

plt.tight_layout()



save_name = r'oil_film_free_end2'
save_extension = '.eps'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')
save_extension = '.png'
save_path = save_folder_path + '\\' + save_name + save_extension
plt.savefig(save_path, bbox_inches='tight')


#%%


##%% Plate overview tests 122, 41
#
#folder_path = r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\test_122_video_image3_C001H001S0001'
#extension = '.tif'
#vmin = 35
#vmax = 175
#shift_x = -8 # negative moves field left
#shift_y = -16 # negative moves field down
#c_u = 200
#c_d = 20
#r_t = 130
#r_b = 170
#scale = 78 # px per diameter
#step = 1
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray1 = copy.deepcopy(imarray)
#r_t1 = copy.deepcopy(r_t)
#height1 = copy.deepcopy(height)
#r_b1 = copy.deepcopy(r_b)
#c_u1 = copy.deepcopy(c_u)
#width1 = copy.deepcopy(width)
#c_d1 = copy.deepcopy(c_d)
#vmin1 = copy.deepcopy(vmin)
#vmax1 = copy.deepcopy(vmax)
#extent_d1 = copy.deepcopy(extent_d)
#x_ticks_d1 = copy.deepcopy(x_ticks_d)
#y_ticks_d1 = copy.deepcopy(y_ticks_d)
#
#folder_path = r'H:\Oil_Film\Square_plate_naturalBL_14m_s\test_41_video_image2_C001H001S0001'
#extension = '.tif'
#vmin = 20
#vmax = 120
#shift_x = 17 # negative moves field left
#shift_y = -11 # negative moves field down
#c_u = 140
#c_d = 0
#r_t = 100
#r_b = 125
#scale = 86 # px per diameter
#step = 1
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray2 = copy.deepcopy(imarray)
#r_t2 = copy.deepcopy(r_t)
#height2 = copy.deepcopy(height)
#r_b2 = copy.deepcopy(r_b)
#c_u2 = copy.deepcopy(c_u)
#width2 = copy.deepcopy(width)
#c_d2 = copy.deepcopy(c_d)
#vmin2 = copy.deepcopy(vmin)
#vmax2 = copy.deepcopy(vmax)
#extent_d2 = copy.deepcopy(extent_d)
#x_ticks_d2 = copy.deepcopy(x_ticks_d)
#y_ticks_d2 = copy.deepcopy(y_ticks_d)
#
#plt.figure(figsize=(6.7, 3.1))
#
#ax = plt.subplot(1, 2, 1, aspect='equal')
#plt.imshow(imarray1[r_t1:height1-r_b1, c_u1:width1-c_d1], vmin=vmin1, vmax=vmax1,
#           extent=extent_d1, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d1)
#plt.yticks(y_ticks_d1)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.97,
#         '(' + string.ascii_lowercase[1 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#ax = plt.subplot(1, 2, 2, aspect='equal')
#plt.imshow(imarray2[r_t2:height2-r_b2, c_u2:width2-c_d2], vmin=vmin2, vmax=vmax2,
#           extent=extent_d2, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d2)
#plt.yticks(y_ticks_d2)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.98,
#         '(' + string.ascii_lowercase[2 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#
#plt.tight_layout()
#
#save_name = r'oil_film_plate1'
#save_extension = '.eps'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#save_extension = '.png'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#
##%% Plate overview tests 122, 41 with lambda2 overlay
#
#folder_path = r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\test_122_video_image3_C001H001S0001'
#extension = '.tif'
#vmin = 35
#vmax = 175
#shift_x = -8 # negative moves field left
#shift_y = -16 # negative moves field down
#c_u = 200
#c_d = 20
#r_t = 130
#r_b = 170
#scale = 78 # px per diameter
#step = 1
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray1 = copy.deepcopy(imarray)
#r_t1 = copy.deepcopy(r_t)
#height1 = copy.deepcopy(height)
#r_b1 = copy.deepcopy(r_b)
#c_u1 = copy.deepcopy(c_u)
#width1 = copy.deepcopy(width)
#c_d1 = copy.deepcopy(c_d)
#vmin1 = copy.deepcopy(vmin)
#vmax1 = copy.deepcopy(vmax)
#extent_d1 = copy.deepcopy(extent_d)
#x_ticks_d1 = copy.deepcopy(x_ticks_d)
#y_ticks_d1 = copy.deepcopy(y_ticks_d)
#
#folder_path = r'H:\Oil_Film\Square_plate_naturalBL_14m_s\test_41_video_image2_C001H001S0001'
#extension = '.tif'
#vmin = 20
#vmax = 120
#shift_x = 17 # negative moves field left
#shift_y = -11 # negative moves field down
#c_u = 140
#c_d = 0
#r_t = 100
#r_b = 125
#scale = 86 # px per diameter
#step = 1
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray2 = copy.deepcopy(imarray)
#r_t2 = copy.deepcopy(r_t)
#height2 = copy.deepcopy(height)
#r_b2 = copy.deepcopy(r_b)
#c_u2 = copy.deepcopy(c_u)
#width2 = copy.deepcopy(width)
#c_d2 = copy.deepcopy(c_d)
#vmin2 = copy.deepcopy(vmin)
#vmax2 = copy.deepcopy(vmax)
#extent_d2 = copy.deepcopy(extent_d)
#x_ticks_d2 = copy.deepcopy(x_ticks_d)
#y_ticks_d2 = copy.deepcopy(y_ticks_d)
#
#plt.figure(figsize=(6.7, 3.1))
#
#ax = plt.subplot(1, 2, 1, aspect='equal')
#plt.imshow(imarray1[r_t1:height1-r_b1, c_u1:width1-c_d1], vmin=vmin1, vmax=vmax1,
#           extent=extent_d1, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d1)
#plt.yticks(y_ticks_d1)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.97,
#         '(' + string.ascii_lowercase[1 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#g = 0
#i = 0
#pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
#                                  M_lambda2_g[g][:, :, i],
#                                  obstacle=obstacles[g],
#                                  plane_type='xy', obstacle_AR=4,
#                                  plot_arrow=False, plot_filled_contours=True,
#                                  label_subplots=False, cmap_name='RdBu',
#                                  plot_contour_lvls=True, add_cb=False,
#                                  plot_field=False, cont_lvl_at_zero=True,
#                                  num_cont_lvls=2, alpha=0.2,
#                                  cont_colour=([1.0, 0.3, 0.3], [1.0, 0, 0]))
#plt.yticks(y_ticks_d1)
#plt.xticks(x_ticks_d1)
#
#ax = plt.subplot(1, 2, 2, aspect='equal')
#plt.imshow(imarray2[r_t2:height2-r_b2, c_u2:width2-c_d2], vmin=vmin2, vmax=vmax2,
#           extent=extent_d2, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d2)
#plt.yticks(y_ticks_d2)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.98,
#         '(' + string.ascii_lowercase[2 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#g = 1
#i = 0
#pltf.plot_normalized_planar_field(x_g[g][:, :, i], y_g[g][:, :, i],
#                                  M_lambda2_g[g][:, :, i],
#                                  obstacle=obstacles[g],
#                                  plane_type='xy', obstacle_AR=4,
#                                  plot_arrow=False, plot_filled_contours=True,
#                                  label_subplots=False, cmap_name='RdBu',
#                                  plot_contour_lvls=True, add_cb=False,
#                                  plot_field=False, cont_lvl_at_zero=True,
#                                  num_cont_lvls=2, alpha=0.3,
#                                  cont_colour=([1.0, 0.3, 0.3], [1.0, 0, 0]))
#plt.yticks(y_ticks_d2)
#plt.xticks(x_ticks_d2)
#
#
#plt.tight_layout()
#
#save_name = r'oil_film_plate_lambda2'
#save_extension = '.eps'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#save_extension = '.png'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#
#
##%% Plate oil film Lambda2 Overlay
#
#
#
##%% Plate overview 2 tests 9, 28
#
#folder_path = r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\test_9_C001H001S0001'
#extension = '.tif'
#vmin = 50
#vmax = 240
#shift_x = 9 # negative moves field left
#shift_y = -2 # negative moves field down
#c_u = 150
#c_d = 0
#r_t = 120
#r_b = 120
#scale = 87 # px per diameter
#step = 1
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray1 = copy.deepcopy(imarray)
#r_t1 = copy.deepcopy(r_t)
#height1 = copy.deepcopy(height)
#r_b1 = copy.deepcopy(r_b)
#c_u1 = copy.deepcopy(c_u)
#width1 = copy.deepcopy(width)
#c_d1 = copy.deepcopy(c_d)
#vmin1 = copy.deepcopy(vmin)
#vmax1 = copy.deepcopy(vmax)
#extent_d1 = copy.deepcopy(extent_d)
#x_ticks_d1 = copy.deepcopy(x_ticks_d)
#y_ticks_d1 = copy.deepcopy(y_ticks_d)
#
#
#folder_path = r'H:\Oil_Film\Square_plate_naturalBL_14m_s\test_28_C001H001S0001'
#extension = '.tif'
#vmin = 75
#vmax = 256
#shift_x = 30 # negative moves field left
#shift_y = -11 # negative moves field down
#c_u = 125
#c_d = 20
#r_t = 105
#r_b = 130
#scale = 88 # px per diameter
#step = 1
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray2 = copy.deepcopy(imarray)
#r_t2 = copy.deepcopy(r_t)
#height2 = copy.deepcopy(height)
#r_b2 = copy.deepcopy(r_b)
#c_u2 = copy.deepcopy(c_u)
#width2 = copy.deepcopy(width)
#c_d2 = copy.deepcopy(c_d)
#vmin2 = copy.deepcopy(vmin)
#vmax2 = copy.deepcopy(vmax)
#extent_d2 = copy.deepcopy(extent_d)
#x_ticks_d2 = copy.deepcopy(x_ticks_d)
#y_ticks_d2 = copy.deepcopy(y_ticks_d)
#
#plt.figure(figsize=(6.7, 3.1))
#
#ax = plt.subplot(1, 2, 1, aspect='equal')
#plt.imshow(imarray1[r_t1:height1-r_b1, c_u1:width1-c_d1], vmin=vmin1, vmax=vmax1,
#           extent=extent_d1, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d1)
#plt.yticks(y_ticks_d1)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.97,
#         '(' + string.ascii_lowercase[1 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#ax = plt.subplot(1, 2, 2, aspect='equal')
#plt.imshow(imarray2[r_t2:height2-r_b2, c_u2:width2-c_d2], vmin=vmin2, vmax=vmax2,
#           extent=extent_d2, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d2)
#plt.yticks(y_ticks_d2)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.98,
#         '(' + string.ascii_lowercase[2 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#
#plt.tight_layout()
#
#save_name = r'oil_film_plate2'
#save_extension = '.eps'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#save_extension = '.png'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#
##%% Obstacle close up tests 114, 58
#
#folder_path = r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\Obstacle_close_up\test_114_video_image1_C001H001S0001'
#extension = '.tif'
#vmin = 20
#vmax = 85
#shift_x = -10 # negative moves field left
#shift_y = 0 # negative moves field down
#c_u = 400
#c_d = 130
#r_t = 350
#r_b = 350
#scale = 132 # px per diameter
#step = 0.5
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray1 = copy.deepcopy(imarray)
#r_t1 = copy.deepcopy(r_t)
#height1 = copy.deepcopy(height)
#r_b1 = copy.deepcopy(r_b)
#c_u1 = copy.deepcopy(c_u)
#width1 = copy.deepcopy(width)
#c_d1 = copy.deepcopy(c_d)
#vmin1 = copy.deepcopy(vmin)
#vmax1 = copy.deepcopy(vmax)
#extent_d1 = copy.deepcopy(extent_d)
#x_ticks_d1 = copy.deepcopy(x_ticks_d)
#y_ticks_d1 = copy.deepcopy(y_ticks_d)
#
#folder_path = r'H:\Oil_Film\Square_plate_naturalBL_14m_s\Obstacle_close_up\test_58_video_image2_C001H001S0001'
#extension = '.tif'
#vmin = 8
#vmax = 59
#shift_x = -20 # negative moves field left
#shift_y = 22 # negative moves field down
#c_u = 300
#c_d = 0
#r_t = 210
#r_b = 160
#scale = 262 # px per diameter
#step = 0.5
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray2 = copy.deepcopy(imarray)
#r_t2 = copy.deepcopy(r_t)
#height2 = copy.deepcopy(height)
#r_b2 = copy.deepcopy(r_b)
#c_u2 = copy.deepcopy(c_u)
#width2 = copy.deepcopy(width)
#c_d2 = copy.deepcopy(c_d)
#vmin2 = copy.deepcopy(vmin)
#vmax2 = copy.deepcopy(vmax)
#extent_d2 = copy.deepcopy(extent_d)
#x_ticks_d2 = copy.deepcopy(x_ticks_d)
#y_ticks_d2 = copy.deepcopy(y_ticks_d)
#
#plt.figure(figsize=(5.25, 2.2))
#gs = gridspec.GridSpec(1, 2, width_ratios=[1.35, 1])
#
#ax = plt.subplot(gs[0], aspect='equal')
##ax = plt.subplot(1, 2, 1, aspect='equal')
#plt.imshow(imarray1[r_t1:height1-r_b1, c_u1:width1-c_d1], vmin=vmin1, vmax=vmax1,
#           extent=extent_d1, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d1)
#plt.yticks(y_ticks_d1)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.97,
#         '(' + string.ascii_lowercase[1 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#ax = plt.subplot(gs[1], aspect='equal')
##ax = plt.subplot(1, 2, 2, aspect='equal')
#plt.imshow(imarray2[r_t2:height2-r_b2, c_u2:width2-c_d2], vmin=vmin2, vmax=vmax2,
#           extent=extent_d2, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d2)
#plt.yticks(y_ticks_d2)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.03, 0.97,
#         '(' + string.ascii_lowercase[2 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#
#plt.tight_layout()
#
#save_name = r'oil_film_obstacle_close_up1'
#save_extension = '.eps'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#save_extension = '.png'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#
#
##%% Obstacle close up tests 114, 58
#
#folder_path = r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\Obstacle_close_up\test_114_video_image1_C001H001S0001'
#extension = '.tif'
#vmin = 20
#vmax = 85
#shift_x = -10 # negative moves field left
#shift_y = 0 # negative moves field down
#c_u = 400
#c_d = 130
#r_t = 350
#r_b = 350
#scale = 132 # px per diameter
#step = 0.5
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray1 = copy.deepcopy(imarray)
#r_t1 = copy.deepcopy(r_t)
#height1 = copy.deepcopy(height)
#r_b1 = copy.deepcopy(r_b)
#c_u1 = copy.deepcopy(c_u)
#width1 = copy.deepcopy(width)
#c_d1 = copy.deepcopy(c_d)
#vmin1 = copy.deepcopy(vmin)
#vmax1 = copy.deepcopy(vmax)
#extent_d1 = copy.deepcopy(extent_d)
#x_ticks_d1 = copy.deepcopy(x_ticks_d)
#y_ticks_d1 = copy.deepcopy(y_ticks_d)
#
#folder_path = r'H:\Oil_Film\Square_plate_naturalBL_14m_s\Obstacle_close_up\test_58_video_image3_C001H001S0001'
#extension = '.tif'
#vmin = 8
#vmax = 59
#shift_x = -28 # negative moves field left
#shift_y = 24 # negative moves field down
#c_u = 300
#c_d = 0
#r_t = 210
#r_b = 160
#scale = 262 # px per diameter
#step = 0.5
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray2 = copy.deepcopy(imarray)
#r_t2 = copy.deepcopy(r_t)
#height2 = copy.deepcopy(height)
#r_b2 = copy.deepcopy(r_b)
#c_u2 = copy.deepcopy(c_u)
#width2 = copy.deepcopy(width)
#c_d2 = copy.deepcopy(c_d)
#vmin2 = copy.deepcopy(vmin)
#vmax2 = copy.deepcopy(vmax)
#extent_d2 = copy.deepcopy(extent_d)
#x_ticks_d2 = copy.deepcopy(x_ticks_d)
#y_ticks_d2 = copy.deepcopy(y_ticks_d)
#
#plt.figure(figsize=(6.7, 3.1))
#
#ax = plt.subplot(1, 2, 1, aspect='equal')
#plt.imshow(imarray1[r_t1:height1-r_b1, c_u1:width1-c_d1], vmin=vmin1, vmax=vmax1,
#           extent=extent_d1, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d1)
#plt.yticks(y_ticks_d1)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.97,
#         '(' + string.ascii_lowercase[1 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#ax = plt.subplot(1, 2, 2, aspect='equal')
#plt.imshow(imarray2[r_t2:height2-r_b2, c_u2:width2-c_d2], vmin=vmin2, vmax=vmax2,
#           extent=extent_d2, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d2)
#plt.yticks(y_ticks_d2)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.98,
#         '(' + string.ascii_lowercase[2 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#
#plt.tight_layout()
#
#save_name = r'oil_film_obstacle_close_up2'
#save_extension = '.eps'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#save_extension = '.png'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#
##%% Plate close up tests 131, 53
#
#folder_path = r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\Reattachment_close_up\test_131_video_image1_C001H001S0001'
#extension = '.tif'
#vmin = 50
#vmax = 140
#shift_x = 1515 # negative moves field left
#shift_y = 24 # negative moves field down
#c_u = 0
#c_d = 0
#r_t = 100
#r_b = 60
#scale = 353 # px per diameter
#step = 0.5
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray1 = copy.deepcopy(imarray)
#r_t1 = copy.deepcopy(r_t)
#height1 = copy.deepcopy(height)
#r_b1 = copy.deepcopy(r_b)
#c_u1 = copy.deepcopy(c_u)
#width1 = copy.deepcopy(width)
#c_d1 = copy.deepcopy(c_d)
#vmin1 = copy.deepcopy(vmin)
#vmax1 = copy.deepcopy(vmax)
#extent_d1 = copy.deepcopy(extent_d)
#x_ticks_d1 = copy.deepcopy(x_ticks_d)
#y_ticks_d1 = copy.deepcopy(y_ticks_d)
#
#folder_path = r'H:\Oil_Film\Square_plate_naturalBL_14m_s\Reattachment_close_up\test_53_video_image1_C001H001S0001'
#extension = '.tif'
#vmin = 8
#vmax = 56
#shift_x = 1262 # negative moves field left
#shift_y = 12 # negative moves field down
#c_u = 0
#c_d = 0
#r_t = 85
#r_b = 50
#scale = 356 # px per diameter
#step = 0.5
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray2 = copy.deepcopy(imarray)
#r_t2 = copy.deepcopy(r_t)
#height2 = copy.deepcopy(height)
#r_b2 = copy.deepcopy(r_b)
#c_u2 = copy.deepcopy(c_u)
#width2 = copy.deepcopy(width)
#c_d2 = copy.deepcopy(c_d)
#vmin2 = copy.deepcopy(vmin)
#vmax2 = copy.deepcopy(vmax)
#extent_d2 = copy.deepcopy(extent_d)
#x_ticks_d2 = copy.deepcopy(x_ticks_d)
#y_ticks_d2 = copy.deepcopy(y_ticks_d)
#
#plt.figure(figsize=(5.25, 2.5))
#
#ax = plt.subplot(1, 2, 1, aspect='equal')
#plt.imshow(imarray1[r_t1:height1-r_b1, c_u1:width1-c_d1], vmin=vmin1, vmax=vmax1,
#           extent=extent_d1, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d1)
#plt.yticks(y_ticks_d1)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.03, 0.97,
#         '(' + string.ascii_lowercase[1 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#ax = plt.subplot(1, 2, 2, aspect='equal')
#plt.imshow(imarray2[r_t2:height2-r_b2, c_u2:width2-c_d2], vmin=vmin2, vmax=vmax2,
#           extent=extent_d2, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d2)
#plt.yticks(y_ticks_d2)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.03, 0.97,
#         '(' + string.ascii_lowercase[2 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#
#plt.tight_layout()
#
#save_name = r'oil_film_plate_close_up1'
#save_extension = '.eps'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#save_extension = '.png'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#
##%% Plate close up tests 2 131, 42
#
#folder_path = r'H:\Oil_Film\Circle_plate_naturalBL_14m_s\Reattachment_close_up\test_131_video_image1_C001H001S0001'
#extension = '.tif'
#vmin = 50
#vmax = 140
#shift_x = 1515 # negative moves field left
#shift_y = 24 # negative moves field down
#c_u = 0
#c_d = 0
#r_t = 100
#r_b = 60
#scale = 353 # px per diameter
#step = 0.5
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray1 = copy.deepcopy(imarray)
#r_t1 = copy.deepcopy(r_t)
#height1 = copy.deepcopy(height)
#r_b1 = copy.deepcopy(r_b)
#c_u1 = copy.deepcopy(c_u)
#width1 = copy.deepcopy(width)
#c_d1 = copy.deepcopy(c_d)
#vmin1 = copy.deepcopy(vmin)
#vmax1 = copy.deepcopy(vmax)
#extent_d1 = copy.deepcopy(extent_d)
#x_ticks_d1 = copy.deepcopy(x_ticks_d)
#y_ticks_d1 = copy.deepcopy(y_ticks_d)
#
#folder_path = r'H:\Oil_Film\Square_plate_naturalBL_14m_s\Reattachment_close_up\test_42_video_image1_C001H001S0001'
#extension = '.tif'
#vmin = 0
#vmax = 256
#shift_x = 550 # negative moves field left
#shift_y = -28 # negative moves field down
#c_u = 330
#c_d = 270
#r_t = 310
#r_b = 365
#scale = 140 # px per diameter
#step = 0.5
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray2 = copy.deepcopy(imarray)
#r_t2 = copy.deepcopy(r_t)
#height2 = copy.deepcopy(height)
#r_b2 = copy.deepcopy(r_b)
#c_u2 = copy.deepcopy(c_u)
#width2 = copy.deepcopy(width)
#c_d2 = copy.deepcopy(c_d)
#vmin2 = copy.deepcopy(vmin)
#vmax2 = copy.deepcopy(vmax)
#extent_d2 = copy.deepcopy(extent_d)
#x_ticks_d2 = copy.deepcopy(x_ticks_d)
#y_ticks_d2 = copy.deepcopy(y_ticks_d)
#
#plt.figure(figsize=(6.7, 3.1))
#
#ax = plt.subplot(1, 2, 1, aspect='equal')
#plt.imshow(imarray1[r_t1:height1-r_b1, c_u1:width1-c_d1], vmin=vmin1, vmax=vmax1,
#           extent=extent_d1, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d1)
#plt.yticks(y_ticks_d1)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.97,
#         '(' + string.ascii_lowercase[1 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#ax = plt.subplot(1, 2, 2, aspect='equal')
#plt.imshow(imarray2[r_t2:height2-r_b2, c_u2:width2-c_d2], vmin=vmin2, vmax=vmax2,
#           extent=extent_d2, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d2)
#plt.yticks(y_ticks_d2)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.98,
#         '(' + string.ascii_lowercase[2 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#
#plt.tight_layout()
#
#save_name = r'oil_film_plate_close_up2'
#save_extension = '.eps'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#save_extension = '.png'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#
#
##%% Free end tests 2 108, 73
#
#folder_path = r'H:\Oil_Film\Circle_free_end_naturalBL_14m_s\test_108_video_image1_C001H001S0001'
#extension = '.tif'
#vmin = 20
#vmax = 100
#shift_x = -8 # negative moves field left
#shift_y = 67 # negative moves field down
#c_u = 340
#c_d = 325
#r_t = 400
#r_b = 266
#scale = 345 # px per diameter
#step = 0.25
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray1 = copy.deepcopy(imarray)
#r_t1 = copy.deepcopy(r_t)
#height1 = copy.deepcopy(height)
#r_b1 = copy.deepcopy(r_b)
#c_u1 = copy.deepcopy(c_u)
#width1 = copy.deepcopy(width)
#c_d1 = copy.deepcopy(c_d)
#vmin1 = copy.deepcopy(vmin)
#vmax1 = copy.deepcopy(vmax)
#extent_d1 = copy.deepcopy(extent_d)
#x_ticks_d1 = copy.deepcopy(x_ticks_d)
#y_ticks_d1 = copy.deepcopy(y_ticks_d)
#
#folder_path = r'H:\Oil_Film\Square_free_end_naturalBL_14m_s\test_73_video_image3_C001H001S0001'
#extension = '.tif'
#vmin = 20
#vmax = 256
#shift_x = -17 # negative moves field left
#shift_y = 28 # negative moves field down
#c_u = 345
#c_d = 311
#r_t = 356
#r_b = 300
#scale = 352 # px per diameter
#step = 0.25
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray2 = copy.deepcopy(imarray)
#r_t2 = copy.deepcopy(r_t)
#height2 = copy.deepcopy(height)
#r_b2 = copy.deepcopy(r_b)
#c_u2 = copy.deepcopy(c_u)
#width2 = copy.deepcopy(width)
#c_d2 = copy.deepcopy(c_d)
#vmin2 = copy.deepcopy(vmin)
#vmax2 = copy.deepcopy(vmax)
#extent_d2 = copy.deepcopy(extent_d)
#x_ticks_d2 = copy.deepcopy(x_ticks_d)
#y_ticks_d2 = copy.deepcopy(y_ticks_d)
#
#plt.figure(figsize=(5.25, 2.5))
#
#ax = plt.subplot(1, 2, 1, aspect='equal')
#plt.imshow(imarray1[r_t1:height1-r_b1, c_u1:width1-c_d1], vmin=vmin1, vmax=vmax1,
#           extent=extent_d1, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d1)
#plt.yticks(y_ticks_d1)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.03, 0.97,
#         '(' + string.ascii_lowercase[1 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#ax = plt.subplot(1, 2, 2, aspect='equal')
#plt.imshow(imarray2[r_t2:height2-r_b2, c_u2:width2-c_d2], vmin=vmin2, vmax=vmax2,
#           extent=extent_d2, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d2)
#plt.yticks(y_ticks_d2)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.03, 0.97,
#         '(' + string.ascii_lowercase[2 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#
#plt.tight_layout()
#
#save_name = r'oil_film_free_end1'
#save_extension = '.eps'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#save_extension = '.png'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#
#
##%% Free end tests 2 108, 76
#
#folder_path = r'H:\Oil_Film\Circle_free_end_naturalBL_14m_s\test_108_video_image1_C001H001S0001'
#extension = '.tif'
#vmin = 20
#vmax = 100
#shift_x = -8 # negative moves field left
#shift_y = 67 # negative moves field down
#c_u = 340
#c_d = 325
#r_t = 400
#r_b = 266
#scale = 345 # px per diameter
#step = 0.25
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray1 = copy.deepcopy(imarray)
#r_t1 = copy.deepcopy(r_t)
#height1 = copy.deepcopy(height)
#r_b1 = copy.deepcopy(r_b)
#c_u1 = copy.deepcopy(c_u)
#width1 = copy.deepcopy(width)
#c_d1 = copy.deepcopy(c_d)
#vmin1 = copy.deepcopy(vmin)
#vmax1 = copy.deepcopy(vmax)
#extent_d1 = copy.deepcopy(extent_d)
#x_ticks_d1 = copy.deepcopy(x_ticks_d)
#y_ticks_d1 = copy.deepcopy(y_ticks_d)
#
#
#folder_path = r'H:\Oil_Film\Square_free_end_naturalBL_14m_s\test_76_video_image4_C001H001S0001'
#extension = '.tif'
#vmin = 20
#vmax = 256
#shift_x = -17 # negative moves field left
#shift_y = 24 # negative moves field down
#c_u = 345
#c_d = 311
#r_t = 352
#r_b = 305
#scale = 352 # px per diameter
#step = 0.25
#
#slash_ind = folder_path[::-1].index('\\')
#file_name = folder_path[-slash_ind:] + '000001'
#file_path = folder_path + '\\' + file_name + extension
#im = Image.open(file_path)
#imarray = np.array(im)
#imarray = np.fliplr(imarray).T
#
#width = np.size(imarray, axis=0)
#height = np.size(imarray, axis=1)
#left_raw = -width/2
#right_raw = left_raw + width
#top_raw = height/2
#bottom_raw = top_raw - height
#extent_raw = [left_raw, right_raw, bottom_raw, top_raw]
#
#left = left_raw + shift_x + c_u
#right = left + width - c_u - c_d
#top = top_raw + shift_y - r_t
#bottom = top - height + r_t + r_b
#extent = [left, right, bottom, top]
#
#left_d = left/scale
#right_d = right/scale
#top_d = top/scale
#bottom_d = bottom/scale
#extent_d = [left_d, right_d, bottom_d, top_d]
#
#x_min_d = np.fix(left_d/step)*step
#x_min_d = np.ceil(left_d/step)*step #*****
#x_max_d = np.fix(right_d/step)*step
#x_ticks_d = np.arange(x_min_d, x_max_d + step, step)
#y_min_d = np.fix(bottom_d/step)*step
#y_max_d = np.fix(top_d/step)*step
#y_ticks_d = np.arange(y_min_d, y_max_d + step, step)
#
#imarray2 = copy.deepcopy(imarray)
#r_t2 = copy.deepcopy(r_t)
#height2 = copy.deepcopy(height)
#r_b2 = copy.deepcopy(r_b)
#c_u2 = copy.deepcopy(c_u)
#width2 = copy.deepcopy(width)
#c_d2 = copy.deepcopy(c_d)
#vmin2 = copy.deepcopy(vmin)
#vmax2 = copy.deepcopy(vmax)
#extent_d2 = copy.deepcopy(extent_d)
#x_ticks_d2 = copy.deepcopy(x_ticks_d)
#y_ticks_d2 = copy.deepcopy(y_ticks_d)
#
#plt.figure(figsize=(6.7, 3.1))
#
#ax = plt.subplot(1, 2, 1, aspect='equal')
#plt.imshow(imarray1[r_t1:height1-r_b1, c_u1:width1-c_d1], vmin=vmin1, vmax=vmax1,
#           extent=extent_d1, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d1)
#plt.yticks(y_ticks_d1)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.97,
#         '(' + string.ascii_lowercase[1 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#ax = plt.subplot(1, 2, 2, aspect='equal')
#plt.imshow(imarray2[r_t2:height2-r_b2, c_u2:width2-c_d2], vmin=vmin2, vmax=vmax2,
#           extent=extent_d2, cmap='binary_r', aspect='equal')
#plt.xticks(x_ticks_d2)
#plt.yticks(y_ticks_d2)
#plt.tick_params(top='on', right='on')
##plt.xlabel(r'$\displaystyle \frac{x}{d}$')
#plt.xlabel(r'$x$')
##plt.ylabel(r'$\displaystyle \frac{y}{d}$', rotation='horizontal')
#plt.ylabel(r'$y$', rotation='horizontal')
#
#bbox_props = {'pad': 2, 'fc': 'w', 'ec': 'w', 'alpha': 1.0}
#transform = ax.transAxes
#plt.text(0.02, 0.98,
#         '(' + string.ascii_lowercase[2 - 1] + ')',
#         ha='left', va='top', bbox=bbox_props,
#         transform=transform)
#
#
#plt.tight_layout()
#
#
#save_name = r'oil_film_free_end2'
#save_extension = '.eps'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#save_extension = '.png'
#save_path = save_folder_path + '\\' + save_name + save_extension
#plt.savefig(save_path, bbox_inches='tight')
#
