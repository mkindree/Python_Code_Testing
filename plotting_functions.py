from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.text as mtext
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib import animation
import matplotlib.transforms as transforms
import string
import copy

import plotting_functions_strm as pltfs

# These options make the figure text match the default LaTex font
font_size = 8.5
plt.rc('text', usetex=True)
plt.rc('axes', **{'titlesize': 'medium'})
plt.rc('font', **{'family': 'serif', 'sans-serif': ['helvetica'],
                  'serif': ['Times'], 'size': font_size})
#plt.rc('mathtext', **{'fontset': 'dejavuserif'})
# times


###############################################################################

def plot_normalized_planar_field(x_g, y_g, field_g, **kwargs):
    cb_label = kwargs.pop('cb_label', '')
    title = kwargs.pop('title', '')
    clim = kwargs.pop('clim', None)
    plane_type = kwargs.pop('plane_type', 'xy')
    label_x = kwargs.pop('label_x', True)
    label_y = kwargs.pop('label_y', True)
    label_subplots = kwargs.pop('label_subplots', True)
    add_cb = kwargs.pop('add_cb', True)
    subplot_ind = kwargs.pop('subplot_ind', 1)
    smooth_colourplot = kwargs.pop('smooth_colourplot', False)
    plot_field = kwargs.pop('plot_field', True)
    plot_streamlines = kwargs.pop('plot_streamlines', False)
    plot_vectors = kwargs.pop('plot_vectors', False)
    strm_U_g = kwargs.pop('strm_U_g', None)
    strm_V_g = kwargs.pop('strm_V_g', None)
    strm_x_g = kwargs.pop('strm_x_g', None)
    strm_y_g = kwargs.pop('strm_y_g', None)
    x_min = kwargs.pop('x_min', None)
    x_max = kwargs.pop('x_max', None)
    y_min = kwargs.pop('y_min', None)
    y_max = kwargs.pop('y_max', None)
    plot_obstacle = kwargs.pop('plot_obstacle', True)
    obstacle = kwargs.pop('obstacle', 'circle')
    obstacle_AR = kwargs.pop('obstacle_AR', 1)
    plot_arrow = kwargs.pop('plot_arrow', True)
    plot_contour_lvls = kwargs.pop('plot_contour_lvls', False)
    cont_lvls = kwargs.pop('cont_lvls', None)
    num_cont_lvls = kwargs.pop('num_cont_lvls', 8)
    cont_lvl_at_zero = kwargs.pop('cont_lvl_at_zero', False)
    cont_lvl_field_g = kwargs.pop('cont_lvl_field_g', None)
    cont_colour = kwargs.pop('cont_colour', 'g')
    plot_filled_contours = kwargs.pop('plot_filled_contours', False)
    cont_lw = kwargs.pop('cont_lw', 1.5)
    cmap_name = kwargs.pop('cmap_name', 'RdBu')
    alpha_cmap = kwargs.pop('alpha_cmap', False)
    alpha_scale = kwargs.pop('alpha_scale', 1.0)
    alpha = kwargs.pop('alpha', 1.0)
    plot_bifurcation_lines = kwargs.pop('plot_bifurcation_lines', False)
    # notice that there is a bug in matplotlib so the bifurcation line does not
    # actually go through the point, please iterate
    # now it should work, need to be newer than matplotlib 2.0
    bifurc_seed_pt_xy = kwargs.pop('bifurc_seed_pt_xy', None)
    bifurc_lw = kwargs.pop('bifurc_lw', 1.5)
    bifurc_colour = kwargs.pop('bifurc_colour', 'g')
    bifurc_arrowsize = kwargs.pop('bifurc_arrowsize', 0.7)
    bifurc_num_arrow_heads = kwargs.pop('bifurc_num_arrow_heads', 2.0)
    title_pad = kwargs.pop('title_pad', None)
    cb_label_0 = kwargs.pop('cb_label_0', True)
    cb_label_1 = kwargs.pop('cb_label_1', False)
    num_extra_cticks = kwargs.pop('num_extra_cticks', 0)
    x_subplot_label = kwargs.pop('x_subplot_label', 0.05)
    y_subplot_label = kwargs.pop('y_subplot_label', 0.95)
    subplot_label_coord = kwargs.pop('subplot_label_coord', 'axes')
    colour_subplot_label = kwargs.pop('colour_subplot_label', True)
    tick_space = kwargs.pop('tick_space', 1.0)
    contrast_strm = kwargs.pop('contrast_strm', False)
    strm_contrast = kwargs.pop('strm_contrast', 'high')
    strm_colour = kwargs.pop('strm_colour', [0.0]*3)
    cmap_greys_name = kwargs.pop('cmap_greys_name', 'binary')
    vect_scale = kwargs.pop('vect_scale', 10.0)
    vect_res = kwargs.pop('vect_res', 1.0)
    strm_dens = kwargs.pop('strm_dens', [0.5, 0.5])
    strm_lw = kwargs.pop('strm_lw', 0.5)
    strm_arrowsize = kwargs.pop('strm_arrowsize', 0.7)
    strm_max_len = kwargs.pop('strm_max_len', 4.0)
    strm_min_len = kwargs.pop('strm_min_len', 0.1)
    strm_num_arrow_heads = kwargs.pop('strm_num_arrow_heads', 2.0)
    arrow_colour = kwargs.pop('arrow_colour', [0.7]*3)
    arrow_scale = kwargs.pop('arrow_scale', 20.0)
    arrow_xlims = kwargs.pop('arrow_xlims', [-2.0, -0.75])
    stretch_cb = kwargs.pop('stretch_cb', False)
    halve_cb = kwargs.pop('halve_cb', False)
    ctick_maxs = kwargs.pop('ctick_maxs', True)
    cb_frac = kwargs.pop('cb_frac', 0.03)
    cb_ar = kwargs.pop('cb_ar', 23.0)
    cb_pad = kwargs.pop('cb_pad', 0.02)
    cb_tick_format = kwargs.pop('cb_tick_format', '%.2f')
    cb_label_pad = kwargs.pop('cb_label_pad', 10.0)
    x_label_pad = kwargs.pop('x_label_pad', 3.0)
    y_label_pad = kwargs.pop('y_label_pad', 6.0)
    mark_max = kwargs.pop('mark_max', False)
    max_mark_symb = kwargs.pop('max_mark_symb', '*')
    max_mark_colour = kwargs.pop('max_mark_colour', 'y')
    max_mark_size = kwargs.pop('max_mark_size', 7.0)
    max_mark_edge_width = kwargs.pop('max_mark_edge_width', 0.5)
    mark_min = kwargs.pop('mark_min', False)
    min_mark_symb = kwargs.pop('min_mark_symb', '*')
    min_mark_colour = kwargs.pop('min_mark_colour', 'y')
    min_mark_size = kwargs.pop('min_mark_size', 7.0)
    min_mark_edge_width = kwargs.pop('min_mark_edge_width', 0.5)
    additional_marker_coords = kwargs.pop('additional_marker_coords', None)
    add_mark_symb = kwargs.pop('add_mark_symb', '*')
    add_mark_colour = kwargs.pop('add_mark_colour', 'y')
    add_mark_size = kwargs.pop('add_mark_size', 7.0)
    add_mark_edge_width = kwargs.pop('add_mark_edge_width', 0.5)

    if np.size(kwargs.keys()) >= 1:
        print 'Unknown kwargs: ', kwargs.keys()

    if not plot_field:
        add_cb = False
    xy_plane_strs = ['xy', 'yx', 'XY', 'YX']
    xz_plane_strs = ['xz', 'zx', 'XZ', 'ZX']
    yz_plane_strs = ['yz', 'zy', 'YZ', 'ZY']
    plane_type_error_str = \
        'Please select either ''xy'', ''xz'', or ''yz'' for plane_type'

    fig = plt.gcf()
    ax = plt.gca()

    # Determining where to place the x limit min        
#    if not x_subplot_label:
    if plane_type in xy_plane_strs:
        if plot_obstacle and plot_arrow:
#                x_subplot_label = (arrow_xlims[0] + arrow_xlims[1])/2
            if not x_min:
                x_min = arrow_xlims[0] - 0.1
            if not y_max:
                y_max = y_g.max()
        elif plot_obstacle:
#            x_subplot_label = 0
            if not x_min:
                x_min = -0.75
            if not y_max:
                y_max = y_g.max()
        elif plot_arrow:
#                x_subplot_label = (arrow_xlims[0] + arrow_xlims[1])/2
            if not x_min:
                x_min = arrow_xlims[0] - 0.1
            if not y_max:
                y_max = y_g.max()
        else:
#                x_subplot_label = x_g.min() + 0.25
            if not x_min:
                x_min = x_g.min()
            if not y_max:
                y_max = y_g.max()
    elif plane_type in xz_plane_strs:
        if plot_obstacle and plot_arrow:
#                x_subplot_label = (arrow_xlims[0] + arrow_xlims[1])/2
            if not x_min:
                x_min = arrow_xlims[0] - 0.1
            if not y_max:
                y_max = y_g.max()
        elif plot_obstacle:
#                x_subplot_label = -0.25
            if label_subplots:
                if not x_min:
                    x_min = -0.75
                if not y_max:
                    y_max = 4.8
            else:
                if not x_min:
                    x_min = -0.75
                if not y_max:
                    y_max = y_g.max()

        elif plot_arrow:
#                x_subplot_label = (arrow_xlims[0] + arrow_xlims[1])/2
            if not x_min:
                x_min = arrow_xlims[0] - 0.1
            if not y_max:
                y_max = y_g.max()
        else:
#                x_subplot_label = x_g.min() + 0.25
            if not x_min:
                x_min = x_g.min()
            if not y_max:
                y_max = y_g.max()
    elif plane_type in yz_plane_strs:
#            x_subplot_label = x_g.min() + 0.25
        if not x_min:
            x_min = x_g.min()
        if not y_max:
            y_max = y_g.max()
    else:
        print plane_type_error_str
    if not x_max:
        x_max = x_g.max()
    if plane_type in xz_plane_strs or plane_type in yz_plane_strs:
        if not y_min:
            y_min = -0.1
        y_plate_min = -0.3
    else:
        if not y_min:
            y_min = y_g.min()
#    if not y_subplot_label:
#        if plane_type in ['xy', 'yx', 'XY', 'YX']:
#            if plot_obstacle and plot_arrow:
#                y_subplot_label = 2
#            elif plot_obstacle:
#                y_subplot_label = 2
#            elif plot_arrow:
#                y_subplot_label = 2
#            else:
#                y_subplot_label = y_g.max() + 0.25
#        elif plane_type in ['xz', 'zx', 'XZ', 'ZX']:
#            if plot_obstacle and plot_arrow:
#                y_subplot_label = obstacle_AR*3/4
#            elif plot_obstacle:
#                y_subplot_label = y_g.max() + 0.25
#            elif plot_arrow:
#                y_subplot_label = obstacle_AR*3/4
#            else:
#                y_subplot_label = y_g.max() + 0.25
#        elif plane_type in ['yz', 'zy', 'YZ', 'ZY']:
#            y_subplot_label = y_g.max() + 0.25
#        else:
#            print 'Please select either ''xy'', ''xz'', or ''yz'' for plane_type' 

    # Plotting obstacle geometry
    if plot_obstacle:
        if obstacle in ['circle', 'Circle']:
            if plane_type in xy_plane_strs:
                ax.add_patch(patches.Circle((0.0, 0.0), radius=0.5, fc='k'))
            elif plane_type in xz_plane_strs:
                ax.add_patch(patches.Rectangle((-0.5, 0.0), 1.0, obstacle_AR,
                                               fc='k', zorder=0))
                ax.add_patch(patches.Rectangle((x_min, y_plate_min),
                                               x_g.max() - x_min,
                                               np.abs(y_plate_min), fc='k', ec='k'))
            elif plane_type in yz_plane_strs:
                ax.add_patch(patches.Rectangle((-0.5, 0.0), 1.0, obstacle_AR,
                                               fc=(1.0, 1.0, 1.0, 0.0), ec='k',
                                               lw=2.5, zorder=0))
                ax.add_patch(patches.Rectangle((x_g.min(), y_plate_min),
                                               x_g.max() - x_g.min(),
                                               np.abs(y_plate_min), fc='k', ec='k'))
            else:
                print plane_type_error_str    
        elif obstacle in ['square', 'Square']:
            if plane_type in xy_plane_strs:
                ax.add_patch(patches.Rectangle((-0.5, -0.5), 1.0, 1.0, fc='k'))
            elif plane_type in xz_plane_strs:
                ax.add_patch(patches.Rectangle((-0.5, 0.0), 1.0, obstacle_AR,
                                               fc='k', zorder=0))
                ax.add_patch(patches.Rectangle((x_min, y_plate_min),
                                               x_g.max() - x_min,
                                               np.abs(y_plate_min), fc='k', ec='k'))
            elif plane_type in yz_plane_strs:
                ax.add_patch(patches.Rectangle((-0.5, 0.0), 1.0, obstacle_AR,
                                               fc=(1.0, 1.0, 1.0, 0.0), ec='k',
                                               lw=2.5, zorder=0))
                ax.add_patch(patches.Rectangle((x_g.min(), y_plate_min),
                                               x_g.max() - x_g.min(),
                                               np.abs(y_plate_min), fc='k', ec='k'))
            else:
                print plane_type_error_str   
        elif obstacle == 'hollow circle':
            ax.add_patch(patches.Circle((0.0, 0.0), radius=0.5, ec='k',
                                        fc=(1.0, 1.0, 1.0, 0.0), lw=2.5))
        elif obstacle == 'hollow square':
            ax.add_patch(patches.Rectangle((-0.5, -0.5), 1.0, 1.0, ec='k',
                                           fc=(1.0, 1.0, 1.0, 0.0), lw=2.5))
        else:
            print 'Add functionality'

    # Plotting an arrow showing approaching flow direction
    if plot_arrow:
        if plane_type in xy_plane_strs:
            ax.add_patch(patches.FancyArrowPatch(
                (arrow_xlims[0], 0.0), (arrow_xlims[1], 0.0),
                arrowstyle='simple', fc=arrow_colour, ec=arrow_colour,
                mutation_scale=arrow_scale))
        elif plane_type in xz_plane_strs:
            ax.add_patch(patches.FancyArrowPatch(
                (arrow_xlims[0], obstacle_AR/2.0),
                (arrow_xlims[1], obstacle_AR/2.0), arrowstyle='simple',
                fc=arrow_colour, ec=arrow_colour, mutation_scale=arrow_scale))
        elif plane_type in yz_plane_strs:
            pass
        else:
            print plane_type_error_str

    # Setting the colour limits
    if not clim:
        clim = np.max(np.abs([field_g.max(), field_g.min()]))
    cmap = plt.get_cmap(cmap_name)
    if alpha_cmap:
        cmap_N = cmap.N
        cmap_alpha = cmap(np.arange(cmap.N))
        cmap_alpha[:, -1] = [(1 - np.sum(cmap_alpha[row, 0:-1])/3)*alpha_scale
                             for row in np.arange(cmap_N)]
        for row in np.arange(cmap_N):
            if cmap_alpha[row, -1] > 1:
                cmap_alpha[row, -1] = 1
            elif cmap_alpha[row, -1] < 0:
                cmap_alpha[row, -1] = 0
        cmap = LinearSegmentedColormap.from_list('cmap_alpha', cmap_alpha)
    if not num_cont_lvls % 2 == 0:
        num_cont_lvls = np.floor(num_cont_lvls/2)*2

    # Choosing the colour map and plotting the field
    if smooth_colourplot:
        smoothing = 'gouraud'
    else:
        smoothing = 'flat'
    if not stretch_cb and not halve_cb:

        clim_min = -clim
        clim_max = clim
        if cont_lvls is None:
            if cont_lvl_at_zero:
                corr_zero_cont_lvl = 1
            else:
                corr_zero_cont_lvl = 0
            cont_lvls = np.linspace(clim_min, clim_max,
                                    num_cont_lvls + corr_zero_cont_lvl)
        if plot_filled_contours:
            if cont_lvl_field_g is None:
                cont_lvl_field_g = field_g
            if plot_contour_lvls:
                cont_lvl_plt = plt.contour(x_g, y_g, cont_lvl_field_g,
                                           levels=cont_lvls,
                                           colors=cont_colour,
                                           linewidths=cont_lw, zorder=10)
            if alpha != 1.0:
                zorder = 9
            else:
                zorder = 0
            if alpha_cmap:
                cont_plt = plt.contourf(x_g, y_g, field_g, levels=cont_lvls,
                                        cmap=cmap, vmin=-clim, vmax=clim,
                                        zorder=1)
            else:
                cont_plt = plt.contourf(x_g, y_g, field_g, levels=cont_lvls,
                                        cmap=cmap, vmin=-clim, vmax=clim,
                                        alpha=alpha, zorder=zorder)

        else:
            if plot_contour_lvls:
                if cont_lvl_field_g is None:
                    cont_lvl_field_g = field_g
                cont_lvl_plt = plt.contour(x_g, y_g, cont_lvl_field_g,
                                           levels=cont_lvls,
                                           colors=cont_colour,
                                           linewidths=cont_lw, zorder=1)
            if plot_field:
                col_plt = plt.pcolormesh(x_g, y_g, field_g, cmap=cmap,
                                         vmin=-clim, vmax=clim,
                                         shading=smoothing, alpha=alpha)
    if stretch_cb:
        clim_min = field_g.min()
        clim_max = field_g.max()
        if cont_lvls is None:
            if cont_lvl_at_zero:
                corr_zero_cont_lvl = 1.0
            else:
                corr_zero_cont_lvl = 0.0
            cont_lvls = np.linspace(clim_min, clim_max,
                                    num_cont_lvls + corr_zero_cont_lvl)
        cmap_trunc_range = np.linspace(field_g.min()/clim,
                                       field_g.max()/clim)/2 + 0.5
        cmap_colour_trunc = cmap(cmap_trunc_range)
        cmap_trunc = LinearSegmentedColormap.from_list('cmap_trunc',
                                                       cmap_colour_trunc)
        if plot_filled_contours:
            if cont_lvl_field_g is None:
                cont_lvl_field_g = field_g
            if plot_contour_lvls:
                cont_lvl_plt = plt.contour(x_g, y_g, cont_lvl_field_g,
                                           levels=cont_lvls,
                                           colors=cont_colour,
                                           linewidths=cont_lw, zorder=10)
            if alpha != 1.0:
                zorder = 9
            else:
                zorder = 0
            cont_plt = plt.contourf(x_g, y_g, field_g, levels=cont_lvls,
                                    cmap=cmap_trunc, alpha=alpha,
                                    zorder=zorder)
        else:
            if plot_contour_lvls:
                if cont_lvl_field_g is None:
                    cont_lvl_field_g = field_g
                cont_lvl_plt = plt.contour(x_g, y_g, cont_lvl_field_g,
                                           levels=cont_lvls,
                                           colors=cont_colour,
                                           linewidths=cont_lw, zorder=10)
            if plot_field:
                col_plt = plt.pcolormesh(x_g, y_g, field_g, cmap=cmap_trunc,
                                         shading=smoothing, alpha=alpha)
    elif halve_cb:
        clim_min = -clim
        clim_max = clim
        if round(field_g.min(), int(cb_tick_format[-2])) >= 0:
            clim_min = 0.0
        if round(field_g.max(), int(cb_tick_format[-2])) <= 0:
            clim_max = 0.0
        if cont_lvls is None:
            if cont_lvl_at_zero:
                corr_zero_cont_lvl = 1.0
            else:
                corr_zero_cont_lvl = 0.0
            cont_lvls = np.linspace(clim_min, clim_max,
                                    num_cont_lvls + corr_zero_cont_lvl)
        cmap_trunc_range = np.linspace(clim_min/clim, clim_max/clim)/2 + 0.5
        cmap_colour_trunc = cmap(cmap_trunc_range)
        cmap_trunc = LinearSegmentedColormap.from_list('cmap_trunc',
                                                       cmap_colour_trunc)
        if plot_filled_contours:
            if cont_lvl_field_g is None:
                cont_lvl_field_g = field_g
            if plot_contour_lvls:
                cont_lvl_plt = plt.contour(x_g, y_g, cont_lvl_field_g,
                                           levels=cont_lvls,
                                           colors=cont_colour,
                                           linewidths=cont_lw, zorder=10)
            if alpha != 1.0:
                zorder = 9
            else:
                zorder = 0
            cont_plt = plt.contourf(x_g, y_g, field_g, levels=cont_lvls,
                                    cmap=cmap_trunc, vmin=clim_min,
                                    vmax=clim_max, alpha=alpha, zorder=zorder)
        else:
            if plot_contour_lvls:
                if cont_lvl_field_g is None:
                    cont_lvl_field_g = field_g
                cont_lvl_plt = plt.contour(x_g, y_g, cont_lvl_field_g,
                                           levels=cont_lvls,
                                           colors=cont_colour,
                                           linewidths=cont_lw, zorder=10)
            if plot_field:
                col_plt = plt.pcolormesh(x_g, y_g, field_g, cmap=cmap_trunc,
                                         vmin=clim_min, vmax=clim_max,
                                         shading=smoothing, alpha=alpha)

    # Adding a colour bar
    if ctick_maxs:
        cticks = [field_g.min(), field_g.max()]
    else:
        cticks = [clim_min, clim_max]
    cticks = np.linspace(cticks[0], cticks[1], 2+num_extra_cticks)
    if cb_label_1:
        if clim > 1.0 and 1.0 not in \
                [round(ctick, int(cb_tick_format[-2])) for ctick in cticks]:
            cticks = np.append(cticks, 1.0)
            cticks = np.unique(cticks)
            cticks.sort()
    if cb_label_0:
        if field_g.max()*field_g.min() < 0.0 and 0.0 not in \
                [round(ctick, int(cb_tick_format[-2])) for ctick in cticks]:
            cticks = np.append(cticks, 0.0)
            cticks = np.unique(cticks)
            cticks.sort()
    if add_cb:
        cb = plt.colorbar(ticks=cticks, format=cb_tick_format,
                          fraction=cb_frac, pad=cb_pad, aspect=cb_ar)
        cb.set_alpha(1)
        cb.draw_all()
#        if plot_contour_lvls or plot_filled_contours:
#            if cont_lvl_field_g.all() == field_g.all():
#                cb.add_lines(cont_lvl_plt)
        if cb_label:
            cb.ax.set_ylabel(cb_label, rotation='horizontal', va='center',
                             ha='center')
            cb.ax.yaxis.labelpad = cb_label_pad
        else:
            cb.ax.yaxis.labelpad = 0.01

    # Plotting streamlines or vectors
    if plot_streamlines or plot_vectors:
        if strm_x_g is None:
            strm_x_g = x_g
        if strm_y_g is None:
            strm_y_g = y_g
        if strm_U_g is not None and strm_V_g is not None:
            if contrast_strm:
                cmap_greys_f = plt.get_cmap(cmap_greys_name)
                cmap_greys_r = plt.get_cmap(cmap_greys_name + '_r')
                cmap_trunc_range_f = np.linspace(0.0,
                                                 np.abs(field_g.max()/clim))
                cmap_trunc_range_r = np.linspace(0.0,
                                                 np.abs(field_g.min()/clim))
                cmap_greys_colours_f = cmap_greys_f(cmap_trunc_range_f)
                cmap_greys_colours_r = cmap_greys_r(cmap_trunc_range_r)
                if strm_contrast == 'high':
                    cmap_greys_colours = np.vstack([cmap_greys_colours_f,
                                                    cmap_greys_colours_r])
                if strm_contrast == 'low':
                    cmap_greys_colours = np.vstack([cmap_greys_colours_r,
                                                    cmap_greys_colours_f])
                cmap_greys = LinearSegmentedColormap.from_list(
                        'cmap_greys', cmap_greys_colours)
                if plot_streamlines:
                    strm_plt = pltfs.streamplot(ax, strm_x_g[0, :],
                                                strm_y_g[:, 0], strm_U_g,
                                                strm_V_g, color=field_g,
                                                cmap=cmap_greys,
                                                density=strm_dens,
                                                linewidth=strm_lw,
                                                arrowsize=strm_arrowsize,
                                                maxlength=strm_max_len,
                                                minlength=strm_min_len,
                                                num_arrow_heads=strm_num_arrow_heads)
                if plot_vectors:
                    vect_plt = pltfs.quiver(strm_x_g[0, 0:-1:vect_res],
                                            strm_y_g[0:-1:vect_res, 0],
                                            strm_U_g[0:-1:vect_res, 0:-1:vect_res],
                                            strm_V_g[0:-1:vect_res, 0:-1:vect_res],
                                            color=field_g[0:-1:vect_res, 0:-1:vect_res],
                                            cmap=cmap_greys, scale=vect_scale)
            else:
                if plot_streamlines:
                    strm_plt = pltfs.streamplot(ax, strm_x_g[0, :],
                                                strm_y_g[:, 0], strm_U_g,
                                                strm_V_g, color=strm_colour,
                                                density=strm_dens,
                                                linewidth=strm_lw,
                                                arrowsize=strm_arrowsize,
                                                maxlength=strm_max_len,
                                                minlength=strm_min_len,
                                                num_arrow_heads=strm_num_arrow_heads)
                if plot_vectors:
                    vect_plt = plt.quiver(strm_x_g[0, 0:-1:vect_res],
                                          strm_y_g[0:-1:vect_res, 0],
                                          strm_U_g[0:-1:vect_res, 0:-1:vect_res],
                                          strm_V_g[0:-1:vect_res, 0:-1:vect_res],
                                          color=strm_colour, scale=vect_scale)
        else:
            print 'Need to provide strm_U_g and strm_V_g for streamlines'
    if plot_bifurcation_lines:
        if strm_x_g is None:
            strm_x_g = x_g
        if strm_y_g is None:
            strm_y_g = y_g
        if strm_U_g is not None and strm_V_g is not None:
            if bifurc_seed_pt_xy is not None:
                for seed_pt_xy in bifurc_seed_pt_xy:
                    bifurc_plt = pltfs.streamplot(ax, strm_x_g[0, :],
                                                  strm_y_g[:, 0], strm_U_g,
                                                  strm_V_g,
                                                  start_points=[seed_pt_xy],
                                                  color=bifurc_colour,
                                                  linewidth=bifurc_lw,
                                                  arrowsize=bifurc_arrowsize,
                                                  num_arrow_heads=bifurc_num_arrow_heads)
            else:
                print 'Need to provide bifurc_seed_pt_xy for bifurcationlines'
        else:
            print 'Need to provide strm_U_g and strm_V_g for bifurcationlines'

    # Finding and marking points
    if mark_max:
        row, col = np.where(field_g == field_g.max())
        plt.plot(x_g[row, col], y_g[row, col], marker=max_mark_symb,
                 mfc=max_mark_colour, mec=max_mark_colour, ms=max_mark_size,
                 mew=max_mark_edge_width)
    if mark_min:
        row, col = np.where(field_g == field_g.min())
        plt.plot(x_g[row, col], y_g[row, col], marker=min_mark_symb,
                 mfc=min_mark_colour, mec=min_mark_colour, ms=min_mark_size,
                 mew=min_mark_edge_width)
    if additional_marker_coords:
        for mark_coord in additional_marker_coords:
            rows, cols = np.where(np.abs(x_g - mark_coord[0]) ==
                                  np.min(np.abs(x_g - mark_coord[0])))
            mark_row = rows[0]
            rows, cols = np.where(np.abs(y_g - mark_coord[1]) ==
                                  np.min(np.abs(y_g - mark_coord[1])))
            mark_col = cols[0]
            plt.plot(x_g[mark_row, mark_col], y_g[mark_row, mark_col],
                     marker=add_mark_symb, mfc=add_mark_colour,
                     mec=add_mark_colour, ms=add_mark_size,
                     mew=add_mark_edge_width)

    # Annotating the plot and adjusting axis limits
    if plane_type in xy_plane_strs:
        y_label_str = r'$\displaystyle \frac{y}{d}$'
        y_label_str = r'$y$'
        y_tick_min = -np.floor(y_g.max()/tick_space)*tick_space
        y_tick_max = np.floor(y_g.max()/tick_space)*tick_space + tick_space
        y_ticks = np.arange(y_tick_min, y_tick_max, tick_space)
    elif plane_type in xz_plane_strs or plane_type in yz_plane_strs:
        y_label_str = r'$\displaystyle \frac{z}{d}$'
        y_label_str = r'$z$'
        y_tick_min = 0.0
        y_tick_max = np.floor(y_g.max()/tick_space)*tick_space + tick_space
        y_ticks = np.arange(y_tick_min, y_tick_max, tick_space)
    else:
        print plane_type_error_str
    plt.yticks(y_ticks)
    if label_y:
        ax.set_ylabel(y_label_str, labelpad=y_label_pad,
                      rotation='horizontal', va='center', ha='center')
    else:
        plt.tick_params(labelleft=False)
        ax.yaxis.labelpad = 0.01
    if plane_type in xy_plane_strs or plane_type in xz_plane_strs:
        x_label_str = r'$\displaystyle \frac{x}{d}$'
        x_label_str = r'$x$'
        if obstacle in ['hollow circle', 'hollow square',
                        'Hollow Circle', 'Hollow Square']:
            x_tick_min = -np.floor(x_g.max()/tick_space)*tick_space
        else:
            x_tick_min = 0.0
        x_tick_max = np.floor(x_g.max()/tick_space)*tick_space + tick_space
        x_ticks = np.arange(x_tick_min, x_tick_max, tick_space)
    elif plane_type in yz_plane_strs:
        x_label_str = r'$\displaystyle \frac{y}{d}$'
        x_label_str = r'$y$'
        x_tick_min = -np.floor(x_g.max()/tick_space)*tick_space
        x_tick_max = np.floor(x_g.max()/tick_space)*tick_space + tick_space
        x_ticks = np.arange(x_tick_min, x_tick_max, tick_space)
    else:
        print plane_type_error_str
    plt.xticks(x_ticks)
    if label_x:
        ax.set_xlabel(x_label_str, labelpad=x_label_pad, va='top',
                      ha='center')
    else:
        plt.tick_params(labelbottom=False)
    if label_subplots:
        bbox_props = {'pad': 0.1, 'fc': 'w', 'ec': 'w', 'alpha': 0.0}
        if colour_subplot_label:
            bbox_props['alpha'] = 1.0
        if subplot_label_coord in ['axes', 'Axes']:
            transform = ax.transAxes
        else:
            transform = ax.transData
        plt.text(x_subplot_label, y_subplot_label,
                 '(' + string.ascii_lowercase[subplot_ind - 1] + ')',
                 ha='left', va='top', bbox=bbox_props,
                 transform=transform)
    plt.xlim([x_min, x_g.max()])
    plt.ylim([y_min, y_max])
    if title:
        if title_pad:
            if title_pad in ['x_axis_ticks']:
                title_pad = ax.get_xaxis().get_tick_padding()
            offset = transforms.ScaledTranslation(0, title_pad/72,
                                                  fig.dpi_scale_trans)  # 72 pts/inch
            offset_transform = ax.transAxes + offset
            plt.text(0.5, 1, title, transform=offset_transform, ha='center',
                     va='baseline')
        else:
            ax.set_title(title, fontdict={'fontsize': font_size})

#    return col_plt,


###############################################################################

def plot_normalized_planar_field_comparison_data_reorg(
        x_g, y_g, field_g_list, strm_x_g, strm_y_g, strm_U_g, strm_V_g,
        cont_lvl_field_g_list, plane_type, num_geos, **kwargs):

    xy_plane_strs = ['xy', 'yx', 'XY', 'YX']
    xz_plane_strs = ['xz', 'zx', 'XZ', 'ZX']
    yz_plane_strs = ['yz', 'zy', 'YZ', 'ZY']
    plane_type_error_str = 'Please select either ''xy'', ''xz'', or ''yz'' for plane_type'

    x_g_raw = copy.deepcopy(x_g)
    y_g_raw = copy.deepcopy(y_g)
    field_g_list_raw = copy.deepcopy(field_g_list)
    strm_x_g_raw = copy.deepcopy(strm_x_g)
    strm_y_g_raw = copy.deepcopy(strm_y_g)
    strm_U_g_raw = copy.deepcopy(strm_U_g)
    strm_V_g_raw = copy.deepcopy(strm_V_g)
    cont_lvl_field_g_list_raw = copy.deepcopy(cont_lvl_field_g_list)


    if plane_type in xy_plane_strs:
        axis = 2
        x_g = [[x_g_raw[geo][:, :, plane]
                for plane in np.arange(np.size(x_g_raw[geo], axis=axis))]
               for geo in np.arange(num_geos)]
        y_g = [[y_g_raw[geo][:, :, plane]
                for plane in np.arange(np.size(y_g_raw[geo], axis=axis))]
               for geo in np.arange(num_geos)]
        field_g_list = [[[field_g_list_raw[field][geo][:, :, plane]
                          for plane in np.arange(np.size(
                                  field_g_list_raw[field][geo], axis=axis))]
                         for geo in np.arange(num_geos)]
                        for field in np.arange(np.size(field_g_list_raw,
                                                       axis=0))]
        strm_x_g = [[strm_x_g_raw[geo][:, :, plane]
                for plane in np.arange(np.size(strm_x_g_raw[geo], axis=axis))]
               for geo in np.arange(num_geos)]
        strm_y_g = [[strm_y_g_raw[geo][:, :, plane]
                for plane in np.arange(np.size(strm_y_g_raw[geo], axis=axis))]
               for geo in np.arange(num_geos)]
        strm_U_g = [[strm_U_g_raw[geo][:, :, plane]
                     for plane in np.arange(np.size(strm_U_g_raw[geo], axis=axis))]
                    for geo in np.arange(num_geos)]
        strm_V_g = [[strm_V_g_raw[geo][:, :, plane]
                     for plane in np.arange(np.size(strm_V_g_raw[geo], axis=axis))]
                    for geo in np.arange(num_geos)]
        cont_lvl_field_g_list = [[[cont_lvl_field_g_list_raw[field][geo][:, :, plane]
                                   for plane in np.arange(np.size(
                                           cont_lvl_field_g_list_raw[field][geo],
                                           axis=axis))]
                                  for geo in np.arange(num_geos)]
                                 for field in np.arange(np.size(
                                         cont_lvl_field_g_list_raw, axis=0))]
    elif plane_type in xz_plane_strs:
        axis = 0
        x_g = [[x_g_raw[geo][plane, :, :]
                for plane in np.arange(np.size(x_g_raw[geo], axis=axis))]
               for geo in np.arange(num_geos)]
        y_g = [[y_g_raw[geo][plane, :, :]
                for plane in np.arange(np.size(y_g_raw[geo], axis=axis))]
               for geo in np.arange(num_geos)]
        field_g_list = [[[field_g_list_raw[field][geo][plane, :, :]
                          for plane in np.arange(np.size(
                                  field_g_list_raw[field][geo], axis=axis))]
                         for geo in np.arange(num_geos)]
                        for field in np.arange(np.size(field_g_list,
                                                       axis=0))]
        strm_x_g = [[strm_x_g_raw[geo][plane, :, :].T
                     for plane in np.arange(np.size(strm_x_g_raw[geo], axis=axis))]
                    for geo in np.arange(num_geos)]
        strm_y_g = [[strm_y_g_raw[geo][plane, :, :].T
                     for plane in np.arange(np.size(strm_y_g_raw[geo], axis=axis))]
                    for geo in np.arange(num_geos)]
        strm_U_g = [[strm_U_g_raw[geo][plane, :, :].T
                     for plane in np.arange(np.size(strm_U_g_raw[geo], axis=axis))]
                    for geo in np.arange(num_geos)]
        strm_V_g = [[strm_V_g_raw[geo][plane, :, :].T
                     for plane in np.arange(np.size(strm_V_g_raw[geo], axis=axis))]
                    for geo in np.arange(num_geos)]
        cont_lvl_field_g_list = [[[cont_lvl_field_g_list_raw[field][geo][plane, :, :]
                                   for plane in np.arange(np.size(
                                           cont_lvl_field_g_list_raw[field][geo],
                                           axis=axis))]
                                  for geo in np.arange(num_geos)]
                                 for field in np.arange(np.size(
                                     cont_lvl_field_g_list_raw, axis=0))]
    elif plane_type in yz_plane_strs:
        axis = 1
        x_g = [[x_g_raw[geo][:, plane, :]
                for plane in np.arange(np.size(x_g_raw[geo], axis=axis))]
               for geo in np.arange(num_geos)]
        y_g = [[y_g_raw[geo][:, plane, :]
                for plane in np.arange(np.size(y_g_raw[geo], axis=axis))]
               for geo in np.arange(num_geos)]
        field_g_list = [[[field_g_list_raw[field][geo][:, plane, :]
                          for plane in np.arange(np.size(
                                  field_g_list_raw[field][geo], axis=axis))]
                         for geo in np.arange(num_geos)]
                        for field in np.arange(np.size(field_g_list_raw,
                                                       axis=0))]
        strm_x_g = [[strm_x_g_raw[geo][:, plane, :].T
                     for plane in np.arange(np.size(strm_x_g_raw[geo], axis=axis))]
                    for geo in np.arange(num_geos)]
        strm_y_g = [[strm_y_g_raw[geo][:, plane, :].T
                     for plane in np.arange(np.size(strm_y_g_raw[geo], axis=axis))]
                    for geo in np.arange(num_geos)]
        strm_U_g = [[strm_U_g_raw[geo][:, plane, :].T
                     for plane in np.arange(np.size(strm_U_g_raw[geo], axis=axis))]
                    for geo in np.arange(num_geos)]
        strm_V_g = [[strm_V_g_raw[geo][:, plane, :].T
                     for plane in np.arange(np.size(strm_V_g_raw[geo], axis=axis))]
                    for geo in np.arange(num_geos)]
        cont_lvl_field_g_list = [[[cont_lvl_field_g_list_raw[field][geo][:, plane, :]
                                   for plane in np.arange(np.size(
                                           cont_lvl_field_g_list_raw[field][geo],
                                           axis=axis))]
                                  for geo in np.arange(num_geos)]
                                 for field in np.arange(np.size(
                                     cont_lvl_field_g_list_raw, axis=0))]
    else:
        print plane_type_error_str

    return [x_g, y_g, field_g_list,
            strm_x_g, strm_y_g, strm_U_g, strm_V_g,
            cont_lvl_field_g_list]


###############################################################################

def plot_normalized_planar_field_comparison(
        x_g, y_g, field_g_list, strm_U_g, strm_V_g, field_labels, titles,
        plane_type, plane_inds, num_geos, obstacles, PUBdir, save_name,
        save_name_suffix, comparison_type='fields', plot_streamlines=True,
        share_clim=False, save_fig=True, close_fig=True, extension='.eps',
        figsize=(6.7, 9), **kwargs):

    plane_comp_strs = ['plane', 'planes', 'Plane', 'Planes']
    field_comp_strs = ['field', 'fields', 'Field', 'Fields']
    plane_field_comp_strs = ['plane and field', 'planes and fields',
                             'Plane and Field', 'Planes and Fields',
                             'field and plane', 'fields and planes',
                             'Field and Plane', 'Fields and Planes']
    comp_type_error_str = 'Please select either ''field'', or ''plane'' for comparison type'

    obstacle_AR = kwargs.pop('obstacle_AR', [1]*num_geos)
    cont_lvl_field_g_list = kwargs.pop('cont_lvl_field_g', field_g_list)
    bifurc_seed_pt_xy = kwargs.pop('bifurc_seed_pt_xy', [None]*num_geos)
    plot_bifurcation_lines = kwargs.pop('plot_bifurcation_lines', [False]*num_geos)
    strm_x_g = kwargs.pop('strm_x_g', x_g)
    strm_y_g = kwargs.pop('strm_y_g', y_g)

    num_planes = np.size(plane_inds[0])
    num_fields = np.size(field_g_list, axis=0)

    cb_label_0 = kwargs.pop('cb_label_0',
                            [[True]*num_fields for geo in np.arange(num_geos)])
    if type(cb_label_0) == bool:
        cb_label_0 = [[cb_label_0]*num_fields for geo in np.arange(num_geos)]
    stretch_cb = kwargs.pop('stretch_cb',
                            [[False]*num_fields for geo in np.arange(num_geos)])

    reorg_data = plot_normalized_planar_field_comparison_data_reorg(
        x_g, y_g, field_g_list, strm_x_g, strm_y_g, strm_U_g, strm_V_g,
        cont_lvl_field_g_list, plane_type, num_geos)
    [x_g, y_g, field_g_list, strm_x_g, strm_y_g, strm_U_g, strm_V_g,
         cont_lvl_field_g_list] = reorg_data

    figs = []
    axs = []
    if comparison_type in plane_comp_strs or comparison_type in plane_field_comp_strs:
        num_fields1 = num_fields
        num_fields2 = 1
    else:
        num_fields1 = 1
        num_fields2 = num_fields
    if comparison_type in plane_field_comp_strs:
        fig = plt.figure(figsize=figsize)
        figs.append(fig)
    for field1 in np.arange(num_fields1):
        if comparison_type in plane_comp_strs:
            fig = plt.figure(figsize=figsize)
            figs.append(fig)
        for plane_ind in np.arange(num_planes):
            if comparison_type in field_comp_strs:
                fig = plt.figure(figsize=figsize)
                figs.append(fig)
            for geo in np.arange(num_geos):
                plane = plane_inds[geo][plane_ind]
                plane_ind = np.where(np.array(plane_inds[geo])==plane)[0][0]
                for field2 in np.arange(num_fields2):
                    field = (field1 + 1)*(field2 + 1) - 1
                    climmax = None
                    if share_clim:
                        climmax_geo = np.zeros(num_geos)
                        for geo2 in np.arange(num_geos):
                            climmax_geo[geo2] = np.max([
                                    np.abs(np.max(field_g_list[field][geo2][plane])),
                                    np.abs(np.min(field_g_list[field][geo2][plane]))
                                    ])
                        climmax = np.max(climmax_geo)
                    if comparison_type in field_comp_strs:
                        subplot_increment = field
                        num_subplots = num_fields
                    elif comparison_type in plane_comp_strs:
                        subplot_increment = plane_ind
                        num_subplots = num_planes
                    elif comparison_type in plane_field_comp_strs:
                        subplot_increment = field*num_planes + plane_ind
                        num_subplots = num_fields*num_planes
                    else:
                        print comp_type_error_str
                    subplot_ind = num_geos*subplot_increment + geo + 1
                    ax = plt.subplot(num_subplots, num_geos, subplot_ind,
                                     aspect='equal')
                    axs.append(ax)
                    if subplot_ind in [num_subplots*num_geos - geo3 
                                       for geo3 in np.arange(num_geos)]:
                        label_x = True
                    else:
                        label_x = False
                    plot_normalized_planar_field(
                            x_g[geo][plane], y_g[geo][plane],
                            field_g_list[field][geo][plane],
                            cb_label=field_labels[field],
                            subplot_ind=subplot_ind, clim=climmax,
                            plane_type=plane_type, obstacle=obstacles[geo],
                            plot_streamlines=plot_streamlines,
                            strm_U_g=strm_U_g[geo][plane],
                            strm_V_g=strm_V_g[geo][plane],
                            strm_x_g=strm_x_g[geo][plane],
                            strm_y_g=strm_y_g[geo][plane],
                            cont_lvl_field_g=cont_lvl_field_g_list[field][geo][plane],
                            title=titles[geo][plane_ind],
                            obstacle_AR=obstacle_AR[geo],
                            label_x=label_x, cb_label_0=cb_label_0[geo][field],
                            stretch_cb=stretch_cb[geo][field],
                            plot_bifurcation_lines=plot_bifurcation_lines[geo],
                            bifurc_seed_pt_xy=bifurc_seed_pt_xy[geo],
                            **kwargs)
            if comparison_type in field_comp_strs:
                plt.tight_layout()
                if save_fig:
                    save_path = PUBdir + '\\' + save_name[plane_ind] + \
                        save_name_suffix + extension
                    plt.savefig(save_path, bbox_inches='tight')
                if close_fig:
                    plt.close()
        if comparison_type in plane_comp_strs:
            plt.tight_layout()
            if save_fig:
                save_path = PUBdir + '\\' + save_name[field1] + \
                    save_name_suffix + extension
                plt.savefig(save_path, bbox_inches='tight')
            if close_fig:
                plt.close()
        if comparison_type in plane_field_comp_strs:
            plt.tight_layout()
            if save_fig:
                save_path = PUBdir + '\\' + \
                    save_name_suffix + extension
                plt.savefig(save_path, bbox_inches='tight')
            if close_fig:
                plt.close()
    return figs, axs


###############################################################################

def plot_spectra(fs, spectras, **kwargs):
    label_spectra = kwargs.pop('label_spectra', False)
    spectra_labels = kwargs.pop('spectra_labels', [''])
    label_spectra_alpha = kwargs.pop('label_spectra_alpha', False)
    label_x = kwargs.pop('label_x', True)
    x_label = kwargs.pop('x_label', 'f')
    label_y = kwargs.pop('label_y', True)
    y_label = kwargs.pop('y_label', 'PSD')
    label_y_ticks = kwargs.pop('label_y_ticks', True)
    x_lims = kwargs.pop('x_lims', [])
    y_lims = kwargs.pop('y_lims', [])
    y_ticks_single = kwargs.pop('y_ticks_single', [])
    num_y_ticks = kwargs.pop('num_y_ticks', 3)
    y_ticks_pos = kwargs.pop('y_ticks_pos', 'left')
    x_ticks = kwargs.pop('x_ticks', [])
    tight = kwargs.pop('tight', False)
    grid_y = kwargs.pop('grid_y', False)
    grid_x = kwargs.pop('grid_x', False)
    subtick_x = kwargs.pop('subtick_x', True)
    colour_spectra_label = kwargs.pop('colour_spectra_label', True)
    harm_freqs = kwargs.pop('harm_freqs', [])
    f_shed = kwargs.pop('f_shed', 1)
    abs_freqs = kwargs.pop('abs_freqs', [])
    abs_freq_labels = kwargs.pop('abs_freq_labels', [])
    abs_freq_has = kwargs.pop('abs_freq_has', [])
    label_axes = kwargs.pop('label_axes', True)
    axes_ind = kwargs.pop('axes_ind', 1)
    axes_label_x = kwargs.pop('axes_label_x', 0.05)
    axes_label_y = kwargs.pop('axes_label_y', 0.95)
    colour_axes_label = kwargs.pop('colour_axes_label', True)
    axes_label_coord = kwargs.pop('axes_label_coord', 'axes')
    label_y_fracs = kwargs.pop('label_y_fracs', 1/2)
    label_x_frac = kwargs.pop('label_x_frac', 1/8)
    label_va = kwargs.pop('label_va', 'top')
    label_ha = kwargs.pop('label_ha', 'left')
    y_lim_pad_frac = kwargs.pop('y_lim_pad_frac', None)
    lw = kwargs.pop('lw', 0.5)
    freq_lc = kwargs.pop('freq_lc', [0.7]*3)
    freq_lsty = kwargs.pop('freq_lsty', ':')
    harm_tick_dir = kwargs.pop('harm_tick_dir', 'out')
    harm_tick = kwargs.pop('harm_tick', True)
    harm_x_label = kwargs.pop('harm_x_label',
                              r'$\displaystyle \frac{f}{f_{sh}}$')
    harm_x_label_pos = kwargs.pop('harm_x_label_pos', None)
    harm_tick_pad = kwargs.pop('harm_tick_pad', None)

    if np.size(kwargs.keys()) >= 1:
        print 'Unknown kwargs: ', kwargs.keys()

    fig = plt.gcf()
    ax1 = plt.gca()

    # Determining plotting parameters
    if np.ndim(spectras) == 1:
        single = True
        num_spectras = 1
        spectras = [spectras]
    elif np.size(spectras, axis=0) == 1:
        single = True
        num_spectras = 1
    else:
        single = False
        num_spectras = np.size(spectras, axis=0)
    spectra_inds = np.arange(num_spectras)
    if label_spectra:
        if type(spectra_labels) == str:
            spectra_labels = [spectra_labels]
        if np.size(spectra_labels) < num_spectras:
            spectra_labels = ['']*num_spectras
    if np.ndim(fs) == 1:
        fs = [fs]*num_spectras
    if fs[0][0] == 0:
        fs = [fs[ind][1:] for ind in spectra_inds]
        spectras = [spectras[ind][1:] for ind in spectra_inds]
    print y_ticks_single
    if np.size(y_ticks_single) == 0:
        y_ticks_max = np.ceil(np.log10(np.max(spectras)))
        y_ticks_min = np.floor(np.log10(np.min(spectras)))
        y_ticks_exp = np.linspace(y_ticks_min, y_ticks_max, num_y_ticks)
        dy_ticks_exp = np.floor(np.abs(y_ticks_exp[1] - y_ticks_exp[0]))
        y_ticks_exp = np.linspace(y_ticks_min,
                                  y_ticks_min + (num_y_ticks - 1)*dy_ticks_exp,
                                  num_y_ticks)
        y_ticks_single = [10**y_tick_exp for y_tick_exp in y_ticks_exp]
    else:
        dy_ticks_exp = np.abs(np.log10(y_ticks_single[1]) -
                              np.log10(y_ticks_single[0]))
    print y_ticks_single, dy_ticks_exp
#    y_single_max = np.max(spectras)
#    multiplier = [np.log10(np.max(spectras[ind])) -
#                  np.log10(np.min(spectras[ind])) for ind in spectra_inds]
#    multiplier = np.max(multiplier)
    y_single_max = np.max(y_ticks_single)
    y_min_data = np.min(y_ticks_single)
    y_max_data = np.max(y_ticks_single)
    multiplier = np.log10(np.max(y_ticks_single)) - np.log10(np.min(y_ticks_single))
    print multiplier
#    if tight:
#        y_min_data = np.min(spectras[-1])
#        y_max_data = np.max(spectras[0])
#    else:
#        multiplier = multiplier + dy_ticks_exp/2
#        y_min_data = np.min(y_ticks_single)
#        y_max_data = np.max(y_ticks_single)
    if not tight:
        multiplier = multiplier + dy_ticks_exp/2
    multiplier = np.ceil(multiplier)
    print multiplier, y_min_data, y_max_data, label_y_fracs
    if np.size(label_y_fracs) == 1:
        label_y_fracs = [label_y_fracs for _ in spectra_inds]
    print label_y_fracs
    if not y_lim_pad_frac:
        y_lim_pad = dy_ticks_exp/2
    else:
        y_lim_pad = np.ceil(multiplier*y_lim_pad_frac)
    if tight: 
        y_lim_pad = 0
    if single:
        y_lim_pad = 1
    if np.size(y_lims) == 2:
        y_min = y_lims[0]
        y_max = y_lims[1]
    else:
        
        y_min = 10**(np.log10(y_min_data) - y_lim_pad)
        y_max = 10**((num_spectras-1)*multiplier + np.log10(y_max_data) + y_lim_pad)
    print y_min, y_max
    text_x = 10**(label_x_frac*np.abs(np.log10(fs[0][-1]) - np.log10(fs[0][0])) +
                  np.log10(fs[0][0]))
    if np.size(spectra_labels) < num_spectras:
        spectra_labels_extra = ['']*(num_spectras - np.size(spectra_labels))
        spectra_labels = spectra_labels + spectra_labels_extra
    if label_spectra_alpha:
        spectra_label_alphas = ['(' + string.ascii_lowercase[spectra_ind - 1] +
                                ')' for spectra_ind in spectra_inds]
        spectra_labels = [spectra_label_alpha + ' - ' + spectra_label
                          for spectra_label_alpha, spectra_label in
                          zip(spectra_label_alphas, spectra_labels)]
    bbox_props = {'pad': 0.1, 'fc': 'w', 'ec': 'w', 'alpha': 0.0}
    if colour_spectra_label:
        bbox_props['alpha'] = 1.0

    # Plot spectra
    for f, spectra, spectra_label, spectra_ind in \
            zip(fs, spectras, spectra_labels, spectra_inds[::-1]):
        spectra_multiplier = 10**(spectra_ind*multiplier)
        plt.loglog(f, spectra_multiplier*spectra, 'k', lw=lw)
        if label_spectra:
            label_y_offset = -multiplier*label_y_fracs[num_spectras - spectra_ind - 1]
            spectra_label_y = spectra_multiplier*y_single_max*10**label_y_offset
            plt.text(text_x, spectra_label_y, spectra_label, ha=label_ha,
                     va=label_va, bbox=bbox_props,)

    # Plot extra frequency identifiers
    if np.size(harm_freqs) > 0:
        for harm_freq in harm_freqs:
            print f_shed, f.max()
#            plt.loglog([harm_freq*f_shed]*2, [y_min, y_max], freq_lsty,
#                       c=freq_lc, zorder=0)
            ax1.axvline(harm_freq*f_shed, ls=freq_lsty, c=freq_lc, zorder=0)
        ax1.tick_params(bottom=True, labelbottom=True, top=False,
                        labeltop=False)
        ax2 = ax1.twiny()
        ax2.set_aspect(ax1.get_aspect(), adjustable='box-forced')
        ax1.set_aspect(ax2.get_aspect(), adjustable='box-forced')
        ax2.set_xscale('log')
        ax2.set_xlim(ax1.get_xlim())
        if not harm_tick_pad:
            harm_tick_pad = ax2.get_xaxis().get_tick_padding()
        ax2.tick_params(bottom=False, labelbottom=False, top=harm_tick,
                        labeltop=True, direction=harm_tick_dir,
                        pad=harm_tick_pad)
        ax2.tick_params(which='minor', top=False)
        if harm_x_label_pos:
            if harm_x_label_pos in ['inline', 'in line', 'in-line']:
                harm_x_label_y = 1
                harm_x_label_x = 0
                offset = transforms.ScaledTranslation(0, harm_tick_pad/72,
                                                      fig.dpi_scale_trans)  # 72 pts/inch
                offset_transform = ax1.transAxes + offset
                ax2.xaxis.set_label_coords(harm_x_label_x, harm_x_label_y,
                                           transform=offset_transform)
                ax2.set_xlabel(harm_x_label, ha='left', va='bottom')
            else:
                print 'Add functionality'
        else:
            ax2.set_xlabel(harm_x_label)
        harm_x_tick_labels = [str(harm_freq) for harm_freq in harm_freqs]
        harm_x_ticks = [harm_freq*f_shed for harm_freq in harm_freqs]
        ax2.set_xticks(harm_x_ticks)
        ax2.set_xticklabels(harm_x_tick_labels)
    if np.size(abs_freqs) > 0:
        num_abs_freqs = np.size(abs_freqs, axis=0)
        if np.size(abs_freq_labels) == 0 \
                or np.size(abs_freq_labels) < num_abs_freqs:
            abs_freq_labels = ['']*num_abs_freqs
        if np.size(abs_freq_has) == 0 \
                or np.size(abs_freq_has) < num_abs_freqs:
            abs_freq_has = ['left']*num_abs_freqs
        for abs_freq, abs_freq_label, abs_freq_ha in \
                zip(abs_freqs, abs_freq_labels, abs_freq_has):
            if abs_freq_ha == 'left':
                ha_offset_mult = 1
            else:
                ha_offset_mult = -1
            pad = ax1.get_xaxis().get_tick_padding()
            offset = transforms.ScaledTranslation(ha_offset_mult*pad/72,
                                                  -pad/72, fig.dpi_scale_trans)  # 72 pts/inch
            offset_transform = ax1.transData + offset
            plt.text(abs_freq, y_max - pad, abs_freq_label, ha=abs_freq_ha,
                     va='top', transform=offset_transform)
#            ax1.loglog([abs_freq]*2, [y_min, y_max], freq_lsty, c=freq_lc,
#                       zorder=0)
            ax1.axvline(abs_freq, ls=freq_lsty, c=freq_lc, zorder=0)

    # Set axis limits, labels, ticks, etc.
    plt.ylim([y_min, y_max])
    if label_y:
        ax1.set_ylabel(y_label)
    if label_x:
        ax1.set_xlabel(x_label)
    else:
        ax1.tick_params(labelbottom=False)
    if label_y_ticks:
        if y_ticks_pos in ['left', 'Left']:
            ax1.tick_params(left=True, labelleft=True, right=False,
                            labelright=False)
        else:
            ax1.tick_params(left=False, labelleft=False, right=True,
                            labelright=True)
        y_ticks = [y_tick*10**(int(spectra_ind*multiplier))
                   for spectra_ind in spectra_inds
                   for y_tick in y_ticks_single]
        y_tick_labels = [r'$10^{%2.0f}$' % np.log10(y_tick)
                         for y_tick in y_ticks_single]*num_spectras
        ax1.set_yticks(y_ticks)
        ax1.set_yticklabels(y_tick_labels)
        if np.size(harm_freqs) > 0:
            ax2.set_ylim(ax1.get_ylim())
    else:
        ax1.tick_params(left=False, labelleft=False)
    if x_ticks:
        ax1.set_xticks(x_ticks)
        if np.size(harm_freqs) > 0:
            ax2.set_xlim(ax1.get_xlim())
    if np.size(x_lims) == 2:
        plt.xlim(x_lims)
    if grid_y:
        ax1.grid(True, axis='y')
    if grid_x:
        ax1.grid(True, axis='x')
    if subtick_x:
        ax1.tick_params(bottom=True, which='minor')
    else:
        ax1.tick_params(bottom=False, which='minor')
    if label_axes:
        bbox_props = {'pad': 0.1, 'fc': 'w', 'ec': 'w', 'alpha': 0.0}
        if colour_axes_label:
            bbox_props['alpha'] = 1.0
        if axes_label_coord in ['axes', 'Axes']:
            transform = ax1.transAxes
        else:
            transform = ax1.transData
        plt.text(axes_label_x, axes_label_y,
                 '(' + string.ascii_lowercase[axes_ind - 1] + ')',
                 ha='left', va='top', bbox=bbox_props,
                 transform=transform)


###############################################################################

def plot_spectra_comparison(
        f, spectras, num_geos, PUBdir, save_name, alignment='cols',
        save_fig=True, close_fig=True, extension='.eps', figsize=(6.7, 9),
        **kwargs):

    spectra_labels = kwargs.pop('spectra_labels', [['']]*num_geos)
    harm_freqs = kwargs.pop('harm_freqs', [[]]*num_geos)
    f_shed = kwargs.pop('f_shed', [1]*num_geos)
    abs_freqs = kwargs.pop('abs_freqs', [[]]*num_geos)
    abs_freq_labels = kwargs.pop('abs_freq_labels', [[]]*num_geos)
    abs_freq_has = kwargs.pop('abs_freq_has', [[]]*num_geos)
    label_y_fracs = kwargs.pop('label_y_fracs', [1/2]*num_geos)

    if np.size(label_y_fracs, axis=0) != num_geos:
        label_y_fracs = [label_y_fracs]*num_geos

    col_alignment_strs = ['cols', 'Cols']
    if alignment in col_alignment_strs:
        num_rows = 1
        num_cols = num_geos
        label_y = [True if geo == 0 else False for geo in np.arange(num_geos)]
        label_x = [True]*num_geos
    else:
        num_rows = num_geos
        num_cols = 1
        label_x = [True if geo == 0 else False for geo in np.arange(num_geos)]
        label_y = [True]*num_geos

    figs = []
    axs = []
    fig = plt.figure(figsize=figsize)
    figs.append(fig)
    geo_inds = np.arange(num_geos)
    for geo in geo_inds:
        subplot_ind = geo + 1
        ax = plt.subplot(num_rows, num_cols, subplot_ind)
        axs.append(ax)

        plot_spectra(f[geo], spectras[geo], spectra_labels=spectra_labels[geo],
                     label_x=label_x[geo], label_y=label_y[geo],
                     harm_freqs=harm_freqs[geo], f_shed=f_shed[geo],
                     abs_freqs=abs_freqs[geo], abs_freq_labels=abs_freq_labels,
                     abs_freq_has=abs_freq_has[geo], axes_ind=subplot_ind,
                     label_y_fracs=label_y_fracs[geo], **kwargs)
    plt.tight_layout()
    if save_fig:
        save_path = PUBdir + '\\' + save_name + extension
        plt.savefig(save_path, bbox_inches='tight')
    if close_fig:
        plt.close()
    return figs, axs


###############################################################################

def annotate_plot(ax, fig, text, x_coord, y_coord, t_x_coord, t_y_coord, 
                  a_x_coord=None, a_y_coord=None, fs=10,
                  mfc='y', mec='k', mew=1, ms=4.5, zorder=10, fc='k', ha='center', va='center',
                  ws=1, hs=1, lsf=0, bsf=0, ec='k', a_fc='y', a_ec='k',
                  a_alpha=0, a_shrink=2, a_width=1, a_h_width=1):
#                  a_style='simple'):

    if a_x_coord == None:
        a_x_coord = t_x_coord
    if a_y_coord == None:
        a_y_coord = t_y_coord
    arrow_props = {'fc': a_fc, 'ec': a_ec, 'alpha': a_alpha,
#                   'shrinkA': a_shrinkA, 'shrinkB': a_shrinkB,
                   'width': a_width, 'headwidth': a_h_width,
                   'shrink': a_shrink, 'headlength': a_h_width}
#                   'arrowstyle': a_style}
    plt.sca(ax)
    transform = ax.transAxes
    plt.plot(x_coord, y_coord, 'o', mfc=mfc, mec=mec, mew=mew, ms=ms)
    an = plt.annotate(text, xy=(x_coord, y_coord),
                      xytext=(t_x_coord, t_y_coord), ha=ha, va=va, size=fs,
                      color=fc)#, arrowprops=arrow_props)
    #             bbox=bbox_props, arrowprops=arrow_props)
    plt.annotate('', xy=(x_coord, y_coord), xytext=(a_x_coord, a_y_coord),
                 arrowprops=arrow_props, zorder=zorder)
    fig.canvas.draw()
    bbox = mtext.Text.get_window_extent(an)
    [[left, bottom], [right, top]] = ax.transData.inverted().transform(bbox)
    width = right - left
    height = top - bottom
    width = width*ws
    height = height*hs
    left = left + lsf*width
    bottom = bottom + bsf*height
    rect = patches.Rectangle((left, bottom), width, height, fc='w', ec=ec,
                             zorder=zorder)
    ax.add_patch(rect)


###############################################################################

def plot_POD_modes(x_g, y_g, Phi_uf_g, Phi_vf_g, Phi_wf_g, af_PSD, lambdaf, lambdaf_comp,
                   f, d, U_inf, St_shed,
                   modes, obstacles, planes, num_geos, PUBdir, save_names,
                   correct_signs, POD_type_str,
                   save_suffix,
                   harm_freqs=[], abs_freqs=[], abs_freq_labels=[],
                   save_fig=True, close_fig=True,
                   extension='.eps', figsize=None, **kwargs):

    if np.ndim(harm_freqs) < num_geos:
        harm_freqs = [harm_freqs]*num_geos
#    if np.ndim(abs_freqs) < num_geos:
#        abs_freqs = [abs_freqs]*num_geos
    print 'add abs freqs'

    num_planes = np.size(planes[0])
    num_modes = np.size(modes[0][0])
    mode_inds = np.arange(num_modes)
    geo_inds = np.arange(num_geos)
    num_cols = 4
    num_rows = num_modes*num_geos

    if not figsize:
        figsize=(6.7, 5/4*num_rows)

    for plane_ind in np.arange(num_planes):
        fig = plt.figure(figsize=figsize)
        
        width_ratios = [0.1] + [1]*num_cols
        height_ratios = [0.1] + [1]*num_rows

        if num_rows == 4:
            main_bottom = 0.03
            main_top = 0.9
            header_bottom = 0.95
        elif num_rows == 2:
            main_bottom = 0.05
            main_top = 0.85
            header_bottom = 0.9
        else:
            print 'May need to adjust vertical spacing of subfigures. '

        gs_top = gridspec.GridSpec(1, num_cols+1, width_ratios=width_ratios)
        gs_top.update(left=0.03, right=0.97, top=0.97, bottom=header_bottom)
        gs_left = gridspec.GridSpec(num_rows+1, 1, height_ratios=height_ratios)
        gs_left.update(left=0.03, right=0.05, top=0.97, bottom=0.03)
        gs_main = gridspec.GridSpec(num_rows, num_cols, hspace=0.4, wspace=0.15)
        gs_main.update(left=0.07, right=0.97, top=main_top, bottom=main_bottom)
    
        ax = plt.subplot(gs_top[1])
#        plt.text(0.5, 0, r'$\displaystyle \frac{\Phi_u}{U_\infty}$', ha='center',
#                 va='bottom')
        plt.text(0.5, 0, r'$\Phi_u$', ha='center', va='bottom')
        plt.plot([0, 1], [0, 0], 'k')
        ax.axis('off')
        ax = plt.subplot(gs_top[2])
#        plt.text(0.5, 0, r'$\displaystyle \frac{\Phi_v}{U_\infty}$', ha='center',
#                 va='bottom')
        plt.text(0.5, 0, r'$\Phi_v$', ha='center', va='bottom')
        plt.plot([0, 1], [0, 0], 'k')
        ax.axis('off')
        ax = plt.subplot(gs_top[3])
#        plt.text(0.5, 0, r'$\displaystyle \frac{\Phi_w}{U_\infty}$', ha='center',
#                 va='bottom')
        plt.text(0.5, 0, r'$\Phi_w$', ha='center', va='bottom')
        plt.plot([0, 1], [0, 0], 'k')
        ax.axis('off')
        ax = plt.subplot(gs_top[4])
        plt.text(0.5, 0, r'PSD of $a$', ha='center', va='bottom')
        plt.plot([0, 1], [0, 0], 'k')
        ax.axis('off')

        for obstacle, geo in zip(obstacles, geo_inds):
            print geo
            plane = planes[geo][plane_ind]
            for mode, mode_ind in zip(modes[geo][plane_ind], mode_inds):
                
                if (mode_ind + geo*num_modes) == num_rows - 1:
                    label_x = True
                else:
                    label_x = False

                Phi_tot_energy = np.sum(Phi_uf_g[geo][plane][:, :, mode]**2) + \
                                 np.sum(Phi_vf_g[geo][plane][:, :, mode]**2) + \
                                 np.sum(Phi_wf_g[geo][plane][:, :, mode]**2)
                mode_energy = lambdaf[geo][plane][mode]/np.sum(lambdaf_comp[geo][plane])
                tke_str = r'$\displaystyle \int_\Omega \frac{\overline{k}}{U_\infty^2} \mathrm{d}x$'
                tke_str = r'$\sum_{i=1}^{N_{POD}} \lambda_i$'
                tke_str = r'tke'
    #            subplot_ind = (mode_ind + geo*num_modes)*(num_cols + 1) + 1 + num_cols
                subplot_ind = (mode_ind + geo*num_modes) + 1
                ax = plt.subplot(gs_left[subplot_ind])
                mode_str = POD_type_str + \
                    'Mode %1.0f \n %2.1f \%% of ' % (mode+1, mode_energy*100) + \
                    tke_str
                plt.text(1, 0.5, mode_str, ha='right', va='center',
                         multialignment='center', rotation=90)
                plt.plot([1, 1], [0, 1], 'k')
                ax.axis('off')
                
                Phi_gs = [Phi_uf_g, Phi_vf_g, Phi_wf_g]
                comp_strs = ['u', 'v', 'w']
                label_ys = [True, False, False]
                for comp_ind, Phi_g, comp_str, label_y in \
                        zip(np.arange(3), Phi_gs, comp_strs, label_ys):
                    subplot_label = (mode_ind + geo*num_modes)*num_cols + 1 + comp_ind
    #                subplot_ind = (mode_ind + geo*num_modes)*(num_cols + 1) + 1 + comp_ind + 1 + num_cols
                    subplot_ind = (mode_ind + geo*num_modes)*num_cols + comp_ind
                    ax = plt.subplot(gs_main[subplot_ind], aspect='equal')
                    ax.patch.set_visible(True)
                    Phi_energy = np.sum(Phi_g[geo][plane][:, :, mode]**2)
                    energy = Phi_energy/Phi_tot_energy*mode_energy
                    energy_str = '%2.1f \%% of ' % (energy*100) + tke_str
                    plot_normalized_planar_field(
                            x_g[geo][:, :, plane], y_g[geo][:, :, plane],
                            correct_signs[geo][plane_ind][mode_ind]*Phi_g[geo][plane][:, :, mode],
                            subplot_ind=subplot_label, title=energy_str,
                            plane_type='xy', obstacle=obstacle, plot_arrow = False,
                            label_x=label_x, label_y=label_y, ctick_maxs=False,
                            colour_subplot_label=False,
                            title_pad='x_axis_ticks', **kwargs)
                subplot_label = (mode_ind + geo*num_modes)*num_cols + 1 + 3
    #            subplot_ind = (mode_ind + geo*num_modes)*(num_cols + 1) + 1 + 3 + 1 + num_cols
                subplot_ind = (mode_ind + geo*num_modes)*num_cols + 3
                ax = fig.add_subplot(gs_main[subplot_ind], aspect=0.5)
                x_label = r'$\displaystyle \frac{fd}{U_\infty}$'
                y_lims = [10**-7, 10**-1]
                y_ticks = [10**-7, 10**-4, 10**-1]
                x_ticks = [10**-2, 10**0]
                if np.size(harm_freqs) == 0:
                    harm_freqs=[1]
                plot_spectra(f[geo]*d[geo]/U_inf[geo][plane],
                             af_PSD[geo][plane][:, mode], x_label=x_label, 
                             label_x=label_x, label_y=False,
                             y_lims=y_lims, y_ticks_single=y_ticks,
                             label_y_ticks=True, y_ticks_pos='right',
                             x_ticks=x_ticks,
                             harm_freqs=harm_freqs[geo],
                             f_shed=St_shed[geo][plane],
#                             abs_freqs=abs_freqs[geo], 
#                             abs_freq_labels=abs_freq_labels,
                             tight=True,
                             harm_tick_dir='in', harm_tick=False,
                             harm_tick_pad=2, harm_x_label=r'$f/f_{sh}$',
                             harm_x_label_pos='inline', subtick_x=True,
                             label_axes=False)
                plt.text(0.05, 0.95,
                         '(' + string.ascii_lowercase[subplot_label - 1] + ')',
                         ha='left', va='top', bbox={'pad' : 0.1, 'fc' : 'w',
                                                    'ec' : 'w', 'alpha' : 0},
                         transform=ax.transAxes)
                
#                subplot_label = (mode_ind + geo*num_modes)*num_cols + 1 + 3
#    #            subplot_ind = (mode_ind + geo*num_modes)*(num_cols + 1) + 1 + 3 + 1 + num_cols
#                subplot_ind = (mode_ind + geo*num_modes)*num_cols + 3
#                ax = fig.add_subplot(gs3[subplot_ind], aspect=0.5)
#                ymax = 10**-1
#                ymin = 10**-7
#                plt.loglog(f[geo]*d[geo]/U_inf[geo][plane],
#                           af_PSD[geo][plane][:, mode], 'k', lw=0.5) 
#                plt.ylim(ymin, ymax)
#                plt.text(0.05, 0.95,
#                         '(' + string.ascii_lowercase[subplot_label - 1] + ')',
#                         ha='left', va='top', bbox={'pad' : 0.1, 'fc' : 'w',
#                                                    'ec' : 'w', 'alpha' : 0},
#                         transform=ax.transAxes)
#        
#                if label_x:
#                    plt.xlabel(r'$\displaystyle \frac{fd}{U_\infty}$')
#                else:
#                    plt.tick_params(labelbottom=False)
#                plt.tick_params(left=False, labelleft=False, right=True,
#                                labelright=True)
#            
#                plt.text(St_shed[geo][plane], ymax, '$f/f_{sh}$ \n ', ha='center',
#                         va='bottom')
##                plt.loglog([St_shed[geo][plane]]*2, [ymin, ymax], ':', c=[0.7]*3,
##                           zorder=0)
#                print harm_freqs
#                if np.size(harm_freqs) > 0:
#                    for harm_freq in harm_freqs[geo]:
#                        if harm_freq < 1:
#                            harm_freq_label_str = '$^1\!/\!_{%1.0f}$' % (1/harm_freq)
#                        else:
#                            harm_freq_label_str = '$%1.0f$' % (harm_freq)
#                        plt.text(harm_freq*St_shed[geo][plane], ymax, harm_freq_label_str,
#                                 ha='center', va='bottom')
#                        plt.loglog([harm_freq*St_shed[geo][plane]]*2, [ymin, ymax],
#                                   ':', c=[0.7]*3, zorder=0)
#                if np.size(abs_freqs) > 0:
#                    for abs_freq in abs_freqs[geo]:
#                        abs_freq_label_str = '$%1.1f$ Hz' % abs_freq
#                        if abs_freq*d[geo]/U_inf[geo][plane] < 1:
#                            abs_freq_ha = 'right'
#                        else:
#                            abs_freq_ha = 'left'
#                        plt.text(abs_freq*d[geo]/U_inf[geo][plane], ymax, abs_freq_label_str,
#                                 ha=abs_freq_ha, va='bottom')
#                        plt.loglog([abs_freq*d[geo]/U_inf[geo][plane]]*2, [ymin, ymax], ':', c=[0.7]*3,
#                                   zorder=0)
##                plt.text(2*St_shed[geo][plane], ymax, '$2$', ha='center',
##                         va='bottom')
##                plt.loglog([2*St_shed[geo][plane]]*2, [ymin, ymax], ':', c=[0.7]*3,
##                           zorder=0)
        
    #    plt.tight_layout()
        if save_fig:
            save_path = PUBdir + '\\' + save_names[plane_ind] + extension
            plt.savefig(save_path, bbox_inches='tight')
        if close_fig:
            plt.close()


###############################################################################

def plot_POD_mode_comparison2(x_g, y_g, Phi_uf_g, Phi_vf_g, Phi_wf_g, af_PSD, lambdaf, lambdaf_comp,
                   f, d, U_inf, St_shed, POD_type_str, correct_signs, 
                   modes, plane, geo_inds, obstacle, PUBdir, save_name,
                   save_fig=True, extension='.eps', figsize=(6.7, 3.5), **kwargs):
# Pretty sure that this is an old version
    fig = plt.figure(figsize=figsize)
    
    num_geos = np.size(geo_inds)
    
    num_cols = 4
    num_rows = num_geos # Not sure but think this is an old version
    
    width_ratios = [0.01] + [1]*num_cols
    height_ratios = [0.01] + [1]*num_rows
#        wspace = [0.05] + [0.1]*num_cols
#        hspace = [0.05] + [0.1]*num_rows
#        wspace = 0.1
#        hspace = 0.1
    gs = gridspec.GridSpec(num_rows+1, num_cols+1, width_ratios=width_ratios,
                           height_ratios=height_ratios)
    
    ax = plt.subplot(gs[1])
    plt.text(0.5, 0, r'$\displaystyle \frac{\Phi_u}{U_\infty}$', ha='center', va='bottom')
    plt.plot([0, 1], [0, 0], 'k')
    ax.axis('off')
    ax = plt.subplot(gs[2])
    plt.text(0.5, 0, r'$\displaystyle \frac{\Phi_v}{U_\infty}$', ha='center', va='bottom')
    plt.plot([0, 1], [0, 0], 'k')
    ax.axis('off')
    ax = plt.subplot(gs[3])
    plt.text(0.5, 0, r'$\displaystyle \frac{\Phi_w}{U_\infty}$', ha='center', va='bottom')
    plt.plot([0, 1], [0, 0], 'k')
    ax.axis('off')
    ax = plt.subplot(gs[4])
    plt.text(0.5, 0, r'PSD of $a$', ha='center', va='bottom')
    plt.plot([0, 1], [0, 0], 'k')
    ax.axis('off')
    
    for (mode, geo, ind) in zip(modes, geo_inds, np.arange(num_geos)):
        print geo, ind
        if ind == num_geos - 1:
            label_x = True
        else:
            label_x = False

        Phi_tot_energy = np.sum(Phi_uf_g[geo][plane][:, :, mode]**2) + \
                         np.sum(Phi_vf_g[geo][plane][:, :, mode]**2) + \
                         np.sum(Phi_wf_g[geo][plane][:, :, mode]**2)
        mode_energy = lambdaf[geo][plane][mode]/np.sum(lambdaf_comp[geo][plane])
        tke_str = r'$\displaystyle \int_\Omega \frac{\overline{k}}{U_\infty^2} \mathrm{d}x$'
        tke_str = r'$\sum_{i=1}^{N_{POD}} \lambda_i$'
        tke_str = r'$tke$'
        
        subplot_ind = ind*(num_cols+1) + 1 + num_cols
        ax = plt.subplot(gs[subplot_ind])
        mode_str = POD_type_str + 'Mode %1.0f \n %2.1f \%% of ' % (mode+1, mode_energy*100) + tke_str
        plt.text(1, 0.5, mode_str, ha='right', va='center', multialignment='center', rotation=90)
        plt.plot([1, 1], [0, 1], 'k')
        ax.axis('off')
        
        Phi_gs = [Phi_uf_g[geo][plane], Phi_vf_g[geo][plane], Phi_wf_g[geo][plane]]
        label_ys = [True, False, False]
        for (comp, Phi_g, label_y) in \
                zip(np.arange(3), Phi_gs, label_ys):
            subplot_label = ind*num_cols + 1 + comp
            subplot_ind = ind*(num_cols+1) + 1 + comp + 1 + num_cols
            ax = plt.subplot(gs[subplot_ind], aspect='equal')
            Phi_energy = np.sum(Phi_g[:, :, mode]**2)
            energy = Phi_energy/Phi_tot_energy*mode_energy
            energy_str = '%2.1f \%% of ' % (energy*100) + tke_str
            plot_normalized_planar_field(x_g[geo][:, :, plane],
                                         y_g[geo][:, :, plane],
                                         correct_signs[geo]*Phi_g[:, :, mode],
                                         subplot_ind=subplot_label,
                                         title=energy_str, plane_type='xy',
                                         obstacle=obstacle[geo], plot_arrow = False,
                                         label_x=label_x, label_y=label_y,
                                         ctick_maxs=False, **kwargs)

        subplot_label = ind*num_cols + 1 + 3
        subplot_ind = ind*(num_cols+1) + 1 + 3 + 1 + num_cols
        ax = fig.add_subplot(gs[subplot_ind], aspect=0.5)
        ymax = 10**-1
        ymin = 10**-7
        plt.loglog(f[geo]*d[geo]/U_inf[geo][plane], af_PSD[geo][plane][:, mode], 'k', lw=0.5)  
        plt.ylim(ymin, ymax)
        plt.text(0.05, 0.95,
                 '(' + string.ascii_lowercase[subplot_label - 1] + ')',
                 ha='left', va='top', bbox={'pad' : 0.1, 'fc' : 'w',
                                            'ec' : 'w', 'alpha' : 0}, transform=ax.transAxes)

        if label_x:
            plt.xlabel(r'$\displaystyle \frac{fd}{U_\infty}$')
        else:
            plt.tick_params(labelbottom=False)
        plt.tick_params(left=False, labelleft=False, right=True, labelright=True)
    
        plt.text(St_shed[geo][plane], ymax, '$f/f_{sh}$ \n 1', ha='center', va='bottom')
        plt.loglog([St_shed[geo][plane]]*2, [ymin, ymax], ':', c=[0.7]*3, zorder=0)
        if geo == 1:
            plt.text(St_shed[geo][plane]/10, ymax, '$^1\!/\!_{10}$', ha='right', va='bottom')
            plt.loglog([St_shed[geo][plane]/10]*2, [ymin, ymax], ':', c=[0.7]*3, zorder=0)
        if geo == 0:
            plt.text(St_shed[geo][plane]/4, ymax, '$^1\!/\!_4$', ha='center', va='bottom')
            plt.loglog([St_shed[geo][plane]/4]*2, [ymin, ymax], ':', c=[0.7]*3, zorder=0)
        plt.text(2*St_shed[geo][plane], ymax, '$2$', ha='center', va='bottom')
        plt.loglog([2*St_shed[geo][plane]]*2, [ymin, ymax], ':', c=[0.7]*3, zorder=0)
        plt.text(3*St_shed[geo][plane], ymax, '$3$', ha='left', va='bottom')
        plt.loglog([3*St_shed[geo][plane]]*2, [ymin, ymax], ':', c=[0.7]*3, zorder=0)

        

        
        
    plt.tight_layout()
    if save_fig:
        save_path = PUBdir + '\\' + save_name + extension
        plt.savefig(save_path, bbox_inches='tight')


###############################################################################

#def animate_field_func(snap, x_g, y_g, field_g, field_label, u_snap_g, v_snap_g, obstacle, f_capture, f_shed, **kwargs):
def animate_field_func(snap, *fargs):
#    x_g = kwargs.pop('x_g', None)
#    y_g = kwargs.pop('y_g', None)
#    field_g = kwargs.pop('field_g', None)
#    u_snap_g = kwargs.pop('u_snap_g', None)
#    v_snap_g = kwargs.pop('v_snap_g', None)
#    field_label = kwargs.pop('field_label', None)
#    f_capture = kwargs.pop('f_capture', None)
#    f_shed = kwargs.pop('f_shed', None)
#    obstacle = kwargs.pop('obstacle', None)

    fargs = np.squeeze(fargs)
    x_g = fargs[0]
    y_g = fargs[1]
    field_g = fargs[2]
    field_label = fargs[3]
    u_snap_g = fargs[4]
    v_snap_g = fargs[5]
    obstacle = fargs[6]
    f_capture = fargs[7]
    f_shed = fargs[8]
    clim = fargs[9]
    add_cb = fargs[10]

    

    
    title_str = r'$t f_{sh}=%.2f$'%(snap/f_capture*f_shed)
    print snap
    ax = plt.gca()
    ax.cla()
    subplot_ind = 1
    plot_normalized_planar_field(x_g, y_g, field_g[:, :, snap],
                                 strm_U_g=u_snap_g[:, :, snap],
                                 strm_V_g=v_snap_g[:, :, snap],
                                 cb_label=field_label,
                                 title=title_str, subplot_ind=subplot_ind,
                                 obstacle=obstacle, add_cb = add_cb, 
                                 plot_streamlines=True, clim=clim, ctick_maxs=False,
                                 plot_arrow=False, strm_dens=0.5, label_subplots=False)
    plt.tight_layout()


###############################################################################

def animate_field(snaps, x_g, y_g, field_g, u_snap_g, v_snap_g, field_label,
                  obstacle, f_capture, f_shed, PUBdir, save_name,
                  plot_streamlines=True, save_anim=True, close_anim=True,
                  figsize=(5, 4.5), fps=2, bitrate=1000, extension='.mp4',
                  **kwargs):
    
    clim = np.max([np.abs(np.min(field_g)), np.abs(np.max(field_g))])
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=bitrate)
    
    fig = plt.figure(figsize=figsize)
    
    ax = plt.subplot(1, 1, 1, aspect='equal')
    fargs=[x_g, y_g, field_g, field_label, u_snap_g, v_snap_g, obstacle,
           f_capture, f_shed, clim, True]
#    kwargs['x_g'] = x_g
#    kwargs['y_g'] = y_g
#    kwargs['field_g'] = field_g
#    kwargs['u_snap_g'] = u_snap_g
#    kwargs['v_snap_g'] = v_snap_g
#    kwargs['field_label'] = field_label
#    kwargs['f_capture'] = f_capture
#    kwargs['f_shed'] = f_shed
#    kwargs['obstacle'] = obstacle
    
    
    animate_field_func(snaps[0], fargs)
    
    fargs[-1] = False
    anim = animation.FuncAnimation(fig, animate_field_func, frames=snaps,
                                   fargs=fargs, blit=False)
    if save_anim:
        save_path = PUBdir + '\\' + save_name + extension
        anim.save(save_path, writer=writer)
    if close_anim:
        plt.close()


