from __future__ import division

import numpy as np
import os
import scipy.io as sio
import grid_conversion as gc


def load_general(WRKSPCdirs, WRKSPCfilenames_both, num_geos, num_planes, zs):

    geo_inds = np.arange(num_geos)
    plane_inds = np.arange(num_planes)

    variable_names = ['B', 'B1', 'f_capture', 'f_shed', 'd']
    B = np.zeros(num_geos, dtype=int)
    B1 = np.zeros(num_geos, dtype=int)
    f_capture = np.zeros(num_geos)
    d = np.zeros(num_geos)
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        matvariables = sio.loadmat(WRKSPCfilenames[0],
                                   variable_names=variable_names)
        B[geo] = int(matvariables['B'])
        B1[geo] = int(matvariables['B1'])
        f_capture[geo] = float(matvariables['f_capture'])
        d[geo] = float(matvariables['d'])
        del matvariables

    variable_names = ['t']
    t = (np.zeros(B[0]), np.zeros(B[1]))
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        matvariables = sio.loadmat(WRKSPCfilenames[0],
                                   variable_names=variable_names)
        t[geo][:] = np.squeeze(matvariables['t'])

    variable_names = ['I', 'J', 'dx', 'dy', 'f_shed', 'St_shed', 'U_inf', 'Re']
    I = [np.zeros(num_planes, dtype=int) for _ in geo_inds]
    J = [np.zeros(num_planes, dtype=int) for _ in geo_inds]
    num_vect = [np.zeros(num_planes, dtype=int) for _ in geo_inds]
    dx = [np.zeros(num_planes) for _ in geo_inds]
    dy = [np.zeros(num_planes) for _ in geo_inds]
    f_shed = [np.zeros(num_planes) for _ in geo_inds]
    St_shed = [np.zeros(num_planes) for _ in geo_inds]
    U_inf = [np.zeros(num_planes) for _ in geo_inds]
    Re = [np.zeros(num_planes) for _ in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            I[geo][plane] = int(matvariables['I'])
            J[geo][plane] = int(matvariables['J'])
            num_vect[geo][plane] = I[geo][plane]*J[geo][plane]
            dx[geo][plane] = float(matvariables['dx'])
            dy[geo][plane] = float(matvariables['dy'])
            f_shed[geo][plane] = float(matvariables['f_shed'])
            St_shed[geo][plane] = float(matvariables['St_shed'])
            U_inf[geo][plane] = float(matvariables['U_inf'])
            Re[geo][plane] = float(matvariables['Re'])
            del matvariables

    variable_names = ['x_g', 'y_g']
    x_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
           for geo in geo_inds]
    y_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
           for geo in geo_inds]
    z_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
           for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            x_g_temp = np.squeeze(matvariables['x_g'])
            y_g_temp = np.squeeze(matvariables['y_g'])

            cut_row = (J[geo][plane] - J[geo].min())//2
            cut_col = (I[geo][plane] - I[geo].min())//2
            cut_row_f = cut_row
            cut_row_b = cut_row
            cut_col_f = cut_col
            cut_col_b = cut_col

            if cut_row_b == 0:
                cut_row_b = -J[geo][plane]
            if cut_col_b == 0:
                cut_col_b = -I[geo][plane]

            x_g[geo][:, :, plane] \
                = x_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            y_g[geo][:, :, plane] \
                = y_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            del matvariables

            z_g[geo][:, :, plane] = np.full((J[geo].min(), I[geo].min()),
                                            zs[plane])/(d[geo]*1000)
    return B, B1, f_capture, d, t, I, J, num_vect, dx, dy, f_shed, St_shed, \
        U_inf, Re, x_g, y_g, z_g


def load_general_old_version(WRKSPCdirs, WRKSPCfilenames_both, num_geos, num_planes, zs):

    geo_inds = np.arange(num_geos)
    plane_inds = np.arange(num_planes)

    variable_names = ['B', 'B1', 'f_capture', 'f_shed', 'd']
    B = np.zeros(num_geos, dtype=int)
    B1 = np.zeros(num_geos, dtype=int)
    f_capture = np.zeros(num_geos)
    d = np.zeros(num_geos)
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        matvariables = sio.loadmat(WRKSPCfilenames[0],
                                   variable_names=variable_names)
        B[geo] = int(matvariables['B'])
        B1[geo] = int(matvariables['B1'])
        f_capture[geo] = float(matvariables['f_capture'])
        d[geo] = float(matvariables['d'])
        del matvariables

#    variable_names = ['t']
#    t = (np.zeros(B[0]), np.zeros(B[1]))
#    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
#                                               WRKSPCfilenames_both):
#        os.chdir(WRKSPCdir)
#        matvariables = sio.loadmat(WRKSPCfilenames[0],
#                                   variable_names=variable_names)
#        t[geo][:] = np.squeeze(matvariables['t'])

    variable_names = ['I', 'J', 'dx', 'dy', 'f_shed', 'St_shed', 'U_inf', 'Re']
    variable_names = ['I', 'J', 'f_shed', 'St_shed', 'U_inf']
    I = [np.zeros(num_planes, dtype=int) for _ in geo_inds]
    J = [np.zeros(num_planes, dtype=int) for _ in geo_inds]
    num_vect = [np.zeros(num_planes, dtype=int) for _ in geo_inds]
#    dx = [np.zeros(num_planes) for _ in geo_inds]
#    dy = [np.zeros(num_planes) for _ in geo_inds]
    f_shed = [np.zeros(num_planes) for _ in geo_inds]
    St_shed = [np.zeros(num_planes) for _ in geo_inds]
    U_inf = [np.zeros(num_planes) for _ in geo_inds]
#    Re = [np.zeros(num_planes) for _ in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            I[geo][plane] = int(matvariables['I'])
            J[geo][plane] = int(matvariables['J'])
            num_vect[geo][plane] = I[geo][plane]*J[geo][plane]
#            dx[geo][plane] = float(matvariables['dx'])
#            dy[geo][plane] = float(matvariables['dy'])
            f_shed[geo][plane] = float(matvariables['f_shed'])
            St_shed[geo][plane] = float(matvariables['St_shed'])
            U_inf[geo][plane] = float(matvariables['U_inf'])
#            Re[geo][plane] = float(matvariables['Re'])
            del matvariables

    variable_names = ['x_g', 'y_g']
    x_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
           for geo in geo_inds]
    y_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
           for geo in geo_inds]
    z_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
           for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            x_g_temp = np.squeeze(matvariables['x_g'])
            y_g_temp = np.squeeze(matvariables['y_g'])

            cut_row = (J[geo][plane] - J[geo].min())//2
            cut_col = (I[geo][plane] - I[geo].min())//2
            cut_row_f = cut_row
            cut_row_b = cut_row
            cut_col_f = cut_col
            cut_col_b = cut_col

            if cut_row_b == 0:
                cut_row_b = -J[geo][plane]
            if cut_col_b == 0:
                cut_col_b = -I[geo][plane]

            x_g[geo][:, :, plane] \
                = x_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            y_g[geo][:, :, plane] \
                = y_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            del matvariables
            print d[geo]
#            z_g[geo][:, :, plane] = np.full((J[geo].min(), I[geo].min()),
#                                            zs[plane])/(d[geo]*1000)
            z_g[geo][:, :, plane] = np.full((J[geo].min(), I[geo].min()),
                                            zs[plane])
    return B, B1, f_capture, d, I, J, num_vect, f_shed, St_shed, \
        U_inf,  x_g, y_g, z_g


def load_mean_field(WRKSPCdirs, WRKSPCfilenames_both, I, J, num_geos,
                    num_planes):

    geo_inds = np.arange(num_geos)
    plane_inds = np.arange(num_planes)

    variable_names = ['M_U_g', 'M_V_g', 'M_W_g',
                      'M_ufuf_g', 'M_ufvf_g', 'M_ufwf_g',
                      'M_vfvf_g', 'M_vfwf_g', 'M_wfwf_g']
    M_U_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
             for geo in geo_inds]
    M_V_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
             for geo in geo_inds]
    M_W_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
             for geo in geo_inds]
    M_ufuf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                for geo in geo_inds]
    M_ufvf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                for geo in geo_inds]
    M_ufwf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                for geo in geo_inds]
    M_vfvf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                for geo in geo_inds]
    M_vfwf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                for geo in geo_inds]
    M_wfwf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                for geo in geo_inds]
    M_tke_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
               for geo in geo_inds]
    M_vfvf_ufuf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                     for geo in geo_inds]
    M_wfwf_ufuf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                     for geo in geo_inds]
    M_wfwf_vfvf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                     for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            M_U_g_temp = np.squeeze(matvariables['M_U_g'])
            M_V_g_temp = np.squeeze(matvariables['M_V_g'])
            M_W_g_temp = np.squeeze(matvariables['M_W_g'])
            M_ufuf_g_temp = np.squeeze(matvariables['M_ufuf_g'])
            M_ufvf_g_temp = np.squeeze(matvariables['M_ufvf_g'])
            M_ufwf_g_temp = np.squeeze(matvariables['M_ufwf_g'])
            M_vfvf_g_temp = np.squeeze(matvariables['M_vfvf_g'])
            M_vfwf_g_temp = np.squeeze(matvariables['M_vfwf_g'])
            M_wfwf_g_temp = np.squeeze(matvariables['M_wfwf_g'])

            cut_row = (J[geo][plane] - J[geo].min())//2
            cut_col = (I[geo][plane] - I[geo].min())//2
            cut_row_f = cut_row
            cut_row_b = cut_row
            cut_col_f = cut_col
            cut_col_b = cut_col

            if cut_row_b == 0:
                cut_row_b = -J[geo][plane]
            if cut_col_b == 0:
                cut_col_b = -I[geo][plane]

            M_U_g[geo][:, :, plane] \
                = M_U_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_V_g[geo][:, :, plane] \
                = M_V_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_W_g[geo][:, :, plane] \
                = M_W_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_ufuf_g[geo][:, :, plane] \
                = M_ufuf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_ufvf_g[geo][:, :, plane] \
                = M_ufvf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_ufwf_g[geo][:, :, plane] \
                = M_ufwf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_vfvf_g[geo][:, :, plane] \
                = M_vfvf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_vfwf_g[geo][:, :, plane] \
                = M_vfwf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_wfwf_g[geo][:, :, plane] \
                = M_wfwf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            del matvariables

            M_tke_g[geo][:, :, plane] = (M_ufuf_g[geo][:, :, plane] +
                                         M_vfvf_g[geo][:, :, plane] +
                                         M_wfwf_g[geo][:, :, plane])/2

            M_vfvf_ufuf_g[geo][:, :, plane] = M_vfvf_g[geo][:, :, plane] / \
                                              M_ufuf_g[geo][:, :, plane]
            M_wfwf_ufuf_g[geo][:, :, plane] = M_wfwf_g[geo][:, :, plane] / \
                                              M_ufuf_g[geo][:, :, plane]
            M_wfwf_vfvf_g[geo][:, :, plane] = M_wfwf_g[geo][:, :, plane] / \
                                              M_vfvf_g[geo][:, :, plane]
    return M_U_g, M_V_g, M_W_g, \
        M_ufuf_g, M_ufvf_g, M_ufwf_g, M_vfvf_g, M_vfwf_g, M_wfwf_g, \
        M_tke_g, \
        M_vfvf_ufuf_g, M_wfwf_ufuf_g, M_wfwf_vfvf_g 


def load_mean_field_2D(WRKSPCdirs, WRKSPCfilenames_both, I, J, num_geos,
                       num_planes):

    geo_inds = np.arange(num_geos)
    plane_inds = np.arange(num_planes)

    variable_names = ['M_U_g', 'M_V_g', 'M_W_g',
                      'M_ufuf_g', 'M_ufvf_g', 'M_ufwf_g',
                      'M_vfvf_g', 'M_vfwf_g', 'M_wfwf_g']
    M_U_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
             for geo in geo_inds]
    M_V_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
             for geo in geo_inds]
    M_ufuf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                for geo in geo_inds]
    M_ufvf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                for geo in geo_inds]
    M_vfvf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                for geo in geo_inds]
    M_tke_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
               for geo in geo_inds]
    M_vfvf_ufuf_g = [np.zeros((J[geo].min(), I[geo].min(), num_planes))
                     for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            M_U_g_temp = np.squeeze(matvariables['M_U_g'])
            M_V_g_temp = np.squeeze(matvariables['M_V_g'])
            M_ufuf_g_temp = np.squeeze(matvariables['M_ufuf_g'])
            M_ufvf_g_temp = np.squeeze(matvariables['M_ufvf_g'])
            M_vfvf_g_temp = np.squeeze(matvariables['M_vfvf_g'])

            cut_row = (J[geo][plane] - J[geo].min())//2
            cut_col = (I[geo][plane] - I[geo].min())//2
            cut_row_f = cut_row
            cut_row_b = cut_row
            cut_col_f = cut_col
            cut_col_b = cut_col

            if cut_row_b == 0:
                cut_row_b = -J[geo][plane]
            if cut_col_b == 0:
                cut_col_b = -I[geo][plane]

            M_U_g[geo][:, :, plane] \
                = M_U_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_V_g[geo][:, :, plane] \
                = M_V_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_ufuf_g[geo][:, :, plane] \
                = M_ufuf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_ufvf_g[geo][:, :, plane] \
                = M_ufvf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            M_vfvf_g[geo][:, :, plane] \
                = M_vfvf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b]
            del matvariables

            M_tke_g[geo][:, :, plane] = (M_ufuf_g[geo][:, :, plane] +
                                         M_vfvf_g[geo][:, :, plane])/2

            M_vfvf_ufuf_g[geo][:, :, plane] = M_vfvf_g[geo][:, :, plane] / \
                                              M_ufuf_g[geo][:, :, plane]
    return M_U_g, M_V_g, \
        M_ufuf_g, M_ufvf_g, M_vfvf_g, \
        M_tke_g, \
        M_vfvf_ufuf_g 


def load_POD(WRKSPCdirs, WRKSPCfilenames_both, I, J, B, num_geos, num_planes,
             num_modes):

    geo_inds = np.arange(num_geos)
    plane_inds = np.arange(num_planes)

    variable_names5 = ['akf', 'lambdaf',
                       'akf_sy', 'lambdaf_sy',
                       'akf_as', 'lambdaf_as',
                       'akf_sy_gaus', 'lambdaf_sy_gaus',
                       'akp_sy', 'lambdap_sy']
    akf = [[np.zeros((B[geo], num_modes))
            for plane in plane_inds] for geo in geo_inds]
    lambdaf = [[np.zeros((B[geo]))
                for plane in plane_inds] for geo in geo_inds]
    akf_sy = [[np.zeros((B[geo], num_modes))
               for plane in plane_inds] for geo in geo_inds]
    lambdaf_sy = [[np.zeros((B[geo]))
                   for plane in plane_inds] for geo in geo_inds]
    akf_as = [[np.zeros((B[geo], num_modes))
               for plane in plane_inds] for geo in geo_inds]
    lambdaf_as = [[np.zeros((B[geo]))
                   for plane in plane_inds] for geo in geo_inds]
    akf_sy_gaus = [[np.zeros((B[geo], num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    lambdaf_sy_gaus = [[np.zeros((B[geo]))
                        for plane in plane_inds] for geo in geo_inds]
    akf_sy_harm = [[np.zeros((B[geo], num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    lambdaf_sy_harm = [[np.zeros((B[geo]))
                        for plane in plane_inds] for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names5)
            print 'geo: %1.0f/%1.0f,  plane: %2.0f/%2.0f' % \
                (geo, num_geos - 1, plane, num_planes - 1)

            akf_temp = np.squeeze(matvariables['akf'])
            akf_sy_temp = np.squeeze(matvariables['akf_sy'])
            akf_as_temp = np.squeeze(matvariables['akf_as'])
            akf_sy_gaus_temp = np.squeeze(matvariables['akf_sy_gaus'])
            akf_sy_harm_temp = np.squeeze(matvariables['akp_sy'])

            akf[geo][plane] = akf_temp[:, 0:num_modes]
            akf_sy[geo][plane] = akf_sy_temp[:, 0:num_modes]
            akf_as[geo][plane] = akf_as_temp[:, 0:num_modes]
            akf_sy_gaus[geo][plane] = akf_sy_gaus_temp[:, 0:num_modes]
            akf_sy_harm[geo][plane] = akf_sy_harm_temp[:, 0:num_modes]

            lambdaf[geo][plane] = np.squeeze(matvariables['lambdaf'])
            lambdaf_sy[geo][plane] = np.squeeze(matvariables['lambdaf_sy'])
            lambdaf_as[geo][plane] = np.squeeze(matvariables['lambdaf_as'])
            lambdaf_sy_gaus[geo][plane] = np.squeeze(matvariables['lambdaf_sy_gaus'])
            lambdaf_sy_harm[geo][plane] = np.squeeze(matvariables['lambdap_sy'])

    variable_names = ['Psi_uf', 'Psi_vf', 'Psi_wf',
                       'Psi_uf_sy', 'Psi_vf_sy', 'Psi_wf_sy',
                       'Psi_uf_as', 'Psi_vf_as', 'Psi_wf_as',
                       'Psi_uf_sy_gaus', 'Psi_vf_sy_gaus', 'Psi_wf_sy_gaus',
                       'Psi_up_sy', 'Psi_vp_sy', 'Psi_wp_sy']
    Psi_uf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                 for plane in plane_inds] for geo in geo_inds]
    Psi_vf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                 for plane in plane_inds] for geo in geo_inds]
    Psi_wf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                 for plane in plane_inds] for geo in geo_inds]
    Psi_uf_sy_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    Psi_vf_sy_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    Psi_wf_sy_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    Psi_uf_as_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    Psi_vf_as_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    Psi_wf_as_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    Psi_uf_sy_gaus_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                         for plane in plane_inds] for geo in geo_inds]
    Psi_vf_sy_gaus_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                         for plane in plane_inds] for geo in geo_inds]
    Psi_wf_sy_gaus_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                         for plane in plane_inds] for geo in geo_inds]
    Psi_uf_sy_harm_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                         for plane in plane_inds] for geo in geo_inds]
    Psi_vf_sy_harm_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                         for plane in plane_inds] for geo in geo_inds]
    Psi_wf_sy_harm_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                         for plane in plane_inds] for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            Psi_uf_temp = np.squeeze(matvariables['Psi_uf'])
            Psi_vf_temp = np.squeeze(matvariables['Psi_vf'])
            Psi_wf_temp = np.squeeze(matvariables['Psi_wf'])
            Psi_uf_sy_temp = np.squeeze(matvariables['Psi_uf_sy'])
            Psi_vf_sy_temp = np.squeeze(matvariables['Psi_vf_sy'])
            Psi_wf_sy_temp = np.squeeze(matvariables['Psi_wf_sy'])
            Psi_uf_as_temp = np.squeeze(matvariables['Psi_uf_as'])
            Psi_vf_as_temp = np.squeeze(matvariables['Psi_vf_as'])
            Psi_wf_as_temp = np.squeeze(matvariables['Psi_wf_as'])
            Psi_uf_sy_gaus_temp = np.squeeze(matvariables['Psi_uf_sy_gaus'])
            Psi_vf_sy_gaus_temp = np.squeeze(matvariables['Psi_vf_sy_gaus'])
            Psi_wf_sy_gaus_temp = np.squeeze(matvariables['Psi_wf_sy_gaus'])
            Psi_uf_sy_harm_temp = np.squeeze(matvariables['Psi_up_sy'])
            Psi_vf_sy_harm_temp = np.squeeze(matvariables['Psi_vp_sy'])
            Psi_wf_sy_harm_temp = np.squeeze(matvariables['Psi_wp_sy'])

            Psi_uf_g_temp = gc.togrid(Psi_uf_temp, J[geo][plane],
                                      I[geo][plane])
            Psi_vf_g_temp = gc.togrid(Psi_vf_temp, J[geo][plane],
                                      I[geo][plane])
            Psi_wf_g_temp = gc.togrid(Psi_wf_temp, J[geo][plane],
                                      I[geo][plane])
            Psi_uf_sy_g_temp = gc.togrid(Psi_uf_sy_temp, J[geo][plane],
                                         I[geo][plane])
            Psi_vf_sy_g_temp = gc.togrid(Psi_vf_sy_temp, J[geo][plane],
                                         I[geo][plane])
            Psi_wf_sy_g_temp = gc.togrid(Psi_wf_sy_temp, J[geo][plane],
                                         I[geo][plane])
            Psi_uf_as_g_temp = gc.togrid(Psi_uf_as_temp, J[geo][plane],
                                         I[geo][plane])
            Psi_vf_as_g_temp = gc.togrid(Psi_vf_as_temp, J[geo][plane],
                                         I[geo][plane])
            Psi_wf_as_g_temp = gc.togrid(Psi_wf_as_temp, J[geo][plane],
                                         I[geo][plane])
            Psi_uf_sy_gaus_g_temp = gc.togrid(Psi_uf_sy_gaus_temp,
                                              J[geo][plane], I[geo][plane])
            Psi_vf_sy_gaus_g_temp = gc.togrid(Psi_vf_sy_gaus_temp,
                                              J[geo][plane], I[geo][plane])
            Psi_wf_sy_gaus_g_temp = gc.togrid(Psi_wf_sy_gaus_temp,
                                              J[geo][plane], I[geo][plane])
            Psi_uf_sy_harm_g_temp = gc.togrid(Psi_uf_sy_harm_temp,
                                              J[geo][plane], I[geo][plane])
            Psi_vf_sy_harm_g_temp = gc.togrid(Psi_vf_sy_harm_temp,
                                              J[geo][plane], I[geo][plane])
            Psi_wf_sy_harm_g_temp = gc.togrid(Psi_wf_sy_harm_temp,
                                              J[geo][plane], I[geo][plane])

            cut_row = (J[geo][plane] - J[geo].min())//2
            cut_col = (I[geo][plane] - I[geo].min())//2
            cut_row_f = cut_row
            cut_row_b = cut_row
            cut_col_f = cut_col
            cut_col_b = cut_col

            if cut_row_b == 0:
                cut_row_b = -J[geo][plane]
            if cut_col_b == 0:
                cut_col_b = -I[geo][plane]

            Psi_uf_g[geo][plane][:, :, :] \
                = Psi_uf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_vf_g[geo][plane][:, :, :] \
                = Psi_vf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_wf_g[geo][plane][:, :, :] \
                = Psi_wf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_uf_sy_g[geo][plane][:, :, :] \
                = Psi_uf_sy_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_vf_sy_g[geo][plane][:, :, :] \
                = Psi_vf_sy_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_wf_sy_g[geo][plane][:, :, :] \
                = Psi_wf_sy_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_uf_as_g[geo][plane][:, :, :] \
                = Psi_uf_as_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_vf_as_g[geo][plane][:, :, :] \
                = Psi_vf_as_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_wf_as_g[geo][plane][:, :, :] \
                = Psi_wf_as_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_uf_sy_gaus_g[geo][plane][:, :, :] \
                = Psi_uf_sy_gaus_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_vf_sy_gaus_g[geo][plane][:, :, :] \
                = Psi_vf_sy_gaus_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_wf_sy_gaus_g[geo][plane][:, :, :] \
                = Psi_wf_sy_gaus_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_uf_sy_harm_g[geo][plane][:, :, :] \
                = Psi_uf_sy_harm_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_vf_sy_harm_g[geo][plane][:, :, :] \
                = Psi_vf_sy_harm_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_wf_sy_harm_g[geo][plane][:, :, :] \
                = Psi_wf_sy_harm_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
    return akf, akf_sy, akf_as, akf_sy_gaus, akf_sy_harm, \
        lambdaf, lambdaf_sy, lambdaf_as, lambdaf_sy_gaus, lambdaf_sy_harm, \
        Psi_uf_g, Psi_uf_sy_g, Psi_uf_as_g, Psi_uf_sy_gaus_g, Psi_uf_sy_harm_g, \
        Psi_vf_g, Psi_vf_sy_g, Psi_vf_as_g, Psi_vf_sy_gaus_g, Psi_vf_sy_harm_g, \
        Psi_wf_g, Psi_wf_sy_g, Psi_wf_as_g, Psi_wf_sy_gaus_g, Psi_wf_sy_harm_g


def load_POD_2D(WRKSPCdirs, WRKSPCfilenames_both, I, J, B, num_geos,
                num_planes, num_modes):

    geo_inds = np.arange(num_geos)
    plane_inds = np.arange(num_planes)

    variable_names5 = ['akf', 'lambdaf',
                       'akf_sy', 'lambdaf_sy',
                       'akf_as', 'lambdaf_as',
                       'akf_sy_gaus', 'lambdaf_sy_gaus',
                       'akp_sy', 'lambdap_sy']
    akf = [[np.zeros((B[geo], num_modes))
            for plane in plane_inds] for geo in geo_inds]
    lambdaf = [[np.zeros((B[geo]))
                for plane in plane_inds] for geo in geo_inds]
    akf_sy = [[np.zeros((B[geo], num_modes))
               for plane in plane_inds] for geo in geo_inds]
    lambdaf_sy = [[np.zeros((B[geo]))
                   for plane in plane_inds] for geo in geo_inds]
    akf_as = [[np.zeros((B[geo], num_modes))
               for plane in plane_inds] for geo in geo_inds]
    lambdaf_as = [[np.zeros((B[geo]))
                   for plane in plane_inds] for geo in geo_inds]
    akf_sy_gaus = [[np.zeros((B[geo], num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    lambdaf_sy_gaus = [[np.zeros((B[geo]))
                        for plane in plane_inds] for geo in geo_inds]
    akf_sy_harm = [[np.zeros((B[geo], num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    lambdaf_sy_harm = [[np.zeros((B[geo]))
                        for plane in plane_inds] for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names5)
            print 'geo: %1.0f/%1.0f,  plane: %2.0f/%2.0f' % \
                (geo, num_geos - 1, plane, num_planes - 1)

            akf_temp = np.squeeze(matvariables['akf'])
            akf_sy_temp = np.squeeze(matvariables['akf_sy'])
            akf_as_temp = np.squeeze(matvariables['akf_as'])
            akf_sy_gaus_temp = np.squeeze(matvariables['akf_sy_gaus'])
            akf_sy_harm_temp = np.squeeze(matvariables['akp_sy'])

            akf[geo][plane] = akf_temp[:, 0:num_modes]
            akf_sy[geo][plane] = akf_sy_temp[:, 0:num_modes]
            akf_as[geo][plane] = akf_as_temp[:, 0:num_modes]
            akf_sy_gaus[geo][plane] = akf_sy_gaus_temp[:, 0:num_modes]
            akf_sy_harm[geo][plane] = akf_sy_harm_temp[:, 0:num_modes]

            lambdaf[geo][plane] = np.squeeze(matvariables['lambdaf'])
            lambdaf_sy[geo][plane] = np.squeeze(matvariables['lambdaf_sy'])
            lambdaf_as[geo][plane] = np.squeeze(matvariables['lambdaf_as'])
            lambdaf_sy_gaus[geo][plane] = np.squeeze(matvariables['lambdaf_sy_gaus'])
            lambdaf_sy_harm[geo][plane] = np.squeeze(matvariables['lambdap_sy'])

    variable_names = ['Psi_uf', 'Psi_vf',# 'Psi_wf',
                       'Psi_uf_sy', 'Psi_vf_sy',# 'Psi_wf_sy',
                       'Psi_uf_as', 'Psi_vf_as',# 'Psi_wf_as',
                       'Psi_uf_sy_gaus', 'Psi_vf_sy_gaus',# 'Psi_wf_sy_gaus',
                       'Psi_up_sy', 'Psi_vp_sy']#, 'Psi_wp_sy']
    Psi_uf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                 for plane in plane_inds] for geo in geo_inds]
    Psi_vf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                 for plane in plane_inds] for geo in geo_inds]
#    Psi_wf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                 for plane in plane_inds] for geo in geo_inds]
    Psi_uf_sy_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    Psi_vf_sy_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
#    Psi_wf_sy_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                    for plane in plane_inds] for geo in geo_inds]
    Psi_uf_as_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
    Psi_vf_as_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                    for plane in plane_inds] for geo in geo_inds]
#    Psi_wf_as_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                    for plane in plane_inds] for geo in geo_inds]
    Psi_uf_sy_gaus_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                         for plane in plane_inds] for geo in geo_inds]
    Psi_vf_sy_gaus_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                         for plane in plane_inds] for geo in geo_inds]
#    Psi_wf_sy_gaus_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                         for plane in plane_inds] for geo in geo_inds]
    Psi_uf_sy_harm_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                         for plane in plane_inds] for geo in geo_inds]
    Psi_vf_sy_harm_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
                         for plane in plane_inds] for geo in geo_inds]
#    Psi_wf_sy_harm_g = [[np.zeros((J[geo].min(), I[geo].min(), num_modes))
#                         for plane in plane_inds] for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            Psi_uf_temp = np.squeeze(matvariables['Psi_uf'])
            Psi_vf_temp = np.squeeze(matvariables['Psi_vf'])
#            Psi_wf_temp = np.squeeze(matvariables['Psi_wf'])
            Psi_uf_sy_temp = np.squeeze(matvariables['Psi_uf_sy'])
            Psi_vf_sy_temp = np.squeeze(matvariables['Psi_vf_sy'])
#            Psi_wf_sy_temp = np.squeeze(matvariables['Psi_wf_sy'])
            Psi_uf_as_temp = np.squeeze(matvariables['Psi_uf_as'])
            Psi_vf_as_temp = np.squeeze(matvariables['Psi_vf_as'])
#            Psi_wf_as_temp = np.squeeze(matvariables['Psi_wf_as'])
            Psi_uf_sy_gaus_temp = np.squeeze(matvariables['Psi_uf_sy_gaus'])
            Psi_vf_sy_gaus_temp = np.squeeze(matvariables['Psi_vf_sy_gaus'])
#            Psi_wf_sy_gaus_temp = np.squeeze(matvariables['Psi_wf_sy_gaus'])
            Psi_uf_sy_harm_temp = np.squeeze(matvariables['Psi_up_sy'])
            Psi_vf_sy_harm_temp = np.squeeze(matvariables['Psi_vp_sy'])
#            Psi_wf_sy_harm_temp = np.squeeze(matvariables['Psi_wp_sy'])

            Psi_uf_g_temp = gc.togrid(Psi_uf_temp, J[geo][plane],
                                      I[geo][plane])
            Psi_vf_g_temp = gc.togrid(Psi_vf_temp, J[geo][plane],
                                      I[geo][plane])
#            Psi_wf_g_temp = gc.togrid(Psi_wf_temp, J[geo][plane],
#                                      I[geo][plane])
            Psi_uf_sy_g_temp = gc.togrid(Psi_uf_sy_temp, J[geo][plane],
                                         I[geo][plane])
            Psi_vf_sy_g_temp = gc.togrid(Psi_vf_sy_temp, J[geo][plane],
                                         I[geo][plane])
#            Psi_wf_sy_g_temp = gc.togrid(Psi_wf_sy_temp, J[geo][plane],
#                                         I[geo][plane])
            Psi_uf_as_g_temp = gc.togrid(Psi_uf_as_temp, J[geo][plane],
                                         I[geo][plane])
            Psi_vf_as_g_temp = gc.togrid(Psi_vf_as_temp, J[geo][plane],
                                         I[geo][plane])
#            Psi_wf_as_g_temp = gc.togrid(Psi_wf_as_temp, J[geo][plane],
#                                         I[geo][plane])
            Psi_uf_sy_gaus_g_temp = gc.togrid(Psi_uf_sy_gaus_temp,
                                              J[geo][plane], I[geo][plane])
            Psi_vf_sy_gaus_g_temp = gc.togrid(Psi_vf_sy_gaus_temp,
                                              J[geo][plane], I[geo][plane])
#            Psi_wf_sy_gaus_g_temp = gc.togrid(Psi_wf_sy_gaus_temp,
#                                              J[geo][plane], I[geo][plane])
            Psi_uf_sy_harm_g_temp = gc.togrid(Psi_uf_sy_harm_temp,
                                              J[geo][plane], I[geo][plane])
            Psi_vf_sy_harm_g_temp = gc.togrid(Psi_vf_sy_harm_temp,
                                              J[geo][plane], I[geo][plane])
#            Psi_wf_sy_harm_g_temp = gc.togrid(Psi_wf_sy_harm_temp,
#                                              J[geo][plane], I[geo][plane])

            cut_row = (J[geo][plane] - J[geo].min())//2
            cut_col = (I[geo][plane] - I[geo].min())//2
            cut_row_f = cut_row
            cut_row_b = cut_row
            cut_col_f = cut_col
            cut_col_b = cut_col

            if cut_row_b == 0:
                cut_row_b = -J[geo][plane]
            if cut_col_b == 0:
                cut_col_b = -I[geo][plane]

            Psi_uf_g[geo][plane][:, :, :] \
                = Psi_uf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_vf_g[geo][plane][:, :, :] \
                = Psi_vf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#            Psi_wf_g[geo][plane][:, :, :] \
#                = Psi_wf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_uf_sy_g[geo][plane][:, :, :] \
                = Psi_uf_sy_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_vf_sy_g[geo][plane][:, :, :] \
                = Psi_vf_sy_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#            Psi_wf_sy_g[geo][plane][:, :, :] \
#                = Psi_wf_sy_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_uf_as_g[geo][plane][:, :, :] \
                = Psi_uf_as_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_vf_as_g[geo][plane][:, :, :] \
                = Psi_vf_as_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#            Psi_wf_as_g[geo][plane][:, :, :] \
#                = Psi_wf_as_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_uf_sy_gaus_g[geo][plane][:, :, :] \
                = Psi_uf_sy_gaus_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_vf_sy_gaus_g[geo][plane][:, :, :] \
                = Psi_vf_sy_gaus_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#            Psi_wf_sy_gaus_g[geo][plane][:, :, :] \
#                = Psi_wf_sy_gaus_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_uf_sy_harm_g[geo][plane][:, :, :] \
                = Psi_uf_sy_harm_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
            Psi_vf_sy_harm_g[geo][plane][:, :, :] \
                = Psi_vf_sy_harm_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
#            Psi_wf_sy_harm_g[geo][plane][:, :, :] \
#                = Psi_wf_sy_harm_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_modes]
    return akf, akf_sy, akf_as, akf_sy_gaus, akf_sy_harm, \
        lambdaf, lambdaf_sy, lambdaf_as, lambdaf_sy_gaus, lambdaf_sy_harm, \
        Psi_uf_g, Psi_uf_sy_g, Psi_uf_as_g, Psi_uf_sy_gaus_g, Psi_uf_sy_harm_g, \
        Psi_vf_g, Psi_vf_sy_g, Psi_vf_as_g, Psi_vf_sy_gaus_g, Psi_vf_sy_harm_g#, \
#        Psi_wf_g, Psi_wf_sy_g, Psi_wf_as_g, Psi_wf_sy_gaus_g, Psi_wf_sy_harm_g


def load_fluctuating_snapshots(WRKSPCdirs, WRKSPCfilenames_both, I, J, B, num_geos, num_planes,
                   num_snaps):

    geo_inds = np.arange(num_geos)
    plane_inds = np.arange(num_planes)

    variable_names = ['uf_g', 'vf_g', 'wf_g']

    uf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
             for plane in plane_inds] for geo in geo_inds]
    vf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
             for plane in plane_inds] for geo in geo_inds]
    wf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
             for plane in plane_inds] for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            print 'loading %s snapshots' % WRKSPCfilename
            uf_g_temp = np.squeeze(matvariables['uf_g'])
            vf_g_temp = np.squeeze(matvariables['vf_g'])
            wf_g_temp = np.squeeze(matvariables['wf_g'])

            cut_row = (J[geo][plane] - J[geo].min())//2
            cut_col = (I[geo][plane] - I[geo].min())//2
            cut_row_f = cut_row
            cut_row_b = cut_row
            cut_col_f = cut_col
            cut_col_b = cut_col

            if cut_row_b == 0:
                cut_row_b = -J[geo][plane]
            if cut_col_b == 0:
                cut_col_b = -I[geo][plane]

            uf_g[geo][plane][:, :, :] \
                = uf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
            vf_g[geo][plane][:, :, :] \
                = vf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
            wf_g[geo][plane][:, :, :] \
                = wf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
    return uf_g, vf_g, wf_g


def load_fluctuating_snapshots_2D(WRKSPCdirs, WRKSPCfilenames_both, I, J, B, num_geos, num_planes,
                   num_snaps):

    geo_inds = np.arange(num_geos)
    plane_inds = np.arange(num_planes)

    variable_names = ['uf_g', 'vf_g']#, 'wf_g']

    uf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
             for plane in plane_inds] for geo in geo_inds]
    vf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
             for plane in plane_inds] for geo in geo_inds]
#    wf_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
#             for plane in plane_inds] for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            print 'loading %s snapshots' % WRKSPCfilename
            uf_g_temp = np.squeeze(matvariables['uf_g'])
            vf_g_temp = np.squeeze(matvariables['vf_g'])
#            wf_g_temp = np.squeeze(matvariables['wf_g'])

            cut_row = (J[geo][plane] - J[geo].min())//2
            cut_col = (I[geo][plane] - I[geo].min())//2
            cut_row_f = cut_row
            cut_row_b = cut_row
            cut_col_f = cut_col
            cut_col_b = cut_col

            if cut_row_b == 0:
                cut_row_b = -J[geo][plane]
            if cut_col_b == 0:
                cut_col_b = -I[geo][plane]

            uf_g[geo][plane][:, :, :] \
                = uf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
            vf_g[geo][plane][:, :, :] \
                = vf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
#            wf_g[geo][plane][:, :, :] \
#                = wf_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
    return uf_g, vf_g#, wf_g


def load_snapshots(WRKSPCdirs, WRKSPCfilenames_both, I, J, B, num_geos, num_planes,
                   num_snaps):

    geo_inds = np.arange(num_geos)
    plane_inds = np.arange(num_planes)

    variable_names = ['u_snap_g', 'v_snap_g', 'w_snap_g']

    u_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
             for plane in plane_inds] for geo in geo_inds]
    v_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
             for plane in plane_inds] for geo in geo_inds]
    w_g = [[np.zeros((J[geo].min(), I[geo].min(), num_snaps))
             for plane in plane_inds] for geo in geo_inds]
    for geo, WRKSPCdir, WRKSPCfilenames in zip(geo_inds, WRKSPCdirs,
                                               WRKSPCfilenames_both):
        os.chdir(WRKSPCdir)
        for plane, WRKSPCfilename in zip(plane_inds, WRKSPCfilenames):
            matvariables = sio.loadmat(WRKSPCfilename,
                                       variable_names=variable_names)
            print 'loading %s snapshots' % WRKSPCfilename
            u_g_temp = np.squeeze(matvariables['u_snap_g'])
            v_g_temp = np.squeeze(matvariables['v_snap_g'])
            w_g_temp = np.squeeze(matvariables['w_snap_g'])

            cut_row = (J[geo][plane] - J[geo].min())//2
            cut_col = (I[geo][plane] - I[geo].min())//2
            cut_row_f = cut_row
            cut_row_b = cut_row
            cut_col_f = cut_col
            cut_col_b = cut_col

            if cut_row_b == 0:
                cut_row_b = -J[geo][plane]
            if cut_col_b == 0:
                cut_col_b = -I[geo][plane]

            u_g[geo][plane][:, :, :] \
                = u_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
            v_g[geo][plane][:, :, :] \
                = v_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
            w_g[geo][plane][:, :, :] \
                = w_g_temp[cut_row_f:-cut_row_b, cut_col_f:-cut_col_b, 0:num_snaps]
    return u_g, v_g, w_g