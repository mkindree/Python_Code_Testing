from __future__ import division
import numpy as np
import os
import h5py

import grid_conversion as gc


###############################################################################


def save_data_to_txt(folder_path, file_name, data, header='', fmt='%.6f',
                     file_ext='.dat'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if data.ndim < 3:
        file_path = folder_path + '\\' + file_name
        print 'Writing: ' + file_path
        np.savetxt(file_path + file_ext, data, fmt=fmt, header=header)
    elif data.ndim == 3:
        file_inds = np.arange(np.size(data, axis=2))
        for file_ind in file_inds:
            file_num = file_ind + 1
            file_path = folder_path + '\\' + file_name % file_num
            print 'Writing: ' + file_path
            np.savetxt(file_path + file_ext, data[:, :, file_ind], fmt=fmt,
                       header=header)


def load_data_from_txt(folder_path, file_name='', file_ext='.dat'):
    if not file_name == '':
        file_path = folder_path + '\\' + file_name
        print 'Reading: ' + file_path
        data = np.loadtxt(file_path)
    else:
        file_cont = os.listdir(folder_path)
        file_names = [folder_cont for folder_cont in folder_conts
                      if folder_cont.endswith(file_ext)]
        file_path0 = folder_path + '\\' + file_names[0]
        file_data0 = np.loadtxt(file_path0)
        length = np.size(file_data0, axis=0)
        width = np.size(file_data0, axis=1)
        depth = np.size(file_names, axis=0)
        data = np.zeros((length, width, depth))
        for file_name, ind in zip(file_names, np.arange(depth)):
            file_path = folder_path + '\\' + file_name
            print 'Reading: ' + file_path
            file_data = np.loadtxt(file_path)
            data[:, :, ind] = file_data
    return data


def save_data_to_hdf5(folder_path, file_name, group_path, data_name, data):
        file_path = folder_path + '\\' + file_name
        if os.path.exists(file_path):
            f = h5py.File(file_path, 'r+')
        else:
            f = h5py.File(file_path, 'w-')
        data_path = group_path + '/' + data_name
        f[data_path] = data
        f.close()


def load_data_from_hdf5(folder_path, file_name, group_path, data_name):
    file_path = folder_path + '\\' + file_name
    f = h5py.File(file_path, 'r')
    data_path = group_path + '/' + data_name
    data = np.array(f[data_path])
    f.close()
    return data


def guess_file_name_hdf5(main_folder_path):
    folder_conts = os.listdir(folder_path)
    poss_file_names = [folder_cont for folder_cont in folder_conts
                       if folder_cont.endswith('.hdf5')]
    sort_poss_file_paths = poss_file_paths.sort(
            key=lambda poss_file_path: os.path.getmtime(poss_file_path)
            )
    file_path = sort_poss_file_paths[0]
    return file_path


###############################################################################
# Geometry


def save_2D_geometry_to_txt(x_g, y_g, num_x, num_y, folder_path, header=True,
                            fmt='%.6f', file_ext='.dat'):
    if header:
        header = 'x(mm), y(mm)'
    else:
        header = ''
    file_name = 'geometry'
    x = gc.fromgrid(x_g, num_x, num_y)
    y = gc.fromgrid(y_g, num_x, num_y)
    data = np.swapaxes(np.vstack([x, y]), 0, 1)
    save_data_to_txt(folder_path, file_name, data, header=header, fmt=fmt,
                     file_ext=file_ext)


def load_2D_geometry_from_txt(num_x, num_y, folder_path, file_ext='.dat'):
    file_name = 'geometry'
    data = load_data_from_txt(folder_path, file_name, file_ext=file_ext)
    x = data[:, 0]
    y = data[:, 1]
    x_g = gc.togrid(x, num_x, num_y)
    y_g = gc.togrid(y, num_x, num_y)
    return x_g, y_g


def save_2D_geometry_to_hdf5(x_g, y_g, folder_path, file_name, group_path):
    save_data_to_hdf5(folder_path, file_name, group_path, 'x_g', x_g)
    save_data_to_hdf5(folder_path, file_name, group_path, 'y_g', y_g)


def load_2D_geometry_from_hdf5(folder_path, file_name, group_path):
    x_g = load_data_from_hdf5(folder_path, file_name, group_path, 'x_g')
    y_g = load_data_from_hdf5(folder_path, file_name, group_path, 'y_g')
    return x_g, y_g


def save_2D_geometry(x_g, y_g, num_x, num_y, main_folder_path, folder_path,
                     save_method='hdf5', file_name='', header=True, fmt='%.6f',
                     file_ext='.dat'):
    if save_method == 'hdf5':
        if not file_name:
            file_name = guess_file_name_hdf5(main_folder)
        group_path = folder_path.replace('\\', '/')
        save_2D_geometry_to_hdf5(x_g, y_g, main_folder_path,
                                 file_name, group_path)
    elif save_method == 'txt':
        folder_path = main_folder + '\\' + folder_path.replace('/', '\\')
        save_2D_geometry_to_txt(x_g, y_g, num_x, num_y, folder_path,
                                header=header, fmt=fmt, file_ext=file_ext)


def load_2D_geometry(main_folder_path, folder_path, save_method='hdf5',
                     file_name='', num_x, num_y, file_ext='.dat'):
    if save_method == 'hdf5':
        if not file_name:
            file_name = guess_file_name_hdf5(main_folder)
        group_path = folder_path.replace('\\', '/')
        x_g, y_g = load_2D_geometry_from_hdf5(main_folder_path, file_name,
                                              group_path)
    elif save_method == 'txt':
        folder_path = main_folder + '\\' + folder_path.replace('/', '\\')
        x_g, y_g = load_2D_geometry_from_txt(num_x, num_y, folder_path,
                                             file_ext=file_ext)
    return x_g, y_g


###############################################################################
# Velocity Snapshots


def save_3C_velocity_snapshots_to_txt(u_g, v_g, w_g, num_x, num_y, folder_path,
                                      header=True, fmt='%.6f', file_ext='.dat'):
    if header:
        header = 'u(m/s), v(m/s), w(m/s)'
    else:
        header = ''
    file_name = 'B%05.0f'
    u = gc.fromgrid(u_g, num_x, num_y)
    v = gc.fromgrid(v_g, num_x, num_y)
    w = gc.fromgrid(w_g, num_x, num_y)
    data = np.swapaxes(np.dstack([u, v, w]), 1, 2)
    save_data_to_txt(folder_path, file_name, data, header=header)


def load_3C_velocity_snapshots_from_txt(num_x, num_y, folder_path,
                                        file_ext='.dat'):
    data = load_data_from_txt(folder_path, file_ext=file_ext)
    u = data[:, 0, :]
    v = data[:, 1, :]
    w = data[:, 2, :]
    u_g = gc.togrid(u, num_x, num_y)
    v_g = gc.togrid(v, num_x, num_y)
    w_g = gc.togrid(w, num_x, num_y)
    return u_g, v_g, w_g


def save_3C_velocity_snapshots_to_hdf5(u_g, v_g, w_g, folder_path, file_name,
                                       group_path):
    save_data_to_hdf5(folder_path, file_name, group_path, 'u_g', u_g)
    save_data_to_hdf5(folder_path, file_name, group_path, 'v_g', v_g)
    save_data_to_hdf5(folder_path, file_name, group_path, 'w_g', w_g)


def load_3C_velocity_snapshots_from_hdf5(folder_path, file_name, group_path):
    u_g = load_data_from_hdf5(folder_path, file_name, group_path, 'u_g')
    v_g = load_data_from_hdf5(folder_path, file_name, group_path, 'v_g')
    w_g = load_data_from_hdf5(folder_path, file_name, group_path, 'w_g')


def save_3C_velocity_snapshots(x_g, y_g, num_x, num_y, main_folder_path,
                               folder_path, save_method='hdf5', file_name='',
                               header=True, fmt='%.6f', file_ext='.dat'):
    if save_method == 'hdf5':
        if not file_name:
            file_name = guess_file_name_hdf5(main_folder)
        group_path = folder_path.replace('\\', '/')
        save_3C_velocity_snapshots_to_hdf5(u_g, v_g, w_g, main_folder_path,
                                           file_name, group_path)
    elif save_method == 'txt':
        folder_path = main_folder + '\\' + folder_path.replace('/', '\\')
        save_3C_velocity_snapshots_to_txt(u_g, v_g, w_g, num_x, num_y,
                                          folder_path, header=header, fmt=fmt,
                                          file_ext=file_ext)


def load_3C_velocity_snapshots(main_folder_path, folder_path,
                               save_method='hdf5', file_name='', num_x, num_y,
                               file_ext='.dat'):
    if save_method == 'hdf5':
        if not file_name:
            file_name = guess_file_name_hdf5(main_folder)
        group_path = folder_path.replace('\\', '/')
        u_g, v_g, w_g = load_3C_velocity_snapshots_from_hdf5(
                main_folder_path, file_name, group_path)
    elif save_method == 'txt':
        folder_path = main_folder + '\\' + folder_path.replace('/', '\\')
        u_g, v_g, w_g = load_3C_velocity_snapshots_from_txt(
                num_x, num_y, folder_path, file_ext=file_ext)
    return u_g, v_g, w_g

