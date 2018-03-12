from __future__ import division
import numpy as np
import os
import h5py

import grid_conversion as gc


###############################################################################


def save_data_to_txt(folder_path, file_name, data, header=''):
    fmt = '%.6f'
    file_ext = '.dat'
    if file_name.count('.') > 0:
        file_name = file_name[:file_name.find('.')]
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
        folder_conts = os.listdir(folder_path)
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
        if data_path in f:
            del f[data_path]
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
    folder_conts = os.listdir(main_folder_path)
    poss_file_names = [folder_cont for folder_cont in folder_conts
                       if folder_cont.endswith('.hdf5')]
    poss_file_paths = main_folder_path + poss_file_names
    sort_poss_file_paths = poss_file_paths.sort(
            key=lambda poss_file_path: os.path.getmtime(poss_file_path)
            )
    file_path = sort_poss_file_paths[0]
    return file_path


###############################################################################
# Geometry


def save_2D_geometry_to_txt(x, y, folder_path, header=True):
    if header:
        header = 'x(mm), y(mm)'
    else:
        header = ''
    file_name = 'geometry'
    data = np.swapaxes(np.vstack([x, y]), 0, 1)
    save_data_to_txt(folder_path, file_name, data, header=header)


def load_2D_geometry_from_txt(folder_path):
    file_name = 'geometry'
    data = load_data_from_txt(folder_path, file_name)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def save_2D_geometry_to_hdf5(x, y, folder_path, file_name, group_path):
    save_data_to_hdf5(folder_path, file_name, group_path, 'x', x)
    save_data_to_hdf5(folder_path, file_name, group_path, 'y', y)


def load_2D_geometry_from_hdf5(folder_path, file_name, group_path):
    x = load_data_from_hdf5(folder_path, file_name, group_path, 'x')
    y = load_data_from_hdf5(folder_path, file_name, group_path, 'y')
    return x, y


def save_2D_geometry(x, y, main_folder_path, folder_path, data_type,
                     save_method='hdf5', file_name='', header=True):
    if save_method == 'hdf5':
        if not file_name:
            file_name = guess_file_name_hdf5(main_folder_path)
        group_path = folder_path.replace('\\', '/')
        if data_type == '2D3C':
            save_2D_geometry_to_hdf5(x, y, main_folder_path, file_name,
                                     group_path)
        else:
            print 'Add functionality here'
    elif save_method == 'txt':
        folder_path = main_folder_path + '\\' + folder_path.replace('/', '\\')
        if data_type == '2D3C':
            save_2D_geometry_to_txt(x, y, folder_path, header=header)
        else:
            print 'Add functionality here'


def load_2D_geometry(main_folder_path, folder_path, data_type,
                     save_method='hdf5', file_name=''):
    if save_method == 'hdf5':
        if not file_name:
            file_name = guess_file_name_hdf5(main_folder_path)
        if file_name.count('.') > 0:
            file_name = file_name[:file_name.find('.')]
        file_name = file_name + '.hdf5'
        group_path = folder_path.replace('\\', '/')
        if data_type == '2D3C':
            x, y = load_2D_geometry_from_hdf5(main_folder_path, file_name,
                                              group_path)
        else:
            print 'Add functionality here'
    elif save_method == 'txt':
        folder_path = main_folder_path + '\\' + folder_path.replace('/', '\\')
        if data_type == '2D3C':
            x, y = load_2D_geometry_from_txt(folder_path)
        else:
            print 'Add functionality here'
    return x, y


###############################################################################
# Velocity Snapshots


def save_3C_velocity_snapshots_to_txt(u, v, w, folder_path, header=True):
    if header:
        header = 'u(m/s), v(m/s), w(m/s)'
    else:
        header = ''
    file_name = 'B%05.0f'
    data = np.swapaxes(np.dstack([u, v, w]), 1, 2)
    save_data_to_txt(folder_path, file_name, data, header=header)


def load_3C_velocity_snapshots_from_txt(folder_path):
    data = load_data_from_txt(folder_path)
    u = data[:, 0, :]
    v = data[:, 1, :]
    w = data[:, 2, :]
    return u, v, w


def save_3C_velocity_snapshots_to_hdf5(u, v, w, folder_path, file_name,
                                       group_path):
    save_data_to_hdf5(folder_path, file_name, group_path, 'u', u)
    save_data_to_hdf5(folder_path, file_name, group_path, 'v', v)
    save_data_to_hdf5(folder_path, file_name, group_path, 'w', w)


def load_3C_velocity_snapshots_from_hdf5(folder_path, file_name, group_path):
    u = load_data_from_hdf5(folder_path, file_name, group_path, 'u')
    v = load_data_from_hdf5(folder_path, file_name, group_path, 'v')
    w = load_data_from_hdf5(folder_path, file_name, group_path, 'w')
    return u, v, w


def save_3C_velocity_snapshots(u, v, w, main_folder_path, data_type,
                               folder_path, save_method='hdf5', file_name='',
                               header=True):
    if save_method == 'hdf5':
        if not file_name:
            file_name = guess_file_name_hdf5(main_folder_path)
        group_path = folder_path.replace('\\', '/')
        if data_type == '2D3C':
            save_3C_velocity_snapshots_to_hdf5(u, v, w, main_folder_path,
                                               file_name, group_path)
        else:
            print 'Add functionality here'
    elif save_method == 'txt':
        folder_path = main_folder_path + '\\' + folder_path.replace('/', '\\')
        if data_type == '2D3C':
            save_3C_velocity_snapshots_to_txt(u, v, w, folder_path,
                                              header=header)
        else:
            print 'Add functionality here'


def load_3C_velocity_snapshots(main_folder_path, folder_path, data_type,
                               save_method='hdf5', file_name=''):
    if save_method == 'hdf5':
        if not file_name:
            file_name = guess_file_name_hdf5(main_folder_path)
        if file_name.count('.') > 0:
            file_name = file_name[:file_name.find('.')]
        file_name = file_name + '.hdf5'
        group_path = folder_path.replace('\\', '/')
        if data_type == '2D3C':
            u, v, w = load_3C_velocity_snapshots_from_hdf5(
                    main_folder_path, file_name, group_path)
        else:
            print 'Add functionality here'
    elif save_method == 'txt':
        folder_path = main_folder_path + '\\' + folder_path.replace('/', '\\')
        if data_type == '2D3C':
            u, v, w = load_3C_velocity_snapshots_from_txt(folder_path)
        else:
            print 'Add functionality here'
    return u, v, w


###############################################################################
# Mean Velocities


def save_3C_mean_velocities_to_txt(M_U, M_V, M_W, folder_path, header=True):
    if header:
        header = 'Mean U(m/s), Mean V(m/s), Mean W(m/s)'
    else:
        header = ''
    file_name = 'B%05.0f'
    data = np.swapaxes(np.dstack([M_U, M_V, M_W]), 1, 2)
    save_data_to_txt(folder_path, file_name, data, header=header)


def load_3C_mean_velocities_from_txt(folder_path):
    data = load_data_from_txt(folder_path)
    M_U = data[:, 0, :]
    M_V = data[:, 1, :]
    M_W = data[:, 2, :]
    return M_U, M_V, M_W


def save_3C_mean_velocities_to_hdf5(M_U, M_V, M_W, folder_path, file_name,
                                    group_path):
    save_data_to_hdf5(folder_path, file_name, group_path, 'M_U', M_U)
    save_data_to_hdf5(folder_path, file_name, group_path, 'M_V', M_V)
    save_data_to_hdf5(folder_path, file_name, group_path, 'M_W', M_W)


def load_3C_mean_velocities_from_hdf5(folder_path, file_name, group_path):
    M_U = load_data_from_hdf5(folder_path, file_name, group_path, 'M_U')
    M_V = load_data_from_hdf5(folder_path, file_name, group_path, 'M_V')
    M_W = load_data_from_hdf5(folder_path, file_name, group_path, 'M_W')
    return M_U, M_V, M_W


def save_3C_mean_velocities(M_U, M_V, M_W, main_folder_path, data_type,
                            folder_path, save_method='hdf5', file_name='',
                            header=True):
    if save_method == 'hdf5':
        if not file_name:
            file_name = guess_file_name_hdf5(main_folder_path)
        group_path = folder_path.replace('\\', '/')
        if data_type == '2D3C':
            save_3C_velocity_snapshots_to_hdf5(M_U, M_V, M_W, main_folder_path,
                                               file_name, group_path)
        else:
            print 'Add functionality here'
    elif save_method == 'txt':
        folder_path = main_folder_path + '\\' + folder_path.replace('/', '\\')
        if data_type == '2D3C':
            save_3C_velocity_snapshots_to_txt(M_U, M_V, M_W, folder_path,
                                              header=header)
        else:
            print 'Add functionality here'


def load_3C_mean_velocities(main_folder_path, folder_path, data_type,
                            save_method='hdf5', file_name=''):
    if save_method == 'hdf5':
        if not file_name:
            file_name = guess_file_name_hdf5(main_folder_path)
        if file_name.count('.') > 0:
            file_name = file_name[:file_name.find('.')]
        file_name = file_name + '.hdf5'
        group_path = folder_path.replace('\\', '/')
        if data_type == '2D3C':
            M_U, M_V, M_W = load_3C_velocity_snapshots_from_hdf5(
                    main_folder_path, file_name, group_path)
        else:
            print 'Add functionality here'
    elif save_method == 'txt':
        folder_path = main_folder_path + '\\' + folder_path.replace('/', '\\')
        if data_type == '2D3C':
            M_U, M_V, M_W = load_3C_velocity_snapshots_from_txt(folder_path)
        else:
            print 'Add functionality here'
    return M_U, M_V, M_W
