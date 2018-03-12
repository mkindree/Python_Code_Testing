from __future__ import division
import sys

CODEdir = r'D:\6 - Circle_AR4_LaminarBL\PIV\Test_1_2_3\PIV\Test_1_2_3_testing\Python Codes'
if CODEdir not in sys.path:
    sys.path.append(CODEdir)

import numpy as np
import matplotlib.pyplot as plt
#import os
import copy as copy
import scipy.interpolate as spinterp
import scipy.signal as spsig

import plotting_functions as pltf
#import grid_conversion as gc
import spectrum_fft_v2 as spect
#import load_from_matlab_workspace as load_mat


#%% User-Defined Variables

mainfolder_p = r'D:\7 - SquareAR4_LaminarBL'  # The main working directory
PUBdir_p = r'D:\EXF Paper\EXF Paper V4\figs'  # Where the .eps figures go

PRESSdirs = (r'G:\Pressure\Pressure Data', r'G:\Pressure\Pressure Data')
PRESSfilenames = [[r'0093', r'0094', r'0095', r'0096', r'0097', r'0099'],
                  [r'0376', r'0377', r'0379', r'0380', r'0381', r'0382']]
#PRESSfilenames = [[r'0093', r'0094', r'0095'], 
#                  [r'0376', r'0377', r'0379']]
ZEROPRESSfilenames = [[r'0090', r'0091', r'0092'], 
                      [r'0373', r'0374', r'0375']]
file_suffix_p = r'_R10240.000_RawV'
file_ext_p = r'.csv'

num_geos_p = 2
obstacles_p = ['circle', 'square']
obstacle_AR_p = np.array([4.01, 3.84])
geo_inds_p = np.arange(num_geos_p, dtype=int)

num_trials_p = [np.size(PRESSfilenames[geo]) for geo in geo_inds_p]
trial_inds_p = [np.arange(num_trials_p[geo], dtype=int) for geo in geo_inds_p]
num_zero_trials_p = [np.size(ZEROPRESSfilenames[geo]) for geo in geo_inds_p]
zero_trial_inds_p = [np.arange(num_zero_trials_p[geo], dtype=int)
                     for geo in geo_inds_p]

PRESSfile_paths = [[PRESSdirs[geo] + '\\' + PRESSfilenames[geo][trial] +
                    file_suffix_p + file_ext_p for trial in trial_inds_p[geo]]
                   for geo in geo_inds_p]
ZEROPRESSfile_paths = [[PRESSdirs[geo] + '\\' +
                        ZEROPRESSfilenames[geo][trial] + file_suffix_p +
                        file_ext_p for trial in zero_trial_inds_p[geo]]
                       for geo in geo_inds_p]


# Parameters
d_p = [0.0127, 0.0133]  # Diameter of obstacle (m)
d_p = [0.01290, 0.01305]  # Diameter of obstacle (m)
f_capture_p = [10240]*num_geos_p  # Pressure sampling frequency (Hz)
calc_rho_and_mew = [True]*num_geos_p  # Calculate rho and mew?
mew = [1.8*10**-5]*num_geos_p  # Viscosity (kg/s/m) Will be overwritten if calc_rho_and_mew = True
rho = [1.05]*num_geos_p  # Density (kg/m^3) Will be overwritten if calc_rho_and_mew = True
P_atm = [89.68, 89.47]  # Atmospheric pressure (kPa). Only used if calc_rho = 1
T_amb = [22.7, 23.9]  # Ambient temperature (deg C). Only used if calc_rho = 1
inchH2OtoPa = 248.84  # Conversion factor to convert inch H2O to Pa

# Pressure trial info
keep_channel = [[True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, False, True]]*num_geos_p
chan_IDs_p = [np.arange(np.size(keep_channel[geo]), dtype=int) + 1
              for geo in geo_inds_p]  # Indexing from 1
chan_IDs_p = [chan_IDs_p[geo][keep_channel[geo]] for geo in geo_inds_p]
chan_IDs_p_g = [np.array([[1, 2, 3, 9, 12, 15, 18],
                          [0, 0, 0, 8, 11, 14, 0],
                          [6, 5, 4, 7, 10, 13, 17]])]*num_geos_p
x_p_g = [np.array([[-0.003175, 0, 0.003175, 0.0127, 0.01905, 0.0254, 0.047625],
                   [-0.00635, 0, 0.00635, 0.0127, 0.01905, 0.0254, 0.047625],
                   [-0.003175, 0, 0.003175, 0.0127, 0.01905, 0.0254, 0.047625]]),
         np.array([[-0.003175, 0, 0.003175, 0.0127, 0.01905, 0.0254, 0.047625],
                   [-0.00635, 0, 0.00635, 0.0127, 0.01905, 0.0254, 0.047625],
                   [-0.003175, 0, 0.003175, 0.0127, 0.01905, 0.0254, 0.047625]])]
y_p_g = [np.array([[0.0055, 0.00635, 0.0055, 0.00635, 0.00635, 0.00635, 0.0127],
                   [0, 0, 0, 0, 0, 0, 0],
                   [-0.0055, -0.00635, -0.0055, -0.00635, -0.00635, -0.00635, -0.0127]]),
         np.array([[0.00635, 0.00635, 0.00635, 0.00635, 0.00635, 0.00635, 0.0127],
                   [0, 0, 0, 0, 0, 0, 0],
                   [-0.00635, -0.00635, -0.00635, -0.00635, -0.00635, -0.00635, -0.0127]])]
z_p_g = [np.array([[0.0127, 0.0254, 0.0381, 0, 0, 0, 0],
                   [0.0127, 0.0254, 0.0381, 0, 0, 0, 0],
                   [0.0127, 0.0254, 0.0381, 0, 0, 0, 0]]),
         np.array([[0.0127, 0.0254, 0.0381, 0, 0, 0, 0],
                   [0.0127, 0.0254, 0.0381, 0, 0, 0, 0],
                   [0.0127, 0.0254, 0.0381, 0, 0, 0, 0]])]
num_x_p = [np.size(x_p_g[geo], axis=1) for geo in geo_inds_p]
num_y_p = [np.size(y_p_g[geo], axis=0) for geo in geo_inds_p]
num_chans_p = [np.sum(keep_channel[geo]) for geo in geo_inds_p]  # Number of pressure channels
chan_inds_p = [np.arange(num_chans_p[geo], dtype=int) for geo in geo_inds_p]
N1_p = [204800]*num_geos_p  # Number of samples in a single trial
N_p = [N1_p[geo]*num_trials_p[geo] for geo in geo_inds_p]
t_p = [np.arange(N_p[geo])/f_capture_p[geo] for geo in geo_inds_p]
N_zero_p = [N1_p[geo]*num_zero_trials_p[geo] for geo in geo_inds_p]
pitot_chan_ID_p = [16]*num_geos_p  # Pitot tube channel. Indexing from 1
pitot_chan_ind_p = [chan_inds_p[geo][chan_IDs_p[geo] == pitot_chan_ID_p[geo]]
                    for geo in geo_inds_p]
power_chan_ID_p = [20]*num_geos_p  # Power supply channel. Indexing from 1
power_chan_ind_p = [chan_inds_p[geo][chan_IDs_p[geo] == power_chan_ID_p[geo]]
                    for geo in geo_inds_p]

# Import from pressure or voltage files?
from_Volts = True  # Read_p pressure from Voltage files?
cal_coeff_m = np.array([
                        [2.5353, 2.5165, 2.5163, 2.5256, 2.5364, 2.5263,
                         2.5441, 2.5265, 2.5388, 2.5328, 2.5367, 2.5119,
                         2.5308, 2.5456, 2.5421, 2.5390, 2.5118, 2.4958,
                         1.0000, 1.0000],
                        [2.4845, 2.4952, 2.4766, 2.4739, 2.4963, 2.4878,
                         2.4880, 2.4910, 2.4908, 2.4823, 2.4930, 2.4817,
                         2.4957, 2.4999, 2.5034, 2.4998, 2.5250, 2.5190,
                         1.0000, 1.0000],
                        [2.5832, 2.5604, 2.5758, 2.5806, 2.5823, 2.5646,
                         2.5854, 2.5602, 2.5805, 2.5724, 2.5674, 2.5510,
                         2.5776, 2.5812, 2.5796, 2.5769, 2.5826, 2.5529,
                         1.0000, 1.0000]
                        ])  # Calibration coefficient m in p = m*V + b. Only used if from_Volts = True                    
cal_coeff_b = np.array([
                        [-5.6500, -5.6161, -5.6251, -5.6194, -5.6596, -5.6391,
                         -5.7024, -5.6391, -5.7499, -5.6609, -5.7049, -5.6260,
                         -5.6371, -5.6575, -5.6824, -5.6514, -5.6109, -5.5359,
                         0.0000, 0.0000],
                        [-5.5999, -5.6339, -5.5997, -5.5661, -5.6318, -5.6154,
                         -5.6350, -5.6223, -5.7027, -5.6087, -5.6688, -5.6217,
                         -5.6197, -5.6178, -5.6584, -5.6261, -5.7002, -5.6451,
                         0.0000, 0.0000],
                        [-5.6930, -5.6481, -5.6944, -5.6766, -5.6979, -5.6580,
                         -5.7197, -5.6480, -5.7779, -5.6829, -5.7064, -5.6470,
                         -5.6731, -5.6713, -5.6973, -5.6681, -5.7273, -5.6200,
                         0.0000, 0.0000]
                        ])  # Calibration coefficient b in p = m*V + b. Only used if from_Volts = True
cal_power_V = np.array([
                        [4.9468, 4.9467, 4.9463, 4.9466, 4.9466, 4.9465,
                         4.9465, 4.9465, 4.9465, 4.9464, 4.9464, 4.9464,
                         4.9464, 4.9464, 4.9465, 4.9465, 4.9377, 4.9380,
                         4.946, 4.946],
                        [5.0026, 5.0027, 5.0027, 5.0026, 5.0027, 5.0026,
                         5.0026, 5.0027, 5.0027, 5.0026, 5.0026, 5.0026,
                         5.0025, 5.0025, 5.0025, 5.0025, 4.9863, 4.9864,
                         5.003, 5.003],
                        [4.8872, 4.8871, 4.8869, 4.8869, 4.8868, 4.8866,
                         4.8866, 4.8866, 4.8865, 4.8865, 4.8863, 4.8864,
                         4.8863, 4.8864, 4.8863, 4.8867, 4.9003, 4.9003,
                         4.886, 4.886]
                         ]) # Power supply voltage used for the corresponding calibration coefficients. Only used if from_Volts = True

# Pressure correction
p_offset = [0]*num_geos_p  # Offset that will be added to the raw pressure data

# Filtering
filter_p = [True]*num_geos_p  # Filter pressure?
f_cutoff_p = [750]*num_geos_p  # Butterworth filter cutoff frequency. Only used if filter_p = True. Will be overwritten if filter_relative = True
order_p = [8]*num_geos_p  # Butterworth filter order_p. Filter is run forwards and backwards to eliminate phase shift so this order_p is doubled when implemented. Only used if filter = True
gustaf_meth_p = [True]*num_geos_p  # Use Gustoffson method to handle the edges of the domain. Useing Gustaffson method will ensure forward-backward equals backward-forward
#filter_relative = [False]*num_geos_p  # Filter pressure relative to the shedding frequency? Only used if filter_p = True
#f_cutoff_p_multiplier = [3.5]*num_geos_p

#%% Calculate Density and Viscosity

R = 287  # J/kg/K
S = 110.4  # K
To = 273  # K
mewo = 1.71e-5  # kg/m/s
if calc_rho_and_mew:
    rho = [(P_atm[geo]*1000)/(R*(T_amb[geo] + 273.15)) for geo in geo_inds_p]
    mew = [((T_amb[geo] + 273.15)/To)**(3/2)*(To + S)/(T_amb[geo] + 273.15 + S)*mewo
           for geo in geo_inds_p]

#%% Create interpolation functions of calibration coefficients

if from_Volts:
    cal_coeff_m_interp_funcs = [[spinterp.interp1d(cal_power_V[:, chan],
                                                   cal_coeff_m[:, chan])
                                 for chan in chan_inds_p[geo]]
                                for geo in geo_inds_p]
    cal_coeff_b_interp_funcs = [[spinterp.interp1d(cal_power_V[:, chan],
                                                   cal_coeff_b[:, chan])
                                 for chan in chan_inds_p[geo]]
                                for geo in geo_inds_p]

#%% Load data

p = [np.zeros([N_p[geo], num_chans_p[geo]]) for geo in geo_inds_p]
p_zero = [np.zeros([N_zero_p[geo], num_chans_p[geo]]) for geo in geo_inds_p]
if from_Volts:
    V_power_zero = [np.zeros([N_zero_p[geo], 1]) for geo in geo_inds_p]
    V_power = [np.zeros([N_p[geo], 1]) for geo in geo_inds_p]
for geo in geo_inds_p:
    # No flow tests
    for trial, file_path in zip(zero_trial_inds_p[geo],
                                ZEROPRESSfile_paths[geo]):
        start_ind = 0 + trial*N1_p[geo]
        end_ind = N1_p[geo] + trial*N1_p[geo]
        print 'Loading: ', file_path[file_path.rfind('\\'):]
        p_inchH2O_zero_trial = np.loadtxt(file_path, delimiter=',')
        p_inchH2O_zero_trial = p_inchH2O_zero_trial[:, keep_channel[geo]]
        # Apply calibration coefficients if reading from voltage file
        if from_Volts:
            V_power_zero[geo][start_ind:end_ind] = \
                p_inchH2O_zero_trial[:, power_chan_ind_p[geo]]
            M_V_trial = np.mean(V_power_zero[geo][start_ind:end_ind])
            cal_coeff_m_temp = [cal_coeff_m_interp_funcs[geo][chan](M_V_trial)
                                for chan in chan_inds_p[geo]]
            cal_coeff_b_temp = [cal_coeff_b_interp_funcs[geo][chan](M_V_trial)
                                for chan in chan_inds_p[geo]]
            p_inchH2O_zero_trial = cal_coeff_m_temp*p_inchH2O_zero_trial + \
                cal_coeff_b_temp
        p_zero[geo][start_ind:end_ind, :] = p_inchH2O_zero_trial*inchH2OtoPa  # Convert inchH2O to Pa
    # Flow tests
    for trial, file_path in zip(trial_inds_p[geo], PRESSfile_paths[geo]):
        start_ind = 0 + trial*N1_p[geo]
        end_ind = N1_p[geo] + trial*N1_p[geo]
        print 'Loading: ', file_path[file_path.rfind('\\'):]
        p_inchH2O_trial = np.loadtxt(file_path, delimiter=',')
        p_inchH2O_trial = p_inchH2O_trial[:, keep_channel[geo]]
        # Apply calibration coefficients if reading from voltage file
        if from_Volts:
            V_power[geo][start_ind:end_ind] = \
                p_inchH2O_trial[:, power_chan_ind_p[geo]]
            M_V_trial = np.mean(V_power[geo][start_ind:end_ind])
            cal_coeff_m_temp = [cal_coeff_m_interp_funcs[geo][chan](M_V_trial)
                                for chan in chan_inds_p[geo]]
            cal_coeff_b_temp = [cal_coeff_b_interp_funcs[geo][chan](M_V_trial)
                                for chan in chan_inds_p[geo]]
            p_inchH2O_trial = cal_coeff_m_temp*p_inchH2O_trial + \
                              cal_coeff_b_temp
        p[geo][start_ind:end_ind, :] = p_inchH2O_trial*inchH2OtoPa  # Convert inchH2O to Pa

#%% Save a copy of the pressure

p_raw = copy.deepcopy(p)
p_zero_raw = copy.deepcopy(p_zero)

#%% Correct Pressure

M_P_zero = [np.mean(p_zero[geo], axis=0) for geo in geo_inds_p]
p_zero = [p_zero_raw[geo] - M_P_zero[geo] + p_offset[geo] for geo in geo_inds_p]
p = [p_raw[geo] - M_P_zero[geo] + p_offset[geo] for geo in geo_inds_p]

#%% Calculate Freestream Velocity and Re 

U_inf_p = [np.sqrt(2*np.mean(p[geo][:, pitot_chan_ind_p])/rho[geo])
           for geo in geo_inds_p]
Re = [U_inf_p[geo]*d_p[geo]*rho[geo]/mew[geo] for geo in geo_inds_p]


#%% Save a copy of pressure before filtering

p_prefilt = copy.deepcopy(p)

#%% Filter Pressure Data

if filter_p:
    f_Nyquist_p = [f_capture_p[geo]/2 for geo in geo_inds_p]
    Wn_p = [f_cutoff_p[geo]/f_Nyquist_p[geo] for geo in geo_inds_p]  # normalized cutoff frequency
    for geo in geo_inds_p:
        # Butterworth filter
        butter_b, butter_a = spsig.butter(order_p[geo], Wn_p[geo], btype='low')
        for trial in trial_inds_p[geo]:
            start_ind = 0 + trial*N1_p[geo]
            end_ind = N1_p[geo] + trial*N1_p[geo]
            if gustaf_meth_p[geo]:
                zero, pole, gain = spsig.tf2zpk(butter_b, butter_a)
                eps = 1e-9
                radius = np.max(np.abs(pole))
                approx_ir_len = int(np.ceil(np.log(eps)/np.log(radius)))
                p_filt_temp = \
                    spsig.filtfilt(butter_b, butter_a,
                                   p_prefilt[geo][start_ind:end_ind, :],
                                   axis=0, method='gust', irlen=approx_ir_len)
            else:
                p_filt_temp = \
                    spsig.filtfilt(butter_b, butter_a,
                                   p_prefilt[geo][start_ind:end_ind, :],
                                   axis=0)
            p[geo][start_ind:end_ind, :] = p_filt_temp

#%% Decompose Pressure Data

M_P = [np.mean(p[geo], axis=0) for geo in geo_inds_p]
pf = [p[geo] - M_P[geo] for geo in geo_inds_p]

#%% Decompose Pressure Data

#M_P2 = [np.zeros([num_trials_p[geo], num_chans_p[geo]]) for geo in geo_inds_p]
#pf2 = copy.deepcopy(p)
#
#for geo in geo_inds_p:
#    for trial in trial_inds_p[geo]:
#        start_ind = 0 + trial*N1_p[geo]
#        end_ind = N1_p[geo] + trial*N1_p[geo]
#        M_P2[geo][trial, :] = np.mean(p[geo][start_ind:end_ind, :], axis=0)
#        pf2[geo][start_ind:end_ind, :] = p[geo][start_ind:end_ind, :] - M_P2[geo][trial, :]

#%% Create grid

pf_g = [np.zeros([num_y_p[geo], num_x_p[geo], N_p[geo]]) for geo in geo_inds_p]
p_g = [np.zeros([num_y_p[geo], num_x_p[geo], N_p[geo]]) for geo in geo_inds_p]
M_P_g = [np.zeros([num_y_p[geo], num_x_p[geo], 1]) for geo in geo_inds_p]
for geo in geo_inds_p:
    for chan_ID in chan_IDs_p[geo]:
        if chan_ID == pitot_chan_ID_p[geo]:
            continue
        elif chan_ID == power_chan_ID_p[geo]:
            continue
        else:
            chan_ind = chan_inds_p[geo][chan_IDs_p[geo] == chan_ID]
            pf_g[geo][chan_IDs_p_g[geo] == chan_ID] = \
                np.squeeze(pf[geo][:, chan_ind])
            p_g[geo][chan_IDs_p_g[geo] == chan_ID] = \
                np.squeeze(p[geo][:, chan_ind])
            M_P_g[geo][chan_IDs_p_g[geo] == chan_ID] = M_P[geo][chan_ind]

#%% Symmetric Antisymmetric split

pf_sy_g = [np.zeros(np.shape(pf_g[geo])) for geo in geo_inds_p]
pf_as_g = [np.zeros(np.shape(pf_g[geo])) for geo in geo_inds_p]
M_P_sy_g = [np.zeros(np.shape(M_P_g[geo])) for geo in geo_inds_p]
M_P_as_g = [np.zeros(np.shape(M_P_g[geo])) for geo in geo_inds_p]
for geo in geo_inds_p:
    pf_sy_g[geo] = 0.5*(pf_g[geo][:, :, :] + pf_g[geo][::-1, :, :])
    pf_as_g[geo] = 0.5*(pf_g[geo][:, :, :] - pf_g[geo][::-1, :, :])
    M_P_sy_g[geo] = 0.5*(M_P_g[geo][:, :] + M_P_g[geo][::-1, :])
    M_P_as_g[geo] = 0.5*(M_P_g[geo][:, :] - M_P_g[geo][::-1, :])

pf_sy = [np.zeros(np.shape(pf[geo])) for geo in geo_inds_p]
pf_as = [np.zeros(np.shape(pf[geo])) for geo in geo_inds_p]
M_P_sy = [np.zeros(np.shape(M_P[geo])) for geo in geo_inds_p]
M_P_as = [np.zeros(np.shape(M_P[geo])) for geo in geo_inds_p]
for geo in geo_inds_p:
    for chan_ID in chan_IDs_p[geo]:
        if chan_ID == pitot_chan_ID_p[geo]:
            continue
        elif chan_ID == power_chan_ID_p[geo]:
            continue
        else:
            chan_ind = chan_inds_p[geo][chan_IDs_p[geo] == chan_ID]
            pf_sy[geo][:, chan_ind] = \
                pf_sy_g[geo][chan_IDs_p_g[geo] == chan_ID].T
            pf_as[geo][:, chan_ind] = \
                pf_as_g[geo][chan_IDs_p_g[geo] == chan_ID].T
            M_P_sy[geo][chan_ind] = M_P_sy_g[geo][chan_IDs_p_g[geo] == chan_ID]
            M_P_as[geo][chan_ind] = M_P_as_g[geo][chan_IDs_p_g[geo] == chan_ID]

#%% Spectral Analysis of pressure signals

pf_PSD = [np.zeros((int(np.floor(N1_p[geo]/2)), num_chans_p[geo]))
          for geo in geo_inds_p]
pf_sy_PSD = [np.zeros((int(np.floor(N1_p[geo]/2)), num_chans_p[geo]))
             for geo in geo_inds_p]
pf_as_PSD = [np.zeros((int(np.floor(N1_p[geo]/2)), num_chans_p[geo]))
             for geo in geo_inds_p]
f_p = [np.zeros((int(np.floor(N1_p[geo]/2)))) for geo in geo_inds_p]
for geo in geo_inds_p:
    for chan in chan_inds_p[geo]:
#        if chan % 10 == 0:
#            print 'geo: %1.0f/%1.0f,  plane: %2.0f/%2.0f,  mode: %4.0f/%4.0f' % \
#        (geo, num_geos_p - 1, plane, num_planes - 1, mode, num_modes - 1)
        pfs = [pf, pf_sy, pf_as]
        num_pfs = np.size(pfs, axis=0)
        pf_inds = np.arange(num_pfs)
        pf_PSDs = [np.zeros((int(np.floor(N1_p[geo]/2)), 1)) for _ in pf_inds]
        for (pf_temp, pf_ind) in zip(pfs, pf_inds):
            pf_PSD_temp = np.zeros((int(np.floor(N1_p[geo]/2)),
                                    num_trials_p[geo]))
            for trial in trial_inds_p[geo]:
                start_ind = 0 + trial*N1_p[geo]
                end_ind = N1_p[geo] + trial*N1_p[geo]
                [f_temp, PSD_temp] = \
                    spect.spectrumFFT_v2(
                        pf_temp[geo][start_ind:end_ind, chan],
                        f_capture_p[geo])
                pf_PSD_temp[:, trial] = PSD_temp
            pf_PSDs[pf_ind] = np.mean(pf_PSD_temp, axis=1)
        pf_PSD[geo][:, chan] = pf_PSDs[0]
        pf_sy_PSD[geo][:, chan] = pf_PSDs[1]
        pf_as_PSD[geo][:, chan] = pf_PSDs[2]
    f_p[geo] = f_temp
#del akfs, akf_PSDs, f_temp, PSD_temp, akf_PSD_temp
fd_U_inf_p = [f_p[geo]*d_p[geo]/U_inf_p[geo] for geo in geo_inds_p]

#%% Finding shedding Frequency (estimate)

pf_as_avg_PSD = [np.mean(pf_as_PSD[geo], axis=1) for geo in geo_inds_p]
f_shed_p = [f_p[geo][pf_as_avg_PSD[geo] == np.max(pf_as_avg_PSD[geo])]
            for geo in geo_inds_p]
St_shed_p = [f_shed_p[geo]*d_p[geo]/U_inf_p[geo] for geo in geo_inds_p]

#%% Plot pressure spectra WHich channels to use

plot_chan_IDs = [[1, 2, 3, 8, 9, 11, 12, 14, 15, 18]]*2
plot_chan_inds = [[int(chan_inds_p[geo][chan_IDs_p[geo] == plot_chan_ID])
                   for plot_chan_ID in plot_chan_IDs[geo]]
                  for geo in geo_inds_p]
spectras_temp = [[pf_PSD[geo][:, plot_chan_ind]
                  for plot_chan_ind in plot_chan_inds[geo]]
                 for geo in geo_inds_p]
inds_g = [[[int(np.where(chan_IDs_p_g[geo] == plot_chan_ID)[0]), 
            int(np.where(chan_IDs_p_g[geo] == plot_chan_ID)[1])]
           for plot_chan_ID in plot_chan_IDs[geo]] for geo in geo_inds_p]
plot_chan_xs = [[x_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_ys = [[y_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_zs = [[z_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
spectra_labels = [[r'$p\textquotesingle$ at $\left(%1.1f, %1.1f, %1.1f\right)$' 
                   % (plot_chan_x, plot_chan_y, plot_chan_z)
                   for plot_chan_x, plot_chan_y, plot_chan_z,
                   in zip(plot_chan_xs[geo], plot_chan_ys[geo],
                          plot_chan_zs[geo])]
                  for geo in geo_inds_p]
x_label = r'$\displaystyle \frac{fd}{U_\infty}$'
y_label = 'PSD'
save_name = 'press_spect_choices'
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
abs_freqs = [[4.3*d_p[geo]/U_inf_p[geo]] for geo in geo_inds_p]
abs_freq_labels=['4.3 Hz']*num_geos_p
abs_freq_has=[['right']]*num_geos_p
pltf.plot_spectra_comparison(fd_U_inf_p, spectras_temp, num_geos_p, PUBdir_p,
                             save_name, label_spectra=True,
                             spectra_labels=spectra_labels,
                             x_label=x_label, y_label=y_label, 
                             harm_freqs=harm_freqs, f_shed=St_shed_p,
                             abs_freqs=abs_freqs,
                             abs_freq_labels=abs_freq_labels,
                             abs_freq_has=abs_freq_has,
                             tight=True, label_y_ticks=False, close_fig=False,
                             axes_label_y=0.99, axes_label_x=0.04,
                             label_x_frac=1/32, label_y_fracs=[0.4, 0.4])

#%% Plot pressure spectra

plot_chan_IDs = [[1, 2, 3, 14, 18]]*2
plot_chan_inds = [[int(chan_inds_p[geo][chan_IDs_p[geo] == plot_chan_ID])
                   for plot_chan_ID in plot_chan_IDs[geo]]
                  for geo in geo_inds_p]
end_freq = f_cutoff_p
end_freq_ind = [int(np.where(f_p[geo] == end_freq[geo])[0])
                for geo in geo_inds_p]
spectras_temp = [[pf_PSD[geo][:end_freq_ind[geo], plot_chan_ind]
                  for plot_chan_ind in plot_chan_inds[geo]]
                 for geo in geo_inds_p]
fd_U_inf_p_temp = [fd_U_inf_p[geo][:end_freq_ind[geo]] for geo in geo_inds_p]
inds_g = [[[int(np.where(chan_IDs_p_g[geo] == plot_chan_ID)[0]), 
            int(np.where(chan_IDs_p_g[geo] == plot_chan_ID)[1])]
           for plot_chan_ID in plot_chan_IDs[geo]] for geo in geo_inds_p]
plot_chan_xs = [[x_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_ys = [[y_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_zs = [[z_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
spectra_labels = [[r'''$p\textquotesingle$ at
                   $\left(%1.1f, %1.1f, %1.1f\right)$'''
                   % (plot_chan_x, plot_chan_y, plot_chan_z)
                   for plot_chan_x, plot_chan_y, plot_chan_z,
                   in zip(plot_chan_xs[geo], plot_chan_ys[geo],
                          plot_chan_zs[geo])]
                  for geo in geo_inds_p]
#x_label = r'$\displaystyle \frac{fd}{U_\infty}$'
x_label = r'$f$'
y_label = 'PSD'
save_name = 'press_spect'
harm_freqs = [[0.25, 1, 2], [0.1, 1, 2]]
pltf.plot_spectra_comparison(fd_U_inf_p_temp, spectras_temp, num_geos_p,
                             PUBdir_p, save_name, label_spectra=True,
                             spectra_labels=spectra_labels,
                             x_label=x_label, y_label=y_label,
                             harm_freqs=harm_freqs, f_shed=St_shed_p,
                             tight=True, label_y_ticks=False, close_fig=False,
                             axes_label_y=0.985, axes_label_x=0.03,
                             y_ticks_single=[10**-6, 10**-2, 10**2],
                             x_ticks = [10**-4, 10**-3, 10**-2, 10**-1, 10**0],
                             label_x_frac=1/32, label_y_fracs=[0.5, 0.45],
                             colour_spectra_label=False,
                             figsize=(3.3, 5), extension='.eps')

#%% Plot pressure spectra symmetric

plot_chan_IDs = [[1, 2, 3, 14, 18]]*2
plot_chan_inds = [[int(chan_inds_p[geo][chan_IDs_p[geo] == plot_chan_ID])
                   for plot_chan_ID in plot_chan_IDs[geo]]
                  for geo in geo_inds_p]
end_freq = f_cutoff_p
end_freq_ind = [int(np.where(f_p[geo] == end_freq[geo])[0])
                for geo in geo_inds_p]
spectras_temp = [[pf_sy_PSD[geo][:end_freq_ind[geo], plot_chan_ind]
                  for plot_chan_ind in plot_chan_inds[geo]]
                 for geo in geo_inds_p]
fd_U_inf_p_temp = [fd_U_inf_p[geo][:end_freq_ind[geo]] for geo in geo_inds_p]
inds_g = [[[int(np.where(chan_IDs_p_g[geo] == plot_chan_ID)[0]), 
            int(np.where(chan_IDs_p_g[geo] == plot_chan_ID)[1])]
           for plot_chan_ID in plot_chan_IDs[geo]] for geo in geo_inds_p]
plot_chan_xs = [[x_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_ys = [[y_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_zs = [[z_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
spectra_labels = [[r'''$p\textquotesingle$ at
                   $\left(%1.1f, %1.1f, %1.1f\right)$''' 
                   % (plot_chan_x, plot_chan_y, plot_chan_z)
                   for plot_chan_x, plot_chan_y, plot_chan_z,
                   in zip(plot_chan_xs[geo], plot_chan_ys[geo],
                          plot_chan_zs[geo])]
                  for geo in geo_inds_p]
#x_label = r'$\displaystyle \frac{fd}{U_\infty}$'
x_label = r'$f$'
y_label = 'PSD'
save_name = 'press_spect_sy'
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
pltf.plot_spectra_comparison(fd_U_inf_p_temp, spectras_temp, num_geos_p,
                             PUBdir_p, save_name, label_spectra=True,
                             spectra_labels=spectra_labels,
                             x_label=x_label, y_label=y_label,
                             harm_freqs=harm_freqs, f_shed=St_shed_p,
                             tight=True, label_y_ticks=False, close_fig=False,
                             axes_label_y=0.98, axes_label_x=0.03,
                             y_ticks_single=[10**-6, 10**-2, 10**2],
                             label_x_frac=1/32, label_y_fracs=[0.25, 0.4],
                             colour_spectra_label=False,
                             figsize=(3.3, 5))

#%% Plot pressure spectra antisymmetric

plot_chan_IDs = [[1, 2, 3, 18]]*2
plot_chan_inds = [[int(chan_inds_p[geo][chan_IDs_p[geo] == plot_chan_ID])
                   for plot_chan_ID in plot_chan_IDs[geo]]
                  for geo in geo_inds_p]
end_freq = f_cutoff_p
end_freq_ind = [int(np.where(f_p[geo] == end_freq[geo])[0])
                for geo in geo_inds_p]
spectras_temp = [[pf_as_PSD[geo][:end_freq_ind[geo], plot_chan_ind]
                  for plot_chan_ind in plot_chan_inds[geo]]
                 for geo in geo_inds_p]
fd_U_inf_p_temp = [fd_U_inf_p[geo][:end_freq_ind[geo]] for geo in geo_inds_p]
inds_g = [[[int(np.where(chan_IDs_p_g[geo] == plot_chan_ID)[0]), 
            int(np.where(chan_IDs_p_g[geo] == plot_chan_ID)[1])]
           for plot_chan_ID in plot_chan_IDs[geo]] for geo in geo_inds_p]
plot_chan_xs = [[x_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo] 
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_ys = [[y_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_zs = [[z_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
spectra_labels = [[r'''$p\textquotesingle$ at
                   $\left(%1.1f, %1.1f, %1.1f\right)$'''
                   % (plot_chan_x, plot_chan_y, plot_chan_z)
                   for plot_chan_x, plot_chan_y, plot_chan_z,
                   in zip(plot_chan_xs[geo], plot_chan_ys[geo],
                          plot_chan_zs[geo])]
                  for geo in geo_inds_p]
#x_label = r'$\displaystyle \frac{fd}{U_\infty}$'
x_label = r'$f$'
y_label = 'PSD'
save_name = 'press_spect_as'
harm_freqs = [[0.25, 1, 2, 3], [0.1, 1, 2, 3]]
pltf.plot_spectra_comparison(fd_U_inf_p_temp, spectras_temp, num_geos_p,
                             PUBdir_p, save_name, label_spectra=True,
                             spectra_labels=spectra_labels,
                             x_label=x_label, y_label=y_label,
                             harm_freqs=harm_freqs, f_shed=St_shed_p,
                             tight=True, label_y_ticks=True, close_fig=False,
                             axes_label_y=0.98, axes_label_x=0.025,
                             y_ticks_single=[10**-6, 10**-2, 10**2],
                             label_x_frac=1/32, label_y_fracs=[0.7, 0.6],
                             colour_spectra_label=False,
                             figsize=(3.3, 4))

#%% Plot symmetric cross_correlations

import string

plot_chan_IDs = [[2, 18]]*2
num_pairs = [np.size(plot_chan_IDs[geo]) for geo in geo_inds_p]
inds_g = [[[int(np.where(chan_IDs_p_g[geo] == plot_chan_ID)[0]), 
            int(np.where(chan_IDs_p_g[geo] == plot_chan_ID)[1])]
           for plot_chan_ID in plot_chan_IDs[geo]] for geo in geo_inds_p]
num_cross = [np.size(plot_chan_IDs[geo], axis=0) for geo in geo_inds_p]
cross_inds = [np.arange(num_cross[geo]) for geo in geo_inds_p]
pair_inds_g = [[[num_y_p[geo] - ind_g[0] - 1, ind_g[1]]
                for ind_g in inds_g[geo]] for geo in geo_inds_p]
#pfs_temp = [[[pf_g[geo][ind_g[0], ind_g[1]]
#              for ind_g in [inds_g[geo][chan_pair], pair_inds_g[geo][chan_pair]]]
#             for chan_pair in  np.arange(num_pairs[geo])] for geo in geo_inds_p]
plot_chan_xs = [[x_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_pair_xs = [[x_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                      for ind_g in pair_inds_g[geo]] for geo in geo_inds_p]
plot_chan_ys = [[y_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_pair_ys = [[y_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                      for ind_g in pair_inds_g[geo]] for geo in geo_inds_p]
plot_chan_zs = [[z_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                 for ind_g in inds_g[geo]] for geo in geo_inds_p]
plot_chan_pair_zs = [[z_p_g[geo][ind_g[0], ind_g[1]]/d_p[geo]
                      for ind_g in pair_inds_g[geo]] for geo in geo_inds_p]
cross_labels = [[r'$\left(%1.1f, \pm%1.1f, %1.1f\right)$' 
                 % (plot_chan_x, plot_chan_y, plot_chan_z)
                 for plot_chan_x, plot_chan_y, plot_chan_z
                 in zip(plot_chan_xs[geo], plot_chan_ys[geo], plot_chan_zs[geo])]
                for geo in geo_inds_p]
x_label = r'$\displaystyle \frac{\tau U_\infty}{d}$'
x_label = r'$\tau f_{sh}$'
y_label = r'$\displaystyle \frac{\left<p\textquotesingle_i\left(t\right), p\textquotesingle_j\left(t+\tau\right)\right>}{\sigma_i \sigma_j}$'
y_label = r'$R_{p\textquotesingle p\textquotesingle}$'
PUBdir =  r'D:\EXF Paper\EXF Paper V3\figs'# Where the .eps figures go
save_name = 'press_cross'
extension = '.eps'
figsize=(5, 2.75)
#figsize=(5.5, 2.75)
font_size=8.5
xmax = [[0.04, 0.04, 0.04, 0.04], [0.04, 0.04, 0.05, 0.05]]
xmax = [[10, 5], [20, 5]]

xmin = [[-xmax[geo][cross] for cross in cross_inds[geo]] for geo in geo_inds_p]
axes_label_coord = 'axes'
colour_axes_label = False
label_axes = True
axes_ind = 0
auto_axes_ind = True
axes_label_x = 0.02
axes_label_y = 0.95
cross_label_x = 0.99
cross_label_y = axes_label_y
label_x = [True, True]
num_rows = np.max(num_cross)
num_cols = num_geos_p

figs = []
axs = []
fig = plt.figure(figsize=figsize)
figs.append(fig)
for geo in geo_inds_p:
    for ind_g, pair_ind_g, cross_label, cross_ind in \
            zip(inds_g[geo], pair_inds_g[geo], cross_labels[geo], cross_inds[geo]):
        subplot_ind = cross_ind*num_geos_p + geo + 1
        ax1 = plt.subplot(num_rows, num_cols, subplot_ind)
        axs.append(ax1)

#        cross_temp = np.zeros((2*N1_p[geo] - 1, num_trials_p[geo]))
        cross_norm_temp = np.zeros((2*N1_p[geo] - 1, num_trials_p[geo]))
        for trial in trial_inds_p[geo]:
            start_ind = 0 + trial*N1_p[geo]
            end_ind = N1_p[geo] + trial*N1_p[geo]
            sig1 = pf_g[geo][ind_g[0], ind_g[1], start_ind:end_ind]
            sig2 = pf_g[geo][pair_ind_g[0], pair_ind_g[1], start_ind:end_ind]
            cross_temp_temp = spsig.correlate(sig1, sig2, mode='full',
                                              method='fft')
#            cross_temp[:, trial] = cross_temp_temp
            norm_fact = np.std(sig1)*np.std(sig2)*N1_p[geo]
            cross_norm_temp[:, trial] = cross_temp_temp/norm_fact
#        cross = np.mean(cross_temp, axis=1)
        cross_norm = np.mean(cross_norm_temp, axis=1)
        tau = np.arange(-(N1_p[geo] - 1), N1_p[geo])/f_capture_p[geo]
        plt.plot(tau*f_shed_p[geo], cross_norm, 'k', lw=0.5)
        plt.plot([0]*2, [-1, 1], ':', c=[0.7]*3, zorder=0)
#        plt.title(cross_label, fontsize=font_size)
        plt.xlim([xmin[geo][cross_ind], xmax[geo][cross_ind]])
        if label_x[cross_ind]:
            plt.xlabel(x_label)
        else:
            plt.tick_params(labelbottom=False)
        plt.ylim([-1, 1])
#        ax2 = ax1.twiny()
#        ax2.set_xlim(ax1.get_xlim())
##        ax2.set_xticks([0, 1, 2])
#        ax2.set_xticklabels([])
        
        if label_axes:
            bbox_props = {'pad': 0.1, 'fc': 'w', 'ec': 'w', 'alpha': 0.0}
            if colour_axes_label:
                bbox_props['alpha'] = 1.0
            if axes_label_coord in ['axes', 'Axes']:
                transform = ax1.transAxes
            else:
                transform = ax1.transData
            if auto_axes_ind:
                axes_ind = subplot_ind
            label = '(' + string.ascii_lowercase[axes_ind - 1] + ')'
            plt.text(axes_label_x, axes_label_y, label, ha='left', va='top',
                     bbox=bbox_props, transform=transform)
        bbox_props = {'pad': 0.1, 'fc': 'w', 'ec': 'w', 'alpha': 0.0}
        plt.text(cross_label_x, cross_label_y, cross_label, ha='right',
                 va='top', transform=ax1.transAxes, bbox=bbox_props)
        if geo == geo_inds_p[0]:
            plt.ylabel(y_label, rotation='horizontal')
plt.tight_layout()
save_path = PUBdir_p + '\\' + save_name + extension
plt.savefig(save_path, bbox_inches='tight')