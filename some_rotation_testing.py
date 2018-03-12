#%%
import scipy.optimize as spopt

iz = 20
def f_exp(x, x0, tau, b):
    return x0*np.exp(x/tau) + b
x_test = np.linspace(0.01, 5)

x0s = np.zeros((I[0], 1)) 
taus = np.zeros((I[0], 1)) 
bs = np.zeros((I[0], 1)) 
for ix in np.arange(I[0]):
    popt, pcov = spopt.curve_fit(f_exp, y_raw_g[0][y_raw_g[0][:, ix, iz]>1.25, ix, iz],
                                 M_U_raw_g3[0][y_raw_g[0][:, ix, iz]>1.25, ix, iz], p0=[-0.5, -0.35, 1])
    x0s[ix] = popt[0]
    taus[ix] = popt[1]
    bs[ix] = popt[2]
    print popt[2], pcov[2]
bs1 = copy.deepcopy(bs)

plt.figure()
for ix in np.arange(I[0]):
    plt.plot(M_U_raw_g3[0][:, ix, iz], y_raw_g[0][:, ix, iz], '-o')
    test = f_exp(x_test, x0s[ix], taus[ix], bs[ix])
    plt.plot(test, x_test)
    plt.xlim([-0.5, 1.5])
  
print
    
x0s = np.zeros((I[0], 1)) 
taus = np.zeros((I[0], 1)) 
bs = np.zeros((I[0], 1)) 
for ix in np.arange(I[0]):
    popt, pcov = spopt.curve_fit(f_exp, y_raw_g[0][y_raw_g[0][:, ix, iz]>1.25, ix, iz],
                                 M_W_raw_g3[0][y_raw_g[0][:, ix, iz]>1.25, ix, iz], p0=[-0.5, -0.35, 0])
    x0s[ix] = popt[0]
    taus[ix] = popt[1]
    bs[ix] = popt[2]
    print popt[2], pcov[2]

plt.figure()
for ix in np.arange(I[0]):
    plt.plot(M_W_raw_g3[0][:, ix, iz], y_raw_g[0][:, ix, iz], '-o')
    test = f_exp(x_test, x0s[ix], taus[ix], bs[ix])
    plt.plot(test, x_test)
    plt.xlim([-0.6, 0.6])

bs2 = copy.deepcopy(bs)


plt.figure()
for ix in np.arange(I[0]):
    plt.plot(M_V_raw_g3[0][:, ix, iz], y_raw_g[0][:, ix, iz], '-o')
    plt.xlim([-0.6, 0.6])

np.mean(bs1[np.logical_and(0.9<bs1, bs1<1.1)])
np.mean(bs2[np.logical_and(-0.1<bs2, bs2<0.1)])

#%%

M_U_raw_g2 = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_W_raw_g2 = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]

theta = -1 # deg
theta_rad = theta/180*np.pi
for geo in geo_inds:
    M_U_raw_g2[geo] = M_U_raw_g[geo]*np.cos(theta_rad) - M_W_raw_g[geo]*np.sin(theta_rad)
    M_W_raw_g2[geo] = M_U_raw_g[geo]*np.sin(theta_rad) + M_W_raw_g[geo]*np.cos(theta_rad) 

#%%

M_V_raw_g3 = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]
M_W_raw_g3 = [np.zeros((J[geo], I[geo], K[geo])) for geo in geo_inds]

theta = 5 # deg
theta_rad = theta/180*np.pi
for geo in geo_inds:
    M_V_raw_g3[geo] = M_V_raw_g[geo]*np.cos(theta_rad) - M_W_raw_g2[geo]*np.sin(theta_rad)
    M_W_raw_g3[geo] = M_V_raw_g[geo]*np.sin(theta_rad) + M_W_raw_g2[geo]*np.cos(theta_rad) 
