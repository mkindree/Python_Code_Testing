from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

#%%

# These options make the figure text match the default LaTex font
plt.rc('text', usetex=True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 10})

#%% Boundary Layer data from spreadsheet

y = np.array([0.57, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50,
              1.70, 1.90, 2.10, 2.30, 2.50, 2.70, 2.90, 3.10, 3.30, 3.50, 4.50])
delta = 2.67
U = np.array([5.1378, 5.4314, 6.1356, 6.8619, 7.4792, 8.1326, 8.6573, 9.2868,
              10.0035, 10.4333, 11.2046, 12.0477, 12.9027, 13.1515, 13.6101,
              13.7471, 13.8719, 13.9032, 13.9653, 13.9828, 13.9917, 14.0058])
U_rms = np.array([0.409, 0.4041, 0.3921, 0.4099, 0.3921, 0.4301, 0.4502, 0.4146,
                  0.4393, 0.3442, 0.4229, 0.3521, 0.2876, 0.26, 0.1927, 0.1878,
                  0.1731, 0.1756, 0.1641, 0.1603, 0.164, 0.1644])
U_inf = 14.01
eta = np.array([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6,
                2.8, 3, 3.2, 3.4, 3.6, 3.8, 4, 4.2, 4.4, 4.6, 4.8, 5])
eta_delta = 4.9
f_prime = np.array([0, 0.06641, 0.13277, 0.19894, 0.26471, 0.32979, 0.39378,
                    0.45627, 0.51676, 0.57477, 0.62977, 0.68132, 0.72899,
                    0.77246, 0.81152, 0.84605, 0.87609, 0.90177, 0.92333,
                    0.94112, 0.95552, 0.96696, 0.97587, 0.98269, 0.98779,
                    0.99155])

#%% Boundary Layer figure

PUBdir = r'D:\EXF Paper\EXF Paper V3\figs'
save_name = 'BL'
save_path = PUBdir + '\\' + save_name + '.eps'

fig = plt.figure(figsize=(3.3, 3))
ax1 = plt.subplot(1, 1, 1)
ax1.scatter(y/delta, U/U_inf, c='k', s=20, marker='o')
ax1.plot(eta/eta_delta, f_prime, 'k')
ax2 = ax1.twinx()
ax2.scatter(y/delta, U_rms/U_inf, c='b', s=20, marker='x')
ax1.set_ylabel(r'$\displaystyle \frac{U}{U_\infty}$', color='k',
               rotation='horizontal', labelpad=14)
ax1.tick_params('y', colors='k')
xy1 = (0, 0.9)
xytext1 = (0.4, 0.9)
ax1.annotate('', xy=xy1, xytext=xytext1, textcoords='data', xycoords='data',
             arrowprops=dict(facecolor='k', edgecolor='k'))
ax2.set_ylabel(r'$\displaystyle \frac{\sqrt{\overline{u^2}}}{U_\infty}$',
               color='b', rotation='horizontal', labelpad=14)
ax2.tick_params('y', colors='b')
xy2 = (1.75, 0.6)
xytext2 = (0.9, 0.6)
ax1.annotate('', xy=xy2, xytext=xytext2,
             arrowprops=dict(facecolor='b', edgecolor='b'))
ax1.set_xlabel(r'$\displaystyle \frac{y}{\delta}$')
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')

#%%

PUBdir = r'D:\NOVA Interview'
save_name = 'BL'
save_path = PUBdir + '\\' + save_name + '.png'

fig = plt.figure(figsize=(3.3, 3))
ax1 = plt.subplot(1, 1, 1)
ax1.plot(eta/eta_delta, f_prime, 'b', zorder=0)
ax1.scatter(y/delta, U/U_inf, c='k', s=20, marker='o')

plt.legend(['Blasius profile', 'LDV measurements'])
#ax2 = ax1.twinx()
#ax2.scatter(y/delta, U_rms/U_inf, c='b', s=20, marker='x')
ax1.set_ylabel(r'$\displaystyle \frac{U}{U_\infty}$', color='k',
               rotation='horizontal', labelpad=14)
ax1.tick_params('y', colors='k')
#xy1 = (0, 0.9)
#xytext1 = (0.4, 0.9)
#ax1.annotate('', xy=xy1, xytext=xytext1, textcoords='data', xycoords='data',
#             arrowprops=dict(facecolor='k', edgecolor='k'))
#ax2.set_ylabel(r'$\displaystyle \frac{\sqrt{\overline{u^2}}}{U_\infty}$',
#               color='b', rotation='horizontal', labelpad=14)
#ax2.tick_params('y', colors='b')
#xy2 = (1.75, 0.6)
#xytext2 = (0.9, 0.6)
#ax1.annotate('', xy=xy2, xytext=xytext2,
#             arrowprops=dict(facecolor='b', edgecolor='b'))
ax1.set_xlabel(r'$\displaystyle \frac{y}{\delta}$')
plt.tight_layout()
plt.savefig(save_path, bbox_inches='tight')
