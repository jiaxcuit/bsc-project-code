#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 00:16:10 2023

@author: liujiaxin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jnp_zeros, jn_zeros
#%%
c = 3e8 # in m/s
R = 0.105 # in m
d_0 = 0.141 # in m

def TMag(m, n, p, d):
    w_TM = c * np.sqrt((jn_zeros(m, n)[-1] / R)**2 + (p * np.pi / d)**2)
    f_TM = w_TM/(2*np.pi) # in Hz
    w_0 = c * np.sqrt((jn_zeros(m, n)[-1] / R)**2 + (p * np.pi / d_0)**2)
    f_0 = w_0/(2*np.pi)
    return w_TM, f_TM, f_0, w_0

def TEle(m, n, p, d):
    w_TE = c * np.sqrt((jnp_zeros(m, n)[-1] / R)**2 + (p * np.pi / d)**2)
    f_TE = w_TE/(2*np.pi)  # in Hz
    w_0 = c * np.sqrt((jnp_zeros(m, n)[-1] / R)**2 + (p * np.pi / d_0)**2)
    f_0 = w_0/(2*np.pi)
    return w_TE, f_TE, f_0, w_0

#%%
d = np.linspace(0.001, 0.300, 1000) # in m


# %%
TM_010_angf, TM_010_f, _, _ = TMag(0, 1, 0, d) # m, n, p. No d dependence when p=0
TM_020_angf, TM_020_f, _, _ = TMag(0, 2, 0, d)

TM_110_angf, TM_110_f, _, _ = TMag(1, 1, 0, d)
TM_120_angf, TM_120_f, _, _ = TMag(1, 2, 0, d)
TM_130_angf, TM_130_f, _, _ = TMag(1, 3, 0, d)

TM_114_angf, TM_114_f, _, _ = TMag(1, 1, 4, d)
TM_548_angf, TM_548_f, _, _ = TMag(5, 4, 8, d)

TE_111_angf, TE_111_f, _, _ = TEle(1, 1, 1, d)  # p != 0 for TE modes (for H_z != 0)
TE_011_angf, TE_011_f, _, _ = TEle(0, 1, 1, d)
TE_011_angf, TE_011_f, _, _ = TEle(0, 1, 1, d)

plt.plot(d, TM_010_f/1e9, label='TM_010')
plt.plot(d, TM_020_f/1e9, label='TM_020')
plt.plot(d, TM_110_f/1e9, label='TM_110')
plt.plot(d, TM_120_f/1e9, label='TM_120')
plt.plot(d, TM_130_f/1e9, label='TM_130')
plt.plot(d, TM_114_f/1e9, label='TM_114')
plt.plot(d, TM_548_f/1e9, label='TM_548')
plt.vlines(0.141, 0, 20, linestyles='dashed')
# plt.plot(d, TE_111_f/1e9, label = 'TE_111')
# plt.plot(d, TE_011_f/1e9, label = 'TE_011')

# plt.plot(d, TM_010, label = 'TM_010')
# plt.plot(d, TM_020, label = 'TM_020')
# plt.plot(d, TE_111, label = 'TE_111')
# plt.plot(d, TE_011, label = 'TE_011')

plt.title(f'$R =$ {R} m')
plt.xlabel('Length of cylinder $d$ (m)')
plt.ylabel('$f_{mnp}$ (GHz)')
plt.grid(True)
# plt.ylabel('$\omega_{mnp}$ (rad/s)')
plt.legend()

plt.ylim([0,20])
plt.show()
#%%

plt.plot(1/d**2, (TM_010)**2, label = 'TM_010')
plt.plot(1/d**2, (TM_020)**2, label = 'TM_020')
plt.plot(1/d**2, (TE_111)**2, label = 'TE_111')
plt.plot(1/d**2, (TE_011)**2, label = 'TE_011')

# plt.plot(1/d**2, (R*TM_010/np.pi)**2, label = 'TM_010')
# plt.plot(1/d**2, (R*TM_020/np.pi)**2, label = 'TM_020')
# plt.plot(1/d**2, (R*TE_111/np.pi)**2, label = 'TE_111')
# plt.plot(1/d**2, (R*TE_011/np.pi)**2, label = 'TE_011')

plt.xlabel('(2Rd)$^2$(m$^4$)')
plt.ylabel('(2Rf)$^2$ (m$^2$/s$^-2$)')
plt.legend()

# plt.ylim([0,2e11])
plt.show()

# %% Auto plot modes

m_max = 2
n_max = 2
p_max = 3

TM_fs = np.zeros((m_max+1, n_max+1, p_max+1), dtype='object')
TE_fs = np.zeros((m_max+1, n_max+1, p_max+1), dtype='object')

TM_f_testc = np.zeros((m_max+1, n_max+1, p_max+1), dtype='float')
TE_f_testc = np.zeros((m_max+1, n_max+1, p_max+1), dtype='float')
# freq at a given d_0 (dimension of the test cavity)

for m in range(0, m_max+1):
    for n in range(0, n_max+1):
        for p in range(0, p_max+1):
            TM_fs[m][n][p] = TMag(m, n+1, p, d)[1]/1e9
            TM_f_testc[m][n][p] = TMag(m, n+1, p, d)[2]/1e9
            TE_fs[m][n][p] = TEle(m, n+1, p, d)[1]/1e9
            TE_f_testc[m][n][p] = TEle(m, n+1, p, d)[2]/1e9


for m in range(0, m_max+1):
    for n in range(0, n_max+1):    
        for p in range(0, p_max+1):
            # plt.plot(d, TM_fs[m][n][p], label=f'TM_{m}{n+1}{p}')
            print(f'Test cavity frequency of mode TM_{m}{n+1}{p} = {TM_f_testc[m][n][p]:.3f} GHz')
            # plt.plot(d, TE_fs[m][n][p], label=f'TE_{m}{n+1}{p}')
            # print(f'Test cavity frequency of mode TE_{m}{n+1}{p} = {TE_f_testc[m][n][p]:.3f} GHz')

# plt.plot(d, TM_fs[1][1][0], label=f'TM_{1}{1+1}{0}')

plt.title(f'$R =$ {R} m')
plt.xlabel('Length of cylinder $d$ (m)')
plt.ylabel('$f_{mnp}$ (GHz)')
plt.grid(True)
# plt.ylabel('$\omega_{mnp}$ (rad/s)')
plt.legend()

plt.ylim([0,20])
plt.xlim([0.13, 0.15])
plt.show()
