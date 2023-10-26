#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 19:21:15 2023

@author: liujiaxin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.special import j0, jv, jvp, jn_zeros, jnp_zeros

# %% Draw cylinder, given dimensions and position
def data_for_cylinder_along_z(center_x, center_y, radius, height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

# from https://stackoverflow.com/a/49311446

# %% test 1

x,y = np.meshgrid(np.linspace(-5,5,10),np.linspace(-5,5,10))

u = x/np.sqrt(x**2 + y**2)
v = y/np.sqrt(x**2 + y**2)

plt.quiver(x,y,u,v)
plt.show()

# %% test 2

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.2),
                      np.arange(-0.8, 1, 0.8))

u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
     np.sin(np.pi * z))

ax.quiver(x, y, z, u, v, w, length=0.1, color = 'black')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
fig.show()


# %% General parameters

c = 3e8
permit_0 = 8.85e-12 # permittivity of free space

R = 0.105 # in m
d = 0.141 # in m

# %% Test with TM_010

p = 0
m = 0
n = 1

E_0 = 0.5
omega_TM_010 = 1.094 * 2 * np.pi # in rad/s
omega = omega_TM_010
gamma_sq = omega**2/c**4 - (p*np.pi/d)**2

arrow_len = 0.05

fig = plt.figure()
ax = fig.gca(projection='3d')

x, y, z = np.meshgrid(np.linspace(-R, R, 10),
                      np.linspace(-R, R, 10),
                      np.linspace(0, d, 4))

r, phi, z = np.meshgrid(np.linspace(0, R, 5),
                          np.linspace(0, 2* np.pi, 10),
                          np.linspace(0, d, 4))
x = r * np.cos(phi)
y = r * np.sin(phi)
rho = np.sqrt(x ** 2 + y ** 2)
grad_Ez = E_0 * 2.405 /R * jvp(v=0, z=2.405 * rho/R, n=1)

E_t_factor = -p*np.pi/(d * gamma_sq) * np.sin(p*np.pi*z/d)
E_x = E_t_factor * grad_Ez * x/rho
E_y = E_t_factor * grad_Ez * y/rho
E_z = E_0 * j0(2.405 * rho/R) # first kind (J) of 0 order
ax.quiver(x, y, z, E_x, E_y, E_z, length=arrow_len, color='black', label='E')


# x, y, z = np.meshgrid(np.linspace(-R, R, 10),
#                       np.linspace(-R, R, 10),
#                       np.linspace(0, d, 4))
r, phi, z = np.meshgrid(np.linspace(0, R, 5),
                          np.linspace(0, 2* np.pi, 10),
                          np.linspace(0, d, 3))

x = r * np.cos(phi)
y = r * np.sin(phi)

rho = np.sqrt(x ** 2 + y ** 2)
grad_Ez = E_0 * 2.405 /R * jvp(v=0, z=2.405 * rho/R, n=1)

H_t_factor = permit_0 * omega/(c * gamma_sq) * np.cos(p*np.pi*z/d)
H_x = H_t_factor * grad_Ez * (-1)*y/rho
H_y = H_t_factor * grad_Ez * x/rho
H_z = np.zeros(x.shape)
ax.quiver(x, y, z, H_x * 3e-15, H_y * 3e-15, H_z, length=arrow_len, color='red', label='H')

Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, R, d)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
fig.show()

# %% Test with arbitrary TM m, n, p
# need to manually adjust the scaling factor for B (or H) field arrows

p = 0
m = 1
n = 1

E_0 = 0.5
omega = TMag(m, n+1, p, d)[3] # in rad/s
gamma_sq = omega**2/c**4 - (p*np.pi/d)**2

arrow_len = 0.1
H_scale_factor = 4e-6

fig = plt.figure()
ax = fig.gca(projection='3d')

# x, y, z = np.meshgrid(np.linspace(-R, R, 10),
#                       np.linspace(-R, R, 10),
#                       np.linspace(0, d, 4))

rho, phi, z = np.meshgrid(np.linspace(0.01, R, 5),
                          np.linspace(0, 2* np.pi, 10),
                          np.linspace(0, d, 4))
x = rho * np.cos(phi)
y = rho * np.sin(phi)

x_mn = jn_zeros(n=m, nt=n)[-1]
grad_Ez_rho = E_0 * x_mn /R * jvp(v=m, z=x_mn * rho/R, n=1) * np.cos(m * phi)
grad_Ez_phi = 1/rho * E_0 * jv(m, x_mn * rho/R) * (-m) * np.sin(m * phi)

E_t_factor = -p*np.pi/(d * gamma_sq) * np.sin(p*np.pi*z/d)
E_x = E_t_factor * (grad_Ez_rho * np.cos(phi) - grad_Ez_phi * np.sin(phi))
E_y = E_t_factor * (grad_Ez_rho * np.sin(phi) + grad_Ez_phi * np.cos(phi))
E_z = E_0 * jv(m, x_mn * rho/R) * np.cos(m * phi) # first kind (J) of v=mth order
ax.quiver(x, y, z, E_x, E_y, E_z, length=arrow_len, color='black', label='E')


# x, y, z = np.meshgrid(np.linspace(-R, R, 10),
#                       np.linspace(-R, R, 10),
#                       np.linspace(0, d, 4))
rho, phi, z = np.meshgrid(np.linspace(0, R, 5),
                          np.linspace(0, 2* np.pi, 10),
                          np.linspace(0, d, 3))

x = rho * np.cos(phi)
y = rho * np.sin(phi)

rho = np.sqrt(x ** 2 + y ** 2)
grad_Ez_rho = E_0 * x_mn /R * jvp(v=m, z=x_mn * rho/R, n=1) * np.cos(m * phi)
grad_Ez_phi = 1/rho * E_0 * jv(m, x_mn * rho/R) * (-m) * np.sin(m * phi)

H_t_factor = permit_0 * omega/(c * gamma_sq) * np.cos(p*np.pi*z/d)
H_x = H_t_factor * ((-1) * grad_Ez_rho * np.sin(phi)
                    + (-1) * grad_Ez_phi * np.cos(phi))
H_y = H_t_factor * (grad_Ez_rho * np.cos(phi)
                    + grad_Ez_phi * (-1) * np.sin(phi))
H_z = np.zeros(x.shape)

ax.quiver(x, y, z, H_x * H_scale_factor, H_y * H_scale_factor, H_z, length=arrow_len, color='red', label='H')

Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, R, d)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.set_title(f'TM_{m}{n}{p}')
fig.show()

# %% Test with TE m, n, p

p = 0
m = 1
n = 1

E_0 = 0.5
omega = TMag(m, n+1, p, d)[3] # in rad/s
gamma_sq = omega**2/c**4 - (p*np.pi/d)**2

arrow_len = 0.1
H_scale_factor = 4e-6

fig = plt.figure()
ax = fig.gca(projection='3d')

# x, y, z = np.meshgrid(np.linspace(-R, R, 10),
#                       np.linspace(-R, R, 10),
#                       np.linspace(0, d, 4))

rho, phi, z = np.meshgrid(np.linspace(0.01, R, 5),
                          np.linspace(0, 2* np.pi, 10),
                          np.linspace(0, d, 4))
x = rho * np.cos(phi)
y = rho * np.sin(phi)

x_mn = jn_zeros(n=m, nt=n)[-1]
grad_Ez_rho = E_0 * x_mn /R * jvp(v=m, z=x_mn * rho/R, n=1) * np.cos(m * phi)
grad_Ez_phi = 1/rho * E_0 * jv(m, x_mn * rho/R) * (-m) * np.sin(m * phi)

E_t_factor = -p*np.pi/(d * gamma_sq) * np.sin(p*np.pi*z/d)
E_x = E_t_factor * (grad_Ez_rho * np.cos(phi) - grad_Ez_phi * np.sin(phi))
E_y = E_t_factor * (grad_Ez_rho * np.sin(phi) + grad_Ez_phi * np.cos(phi))
E_z = E_0 * jv(m, x_mn * rho/R) * np.cos(m * phi) # first kind (J) of v=mth order
ax.quiver(x, y, z, E_x, E_y, E_z, length=arrow_len, color='black', label='E')


# x, y, z = np.meshgrid(np.linspace(-R, R, 10),
#                       np.linspace(-R, R, 10),
#                       np.linspace(0, d, 4))
rho, phi, z = np.meshgrid(np.linspace(0, R, 5),
                          np.linspace(0, 2* np.pi, 10),
                          np.linspace(0, d, 3))

x = rho * np.cos(phi)
y = rho * np.sin(phi)

rho = np.sqrt(x ** 2 + y ** 2)
grad_Ez_rho = E_0 * x_mn /R * jvp(v=m, z=x_mn * rho/R, n=1) * np.cos(m * phi)
grad_Ez_phi = 1/rho * E_0 * jv(m, x_mn * rho/R) * (-m) * np.sin(m * phi)

H_t_factor = permit_0 * omega/(c * gamma_sq) * np.cos(p*np.pi*z/d)
H_x = H_t_factor * ((-1) * grad_Ez_rho * np.sin(phi)
                    + (-1) * grad_Ez_phi * np.cos(phi))
H_y = H_t_factor * (grad_Ez_rho * np.cos(phi)
                    + grad_Ez_phi * (-1) * np.sin(phi))
H_z = np.zeros(x.shape)

ax.quiver(x, y, z, H_x * H_scale_factor, H_y * H_scale_factor, H_z, length=arrow_len, color='red', label='H')

Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, R, d)
ax.plot_surface(Xc, Yc, Zc, alpha=0.5)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.set_title(f'TM_{m}{n}{p}')
fig.show()