#!/usr/bin/env python
# coding=utf-8

import numpy as np
from scipy.stats import linregress


def neutral_prof(u_wind, alt_wind, u_star, grid_z):

    kappa = 0.41

    z0 = alt_wind / np.exp((u_wind * kappa) / u_star)

    wind_array = np.zeros(len(grid_z), np.float)

    for k, z in enumerate(grid_z):
        wind_array[k] = (u_star / kappa) * (np.log(z / z0))


    return wind_array


def stable_prof(u_wind, alt_wind, u_star, L, grid_z):

    kappa = 0.41

    exp_term = np.exp((u_wind - (6 * alt_wind / L) * (kappa / u_star)))

    z0 = alt_wind / exp_term

    wind_array = np.zeros(len(grid_z), np.float)

    for k, z in enumerate(grid_z):
        tmp1 = u_star / kappa
        tmp2 = np.log(z / z0)
        tmp3 = 6 * (z / L)
        wind_array[k] = tmp1 * tmp2 +  tmp3

    return wind_array


def x_factor(z, L):

    x = (1 - (15 * z / L))**0.25

    return x


def phi_factor(x):

    tmp1 = 2 * np.log((1 + x) / 2)
    tmp2 = np.log((1 + x**2) / 2)
    tmp3 = -2 * np.arctan(x) + (np.pi / 2)

    phi = tmp1 + tmp2 + tmp3

    return phi


def unstable_prof(u_wind, alt_wind, u_star, L, grid_z):

    kappa = 0.41
    x = x_factor(alt_wind, L)
    phi = phi_factor(x)

    exp_term = np.exp((u_wind + phi) * (kappa / u_star))
    z0 = alt_wind / exp_term

    wind_array = np.zeros(len(grid_z), np.float)

    for k, z in enumerate(grid_z):
        tmp1 = u_star / kappa
        tmp2 = np.log(z / z0)

        x_grid = x_factor(z, L)

        tmp3 = phi_factor(x_grid)

        wind_array[k] = tmp1 * tmp2 - tmp3

    return wind_array


def swise_correction(wd, m):

    ## Put wind direction and transect angle into one coordinate system
    ##      0
    ##  na     90
    ##     180

    wd = (wd + 360) % 180 ## Angle from north up to 180 only
    transect_angle = (np.rad2deg(np.arctan(1/m)) + 360) % 180 ## Angle from the north line
    
    streamwise_factor = np.abs(np.sin(np.deg2rad(wd - transect_angle)))

    return float(streamwise_factor)

def xwise_correction(wd, m):
    wd = (wd + 360) % 180 ## Angle from north up to 180 only
    transect_angle = (np.rad2deg(np.arctan(1/m)) + 360) % 180 ## Angle from the north line

    theta_diff = wd - transect_angle
    # theta_diff = 180 - (wd - transect_angle)
    crosswise_factor = np.cos(np.deg2rad(theta_diff))

    return crosswise_factor
