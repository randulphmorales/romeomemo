#!/usr/bin/env python
# coding=utf-8

import numpy as np

def ppm2micro(conc, mass, temp=273.15, pres=1013.25):
    """
    Returns micrograms per cubic meter from ppb measurements

    Parameters
    ----------
    conc: float
        measured molar fraction of species
    temp: float
        ambient temperature in Kelvin
    pres: float
        ambient pressure in hPa
    mass: float
        molar mass of species (e.g. CH4 = 16.04 g/mol)

    Returns
    -------
    float
       concentration in ug / m3

    """

    R = 8.31446
    conc_out = pres / R / temp * 1e2 * mass * conc
    # conc_out[conc <= 0] = np.nan
    
    return conc_out


def micro2ppm(conc, mass, temp=273.15, pres=1013.25):
    """
    Returns ppm from micrograms per cubic meter of species

    Parameters
    ----------
    conc: float
        measured molar fraction of species
    temp: float
        ambient temperature in degrees Celcius
    pres: float
        ambient pressure in hPa
    mass: float
        molar mass of species (e.g. CH4 = 16.04 g/mol)

    Returns
    -------
    float
       concentration in kg / m3

    """

    R = 8.31446
    conc_out = conc * R * temp * 1e-2 / (mass * pres)

    return conc_out



def mass2volumeflow(mass_rate, mass, temp=273.15, pres=101.325):
    """
    Returns mass rate in g/s to volume L/min

    Parameters
    ----------
    mass_rate : float
        emission flux in g/s
    mass : float
        molar mass of species in g/mol(e.g. CH4 = 16.04 g/mol)
    temp : float
        temperature in Kelvin
    pres : float
        pressure in kPa
    """
    R = 8.31446

    ## conversion factor in m3/mol
    conv_factor = (R * temp) / pres

    ## convert g/s into L/min
    volume_rate = mass_rate * (1 / mass) * conv_factor * 60

    return volume_rate


def volume2massflow(volume_rate, mass, temp=273.15, pres=101.325):
    """
    Returns volume rate in L/min into mass rate in g/s

    Parameters
    ----------
    mass_rate : float
        emission flux in g/s
    mass : float
        molar mass of species in g/mol(e.g. CH4 = 16.04 g/mol)
    temp : float
        temperature in Kelvin
    pres : float
        pressure in kPa
    """

    R = 8.31446

    ## conversion factor in m3/mol
    conv_factor = (R * temp) / pres

    ## convert L/min into g/s
    mass_rate = volume_rate * mass * (1 / conv_factor) * (1 / 60)

    return mass_rate
