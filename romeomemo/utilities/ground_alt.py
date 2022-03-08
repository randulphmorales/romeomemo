#!/usr/bin/env python
# coding=utf-8

import numpy as np
from scipy.stats import linregress

def alt_regression(agl, asl):
    """ Computes constant ground altitude
    Parameters
    ----------
    agl : numpy.array
        above take off from landing
    asl : numpy.array
        above sea level

    Returns
    -------
    g_alt : constant offset for altitude above sea level
    """
    agl = np.asarray(agl)
    asl = np.asarray(asl)

    m, g_alt, r_value, p_value, std_err = linregress(agl, asl)

    return g_alt
