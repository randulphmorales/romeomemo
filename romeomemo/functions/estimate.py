#!/usr/bin/env python
# coding=utf-8

import os
import sys

import numpy as np


def kriging_estimate(conc_field, wind_field):

    dimension = conc_field.shape[0] * conc_field.shape[1]

    conc_vec = conc_field.flatten().reshape(dimension, 1)
    wind_vec = wind_field.flatten().reshape(dimension, 1)

    flux_estimate = np.dot(conc_vec.T, wind_vec)

    return flux_estimate[0][0]


def cluster_covariance(prob_field, cluster_cov):
    """
    """

    prob_arr = prob_field.flatten().reshape(-1, 1)
    prob_mat = np.dot(prob_arr, prob_arr.T)

    sigma_cluster = np.multiply(prob_mat, cluster_cov)

    return sigma_cluster



def kriging_uncertainty(conc_field, wind_field, conc_cov_mat, wind_cov_mat):
    """
    """

    conc_vec = conc_field.flatten().reshape(-1, 1)
    wind_vec = wind_field.flatten().reshape(-1, 1)

    var_c = np.dot(conc_vec.T, np.dot(wind_cov_mat, conc_vec))
    var_w = np.dot(wind_vec.T, np.dot(conc_cov_mat, wind_vec))

    var_tot = var_c + var_w

    return var_tot[0][0]
