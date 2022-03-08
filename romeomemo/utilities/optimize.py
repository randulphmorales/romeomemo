#!/usr/bin/env python
# coding=utf-8

import numpy as np
from scipy.optimize import minimize

def weight_matrix(weights, one, two):

    one = one.flatten()
    two = two.flatten()

    one_matrix = []
    two_matrix = []

    for i,j in zip(one, two):
        def optimize(weights):
            w1, w2 = weights
            return w1**2 * i + w2**2 * j

        def constraint(weights):
            w1, w2 = weights
            return w1 + w2 -1

        sol = minimize(optimize, weights, constraints={"type":"eq", "fun":constraint})
        one_matrix.append(sol.x[0])
        two_matrix.append(sol.x[1])

    one_matrix = np.asarray(one_matrix)
    two_matrix = np.asarray(two_matrix)

    return one_matrix.reshape(one.shape), two_matrix.reshape(two.shape)
