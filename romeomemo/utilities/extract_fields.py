#!/usr/bin/env python
# coding=utf-8

import numpy as np
import pandas as pd
import os

from scipy.interpolate import interp2d
from romeomemo.utilities import grids


class DroneSampling:
    def __init__(self, ymin, zmin, dy, dz, ny, nz, conc_file):
        self.ymin = ymin
        self.zmin = zmin
        self.dy = dy
        self.dz = dz
        self.ny = ny
        self.nz = nz
        self.conc_file = conc_file

    def interpolation_fxn(self):

        curtain_grid = grids.Grid(self.ny, self.nz, self.dz, self.dy,
                                  self.ymin, self.zmin)

        grid_x = curtain_grid.xdist_range()
        grid_z = curtain_grid.alt_range()

        interp_fxn = interp2d(grid_x, grid_z, self.conc_file)

        return interp_fxn


    def interpolate_points(self, distance, alt_above_ground):

        interp_fxn = self.interpolation_fxn()

        distance = np.asarray(distance)
        alt_above_ground = np.asarray(alt_above_ground)

        obs_pred = []
        for dist, alt in zip(distance, alt_above_ground):
            obs_point = interp_fxn(dist, alt)
            obs_pred.append(obs_point)

        obs_pred = np.asarray(obs_pred)

        return obs_pred


