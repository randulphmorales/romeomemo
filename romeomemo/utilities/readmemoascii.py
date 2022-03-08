#!/usr/bin/env python
# coding=utf-8

import pandas as pd
import numpy as np

from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from scipy.interpolate import interp1d

from romeomemo.utilities import grids


utils = importr("utils")
# utils.install_packages("IDPmisc", cleanup=True)
IDPmisc = importr("IDPmisc")

def compute_bg(x_val, y_val, NoXP, b):

    pandas2ri.activate()
    baseline = IDPmisc.rfbaseline(x_val, y_val, NoXP=NoXP, b=b, maxit=np.array([10,10]))

    background = np.array(baseline[2])

    return background

class memoCDF:

    def __init__(self, memofile, start=None, end=None):
        """
        Parameters
        ----------
        memofile : string
            path and filename of the memo ascii common data format

        start : string
             YYYY-MM-DD HH:MM:SS starting time of the measurement flight
        end : string
             YYYY-MM-DD HH:MM:SS end time of the measurement flight
        """
        self.memofile = memofile
        self.start = start
        self.end = end


    def dataFrame(self):
        """ Reads the memo2 ascii common data format
        Parameters
        ----------
        memo_file : string
            contains the path and the filename of the memo ascii common data format
        start : string
            datetime of the start of the mass balance. Format  "%Y-%m-%d %H:%M:%S"
        end : string
            datetime of the end of the mass balance. Format  "%Y-%m-%d %H:%M:%S"

        Returns
        -------
        tuple(pd.DataFrame)
        """

        memo_df = pd.read_csv(self.memofile, sep=";", header=0, comment="#",
                              index_col=0, parse_dates=[["Date_UTC", "Time_UTC"]])

        memo_df["CH4_spec_corr"] = np.asarray(memo_df["CH4_spec_corr"]) / 1000.0

        return memo_df


    def dataFrameRUG(self):
        """ Reads the memo2 ascii common data format
        Parameters
        ----------
        memo_file : string
            contains the path and the filename of the memo ascii common data format
        start : string
            datetime of the start of the mass balance. Format  "%Y-%m-%d %H:%M:%S"
        end : string
            datetime of the end of the mass balance. Format  "%Y-%m-%d %H:%M:%S"

        Returns
        -------
        tuple(pd.Series)
        """

        rug_df = pd.read_csv(self.memofile, index_col=1, parse_dates=True)

        columns = ["ch4.corr", "lon", "lat", "alt"]
        rename_col = {"lon":"Longitude", "lat":"Latitude",
                      "alt":"Altitude", "ch4.corr":"CH4_spec_corr"}

        # columns = ["lon", "lat", "alt.uav", "ch4"]
        # rename_col = {"lon":"Longitude", "lat":"Latitude",
        #               "alt.uav":"Altitude", "ch4":"CH4_spec_corr"}

        rug_df = rug_df[columns]
        rug_df = rug_df.rename(columns=rename_col)
        # rug_df = rug_df[rug_df["Longitude"] != 0]
        rug_df["CH4_spec_corr"] /= 1000.00

        return rug_df



    def remove_background(self, NoXP, b):

        ## COMPUTE BACKGROUND
        memo_df = self.dataFrame()
        ch4_index = np.arange(len(memo_df.index))
        ch4_conc = np.array(memo_df["CH4_spec_corr"])

        ch4_bg_array = compute_bg(ch4_index, ch4_conc, NoXP, b)
        ch4_above_bg = ch4_conc - ch4_bg_array

        above_df = pd.Series(data=ch4_above_bg, index=memo_df.index)

        return above_df


    def massbalance_data(self,  NoXP=150.0, b=3.5):

        memo_df = self.dataFrame()
        memo_df = memo_df[(memo_df.index >= self.start) & (memo_df.index <= self.end)]
        memo_df = memo_df.resample("1s").interpolate()

        con_sc = np.asarray(memo_df["CH4_spec_corr"])
        con_bg = compute_bg(np.arange(len(memo_df.index)), con_sc, NoXP, b)
        con_ab = con_sc - con_bg

        memo_df["con_bg"] = con_bg
        memo_df["con_ab"] = con_ab

        lon = np.asarray(memo_df["Longitude"])
        lat = np.asarray(memo_df["Latitude"])

        proc_gps = grids.ProcessCoordinates(lon, lat)
        m, b, r_value = proc_gps.regression_coeffs()
        proj_lon, proj_lat = proc_gps.regression_vector()
        lon_anchor, lat_anchor = proc_gps.anchorpoint(proj_lon, proj_lat)

        distance = proc_gps.projected_distance(lon_anchor, lat_anchor)
        memo_df["Distance"] = distance
        memo_df = memo_df.drop(["Date_Loc", "Time_Loc", "Flag"], axis=1)

        return memo_df


    def massbalance_rugdata(self, NoXP=150.0, b=3.5):

        memo_df = self.dataFrameRUG()

        memo_df = memo_df[memo_df["CH4_spec_corr"] != 0]
        memo_df = memo_df[memo_df["Longitude"] != 0]

        memo_df = memo_df.resample("1s").interpolate()
        memo_df = memo_df.dropna()

        con_sc = np.asarray(memo_df["CH4_spec_corr"])
        con_bg = compute_bg(np.arange(len(memo_df.index)), con_sc, NoXP, b)
        con_ab = con_sc - con_bg

        memo_df["con_bg"] = con_bg
        memo_df["con_ab"] = con_ab
        memo_df = memo_df[(memo_df.index >= self.start) & (memo_df.index <= self.end)]

        lon = np.asarray(memo_df["Longitude"])
        lat = np.asarray(memo_df["Latitude"])

        proc_gps = grids.ProcessCoordinates(lon, lat)
        m, b, r_value = proc_gps.regression_coeffs()
        proj_lon, proj_lat = proc_gps.regression_vector()
        lon_anchor, lat_anchor = proc_gps.anchorpoint(proj_lon, proj_lat)

        distance = proc_gps.projected_distance(lon_anchor, lat_anchor)
        memo_df["Distance"] = distance

        return memo_df
