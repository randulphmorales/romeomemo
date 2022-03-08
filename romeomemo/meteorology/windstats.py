#!/usr/bin/env python
# coding=utf-8

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import massbalance.mixturemodel as mm
from meteorology import anemo, wcomponent
from utilities import conversion, grids, plotting, projection
from utilities import readconfig, readmemoascii

import gstools as gs
from pykrige.ok import OrdinaryKriging


def main(cfg_path):

    cfg = readconfig.load_cfg(cfg_path)

    memo_file = cfg.memo_file
    meteo_file = cfg.meteo_file
    flight_code = cfg.flight_code

    # horizontal and vertical grid steps
    dx = cfg.dx
    dz = cfg.dz

    ## start and end time of mass balance
    start = cfg.start
    end = cfg.end

    ## REBS parameters
    NoXP = cfg.rebs_NoXP
    b = cfg.rebs_b

    ## LOCATION OF THE SOURCE
    source_lon = cfg.source_lon
    source_lat = cfg.source_lat

    ## Meteorology configuration
    meteo_file = cfg.meteo_file

    # Define target path
    target_path = cfg.model_path

    ## INITIALIZE FILES
    krige_fname = "_".join(["CK", flight_code])
    fkrige_results = os.path.join(target_path, krige_fname + ".nc")
    cfg.fkrige_results = fkrige_results

    memo = readmemoascii.memoCDF(memo_file, start, end)
    ground_alt, _ = memo.ground_altitude()
    memo_df = memo.massbalance_data(NoXP, b)

    con_ab = memo_df["con_ab"]
    ## Align altitude of GPS with inlet of instrument
    # Distance between the GPS and the altitude is 72 cm
    agl = (memo_df["Altitude"] - ground_alt)

    ## COMPUTE PERPENDICULAR DISTANCE BETWEEN SOURCE AND CURTAIN
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]

    proc_gps = grids.ProcessCoordinates(lon, lat)
    m, b, r_value = proc_gps.regression_coeffs()

    perp_distance = grids.perpendicular_distance(lon, lat, source_lon,
                                                 source_lat)

    meteo_df = anemo.meteorology_data(meteo_file, start, end)
    sw_corr_factor = [wcomponent.swise_correction(wd, m) for wd in meteo_df["wd"]]
    meteo_df["sw_factor"] = sw_corr_factor

    meteo_df["stream_wind"] = meteo_df["ws"] * meteo_df["sw_factor"]

    print("WS : {:.4f}".format(meteo_df.ws.mean()))
    print("WS_STD : {:.4f}".format(meteo_df.ws.std()))
    print("WD : {:.4f}".format(meteo_df.wd.mean()))
    print("WD_STD : {:.4f}".format(meteo_df.wd.std()))
    print("stream_ws : {:.4f}".format(meteo_df.stream_wind.mean()))
    print("stream_std : {:.4f}".format(meteo_df.stream_wind.std()))
    print("U_star : {:.4f}". format(meteo_df["u.star"].mean()))

    return


