#!/usr/bin/env python
# coding=utf-8

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

from romeomemo.utilities import conversion, grids, plotting, projection
from romeomemo.utilities import extract_fields, readconfig, readmemoascii

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
    krige_fname = "_".join(["OK", flight_code])
    fkrige_results = os.path.join(target_path, krige_fname + ".nc")
    cfg.fkrige_results = fkrige_results

    ## Load netCDF file from kriging results
    krige_results =  xr.open_dataset(fkrige_results)
    nx = krige_results.dims["x"]
    nz = krige_results.dims["z"]

    dx = np.around(krige_results.x[1] - krige_results.x[0], 2)
    dz = np.around(krige_results.z[1] - krige_results.z[0], 2)

    xmin = np.asarray(krige_results.x)[0] - dx/2
    zmin = np.asarray(krige_results.z)[0] - dz/2

    obs_pred = np.asarray(krige_results.ppm_mean)
    obs_var = np.asarray(krige_results.ppm_variance)

    memo = readmemoascii.memoCDF(cfg.memo_file, cfg.start, cfg.end)
    memo_df = memo.massbalance_data(NoXP, b)
    con_ab = memo_df["con_ab"]
    ## Align altitude of GPS with inlet of instrument
    # Distance between the GPS and the altitude is 72 cm
    agl = memo_df["Altitude"]

    ## COMPUTE PERPENDICULAR DISTANCE BETWEEN SOURCE AND CURTAIN
    dtm = memo_df.index
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]

    obs_interp = extract_fields.DroneSampling(xmin, zmin, dx, dz, nx, nz, obs_pred)
    var_interp = extract_fields.DroneSampling(xmin, zmin, dx, dz, nx, nz, obs_var)

    obs_points = obs_interp.interpolate_points(dist, agl)
    var_points = var_interp.interpolate_points(dist, agl)

    interp_df = pd.DataFrame(index = memo_df.index)
    interp_df["obs_pred"] = obs_points
    interp_df["var_pred"] = var_points

    dtm = memo_df.index
    obs = con_ab
    obs_pred = interp_df["obs_pred"]
    var_pred = interp_df["var_pred"]

    tseries_fig = plotting.extract_con_plots(dtm, obs, obs_pred, var_pred)
    scatter_fig = plotting.plot_scatter(obs, obs_pred, var_pred)

    return
