#!/usr/bin/env python
# coding=utf-8

import os
import sys
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from romeomemo.massbalance import mixturemodel as mm
from romeomemo.meteorology import anemo, wcomponent
from romeomemo.utilities import conversion, grids, plotting, projection
from romeomemo.utilities import readconfig, readmemoascii

from romeomemo.functions import estimate

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

    ## Log wind construction
    wind_z = cfg.wind_z
    wind_OL = cfg.wind_OL
    wind_u_star = cfg.wind_u_star

    ## LOCATION OF THE SOURCE
    source_lon = cfg.source_lon
    source_lat = cfg.source_lat

    ## Meteorology configuration
    meteo_file = cfg.meteo_file

    # Define target path
    target_path = cfg.model_path

    ## INITIALIZE FILES
    wind_fname = "_".join(["MET", flight_code])
    fwind_results = os.path.join(target_path, wind_fname + ".nc")
    cfg.fwind_results = fwind_results

    memo = readmemoascii.memoCDF(memo_file, start, end)
    memo_df = memo.massbalance_data(NoXP, b)

    con_ab = memo_df["con_ab"]
    ## Align altitude of GPS with inlet of instrument
    # Distance between the GPS and the altitude is 72 cm
    agl = memo_df["Altitude"]

    ## COMPUTE PERPENDICULAR DISTANCE BETWEEN SOURCE AND CURTAIN
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]

    proc_gps = grids.ProcessCoordinates(lon, lat)
    m, b, r_value = proc_gps.regression_coeffs()

    perp_distance = grids.perpendicular_distance(lon, lat, source_lon, source_lat)

    xmin = 0.0
    zmin = 0.0
    nx = int(np.floor(np.max(dist) / dx) + 1)
    nz = int(np.floor(np.max(agl) / dz) + 1)

    ## CREATE TARGET GRID
    target_grid = grids.Grid(nx, nz, dx, dz, xmin, zmin)
    range_x = target_grid.xdist_range()
    range_z = target_grid.alt_range()

    krige_ppm = pd.DataFrame(index=dist.index)
    krige_ppm["dist"] = dist
    krige_ppm["agl"] = agl
    krige_ppm["obs_data"] = con_ab

    max_dist = np.max(krige_ppm["dist"])
    agl_series = pd.Series(krige_ppm["agl"]).diff()
    agl_max = agl_series.max()

    ## Projected Wind profile
    meteo_df = anemo.meteorology_data(meteo_file, start, end)
    sw_corr_factor = [wcomponent.swise_correction(wd, m) for wd in meteo_df["wd"]]
    meteo_df["sw_factor"] = sw_corr_factor

    meteo_df["stream_wind"] = meteo_df["ws"] * meteo_df["sw_factor"]
    meteo_df["dist"] = krige_ppm["dist"]
    meteo_df["agl"] = krige_ppm["agl"]
    meteo_df = meteo_df.dropna()

    ## Domininant perpendicular wind
    mean_perp_wind = meteo_df["stream_wind"].mean()
    std_perp_wind = meteo_df["stream_wind"].std()
    ste_perp_wind = std_perp_wind / np.sqrt(len(meteo_df))

    dom_wind_mean = np.full((nz,nx), mean_perp_wind)
    dom_wind_ss = np.full((nz,nx), std_perp_wind**2)
    dom_wind_se = np.full((nz,nx), ste_perp_wind)
    print("Average mean wind speed: {:.4f}m/s".format(mean_perp_wind))
    print("Standard deviation: {:.4f}m/s".format(std_perp_wind))
    print("Standard error: {:.4f}m/s".format(ste_perp_wind))

    ## Logarithmic profile wind
    stab_param = wind_z / wind_OL

    if np.isclose(stab_param, 0, atol=0.5):
        print("Neutral")
        log_wind = wcomponent.neutral_prof(mean_perp_wind, wind_z, wind_u_star,
                                           range_z)
        std_log = wcomponent.neutral_prof(std_perp_wind, wind_z, wind_u_star,
                                          range_z)
    elif stab_param > 0.5:
        print("Stable")
        log_wind = wcomponent.stable_prof(mean_perp_wind, wind_z,
                                          wind_u_star,wind_OL, range_z)
        std_log = wcomponent.stable_prof(std_perp_wind, wind_z, wind_u_star,
                                         wind_OL, range_z)
    elif stab_param < -0.5:
        print("Unstable")
        log_wind = wcomponent.unstable_prof(mean_perp_wind, wind_z,
                                            wind_u_star, wind_OL, range_z)
        std_log = wcomponent.unstable_prof(std_perp_wind, wind_z, wind_u_star,
                                           wind_OL, range_z)
    else:
        print("Check stability parameter value:{:.4f}".format(stab_param))

    min_wind = np.min(log_wind[log_wind >= 0])
    log_wind[log_wind < 0] = min_wind

    log_mean_curt = np.ones((nz, nx)) * log_wind[:, None]
    std_mean_curt = np.ones((nz, nx)) * std_log[:, None]


    print("================================")
    print("Writing meteorology files.\n\n")

    wind_results = xr.Dataset({
                "z" : xr.DataArray(range_z, dims=("z",)),
                "x" : xr.DataArray(range_x, dims=("x",)),
                "scalar_wind_mean" : xr.DataArray(dom_wind_mean, dims=("z", "x")),
                "scalar_wind_std" : xr.DataArray(dom_wind_ss, dims=("z", "x")),
                "scalar_wind_ste" : xr.DataArray(dom_wind_se, dims=("z", "x")),
                "log_wind_mean" : xr.DataArray(log_mean_curt, dims=("z", "x")),
                "log_wind_std" : xr.DataArray(std_mean_curt, dims=("z", "x")),
                })

    wind_results.to_netcdf(fwind_results)

    print("Meteorology curtain file written")

    return

if __name__ ==  "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        raise RuntimeError("Please supply a config file.")
    main(config_path)

