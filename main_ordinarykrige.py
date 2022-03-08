#!/usr/bin/env python
# coding=utf-8

import os
import sys

import numpy as np
import pandas as pd
import xarray as xr


import gstools as gs
from pykrige.ok import OrdinaryKriging

from romeomemo.massbalance import mixturemodel as mm
from romeomemo.meteorology import anemo, wcomponent
from romeomemo.utilities import conversion, grids, plotting, projection
from romeomemo.utilities import readconfig, readmemoascii

from romeomemo.functions import estimate



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

    memo = readmemoascii.memoCDF(memo_file, start, end)
    memo_df = memo.massbalance_data(NoXP, b)

    con_ab = memo_df["con_ab"]
    ## Align altitude of GPS with inlet of instrument
    # Distance between the GPS and the altitude is 72 cm
    agl = memo_df["Altitude"]

    ## Load spatial parameters
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]

    ## Load environmental parameters
    pres = memo_df["Pressure"]
    temp = memo_df["Temperature"]

    ## Replace NANs with STP values
    pres[pres == -999.99] = 1013.25
    temp[temp == -999.99] = 273.15

    ## Convert mole fraction to concentration
    # con_ab = conversion.ppm2micro(ppm_ab, temp=temp, pres=pres, mass=16.04)

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

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    save_arrays = True
    if os.path.isfile(fkrige_results):
        print("Overwite interpolated arrays in %s? " % target_path)
        s = input("y/[n] \n")
        save_arrays = (s == "y")

    if save_arrays:

        krige_ppm = pd.DataFrame(index=dist.index)
        krige_ppm["dist"] = dist
        krige_ppm["agl"] = agl
        krige_ppm["obs_data"] = con_ab

        max_dist = np.max(krige_ppm["dist"])
        # max_agl = np.max(krige_ppm["dist"])
        # max_disp = np.hypot(max_dist, max_agl) / 2
        agl_series = pd.Series(krige_ppm["agl"]).diff()
        agl_max = agl_series.max()

        ## Standard Kriging for measurements
        X_one = np.asarray(krige_ppm["dist"])
        Y_one = np.asarray(krige_ppm["agl"])
        Z_one = np.asarray(krige_ppm["obs_data"])

        init_gp1_l1 = mm.init_lengthscale(krige_ppm["dist"], 1)
        init_gp1_l2 = mm.init_lengthscale(krige_ppm["agl"], 1)
        init_gp1_sf = mm.init_variance(krige_ppm["obs_data"], 1)
        init_gp1_sn = 0.25 * init_gp1_sf

        nu = 1.5
        gp1_kernel = mm.matern_kernel(init_gp1_sf, [init_gp1_l1, init_gp1_l2],
                                      nu, agl_max, max_dist)
        gp1_sf, gp1_l1, gp1_l2 = mm.EM_hyperparam(X_one, Y_one, Z_one, gp1_kernel, init_gp1_sn)
        gp1_sn = np.mean(gp1_sf)

        print("Horizontal length scale: {:.4f}".format(gp1_l1))
        print("Vertical length scale: {:.4f}".format(gp1_l2))

        gp1_l1 = 2.9610
        gp1_l2 = 2.1607

        gp1_cov = gs.Matern(dim=2, var=gp1_sf, len_scale=[gp1_l1, gp1_l2],
                            nugget = gp1_sn, nu=nu)

        krige_one = OrdinaryKriging(X_one, Y_one, Z_one, variogram_model=gp1_cov)
        pred_one, var_one = krige_one.execute("grid", range_x, range_z)

        # Kriging wind data (Standard ordinary kriging)
        meteo_df = anemo.meteorology_data(meteo_file, start, end)
        sw_corr_factor = [wcomponent.swise_correction(wd, m) for wd in meteo_df["wd"]]
        meteo_df["sw_factor"] = sw_corr_factor

        meteo_df["stream_wind"] = meteo_df["ws"] * meteo_df["sw_factor"]
        meteo_df["dist"] = krige_ppm["dist"]
        meteo_df["agl"] = krige_ppm["agl"]
        meteo_df = meteo_df.dropna()

        init_gpw_l1 = mm.init_lengthscale(meteo_df["dist"], 1)
        init_gpw_l2 = mm.init_lengthscale(meteo_df["agl"], 1)
        init_gpw_sf = mm.init_variance(meteo_df["stream_wind"], 1)
        init_gpw_sn = 0.25 * init_gpw_sf

        gpw_kernel = mm.matern_kernel(init_gpw_sf, [init_gpw_l1, init_gpw_l2],
                                      nu, 1e-3, max_dist)
        gpw_sf, gpw_l1, gpw_l2 = mm.EM_hyperparam(meteo_df["dist"],
                                                  meteo_df["agl"],
                                                  meteo_df["stream_wind"],
                                                  gpw_kernel, init_gpw_sn)
        gpw_sn = 0.25 * gpw_sf

        gpw_cov = gs.Matern(dim=2, var=gpw_sf, len_scale=[gpw_l1, gpw_l2],
                            nugget=gpw_sn, nu=nu)

        wind_krige = OrdinaryKriging(meteo_df["dist"], meteo_df["agl"],
                                     meteo_df["stream_wind"], variogram_model=gpw_cov)
        wind_prof, wind_ss = wind_krige.execute("grid", range_x, range_z)

        ## Emission computation
        mean_pres = np.mean(pres)
        mean_temp = np.mean(temp)
        mic_meas = conversion.ppm2micro(pred_one, mass=16.04, temp=mean_temp, pres=mean_pres)

        krige_meas = estimate.kriging_estimate(mic_meas , wind_prof)
        okpw_flux = krige_meas * cfg.dx * cfg.dz * 1e-6

        ## Uncertainty computation 
        square_dist = target_grid.square_matrix()
        range_covx = np.arange(square_dist.shape[0])
        range_covz = np.arange(square_dist.shape[1])

        ppm_cov = gp1_cov.covariance(square_dist)
        conc_cov = conversion.ppm2micro(ppm_cov, mass=16.04, temp=mean_temp, pres=mean_pres)
        conc_cov = conversion.ppm2micro(conc_cov, mass=16.04, temp=mean_temp, pres=mean_pres)

        wind_cov = gpw_cov.covariance(square_dist)

        okpw_var = estimate.kriging_uncertainty(mic_meas, wind_prof, conc_cov, wind_cov)
        okpw_std = np.sqrt(okpw_var) * cfg.dx * cfg.dz * 1e-6

        print("================================")
        print("\n\nOKPW estimate is: {:.4f}".format(okpw_flux))
        print("\nOKPW Uncertainty estimate is: {:.4f}".format(okpw_std))

        ## OKSW
        ## Domininant perpendicular wind
        # mean_perp_wind = meteo_df["stream_wind"].mean()
        # std_perp_wind = meteo_df["stream_wind"].std()

        # sw_mean = np.full((nz,nx), mean_perp_wind)
        # sw_ss = np.full((nz,nx), std_perp_wind**2)

        # oksw_mean = estimate.kriging_estimate(mic_meas, sw_mean)
        # oksw_flux = oksw_mean * dx * dz * 1e-6

        # oksw_var = estimate.kriging_uncertainty(mic_meas, sw_mean, conc_cov, std_perp_wind)
        # oksw_std = np.sqrt(oksw_var) * dx * dz * 1e-6

        # print("================================")
        # print("\n\nOKPW estimate is: {:.4f}".format(oksw_flux))
        # print("\nOKPW Uncertainty estimate is: {:.4f}".format(oksw_std))


        krige_results = xr.Dataset({
                     "z" : xr.DataArray(range_z, dims=("z",)),
                     "x" : xr.DataArray(range_x, dims=("x",)),
                     "cov_z" : xr.DataArray(range_covz, dims=("cov_z")),
                     "cov_x" : xr.DataArray(range_covx, dims=("cov_x")),
                     "ppm_mean" : xr.DataArray(pred_one, dims=("z", "x")),
                     "ppm_variance" : xr.DataArray(var_one, dims=("z", "x")),
                     "wind_mean" : xr.DataArray(wind_prof, dims=("z", "x")),
                     "wind_variance" : xr.DataArray(wind_ss, dims=("z", "x")),
                     "concentration" : xr.DataArray(mic_meas, dims=("z","x")),
                     "ppm_covmat" : xr.DataArray(ppm_cov, dims=("cov_z", "cov_x")),
                     "conc_covmat" : xr.DataArray(conc_cov, dims=("cov_z", "cov_x")),
                     "wind_covmat" : xr.DataArray(wind_cov, dims=("cov_z", "cov_x")),
                     })

        krige_results.attrs["curtain_r2"] = r_value
        krige_results.attrs["perp_distance"] = perp_distance
        krige_results.attrs["mass_emission"] = okpw_flux
        krige_results.attrs["std_mass_emission"] = okpw_std
        krige_results.attrs["flight_code"] = flight_code
        krige_results.attrs["source_lon"] = source_lon
        krige_results.attrs["source_lat"] = source_lat
        krige_results.attrs["mean_pres"] = mean_pres
        krige_results.attrs["mean_temp"] = mean_temp

        krige_results.to_netcdf(fkrige_results, mode="w")

        print("Successfully written the emission file")

    else:
        pass

    return cfg



if __name__ ==  "__main__":
    try:
        config_path = sys.argv[1]
    except IndexError:
        raise RuntimeError("Please supply a config file.")

    print("Computing methane fluxes using mass balance...")
    cfg = main(config_path)

    ## Initialize path for saving files
    target_path = cfg.model_path
    fkrige_results = cfg.fkrige_results

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
    mic_meas = np.asarray(krige_results.concentration)
    wind_mean = np.asarray(krige_results.wind_mean)
    wind_var = np.asarray(krige_results.wind_variance)

    source_lon = krige_results.attrs["source_lon"]
    source_lat = krige_results.attrs["source_lat"]
    r_value = krige_results.attrs["curtain_r2"]
    perp_distance = krige_results.attrs["perp_distance"]

    NoXP = cfg.rebs_NoXP
    b = cfg.rebs_b

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

    im_meas_scatter = os.path.join(target_path,"measurement_scatter.png")

    im_obs_pred = os.path.join(target_path, "ppm_meas.png")
    im_obs_pred_scatter = os.path.join(target_path, "ppm_meas_scatter.png")
    im_obs_var = os.path.join(target_path, "ppm_ss.png")
    im_mic_meas = os.path.join(target_path, "mic_meas.png")

    im_proj_meas = os.path.join(target_path, "projected_conc_map.png")
    im_wind_prof = os.path.join(target_path, "wind_prof.png")
    im_above_bg = os.path.join(target_path, "above_background.png")

    save_images = True
    if os.path.isfile(im_meas_scatter):
        print("Overwrite the plots in %s?" % target_path)
        s = input("y/[n] \n")
        save_images = (s=="y")

    if save_images:
        curtain_plot = plotting.GridPlotting(nx, nz, dx, dz, xmin, zmin)

        ## Scatter plot displaying measured values with respect to height
        fig = plotting.geo_scatter(dtm, dist, agl, con_ab, "Distance [m]",
                                   "Altitude [m]", "CH$_4$ [ppm]")

        ## curtain plot of prediction field of methane molar fraction
        fig2 = curtain_plot.curtainPlots(obs_pred, units = "CH$_4$ [ppm]",
                                         title = "Predicted Measured Molar Fraction")

        ## curtain plot of prediction variance field of methane molar fraction
        fig3 = curtain_plot.curtainPlots(obs_var, units = "CH$_4$ [ppm$^2$]",
                                         title = "Prediction variance")

        ## curtain plot of prediction field overlaid with measurement points
        fig4 = curtain_plot.curtainScatterPlots(dist, agl, con_ab, obs_pred,
                                                units = "CH$_4$ [ppm]", title =
                                                "Predicted Measured Molar Fraction")

        ## curtain plot displaying krige measured concentration
        fig5 = curtain_plot.curtainPlots(mic_meas, units = "CH$_4$ [$\mu$g/m$^3$]",
                                         title = "Predicted Measured Concentration")

        ## Wind profile plot
        fig6 = curtain_plot.curtainPlots(wind_mean, units="[m/s]",
                                         title="Kriging streamwise wind")

        ## Map plot of point source and projected points
        fig7 = curtain_plot.projected_map(source_lon, source_lat, lon, lat,
                                          con_ab, r_value, perp_distance,
                                          units="CH4 [ppm]")


        ## Time series of methane elevations from background
        fig8 = plotting.measurevsbg(dtm, con_ab, agl)


        ## SAVE FIGURE
        fig.savefig(im_meas_scatter, dpi=300)
        fig2.savefig(im_obs_pred, dpi=300)
        fig3.savefig(im_obs_var, dpi=300)
        fig4.savefig(im_obs_pred_scatter, dpi=300)
        fig5.savefig(im_mic_meas, dpi=300)
        fig6.savefig(im_wind_prof, dpi=300)
        fig7.savefig(im_proj_meas, dpi=300)
        fig8.savefig(im_above_bg, dpi=300)
