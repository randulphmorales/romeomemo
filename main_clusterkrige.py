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
from romeomemo.utilities import conversion, grids, plotting
from romeomemo.utilities import readconfig, readmemoascii

from romeomemo.functions import estimate


def main(cfg_path):
    
    cfg = readconfig.load_cfg(cfg_path)

    memo_file = cfg.memo_file
    rug_file = cfg.rug_file
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
    fkrige_results = os.path.join(target_path, krige_fname + "v2.nc")
    cfg.fkrige_results = fkrige_results

    ### UNCOMMENT BLOCK FOR PROCESSNG QCL FILES ###
    ##################################################
    memo = readmemoascii.memoCDF(memo_file, start, end)
    memo_df = memo.massbalance_data(NoXP, b)

    con_ab = memo_df["con_ab"]

    ## Load spatial parameters
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]

    ## Load environmental parameters
    pres = memo_df["Pressure"]
    temp = memo_df["Temperature"]

    agl = memo_df["Altitude"]
    ##################################################

    ## UNCOMMENT BLOCK FOR PROCESSING RUG FILES ##
    ##############################################
    # memo = readmemoascii.memoCDF(memo_file, start, end)
    # memo_qcl_df = memo.massbalance_data(NoXP, b)

    # memo = readmemoascii.memoCDF(rug_file, start, end)
    # memo_df = memo.massbalance_rugdata(NoXP, b)

    # memo_qcl_df = memo_qcl_df[memo_qcl_df.index.isin(memo_df.index)]
    # memo_df = memo_df[memo_df.index.isin(memo_qcl_df.index)]

    # con_ab = memo_df["con_ab"]
    # ## Align altitude of GPS with inlet of instrument
    # # Distance between the GPS and the altitude is 72 cm
    # # agl = memo_df["Altitude"]

    # ## Load spatial parameters
    # lon = memo_df["Longitude"]
    # lat = memo_df["Latitude"]
    # dist = memo_df["Distance"]

    # # ## Load environmental parameters
    # pres = memo_qcl_df["Pressure"]
    # temp = memo_qcl_df["Temperature"]

    # agl = memo_qcl_df["Altitude"]
    ###############################################

    ### UNCOMMENT BLOCK FOR PROCESSING FIT FILES ##
    # memo = readmemoascii.memoCDF(memo_file, start, end)
    # memo_df = memo.massbalance_data(NoXP, b)

    # try:
    #     fit_file = cfg.fit_file
    # except NameError:
    #     raise FileNotFoundError(f"Fitted file does not exist")

    # fit_ds = xr.open_dataset(fit_file)
    # fit_df = fit_ds.to_dataframe()
    # fit_df = fit_df.dropna()
    # con_ab = fit_df["y_ac"]
    # lon = fit_df["longitude"]
    # lat = fit_df["latitude"]
    # alt = fit_df["altitude"]

    # dist = fit_df["distance"]

    # ## Load environmental parameters
    # pres = memo_df["Pressure"]
    # temp = memo_df["Temperature"]
    # agl = memo_df["Altitude"]
    # ###############################################

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

        obs_x = np.asarray(list(zip(dist, agl)))
        obs_y = np.asarray(con_ab)

        gmm = mm.GMM(n_clusters=2)
        gmm.set_training_values(obs_x, obs_y)
        gmm.train()

        prob_clust, hard_clust = gmm.cluster_data()
        weights = gmm.weights()

        ## Compute posteriori membership probabilities
        post_points = []
        for j, z in enumerate(range_z):
            for i, x in enumerate(range_x):
                post_points.append([x,z])
        post_points = np.asarray(post_points)

        membership_prob = gmm.prob_membership(post_points)

        prob_one = membership_prob[:,1].reshape(nz, nx)
        prob_two = membership_prob[:,0].reshape(nz, nx)
        # opp_clust = np.where((hard_clust==0)|(hard_clust==1), hard_clust^1, hard_clust)

        krige_ppm = pd.DataFrame(index=dist.index)
        krige_ppm["dist"] = dist
        krige_ppm["agl"] = agl
        krige_ppm["obs_data"] = con_ab
        krige_ppm["hard_cluster"] = hard_clust
        krige_ppm["prob"] = prob_clust[:,1]

        max_dist = np.max(krige_ppm["dist"])
        agl_series = pd.Series(krige_ppm["agl"]).diff()
        agl_max = agl_series.max()

        ## Component zero
        cl_one = krige_ppm[krige_ppm["hard_cluster"]==1]
        X_one = np.asarray(cl_one["dist"])
        Y_one = np.asarray(cl_one["agl"])
        Z_one = np.asarray(cl_one["obs_data"])

        init_gp1_l1 = mm.init_lengthscale(cl_one["dist"], cl_one["prob"])
        init_gp1_l2 = mm.init_lengthscale(cl_one["agl"], cl_one["prob"])
        init_gp1_sf = mm.init_variance(cl_one["obs_data"], cl_one["prob"])
        init_gp1_sn = 0.25 * init_gp1_sf

        nu = 1.5
        gp1_kernel = mm.matern_kernel(init_gp1_sf, [init_gp1_l1, init_gp1_l2],
                                      nu, agl_max, max_dist)
        gp1_sf, gp1_l1, gp1_l2 = mm.EM_hyperparam(X_one, Y_one, Z_one, gp1_kernel, init_gp1_sn)
        gp1_sn = np.mean(gp1_sf/np.asarray(cl_one["prob"]))

        print("Horizontal length scale GP1: {:.4f}".format(gp1_l1))
        print("Vertical length scale GP1: {:.4f}".format(gp1_l2))


        gp1_cov = gs.Matern(dim=2, var=gp1_sf, len_scale=[gp1_l1, gp1_l2],
                            nugget = gp1_sn, nu=nu)

        krige_one = OrdinaryKriging(X_one, Y_one, Z_one, variogram_model=gp1_cov)
        pred_one, var_one = krige_one.execute("grid", range_x, range_z)

        ## Component two
        comp_clust = 1 - krige_ppm["hard_cluster"]
        comp_prob = 1 - krige_ppm["prob"]
        krige_ppm["comp_prob"] = comp_prob
        cl_two = krige_ppm[comp_clust==1]
        X_two = np.asarray(cl_two["dist"])
        Y_two = np.asarray(cl_two["agl"])
        Z_two = np.asarray(cl_two["obs_data"])

        init_gp2_l1 = mm.init_lengthscale(cl_two["dist"], cl_two["comp_prob"])
        init_gp2_l2 = mm.init_lengthscale(cl_two["agl"], cl_two["comp_prob"])
        init_gp2_sf = mm.init_variance(cl_two["obs_data"], cl_two["comp_prob"])
        init_gp2_sn = 0.25 * init_gp2_sf
        init_gp2_sn /= np.asarray(cl_two["comp_prob"])

        gp2_kernel = mm.matern_kernel(init_gp2_sf, [init_gp2_l1, init_gp2_l2],
                                      nu, agl_max, max_dist)
        gp2_sf, gp2_l1, gp2_l2 = mm.EM_hyperparam(X_two, Y_two, Z_two, gp2_kernel, init_gp2_sn)
        gp2_sn = np.mean(gp2_sf/np.asarray(cl_two["comp_prob"]))

        print("Horizontal length scale GP2: {:.4f}".format(gp2_l1))
        print("Vertical length scale GP2: {:.4f}".format(gp2_l2))

        gp2_cov = gs.Matern(dim=2, var=gp2_sf, len_scale=[gp2_l1, gp2_l2],
                            nugget=gp2_sn, nu=nu)

        krige_two = OrdinaryKriging(X_two, Y_two, Z_two, variogram_model=gp2_cov)
        pred_two, var_two = krige_two.execute("grid", range_x, range_z)

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
                                      nu, agl_max, max_dist)
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


        ## Final prediction
        obs_pred = (prob_one * pred_one) + (prob_two * pred_two)

        ## Final variance
        res_one = prob_one * (var_one + pred_one**2)
        res_two = prob_two * (var_two + pred_two**2)

        obs_var = res_one + res_two - obs_pred ** 2

        ## Emission computation
        mean_pres = np.mean(pres)
        mean_temp = np.mean(temp)
        mic_meas = conversion.ppm2micro(obs_pred, mass=16.04, temp=mean_temp, pres=mean_pres)

        krige_meas = estimate.kriging_estimate(mic_meas, wind_prof)
        flux_estimate = krige_meas * cfg.dx * cfg.dz * 1e-6

        ## Uncertainty computation 
        square_dist = target_grid.square_matrix()
        range_covx = np.arange(square_dist.shape[0])
        range_covz = np.arange(square_dist.shape[1])

        gp1_covmat = gp1_cov.covariance(square_dist)
        gp2_covmat = gp2_cov.covariance(square_dist)

        sigma_c1 = estimate.cluster_covariance(prob_one, gp1_covmat)
        sigma_c2 = estimate.cluster_covariance(prob_two, gp2_covmat)

        ppm_cov = sigma_c1 + sigma_c2
        conc_cov = conversion.ppm2micro(ppm_cov, mass=16.04, temp=mean_temp, pres=mean_pres)
        conc_cov = conversion.ppm2micro(conc_cov, mass=16.04, temp=mean_temp, pres=mean_pres)

        wind_cov = gpw_cov.covariance(square_dist)

        var_flux = estimate.kriging_uncertainty(mic_meas, wind_prof, conc_cov, wind_cov)
        sigma_flux = np.sqrt(var_flux) * cfg.dx * cfg.dz * 1e-6


        print("================================")
        print("\n\nEmission estimate is: {:.4f}".format(flux_estimate))
        print("\n\nUncertainty estimate is: {:.4f}".format(sigma_flux))

        krige_results = xr.Dataset({
                     "z" : xr.DataArray(range_z, dims=("z",)),
                     "x" : xr.DataArray(range_x, dims=("x",)),
                     "cov_z" : xr.DataArray(range_covz, dims=("cov_z")),
                     "cov_x" : xr.DataArray(range_covx, dims=("cov_x")),
                     "ppm_cl_one" : xr.DataArray(pred_one, dims=("z", "x")),
                     "ppm_cl_two" : xr.DataArray(pred_two, dims=("z", "x")),
                     "var_cl_one" : xr.DataArray(var_one, dims=("z", "x")),
                     "var_cl_two" : xr.DataArray(var_two, dims=("z", "x")),
                     "prob_cl_one" : xr.DataArray(prob_one, dims=("z", "x")),
                     "prob_cl_two" : xr.DataArray(prob_two, dims=("z", "x")),
                     "ppm_mean" : xr.DataArray(obs_pred, dims=("z", "x")),
                     "ppm_variance" : xr.DataArray(obs_var, dims=("z", "x")),
                     "wind_mean" : xr.DataArray(wind_prof, dims=("z", "x")),
                     "wind_variance" : xr.DataArray(wind_ss, dims=("z", "x")),
                     "concentration" : xr.DataArray(mic_meas, dims=("z","x")),
                     "cov_one" : xr.DataArray(gp1_covmat, dims=("cov_z", "cov_x")),
                     "cov_two" : xr.DataArray(gp2_covmat, dims=("cov_z", "cov_x")),
                     "ppm_covmat" : xr.DataArray(ppm_cov, dims=("cov_z", "cov_x")),
                     "conc_covmat" : xr.DataArray(conc_cov, dims=("cov_z", "cov_x")),
                     "wind_covmat" : xr.DataArray(wind_cov, dims=("cov_z", "cov_x")),
                     })

        krige_results.attrs["curtain_r2"] = r_value
        krige_results.attrs["perp_distance"] = perp_distance
        krige_results.attrs["mass_emission"] = flux_estimate
        krige_results.attrs["std_mass_emission"] = sigma_flux
        krige_results.attrs["flight_code"] = flight_code
        krige_results.attrs["source_lon"] = source_lon
        krige_results.attrs["source_lat"] = source_lat
        krige_results.attrs["cl_one_lscale"] = [gp1_l1, gp1_l2]
        krige_results.attrs["cl_two_lscale"] = [gp2_l1, gp2_l2]
        krige_results.attrs["mean_pres"] = mean_pres
        krige_results.attrs["mean_temp"] = mean_temp

        krige_results.to_netcdf(fkrige_results)

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

    dx = cfg.dx
    dz = cfg.dz

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


    ### UNCOMMENT BLOCK FOR PROCESSNG QCL FILES ###
    ##################################################
    memo = readmemoascii.memoCDF(cfg.memo_file, cfg.start, cfg.end)
    memo_df = memo.massbalance_data(NoXP, b)

    con_ab = memo_df["con_ab"]
    ## Align altitude of GPS with inlet of instrument
    # Distance between the GPS and the altitude is 72 cm
    agl = memo_df["Altitude"]

    dtm = memo_df.index
    lon = memo_df["Longitude"]
    lat = memo_df["Latitude"]
    dist = memo_df["Distance"]
    ##################################################

    ### UNCOMMENT BLOCK FOR PROCESSNG RUG FILES ###
    ##################################################
    # memo = readmemoascii.memoCDF(cfg.memo_file, cfg.start, cfg.end)
    # memo_qcl_df = memo.massbalance_data(NoXP, b)

    # memo = readmemoascii.memoCDF(cfg.rug_file, cfg.start, cfg.end)
    # memo_df = memo.massbalance_rugdata(NoXP, b)

    # memo_qcl_df = memo_qcl_df[memo_qcl_df.index.isin(memo_df.index)]
    # memo_df = memo_df[memo_df.index.isin(memo_qcl_df.index)]

    # con_ab = memo_df["con_ab"]

    # agl = memo_qcl_df["Altitude"]

    # ## Load spatial parameters
    # dtm = memo_df.index
    # lon = memo_df["Longitude"]
    # lat = memo_df["Latitude"]
    # dist = memo_df["Distance"]
    ##################################################

    im_meas_scatter = os.path.join(target_path,"measurement_scatter.png")

    im_obs_pred = os.path.join(target_path, "ppm_meas.png")
    im_obs_pred_scatter = os.path.join(target_path, "ppm_meas_scatter.png")
    im_obs_var = os.path.join(target_path, "ppm_ss.png")
    im_mic_meas = os.path.join(target_path, "mic_meas.png")

    im_proj_meas = os.path.join(target_path, "projected_conc_map.png")
    im_wind_prof = os.path.join(target_path, "wind_prof.png")
    im_above_bg = os.path.join(target_path, "above_background.png")
    im_area_conc = os.path.join(target_path, "area_concentrion.png")

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

        ## curtain plot of prediction field of methane Mole fraction
        fig2 = curtain_plot.curtainPlots(obs_pred, units = "CH$_4$ [ppm]",
                                         title = "Predicted Measured Mole Fraction")

        ## curtain plot of prediction variance field of methane Mole fraction
        fig3 = curtain_plot.curtainPlots(obs_var, units = "CH$_4$ [ppm$^2$]",
                                         title = "Prediction variance")

        ## curtain plot of prediction field overlaid with measurement points
        fig4 = curtain_plot.curtainScatterPlots(dist, agl, con_ab, obs_pred,
                                                units = "CH$_4$ [ppm]", title =
                                                "Predicted Measured Mole Fraction")

        ## curtain plot displaying krige measured concentration
        fig5 = curtain_plot.curtainPlots(mic_meas, units = "CH$_4$ [$\mu$g/m$^3$]",
                                         title = "Predicted Measured Concentration")

        ## Wind profile plot
        fig6 = curtain_plot.curtainPlots(wind_mean, units="[m/s]",
                                         title="Kriging streamwise wind")

        ## Map plot of point source and projected points
        fig7 = curtain_plot.projected_map(source_lon, source_lat, lon, lat,
                                          con_ab, r_value, perp_distance,
                                          units="CH$_4$ - CH$_{4\mathrm{bg}}}$ [ppm]")


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

