#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib
import matplotlib.pylab as pylab
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FormatStrFormatter
from salem import get_demo_file, GoogleVisibleMap, Map
from scipy.stats import linregress
from windrose import WindroseAxes

from romeomemo.utilities import projection, grids

cmap = cm.Spectral_r
plt.style.use("seaborn-ticks")

myFmt = mdates.DateFormatter("%H:%M")
params = {"legend.fontsize" : "large",
          "axes.labelsize" : "large",
          "axes.titlesize" : "large",
          "xtick.labelsize" : "large",
          "ytick.labelsize" : "large"}
pylab.rcParams.update(params)

class GridPlotting:


    def __init__(self, nx, nz, dx, dz, xmin, zmin):

        self.nx = nx
        self.nz = nz
        self.dx = dx
        self.dz = dz
        self.xmin = xmin
        self.zmin = zmin
        self.grid_x = np.array([self.xmin + i * self.dx + self.dx / 2 for i in
                                range(self.nx)])
        self.grid_z = np.array([self.zmin + i * self.dz + self.dz / 2 for i in
                                range(self.nz)])

    def curtainPlots(self, curtain_array, **labels):

        fig, ax= plt.subplots()
        # cax = fig.add_axes([0.125, 0.25, 0.77, 0.03])

        extent  = [self.grid_x[0], self.grid_x[-1], self.grid_z[0], self.grid_z[-1]]

        # vmin = np.min(np.min(curtain_array), 0.0001)
        # vmax = np.max(curtain_array)
        img = ax.imshow(curtain_array, extent=extent, origin="lower", aspect=2.0, cmap=cmap)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Altitude [m]")
        ax.tick_params(axis="both", which="major")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cb = plt.colorbar(img, orientation="horizontal")
        cb.ax.tick_params()

        if labels:
            for key, lab in labels.items():
                if key == "title":
                    ax.set_title(lab)
                elif key == "units":
                    cb.set_label(lab)
                else:
                    pass

        return fig


    def curtainScatterPlots(self, x_data, y_data, z_data, curtain_array, **labels):

        fig, ax= plt.subplots()
        # cax = fig.add_axes([0.125, 0.25, 0.77, 0.03])

        extent  = [self.grid_x[0], self.grid_x[-1], self.grid_z[0], self.grid_z[-1]]

        cb_values = np.asarray(z_data)
        cmin, cmax = cb_values.min(), cb_values.max()
        norm = plt.Normalize(cmin, cmax)

        ax.imshow(curtain_array, extent=extent, origin="lower", aspect=2.0,
                  cmap=cmap, norm=norm)
        img = ax.scatter(x_data, y_data, c=z_data, cmap=cmap, s=7, norm=norm)
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Altitude [m]")
        ax.set_xlim([self.grid_x[0], self.grid_x[-1]])
        ax.set_ylim([self.grid_z[0], self.grid_z[-1]])
        ax.tick_params(axis="both", which="major")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cb = plt.colorbar(img, orientation="horizontal")
        cb.ax.tick_params()

        if labels:
            for key, lab in labels.items():
                if key == "title":
                    ax.set_title(lab)
                elif key == "units":
                    cb.set_label(lab)
                else:
                    pass

        return fig

    def projected_map(self, source_lon, source_lat, orig_lon, orig_lat, data,
                      *stats, **labels):

        self.source_lon = source_lon
        self.source_lat = source_lat

        lons = np.append(orig_lon, self.source_lon)
        lats = np.append(orig_lat, self.source_lat)

        ## COMPUTE REGRESSION LINE AND PERP DISTANCE FOR VISUALIZATION PURPOSE
        g = GoogleVisibleMap(x=lons, y=lats, scale=2, maptype="satellite")
        ggl_img = g.get_vardata()
        sm = Map(g.grid, factor=1, countries=False)
        orig_x, orig_y = sm.grid.transform(orig_lon, orig_lat)

        proc_gps = grids.ProcessCoordinates(orig_lon, orig_lat)
        proj_lon, proj_lat = proc_gps.regression_vector()
        lon_anchor, lat_anchor = proc_gps.anchorpoint(proj_lon, proj_lat)

        proj_x, proj_y = sm.grid.transform(proj_lon, proj_lat)
        anchor_x, anchor_y = sm.grid.transform(lon_anchor, lat_anchor)

        m, b, _, _, _ = linregress(orig_x, orig_y)
        source_x, source_y = sm.grid.transform(self.source_lon, self.source_lat)

        perp_slope = - 1 / m
        perp_b = source_y - perp_slope * source_x

        perp_x1 = source_x

        perp_y1 = perp_slope * perp_x1 + perp_b

        perp_x2 = (perp_b - b) / (m - perp_slope)
        perp_y2 = (perp_slope * perp_x2 + perp_b)
            
        # textstr = '\n'.join([r"R$^2$ = {:.2f}".format(stats[0]),
        #                     r"distance [m] = {:.2f}".format(stats[1])])

        ## CREATE PLOT
        fig, ax = plt.subplots()
        cax = fig.add_axes()

        # scalebar = AnchoredSizeBar(ax.transData, 30, "30 m", "lower left",
        #                            pad=0.1, color="white", frameon=False,
        #                            size_vertical=1)

        vmin = np.min(data)
        vmax = np.max(data)

        box_props = dict(facecolor="white", alpha=0.5)

        ax.imshow(ggl_img)
        ax.plot(proj_x, proj_y)
        # ax.plot(anchor_x, anchor_y, marker="o", fillstyle="none", markersize=3)
        ax.plot(source_x, source_y, marker="x", color="red")
        ax.plot([perp_x1, perp_x2], [perp_y1, perp_y2], color="orange")
        # ax.text(0.05, 0.85, textstr, transform = ax.transAxes, bbox=box_props,
        #        fontsize="large")
        im = ax.scatter(orig_x, orig_y, s=7, c = data, cmap=cmap)
        ax.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)
        # ax.add_artist(scalebar)
        plt.xticks([]), plt.yticks([])
        cb = plt.colorbar(im, cax=cax)

        if labels:
            for key, lab in labels.items():
                if key == "units":
                    cb.set_label(lab)
                elif key == "title":
                    ax.set_title(lab)
                else:
                    pass

        return fig


    def crosswind_curtain(self, curtain_array, x_data, y_data, x_wind, **labels):

        fig, ax= plt.subplots(figsize=(9.6, 7.2))
        # cax = fig.add_axes([0.125, 0.25, 0.77, 0.03])

        extent  = [self.grid_x[0], self.grid_x[-1], self.grid_z[0], self.grid_z[-1]]

        img = ax.imshow(curtain_array, extent=extent, origin="lower", aspect=2.0, cmap=cmap)
        ax.quiver(x_data[::3], y_data[::3], x_wind[::3], 0, units="xy", width=0.2, color="#444444")
        ax.set_xlabel("Distance [m]")
        ax.set_ylabel("Altitude [m]")
        ax.set_xlim([self.grid_x[0], self.grid_x[-1]])
        ax.set_ylim([self.grid_z[0], self.grid_z[-1]])
        ax.tick_params(axis="both", which="major")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cb = plt.colorbar(img, orientation="horizontal")
        cb.ax.tick_params()

        if labels:
            for key, lab in labels.items():
                if key == "title":
                    ax.set_title(lab)
                elif key == "units":
                    cb.set_label(lab)
                else:
                    pass

        return fig



def windrose_plot(wd, ws):

        textstr = r"$\overline{ws}$ : %0.2f $\pm$ %0.2f" %(np.mean(ws),
                                                           np.std(ws))
        fig = plt.figure()
        ax = fig.add_axes()
        wrax = WindroseAxes.from_ax(ax = ax, fig=fig)
        wrax.bar(wd, ws, normed=True, cmap=cmap)
        wrax.set_legend(frameon=True, edgecolor="#444444", fontsize="large",
                        bbox_to_anchor=(-0.3, 0.0))
        wrax.text(0.35, 0.3, textstr, transform = wrax.transAxes,
                  fontsize="large")

        return fig



def plot_scatter(observation, model, mod_ss):

    fig, ax = plt.subplots(figsize=(6,6))

    N = len(observation)
    actual = np.asarray(observation)
    predictor = np.asarray(model)
    mse = np.sum((predictor-actual)**2)
    rmse = round(np.sqrt(mse/N), 2)

    # negative log predictive density
    sigma_term = np.log(mod_ss) + ((observation-model)**2 / mod_ss)
    nlpd = (1/ (2 * N)) * np.sum(sigma_term) + (0.5 * np.log(2 * np.pi))

    ## plot other observations with other models
    ax.scatter(actual, predictor, s=7)

    ## calculate regression line
    m, b, r_value, p_value, std_err = linregress(actual, predictor)

    r_square = r_value**2

    ## get limits
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]

    ## create 1:1 line
    line_11 = [1*i for i in lims]
    ax.plot(lims, line_11, linewidth=0.5, color="red", alpha=0.5)

    line_y = [m*i + b for i in lims]
    ax.plot(lims, line_y, linewidth=0.5)

    ## Prepare statistics for printing on plot
    rsq_txt = "R$^2$ = {:.2f}".format(r_square)
    N_txt = "N = {:d}".format(N)
    rmse_txt = "RMSE = {:.2f}".format(rmse)
    line_txt = "y = {:.2f}x {:+.2f}".format(m,b)
    nlpd_txt = "NLPD = {:.2f}".format(nlpd)

    stat_texts = "\n".join([rsq_txt, N_txt, rmse_txt, line_txt, nlpd_txt])

    ax.grid(linewidth=0.2, color="grey", linestyle="dotted")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Observation")
    ax.set_ylabel("Model")
    ax.text(0.05, 0.85, stat_texts, transform=ax.transAxes, fontsize=8)

    return fig


def geo_scatter(dtm, x_data, y_data, c_data, *labels):

    fig, ax = plt.subplots()
    cax = fig.add_axes()
    im = ax.scatter(x_data, y_data, s=7, c=c_data, cmap=cmap)
    ax.set_xlim([np.min(x_data), np.max(x_data)])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major")

    cb = plt.colorbar(im, cax=cax)

    if labels:
        label_list = []
        for label in labels:
            label_list.append(label)

        ax.set_xlabel(label_list[0])
        ax.set_ylabel(label_list[1])
        cb.set_label(label_list[2])


    for i in range(1, len(dtm)):
        if (dtm[i].time().minute - dtm[i-1].time().minute) == 1:
            ax.annotate(dtm[i].strftime("%H:%M:%S"), (x_data[i], y_data[i]), fontsize=6)
        else:
            pass

    return fig


def crosswind_scatter(dtm, x_data, y_data, c_data, xwind, *labels):

    fig, ax = plt.subplots(figsize=(9.6, 7.2))
    cax = fig.add_axes()
    ax.quiver(x_data[::3], y_data[::3], xwind[::3], 0, width=0.003, headwidth=5,
              headlength=6, headaxislength=3.5, color="#444444")
    im = ax.scatter(x_data, y_data, s=7, c=c_data, cmap=cmap)
    ax.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    cb = plt.colorbar(im, cax=cax)

    if labels:
        label_list = []
        for label in labels:
            label_list.append(label)

        ax.set_xlabel(label_list[0])
        ax.set_ylabel(label_list[1])
        cb.set_label(label_list[2])


    for i in range(1, len(dtm)):
        if (dtm[i].time().minute - dtm[i-1].time().minute) == 1:
            ax.annotate(dtm[i].strftime("%H:%M:%S"), (x_data[i], y_data[i]), fontsize=6)
        else:
            pass

    return fig


def measurevsbg(dtm, con_ab, altitude):
    """
    """

    df = pd.DataFrame(index = dtm)
    df["con_ab"] = con_ab
    df["Altitude"] = altitude

    fig, ax1 = plt.subplots()
    ax1.plot(df.index, df["con_ab"], marker=".", markersize=1.0,
             linewidth=0.75, label="CH$_4$")
    ax1.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)
    ax1.tick_params(axis="both", which="major")
    ax1.set_xlabel("Datetime [UTC]")
    ax1.set_ylabel("CH$_4$ [ppm]")
    ax1.xaxis.set_major_formatter(myFmt)

    ax2 = ax1.twinx()
    colorax2 = "tab:red"
    ax2.plot(df.index, df["Altitude"], linestyle="dashed", linewidth=0.75,
             color=colorax2)
    ax2.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)
    ax2.tick_params(axis="both", which="major", labelcolor=colorax2)
    ax2.set_ylabel("Altitude [m]", color=colorax2)
    ax2.spines["right"].set_color(colorax2)

    return fig

def densityhist(emission_list):

    fig, ax = plt.subplots()
    ax = sns.distplot(emission_list)
    ax.set_xlabel("CH$_4$ emission estimate [g/s]")
    ax.set_ylabel("Normalized frequency")
    ax.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)

    return fig


def projected_wind(source_lon, source_lat, orig_lon, orig_lat, meteo_u,
                   meteo_v, ch4, **labels):

        source_lon = np.asarray(source_lon)
        source_lat = np.asarray(source_lat)

        orig_lon = np.asarray(orig_lon)
        orig_lat = np.asarray(orig_lat)

        meteo_u = np.asarray(meteo_u)
        meteo_v = np.asarray(meteo_v)

        ## CONVERT LONS AND LATS INTO UTM COORDINATES
        utm_x, utm_y = projection.gps2proj(orig_lon, orig_lat)
        source_x, source_y = projection.gps2proj(source_lon, source_lat)


        ## CREATE PLOT
        fig, ax = plt.subplots()
        cax = fig.add_axes()

        vmin = np.min(ch4)
        vmax = np.max(ch4)

        ax.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)
        ax.plot(source_x, source_y, marker="x")
        ax.quiver(utm_x[::3], utm_y[::3], meteo_u[::3], meteo_v[::3], alpha=0.7, pivot="tip")
        im = ax.scatter(utm_x, utm_y, s=7, c = ch4, cmap=cmap)
        cb = plt.colorbar(im, cax=cax)

        if labels:
            for key, lab in labels.items():
                if key == "units":
                    cb.set_label(lab)
                elif key == "title":
                    ax.set_title(lab)
                else:
                    pass

        return fig

def quick_cmap(im_field):
    fig, ax = plt.subplots()
    cax = fig.add_axes()
    im = ax.imshow(im_field, origin="lower", cmap=cmap)
    plt.colorbar(im, cax=cax, orientation="horizontal")

    return ax


def paper_map(source_lon, source_lat, lon, lat, data):
    
    perp_distance = grids.perpendicular_distance(lon, lat, source_lon,
                                                 source_lat)

    fig, ax1 = plt.subplots()
    cax = fig.add_axes()

    # Extract google image
    g = GoogleVisibleMap(x=np.asarray(lon), y= np.asarray(lat), scale=2,
                         maptype="satellite")
    ggl_img = g.get_vardata()
    sm = Map(g.grid, factor=1, countries=False)
    sm.set_rgb(ggl_img)
    sm.set_scale_bar((0.1,0.05), 30, linewidth=1.0, color="white")
    sm.visualize(ax=ax1)

    # Transform GPS points into plottable data
    x, y = sm.grid.transform(lon, lat)
    source_x, source_y = sm.grid.transform(source_lon, source_lat)
    
    proc_gps = grids.ProcessCoordinates(lon, lat)
    proj_lon, proj_lat = proc_gps.regression_vector()
    lon_anchor, lat_anchor = proc_gps.anchorpoint(proj_lon, proj_lat)

    proj_x, proj_y = sm.grid.transform(proj_lon, proj_lat)
    anchor_x, anchor_y = sm.grid.transform(lon_anchor, lat_anchor)

    m, b, _, _, _ = linregress(x, y)
    source_x, source_y = sm.grid.transform(source_lon, source_lat)

    perp_slope = - 1 / m
    perp_b = source_y - perp_slope * source_x

    perp_x1 = source_x

    perp_y1 = perp_slope * perp_x1 + perp_b

    perp_x2 = (perp_b - b) / (m - perp_slope)
    perp_y2 = (perp_slope * perp_x2 + perp_b)


    box_props = dict(facecolor="white", alpha=0.5)
    textstr = "source-transect \ndistance : {:d} m".format(int(perp_distance))

    im = ax1.scatter(x, y, s = 7, c = data, cmap=cmap)
    ax1.plot(proj_x, proj_y)
    ax1.plot(source_x, source_y, marker="x", color="red", label="Source")
    ax1.plot([perp_x1, perp_x2], [perp_y1, perp_y2])
    ax1.text(0.75, 0.05, textstr, transform = ax1.transAxes, color="white",
             fontsize=9)
    plt.xticks([]), plt.yticks([])
    cb = plt.colorbar(im, cax=cax)
    cb.set_label("CH$_4$ [ppm]")

    return fig

def obs_tseries(ax, dtm, obs, prob, **labels):

    # timeseries plots
    ax.scatter(dtm, obs, s=20, c=prob, cmap=cmap, linewidth=0.5)
    ax.set_xlim([min(dtm), max(dtm)])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major")
    ax.xaxis.set_major_formatter(myFmt)

    if labels:
        for key, lab in labels.items():
            if key == "title":
                ax.set_title(lab, loc="left")
            elif key == "units":
                cb.set_label(lab)
            elif key == "xlabel":
                ax.set_xlabel(lab)
            elif key == "ylabel":
                ax.set_ylabel(lab)

    return


def prob_tseries(ax, dtm, obs, prob, **labels):

    cmap = cm.Blues

    # timeseries plots
    ax.scatter(dtm, obs, s=20, c=prob, cmap=cmap, linewidth=0.5)
    ax.set_xlim([min(dtm), max(dtm)])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)
    ax.tick_params(axis="both", which="major")
    ax.xaxis.set_major_formatter(myFmt)

    if labels:
        for key, lab in labels.items():
            if key == "title":
                ax.set_title(lab, loc="left")
            elif key == "units":
                cb.set_label(lab)
            elif key == "xlabel":
                ax.set_xlabel(lab)
            elif key == "ylabel":
                ax.set_ylabel(lab)

    return

def obs_scatter(ax, dist, agl, prob, **labels):

    im = ax.scatter(dist, agl, s=20, c=prob, linewidth=0.2, cmap=cmap)
    ax.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)

    divider = make_axes_locatable(ax)
    ## Figure 5a
    cax = divider.append_axes("right", size="5%", pad="10%")
    cb = plt.colorbar(im, cax=cax)

    ## Figure 5c
    # cax = divider.append_axes("bottom", size="7%", pad="35%")
    # cb = plt.colorbar(im, cax=cax, orientation="horizontal")
    # cb.ax.tick_params()

    if labels:
        for key, lab in labels.items():
            if key == "title":
                ax.set_title(lab, loc="left")
            elif key == "units":
                cb.set_label(lab)
            elif key == "xlabel":
                ax.set_xlabel(lab)
            elif key == "ylabel":
                ax.set_ylabel(lab)

    return


def prob_scatter(ax, dist, agl, prob, **labels):

    cmap = cm.Blues

    im = ax.scatter(dist, agl, s=20, c=prob, linewidth=0.2, cmap=cmap)
    ax.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)

    cb = plt.colorbar(im, cax=cax)

    if labels:
        for key, lab in labels.items():
            if key == "title":
                ax.set_title(lab, loc="left")
            elif key == "units":
                cb.set_label(lab)
            elif key == "xlabel":
                ax.set_xlabel(lab)
            elif key == "ylabel":
                ax.set_ylabel(lab)

    return


def curtain_plot(ax, grid_x, grid_z, curtain_array, **labels):


    extent  = [grid_x[0], grid_x[-1], grid_z[0], grid_z[-1]]

    img = ax.pcolormesh(grid_x, grid_z, curtain_array, cmap=cmap)
    ax.tick_params(axis="both", which="major")
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    divider = make_axes_locatable(ax)

    ## Figure 6A
    # cax = divider.append_axes("bottom", size="7%", pad="50%")

    # ## Figure 6B
    # cax = divider.append_axes("bottom", size="7%", pad="35%")

    ## Figure 6B
    cax = divider.append_axes("bottom", size="7%", pad="60%")

    cb = plt.colorbar(img, cax=cax, orientation="horizontal")
    cb.ax.tick_params()

    if labels:
        for key, lab in labels.items():
            if key == "title":
                ax.set_title(lab, loc="left")
            elif key == "units":
                cb.set_label(lab)
            elif key == "xlabel":
                ax.set_xlabel(lab)
            elif key == "ylabel":
                ax.set_ylabel(lab)

    return


def extract_con_plots(dtm, obs, obs_pred, var_pred):

    cmap = cm.tab20
    cmaplist = [cmap(i) for i in range(cmap.N)]

    std_pred = np.sqrt(var_pred)

    low_curve = obs_pred - std_pred
    high_curve = obs_pred + std_pred

    fig, ax = plt.subplots()
    ax.plot(dtm, obs, color=cmaplist[0], label="Obs")
    ax.plot(dtm, obs_pred, color= cmaplist[2], label="Model")
    ax.fill_between(dtm, low_curve, high_curve, color = cmaplist[3], alpha=0.6)
    ax.grid(linewidth=0.5, color="grey", linestyle="dotted", alpha=0.5)
    ax.set_xlabel("Datetime [UTC]")
    ax.set_ylabel("CH$_4$ [ppm]")
    ax.xaxis.set_major_formatter(myFmt)
    ax.legend()

    return fig

