#!/usr/bin/env python
# coding=utf-8

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from romeomemo.utilities import plotting


myFmt = mdates.DateFormatter("%H:%M:%S")
params = {"legend.fontsize" : "large",
          "axes.labelsize" : "xx-large",
          "axes.titlesize" : "xx-large",
          "xtick.labelsize" : "x-large",
          "ytick.labelsize" : "x-large"}
pylab.rcParams.update(params)

def cluster_curtain(grid_x, grid_y, elev, bg, elev_prob, bg_prob):

    ch4_text = "CH$_4$ - CH$_{4{\mathrm{bg}}}$ [ppm]"

    cl_labels = {"xlabel" : "Distance [m]", "ylabel" : "Altitude [m]", "units":ch4_text}
    pr_labels = {"xlabel" : "Distance [m]", "ylabel" : "Altitude [m]",
                 "units":"Probability"}
 
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    plotting.curtain_plot(ax1, grid_x, grid_y, elev, xlabel="Distance [m]",
                          ylabel="Altitude [m]",
                          title="Background methane mole fraction")

    plotting.curtain_plot(ax2, grid_x, grid_y, elev_prob,
                          xlabel="Distance [m]", ylabel="Altitude [m]",
                          title="Background membership probability")

    plotting.curtain_plot(ax3, grid_x, grid_y, bg,
                          title="Elevated methane mole fraction", **cl_labels)

    plotting.curtain_plot(ax4, grid_x, grid_y, bg_prob,
                          title="Elevated membership probability", **pr_labels)

    plt.tight_layout()

    return

def prediction_curtain(grid_x, grid_y, dist, agl, obs_data, pred, ss):

    ch4_text = "CH$_4$ - CH$_{4{\mathrm{bg}}}$ [ppm]"


    fig = plt.figure(figsize=(16,4))
    ax1 = fig.add_subplot(1, 5, 1)
    ax2 = fig.add_subplot(1, 5, (2,3))
    ax3 = fig.add_subplot(1, 5, (4,5))

    plotting.obs_scatter(ax1, dist, agl, obs_data, ylabel="Altitude [m]",
                         xlabel="Distance [m]", units=ch4_text)
    plotting.curtain_plot(ax2, grid_x, grid_y, pred, xlabel="Distance [m]",
                          units="CH$_4$ [ppm]",
                          title="Predicted measure mole fraction")
    plotting.curtain_plot(ax3, grid_x, grid_y, ss, xlabel="Distance [m]",
                          units="CH$_4$ [ppm]", title="Prediction uncertainty")

    plt.tight_layout()

    return


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)
plotting.curtain_plot(ax1, grid_x, grid_z, ck_krige.ppm_mean, xlabel="Distance [m]",
                      ylabel="Altitude [m]",
                      title="Cluster kriging")

plotting.curtain_plot(ax2, grid_x, grid_z, ck_krige.ppm_cl_one,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="Background cluster [l$_h$=5.38, l$_v$=1.63]")

plotting.curtain_plot(ax4, grid_x, grid_z, ck_krige.ppm_cl_two,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="Elevated cluster [l$_h$=2.96, l$_v$=2.16]")

plotting.curtain_plot(ax3, grid_x, grid_z, ok_krige.ppm_mean,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="Ordinary kriging [l$_h$=80.76, l$_v$=80.76]", units="CH$_4$ [ppm]")

plotting.curtain_plot(ax5, grid_x, grid_z, ok_forced_krige.ppm_mean,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="Forced ordinary kriging [l$_h$=2.96, l$_v$=2.16]", units="CH$_4$ [ppm]")
plt.tight_layout()
