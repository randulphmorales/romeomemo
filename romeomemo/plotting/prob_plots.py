#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from romeomemo.utilities import plotting


params = {"legend.fontsize" : "large",
          "axes.labelsize" : "xx-large",
          "axes.titlesize" : "xx-large",
          "xtick.labelsize" : "x-large",
          "ytick.labelsize" : "x-large"}
pylab.rcParams.update(params)

# dtm = krige_ppm.index
# dist = krige_ppm.dist
# agl = krige_ppm.agl
# obs = krige_ppm.obs_data
# prob = krige_ppm.prob
# comp_prob = krige_ppm.comp_prob

def prob_figure(dtm, dist, agl, obs, prob, comp_prob):

    ch4_text = "CH$_4$ - CH$_{4{\mathrm{bg}}}$ [ppm]"

    tser_labels = {"xlabel" : "Datetime [UTC]", "ylabel" : ch4_text}
    scat_labels = {"xlabel" : "Distance [m]", "ylabel":"Altitude [m]",
                   "units" : "Probability"}

    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(3, 3, (1,2))
    ax2 = fig.add_subplot(3, 3, 3)
    ax3 = fig.add_subplot(3, 3, (4,5))
    ax4 = fig.add_subplot(3, 3, 6)
    ax5 = fig.add_subplot(3, 3, (7,8))
    ax6 = fig.add_subplot(3, 3, 9)

    plotting.obs_tseries(ax1, dtm, obs, obs, ylabel=ch4_text, 
                         title="Flight Measurement")
    plotting.obs_scatter(ax2, dist, agl, obs, ylabel="Altitude [m]", units=ch4_text)
    plotting.prob_tseries(ax3, dtm, obs, prob, ylabel=ch4_text,
                          title="Background Cluster")
    plotting.prob_scatter(ax4, dist, agl, prob, ylabel="Altitude [m]",
                          units="Probability")
    plotting.prob_tseries(ax5, dtm, obs, comp_prob, title="Elevated Cluster",
                          **tser_labels)
    plotting.prob_scatter(ax6, dist, agl, comp_prob, **scat_labels)
    plt.tight_layout()

    return fig
