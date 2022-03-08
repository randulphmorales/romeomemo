#!/usr/bin/env python
# coding=utf-8

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

from romeomemo.utilities import plotting

import xarray as xr


myFmt = mdates.DateFormatter("%H:%M:%S")
params = {"legend.fontsize" : "large",
          "axes.labelsize" : "large",
          "axes.titlesize" : "large",
          "xtick.labelsize" : "large",
          "ytick.labelsize" : "large"}
pylab.rcParams.update(params)

gauss_303 = "/project/mrp/GAUSSIAN/DU_20200313/CURTAIN_3/gaussian_plume.nc"
gauss_304 = "/project/mrp/GAUSSIAN/DU_20200313/CURTAIN_4/gaussian_plume.nc"
gauss_305 = "/project/mrp/GAUSSIAN/DU_20200313/CURTAIN_5/gaussian_plume.nc"

cknc_303 = "/project/mrp/DUREX/313/CURTAIN_3/MASS_BALANCE/CK/CK_313_03v2.nc"
cknc_304 = "/project/mrp/DUREX/313/CURTAIN_4/MASS_BALANCE/CK/CK_313_04v2.nc"
cknc_305 = "/project/mrp/DUREX/313/CURTAIN_5/MASS_BALANCE/CK/CK_313_05v2.nc"

gds_303 = xr.open_dataset(gauss_303)
gds_304 = xr.open_dataset(gauss_304)
gds_305 = xr.open_dataset(gauss_305)

ckds_303 = xr.open_dataset(cknc_303)
ckds_304 = xr.open_dataset(cknc_304)
ckds_305 = xr.open_dataset(cknc_305)

gds_303.close()
gds_304.close()
gds_305.close()

ckds_303.close()
ckds_304.close()
ckds_305.close()

def sigma_str(sigma_y, sigma_z):

    y_str = "sigma_y : {:.2f}".format(sigma_y)
    z_str = "sigma_z : {:.2f}".format(sigma_z)

    sig_str = "\n".join([y_str, z_str])

    return sig_str

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)
ax6 = fig.add_subplot(3, 2, 6)

plotting.curtain_plot(ax1, gds_303.y, gds_303.z, gds_303.yz_ppm, xlabel="Distance [m]",
                      ylabel="Altitude [m]", title="Gaussian Plume : 313_03")
gds_str303 = sigma_str(gds_303.attrs["sy [m]"], gds_303.attrs["sz [m]"])
ax1.text(0.77, 0.70, gds_str303, transform=ax1.transAxes, color="white")

plotting.curtain_plot(ax2, ckds_303.x, ckds_303.z, ckds_303.ppm_mean,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="Cluster Kriging : 313_03")

plotting.curtain_plot(ax3, gds_304.y, gds_304.z, gds_304.yz_ppm,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="313_04")
gds_str304 = sigma_str(gds_304.attrs["sy [m]"], gds_304.attrs["sz [m]"])
ax3.text(0.77, 0.70, gds_str304, transform=ax3.transAxes, color="white")

plotting.curtain_plot(ax4, ckds_304.x, ckds_304.z, ckds_304.ppm_mean,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title= "313_04")

plotting.curtain_plot(ax5, gds_305.y, gds_305.z, gds_305.yz_ppm,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="313_05", units="CH$_4$ [ppm]")
gds_str305 = sigma_str(gds_305.attrs["sy [m]"], gds_305.attrs["sz [m]"])
ax5.text(0.77, 0.70, gds_str305, transform=ax5.transAxes, color="white")

plotting.curtain_plot(ax6, ckds_305.x, ckds_305.z, ckds_305.ppm_mean,
                      xlabel= "Distance [m]", ylabel="Altitude [m]",
                      title= "313_05", units="CH$_4$ [ppm]")
plt.tight_layout()

supp_fig = "/project/mrp/DUREX/PAPER_ONE/FIG/low_wind_supplement.png"

fig.savefig(supp_fig, dpi=400)
