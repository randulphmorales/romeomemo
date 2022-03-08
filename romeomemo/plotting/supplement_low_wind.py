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

gauss_401 = "/project/mrp/GAUSSIAN/DU_20200314/CURTAIN_1/gaussian_plume.nc"
gauss_402 = "/project/mrp/GAUSSIAN/DU_20200314/CURTAIN_2/gaussian_plume.nc"
gauss_403 = "/project/mrp/GAUSSIAN/DU_20200314/CURTAIN_3/gaussian_plume.nc"

cknc_401 = "/project/mrp/DUREX/314/CURTAIN_1/MASS_BALANCE/CK/CK_314_01v2.nc"
cknc_402 = "/project/mrp/DUREX/314/CURTAIN_2/MASS_BALANCE/CK/CK_314_02v2.nc"
cknc_403 = "/project/mrp/DUREX/314/CURTAIN_3/MASS_BALANCE/CK/CK_314_03v2.nc"

gds_401 = xr.open_dataset(gauss_401)
gds_402 = xr.open_dataset(gauss_402)
gds_403 = xr.open_dataset(gauss_403)

ckds_401 = xr.open_dataset(cknc_401)
ckds_402 = xr.open_dataset(cknc_402)
ckds_403 = xr.open_dataset(cknc_403)

gds_401.close()
gds_402.close()
gds_403.close()

ckds_401.close()
ckds_402.close()
ckds_403.close()

def sigma_str(sigma_y, sigma_z):

    y_str = "sigma_y : {:.2f}".format(sigma_y)
    z_str = "sigma_z : {:.2f}".format(sigma_z)

    sig_str = "\n".join([y_str, z_str])

    return sig_str

fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)
ax6 = fig.add_subplot(3, 2, 6)

plotting.curtain_plot(ax1, gds_401.y, gds_401.z, gds_401.yz_ppm, xlabel="Distance [m]",
                      ylabel="Altitude [m]", title="Gaussian Plume : 314_01")
gds_str403 = sigma_str(gds_403.attrs["sy [m]"], gds_403.attrs["sz [m]"])
ax1.text(0.77, 0.70, gds_str403, transform=ax1.transAxes, color="white")

plotting.curtain_plot(ax2, ckds_401.x, ckds_401.z, ckds_401.ppm_mean,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="Cluster Kriging : 314_01")

plotting.curtain_plot(ax3, gds_402.y, gds_402.z, gds_402.yz_ppm,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="314_01")
gds_str402 = sigma_str(gds_402.attrs["sy [m]"], gds_402.attrs["sz [m]"])
ax3.text(0.77, 0.70, gds_str402, transform=ax3.transAxes, color="white")

plotting.curtain_plot(ax4, ckds_402.x, ckds_402.z, ckds_402.ppm_mean,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title= "313_04")

plotting.curtain_plot(ax5, gds_403.y, gds_403.z, gds_403.yz_ppm,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="313_05", units="CH$_4$ [ppm]")
gds_str403 = sigma_str(gds_403.attrs["sy [m]"], gds_403.attrs["sz [m]"])
ax5.text(0.77, 0.70, gds_str403, transform=ax5.transAxes, color="white")

plotting.curtain_plot(ax6, ckds_403.x, ckds_403.z, ckds_403.ppm_mean,
                      xlabel= "Distance [m]", ylabel="Altitude [m]",
                      title= "313_05", units="CH$_4$ [ppm]")
plt.tight_layout()

supp_fig = "/project/mrp/DUREX/PAPER_ONE/FIG/low_wind_supplement.png"

fig.savefig(supp_fig, dpi=400)
