#!/usr/bin/env python
# coding=utf-8

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm

from romeomemo.utilities import plotting

import xarray as xr


myFmt = mdates.DateFormatter("%H:%M:%S")
params = {"legend.fontsize" : "large",
          "axes.labelsize" : "large",
          "axes.titlesize" : "large",
          "xtick.labelsize" : "large",
          "ytick.labelsize" : "large"}
pylab.rcParams.update(params)

uavnc_302 = "/project/mrp/DUREX/313/CURTAIN_2/MASS_BALANCE/UAV/CK_313_02v2.nc"
uavnc_305 = "/project/mrp/DUREX/313/CURTAIN_5/MASS_BALANCE/UAV/CK_313_05v2.nc"

rtknc_302 = "/project/mrp/DUREX/313/CURTAIN_2/MASS_BALANCE/CK/CK_313_02v2.nc"
rtknc_305 = "/project/mrp/DUREX/313/CURTAIN_5/MASS_BALANCE/CK/CK_313_05v2.nc"


uavds_302 = xr.open_dataset(uavnc_302)
uavds_305 = xr.open_dataset(uavnc_305)

rtkds_302 = xr.open_dataset(rtknc_302)
rtkds_305 = xr.open_dataset(rtknc_305)

uavds_302.close()
uavds_305.close()

rtkds_302.close()
rtkds_305.close()

diff302 = uavds_302.ppm_mean - rtkds_302.ppm_mean
diff305 = uavds_305.ppm_mean - rtkds_305.ppm_mean

coolwarm = cm.seismic
spectral = cm.Spectral_r


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)
ax6 = fig.add_subplot(3, 2, 6)

plotting.curtain_plot(ax1, rtkds_302.x, rtkds_302.z, rtkds_302.ppm_mean,
                      cmap = spectral,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="313_02 : Mapped CH$_4$ (RTK-GPS)", units="ppm")

plotting.curtain_plot(ax2, rtkds_305.x, rtkds_305.z, rtkds_305.ppm_mean,
                      cmap = spectral,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="313_05 : Mapped CH$_4$ (RTK-GPS)", units="ppm")

plotting.curtain_plot(ax3, uavds_302.x, uavds_302.z, uavds_302.ppm_mean,
                      cmap = spectral,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="313_02 : Mapped CH$_4$ (UAV-GPS)", units="ppm")

plotting.curtain_plot(ax4, uavds_305.x, uavds_305.z, uavds_305.ppm_mean,
                      cmap = spectral,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title= "313_02 : Mapped CH$_4$ (UAV-GPS)", units="ppm")

plotting.curtain_plot(ax5, rtkds_302.x, rtkds_302.z, diff302,
                      cmap = coolwarm,
                      xlabel="Distance [m]", ylabel="Altitude [m]",
                      title="313_02 : CH$_4$ [UAV-RTK] ", units="ppm")

plotting.curtain_plot(ax6, rtkds_305.x, rtkds_305.z, diff305,
                      cmap = coolwarm,
                      xlabel= "Distance [m]", ylabel="Altitude [m]",
                      title= "313_05 : CH$_4$ [UAV-RTK]", units="ppm")
plt.tight_layout()

supp_fig = "/project/mrp/DUREX/PAPER_ONE/FIG/uav_rtk_supplement.pdf"

fig.savefig(supp_fig, dpi=400)
