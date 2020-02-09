# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:05:37 2019

@author: Maria
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import netCDF4 as nc
import cartopy.crs as ccrs
from cartopy._crs import CRS, Geodetic, Globe, PROJ4_VERSION
from cartopy._crs import Geocentric  # noqa: F401 (flake8 = unused import)
import cartopy.trace
from sklearn.decomposition import PCA
import cartopy.feature as cfeature
import matplotlib as mpl
from matplotlib.colors import LogNorm



data = nc.Dataset("dt_pac_allsat_msla_h_y1993_2018_05deg.nc")
#for var in data.variables:
#    print(var)
time = data["time"][:]
lon = data["lon"][:]
lat = data["lat"][:]
sla = data["sla"][:]        #sea level anomaly 

#question 1

#Inspect the data by plotting a number of months
#plots1 = np.round(np.arange(1,311,60))
#plots1 = plots1.astype(int)
#plots2 = np.round(np.arange(1,311,65))
#plots2 = plots2.astype(int)
#names1 = ['1993','1999','2005','2011','2017']
#names2 = ['1993','1999','2005','2011','2017']
#fig1 = plt.figure(figsize=(20,10))
#fig1.suptitle('January', fontsize=24)
#for k in range(1,6):
#    fig1 = plt.subplot(2, 5, k)
#    fig1.contourf(data["lon"][:],data["lat"][:],sla[plots1[k-1],:,:])
#    fig1.set_title(names1[k-1])
#fig2 = plt.figure(figsize=(20,10))
#fig2.suptitle('July', fontsize=24)  
#for k in range(1,6):
#    fig2 = plt.subplot(2, 5, k)
#    fig2.contourf(data["lon"][:],data["lat"][:],sla[plots2[k-1],:,:])
#    fig2.set_title(names2[k-1])

#plot for Dec 2018
fig = plt.figure()
ax = plt.subplot(projection=ccrs.PlateCarree(180))
ax1 = plt.contourf(data["lon"][:]-180,data["lat"][:],sla[311,:,:], cmap="bwr",levels=np.linspace(-0.4,0.4,40))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
cbar = plt.colorbar(ax1,orientation='horizontal')
cbar.set_ticks(np.linspace(-0.4,0.4,9))
plt.title('December 2018')



#cbar.set_ticks([-0.4,-0.1,0,0.1,0.4])
#cbar.set_ticklabels([-0.4,-0.1,0,0.1,0.4])

