#author: Maria Eirini Tzampazidou

import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
from sklearn.decomposition import PCA
import cartopy.feature as cfeature
import cartopy.crs as ccrs

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
plt.figure(figsize=(5,5))
ax = plt.subplot(projection=ccrs.PlateCarree(180))
ax1 = plt.contourf(data["lon"][:]-180,data["lat"][:],sla[311,:,:], cmap="bwr",levels=np.linspace(-0.4,0.4,40))
ax.add_feature(cfeature.LAND, facecolor="black")
ax.add_feature(cfeature.COASTLINE)
ax.set_xticks(np.arange(100,281,20)-180)
ax.set_xticklabels(np.arange(100,281,20))
ax.set_yticks(np.arange(-30,31,10))
cbar = plt.colorbar(ax1,orientation='horizontal')
cbar.set_ticks(np.linspace(-0.4,0.4,9))
plt.xlabel('Longitude in $\degree$')
plt.ylabel('Latitude in $\degree$')
plt.title('December 2018')
plt.grid()
plt.savefig('December2018(initial)')

#data matrix where each row contains the grid data of a time step.
sla = np.reshape(sla, (time.size, lon.size * lat.size))

sla = np.ma.filled(sla, fill_value=0)

#remove mean
for i in range(312):
    sla[i,:] = sla[i,:]-np.mean(sla[i,:])


#question2,3

# 6 modes
pca = PCA(n_components=6)
pca.fit(sla)
eigenvectors6 = pca.components_ 

#How each EOF field evolves in time, can be found by projecting the eigenvector on the data matrix: we get the PCtimeseries
#matrix multiplication is projection; the distance of the data from the eigenvector 
PC_6 = np.matmul(sla,np.transpose(eigenvectors6))

#EOFs and PCs, order them according to the variance: already done by PCA
variances3 = pca.explained_variance_

PC_6_norm = PC_6 / np.max(np.abs(PC_6), axis=0)  #takes the max from each column, and not from each row 

eigenvectors_6_norm = eigenvectors6.T *np.max(np.abs(PC_6), axis=0)

for i in range(3):
    #Plot the EOFs for mode 2 and 3
    plt.figure(figsize=(5,5))
    ax = plt.subplot(projection=ccrs.PlateCarree(180))
    ax1 =plt.contourf(lon-180,lat,np.reshape(eigenvectors_6_norm[:,i],(lat.size,lon.size)),cmap="bwr",levels=np.linspace(-0.4,0.4,40))
    cbar = plt.colorbar(ax1,orientation='horizontal')
    cbar.set_ticks(np.linspace(-0.4,0.4,9))
    ax.set_xticks(np.arange(100,281,20)-180)
    ax.set_xticklabels(np.arange(100,281,20))
    ax.set_yticks(np.arange(-30,31,10))
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, facecolor="black")
    plt.xlabel('Longitude in $\degree$')
    plt.ylabel('Latitude in $\degree$')
    plt.grid()
    plt.title('EOFs Mode ' +str(i+1))
    plt.savefig('EOFmode'+str(i+1)+'.png')
    #Plot the PCs for mode 2 and 3
    plt.figure(figsize=(8, 2))
    plt.xlabel('Time (years)')
    plt.ylabel('PCs')
    plt.plot(np.linspace(1993,2018,num=312),PC_6_norm[:,i])
    plt.title('PCs Mode ' +str(i+1))
    plt.xticks(np.arange(1993,2018,3))
    plt.grid()
    plt.savefig('PCsmode'+str(i+1)+'.png')

#Fourier Transforrm
#After fourier transform we get both positive and negative freq. we neglect the negative because they dont make sense in timeseries, and also the first freq because it is 0
Fourier = np.fft.fft(PC_6_norm.T)[:,1:int(312/2)]      
freq = np.fft.fftfreq(312, d=31)[1:int(312/2)]   

#spectra power
def spectrum_power(fourier):
    spec = (np.absolute(fourier))**2
    return spec

spectrum1 = spectrum_power(Fourier[0,:])
plt.figure(figsize=(8, 2))
plt.plot(freq*365,spectrum1/365**2)
plt.title('Spectrum Mode 1')
plt.xlabel('Frequency (yr$^{-1}$)')
plt.ylabel('Power (years$^2$)')
plt.savefig('spectrum1',bbox_inches="tight")

spectrum2 = spectrum_power(Fourier[1,:].T)
plt.figure(figsize=(8, 2))
plt.plot(freq*365,spectrum2/365**2)
plt.title('Spectrum Mode 2')
plt.xlabel('Frequency (yr$^{-1}$)')
plt.ylabel('Power (years$^2$)')
plt.savefig('Spectrum2',bbox_inches="tight")

spectrum3 = spectrum_power(Fourier[2,:].T)
plt.figure(figsize=(8, 2))
plt.plot(freq*365,spectrum3/365**2)
plt.title('Spectrum Mode 3')
plt.xlabel('Frequency (yr$^{-1}$)')
plt.ylabel('Power (years$^2$)')
plt.savefig('spectrum3',bbox_inches="tight")

#1order spectrum1 from min to max
order = np.argsort(spectrum1)
#From spectrum we get the numbers in this specific 'order'
y = spectrum1[order]
x = freq[order]
#we obtain the accurate position of the largest frequency


#question 4
#Make plot of the cumulative fraction of explained variance of all modes
pca4 = PCA()
pca4.fit(sla)
ratio = pca4.explained_variance_ratio_

for i in range(1,np.size(ratio)):
    ratio[i] = ratio[i-1] + ratio[i]

plt.figure(figsize=(6, 2))
plt.plot(ratio)
plt.title('Cummulative Fraction')
plt.xlabel('Modes')
plt.xscale('log')
plt.xticks([1, 6, 10, 100, 300],["1", "6", "10", "100", "300"])
plt.grid()

plt.savefig('ACCUMULATIVEfrac')

#question5
#Rebuild the data set using modes 1-6.
dec=np.zeros((lat.size,lon.size))
plt.figure()
eigen6= np.reshape(eigenvectors_6_norm.T,(6,lat.size,lon.size))
for i in range(5):
    dec += PC_6_norm[311,i]*eigen6[i,:,:]
       
    
#plot of the reconstructed data for Dec 2018    
plt.figure(figsize=(5,5)) 
ax = plt.subplot(projection=ccrs.PlateCarree(180))  
decc = plt.contourf(lon-180,lat,dec,cmap="bwr",levels=np.linspace(-0.4,0.4,40))
cbar = plt.colorbar(decc,orientation='horizontal')
cbar.set_ticks(np.linspace(-0.4,0.4,9))
ax.set_xticks(np.arange(100,281,20)-180)
ax.set_xticklabels(np.arange(100,281,20))
ax.set_yticks(np.arange(-30,31,10))
ax.add_feature(cfeature.LAND, facecolor="black")
ax.add_feature(cfeature.COASTLINE)
plt.xlabel('Longitude in $\degree$')
plt.ylabel('Latitude in $\degree$')
plt.title('Reconstructed December 2018')
plt.grid()
plt.savefig('reconstructedDec')

#difference between the original and reconstructed data
diff = data["sla"][311,:,:] - dec
plt.figure(figsize=(5,5))
ax = plt.subplot(projection=ccrs.PlateCarree(180))  
diffe = plt.contourf(lon-180,lat,diff,cmap="bwr",levels=np.linspace(-0.4,0.4,40))
cbar = plt.colorbar(diffe,orientation='horizontal')
cbar.set_ticks(np.linspace(-0.4,0.4,9))
ax.add_feature(cfeature.LAND, facecolor="black")
ax.add_feature(cfeature.COASTLINE)
ax.set_xticks(np.arange(100,281,20)-180)
ax.set_xticklabels(np.arange(100,281,20))
ax.set_yticks(np.arange(-30,31,10))
plt.xlabel('Longitude in $\degree$')
plt.ylabel('Latitude in $\degree$')
plt.title('Difference 2018')
plt.grid()
plt.savefig('difference')



   
