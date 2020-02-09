import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs
from sklearn.decomposition import PCA
import cartopy.feature as cfeature

data = nc.Dataset("dt_pac_allsat_msla_h_y1993_2018_05deg.nc")
time = data["time"][:]
lon = data["lon"][:]
lat = data["lat"][:]
sla = data["sla"][:] 

#data matrix where each row contains the grid data of a time step.
sla = np.reshape(sla, (time.size, lon.size * lat.size))

sla = np.ma.filled(sla, fill_value=0)

#remove mean
for i in range(312):
    sla[i,:] = sla[i,:]-np.mean(sla[i,:])


#covariance matrix
#covariance is a measure of the joint variability of two random variables.
pca = PCA(n_components=1)
pca.fit(sla)
eigenvectors = pca.components_ 

#How each EOF field evolves in time, can be found by projecting the eigenvector on the data matrix: we get the PCtimeseries
PC_1 = np.matmul(sla,np.transpose(eigenvectors))

#EOFs and PCs, order them according to the variance: already done by PCA
variances = pca.explained_variance_

#apply a normalization to the EOFs(=eigenvectors) and PCs(=principal component(PC) timeseries so that the absolute maximum value 
#of each PC equals 1 (keep in mind that if you scale the PC by a factor x, the EOF needs to be scaled by 1/x).

PC_1_norm = PC_1/np.max(PC_1)
eigenvectors_norm = eigenvectors*np.max(PC_1)


#Make a plot of the first EOFs (i.e., the spatial maps) and its PCs (time series).

plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.title('PC_1_norm')
plt.plot(np.linspace(1993,2018,num=312),PC_1_norm)
plt.subplot(2,1,2)
plt.contourf(np.reshape(eigenvectors_norm,(lat.size,lon.size)))
plt.title('Eigenvectors')
plt.subplots_adjust(left=0.2, wspace=0.6, hspace=0.8,)

# 2 modes
pca = PCA(n_components=2)
pca.fit(sla)
eigenvectors2 = pca.components_ 

#How each EOF field evolves in time, can be found by projecting the eigenvector on the data matrix: we get the PCtimeseries
#matrix multiplication is projection; the distance of the data from the eigenvector 
PC_2 = np.matmul(sla,np.transpose(eigenvectors2))

#EOFs and PCs, order them according to the variance: already done by PCA
variances2 = pca.explained_variance_

PC_2_norm = PC_2 / np.max(np.abs(PC_2), axis=0)  #takes the max from each column, and not from each row 

eigenvectors_2_norm = eigenvectors2.T *np.max(np.abs(PC_2), axis=0)

# 3 modes
pca = PCA(n_components=3)
pca.fit(sla)
eigenvectors3 = pca.components_ 

#How each EOF field evolves in time, can be found by projecting the eigenvector on the data matrix: we get the PCtimeseries
#matrix multiplication is projection; the distance of the data from the eigenvector 
PC_3 = np.matmul(sla,np.transpose(eigenvectors3))

#EOFs and PCs, order them according to the variance: already done by PCA
variances3 = pca.explained_variance_

PC_3_norm = PC_3 / np.max(np.abs(PC_3), axis=0)  #takes the max from each column, and not from each row 

eigenvectors_3_norm = eigenvectors3.T *np.max(np.abs(PC_3), axis=0)

#question3
# 3 modes
pca = PCA(n_components=6)
pca.fit(sla)
eigenvectors3 = pca.components_ 

#How each EOF field evolves in time, can be found by projecting the eigenvector on the data matrix: we get the PCtimeseries
#matrix multiplication is projection; the distance of the data from the eigenvector 
PC_3 = np.matmul(sla,np.transpose(eigenvectors3))

#EOFs and PCs, order them according to the variance: already done by PCA
variances3 = pca.explained_variance_

PC_3_norm = PC_3 / np.max(np.abs(PC_3), axis=0)  #takes the max from each column, and not from each row 

eigenvectors_3_norm = eigenvectors3.T *np.max(np.abs(PC_3), axis=0)

#Fourier Transforrm
#After fourier transform we get both positive and negative freq. we neglect the negative because they dont make sense in timeseries, and also the first freq because it is 0
Fourier3 = np.fft.fft(PC_3_norm.T)[:,1:int(312/2)]      
freq = np.fft.fftfreq(312, d=31)[1:int(312/2)]   

#spectra power
def spectrum_power(fourier):
    spec = (np.absolute(fourier))**2
    return spec

spectrum1 = spectrum_power(Fourier3[0,:])
plt.figure()
plt.plot(freq*365,spectrum1/365**2)
plt.title('Spectrum Mode 1')
plt.xlabel('Frequency (yr$^{-1}$)')
#plt.ylabel('Power ())

spectrum2 = spectrum_power(Fourier3[1,:].T)
plt.figure()
plt.plot(freq*365,spectrum2/365**2)
plt.title('Spectrum Mode 2')
plt.xlabel('Frequency (yr$^{-1}$)')
#plt.ylabel('Power ())

spectrum3 = spectrum_power(Fourier3[2,:].T)
plt.figure()
plt.plot(freq*365,spectrum3/365**2)
plt.title('Spectrum Mode 3')
plt.xlabel('Frequency (yr$^{-1}$)')
#plt.ylabel('Power ())

#1.order spectrum1 from min to max
order = np.argsort(spectrum1)
#From spectrum we get the numbers in this specific 'order'
y = spectrum1[order]
x = freq[order]



#question 4
#Make plot of the cumulative fraction of explained variance of all modes
pca4 = PCA()
pca4.fit(sla)
ratio = pca4.explained_variance_ratio_

for i in range(1,np.size(ratio)):
    ratio[i] = ratio[i-1] + ratio[i]

plt.figure()
plt.plot(ratio)


dec=np.zeros((lat.size,lon.size))
plt.figure()
eigen6= np.reshape(eigenvectors_3_norm.T,(6,lat.size,lon.size))
for i in range(5):
    dec += PC_3_norm[311,i]*eigen6[i,:,:]

plt.figure()  
ax = plt.subplot(projection=ccrs.PlateCarree(180))  
dec = plt.contourf(lon-180,lat,dec,cmap="bwr",levels=np.linspace(-0.4,0.4,40))
cbar = plt.colorbar(dec,orientation='horizontal')
cbar.set_ticks(np.linspace(-0.4,0.4,9))
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)
ax.set_xticks(np.arange(100,281,20)-180)
ax.set_xticklabels(np.arange(100,281,20))
ax.set_yticks(np.arange(-30,31,10))
ax.set_yticks(np.arange(-30,31,10))

diff = data["sla"][311,:,:] - dec
plt.figure()
plt.contourf(diff)

