import numpy as np
import netCDF4 as nc
import xarray as xr

# set up weights for T_upper calculation
lats = np.arange(-90, 91, 0.1)
weights = np.cos(np.radians(lats))

def calc_T_deep(GMST):
    return (GMST - 15.4) / 0.76

def calc_T_upper(gmst, T_deep):
    weighted_sum = np.sum(weights * weights**2)
    average_weight = np.sum(weights)
    T_upper = (gmst - T_deep) / (weighted_sum / average_weight)
    return T_upper



ocgrid = 'grid.nc'
old_ocean_c1 = 'ocean_ann_c1.nc'

f = nc.Dataset(ocgrid,'r')
geolon = f.variables['geolon_t'][:]
geolat = f.variables['geolat_t'][:]
f.close()

f = nc.Dataset(old_ocean_c1,'r')
depth = f.variables['st_ocean'][:]
f.close()

ny, nx = geolon.shape
nz = depth.shape[0]

depthCutoff = 2500.
shallowIdx = np.where(depth <= depthCutoff)
shallowWeights = (depthCutoff-depth[shallowIdx[0]]) / depthCutoff
nshal = shallowWeights.shape[0]
shallowWeights = np.reshape(shallowWeights,(nshal, 1, 1)) # Here we reshape so we can broadcast to 3D without for loops

salt_val = 34.7

# These averages are taken from 100 year means of GFDL DeepMIP runs 1x and 3x CO2
gmst_c3 = 25.3644
gmst_c1 = 19.2343

deep_c3 = calc_T_deep(gmst_c3)
upper_c3 = calc_T_upper(gmst_c3, deep_c3)
deep_c1 = calc_T_deep(gmst_c1)
upper_c1 = calc_T_upper(gmst_c1, deep_c1)

print('c3: deep:', deep_c3, 'upper:', upper_c3)
print('c1: deep:', deep_c1, 'upper:', upper_c1)

temp_c3 = np.ones((1, nz, ny, nx), 'f8') * deep_c3
temp_c3[0,shallowIdx,:,:] = ( shallowWeights**6 * upper_c3 * np.cos(np.deg2rad(geolat))**2 ) + deep_c3

temp_c1 = np.ones((1, nz, ny, nx), 'f8') * deep_c1
temp_c1[0,shallowIdx,:,:] = ( shallowWeights**6 * upper_c1 * np.cos(np.deg2rad(geolat))**2 ) + deep_c1

for co2 in ['c1','c3']:

    outfile = f'ocean_temp_salt.res.steinig.{co2}.nc'

    fo = nc.Dataset(outfile,'w')
    fo.history = f'oc_restart_steinig.py \n '
    fo.title = outfile

    fo.createDimension('xaxis_1',nx)
    fo.createDimension('yaxis_1',ny)
    fo.createDimension('zaxis_1',nz)
    fo.createDimension('Time',0)

    x_o = fo.createVariable('xaxis_1','f4',('xaxis_1'))
    x_o.cartesian_axis = 'X'
    x_o[:] = np.arange(nx)

    y_o = fo.createVariable('yaxis_1','f4',('yaxis_1'))
    y_o.cartesian_axis = 'Y'
    y_o[:] = np.arange(ny)

    z_o = fo.createVariable('zaxis_1','f4',('zaxis_1'))
    z_o.cartesian_axis = 'Z'
    z_o[:] = np.arange(nz)

    t_o = fo.createVariable('Time','f8',('Time'))
    t_o.cartesian_axis = 'T'
    t_o[:] = 1

    t_o = fo.createVariable('temp','f8',('Time','zaxis_1','yaxis_1','xaxis_1'))
    t_o[:] = eval(f'temp_{co2}[:]')

    s_o = fo.createVariable('salt','f8',('Time','zaxis_1','yaxis_1','xaxis_1'))
    s_o[:] = salt_val

    fo.close()