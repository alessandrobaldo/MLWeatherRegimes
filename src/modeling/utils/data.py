import xarray as xr
import numpy as np
import pandas as pd
import os
from modeling.utils.tools import *
from modeling.utils.sigma_vae import *
from sklearn_xarray import wrap
from sklearn.decomposition import PCA
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)


READ_PATH = 'P:\CH\Weather Data\ERA-5\GEOPOTENTIAL'
#WRITE_PATH = '../dataset/'
G = 9.80665
freq = 'hourly' # 'monthly'
months = 'MayJunJulAugSep'#'DecJanFeb'
obs_years = '1979-2021'#'1979-2020'
LAT, LONG = (20.,80.), (-90., 30.)


@timing
def read_nc(path_to_file):
    '''
    Args:
    - path_to_file: string of the file path

    Returns:
    - an nc.Dataset variable
    '''
    # return nc.Dataset(path_to_file)
    return xr.open_dataset(path_to_file)


@timing
def to_df(dataset, col_name='z'):
    '''
    Args:
    - dataset: an xr.Dataset

    Returns:
    - a Pandas df
    '''
    df = dataset.to_dataframe().reset_index()
    df = df.astype({'latitude': 'float32', 'longitude': 'float32', col_name: 'float32'})
    df.set_index('time', inplace=True)
    if col_name == 'z':
        df['z'] = df['z'] / G
    return df


@timing
def to_nc(dt, variable='z'):
    '''
    Method to push a pandas DataFrame to a .nc file
    Args:
    - dt: an xarray Dataset
    '''
    dt.to_netcdf(READ_PATH + '/ERA-5_{}_Geopotential-500hPa_{}_{}_{}.nc'. \
                 format(freq if freq != 'hourly' else 'daily', months, obs_years, variable),
                 engine="netcdf4")


@timing
def limit_geography(dt, lats, longs):
    '''
    Args:
    - dt: an xarray Dataset containing data
    - lats: extreme values of latitude
    - longs: extreme values of longitude

    Returns:
    - a Pandas df containing the retained rows falling in the geographical area
    '''
    return dt.where(lambda x: ((lats[0] <= x.latitude) & (x.latitude <= lats[1]) &
                               (longs[0] <= x.longitude) & (x.longitude <= longs[1])), drop=True)
    # return dt.loc[dict(latitude=slice(lats[1], lats[0]), longitude=slice(longs[0],longs[1]))]


@timing
def limit_years(df, start_year=1991):
    '''
    Args:
    - df: a pandas DataFrame  containing historical data
    - start_year: the year from which to start to compute the normal

    Returns:
    - a pandas DataFrame containing the historical series of the normal starting from start_year
    '''
    print("Dataframe length before limiting years: %d" % len(df))
    new_df = df[df['year'] >= start_year]
    print("Dataframe length after limiting years: %d" % len(new_df))
    return new_df


@timing
def hourly_to_daily(ds):
    '''
    Args:
    - df: an xarray Dataset containing hourly historical data

    Returns:
    - an xarray Dataset containing daily historical data
    '''
    #ds = read_nc(READ_PATH + '/ERA-5_{}_Geopotential-500hPa_{}_{}.nc'.format(freq, months, obs_years))
    ds.coords['time'] = ds.time.dt.floor('1D')
    ds = ds.groupby('time').mean()
    ds.to_netcdf(READ_PATH + '/ERA-5_{}_Geopotential-500hPa_{}_{}.nc'.format('daily', months, obs_years))
    return ds


@timing
def evaluate_normal(dt, domain='local', mode='flat', freq = 'm', start_year=None):
    '''
    Args:
    - dt: an xarray Dataset containing historical data
    - domain: either 'local'  or 'global'. In the first case a normal for each location is computed, in the latter
      the normal is computed averaging all the positions
    - mode: either 'flat' or 'dynamic'. In the first case the normal is the same for each day, in the latter
      a normal for each day is computed
    - freq:
    - start_year: the starting year to consider for evaluating the normal.
      If not specified, all the dates in the passed DataFrame are used

    Returns:
    - a pandas DataFrame containing the historical series of the normal
    '''

    if start_year is not None:
        dt = dt.loc[dict(time=slice("01-01-{}".format(start_year), None))]

    if domain == 'local' and mode == 'dynamic':
        #month_day = pd.MultiIndex.from_arrays([dt['time.month'].values, dt['time.day'].values])
        #dt.coords['month_day'] = ('time', month_day)
        if freq == 'm':
            return dt.groupby("time.month").mean()
        elif freq == 'w':
            return dt.groupby('time.week').mean()
        else:
            return dt.groupby('time.day').mean()

    elif domain == 'local' and mode == 'flat':
        return dt.mean(dim=["time"])

    elif domain == 'global' and mode == 'flat':
        dt = dt.to_array().values
        return dt.min(), dt.mean(), dt.max()

    else:
        #month_day = pd.MultiIndex.from_arrays([dt['time.month'].values, dt['time.day'].values])
        #dt.coords['month_day'] = ('time', month_day)
        if freq == 'm':
            return dt.groupby("time.month").min(), dt.groupby("time.month").mean(), dt.groupby("time.month").max()
        elif freq == 'w':
            return dt.groupby("time.week").min(), dt.groupby("time.week").mean(), dt.groupby("time.week").max()
        else:
            return dt.groupby("time.day").min(), dt.groupby("time.day").mean(), dt.groupby("time.day").max()


@timing
def evaluate_anomaly(observation, normal, mode='flat', freq = 'm'):
    '''
    Args:
    - observation: an xarray Dataset containing observation data
    - normal: an xarray Datasete containing the historical series of the normal
    - mode:
    - freq:

    Returns:
    - an xarray Dataset containing the anomalous data
    '''

    if mode == 'flat':
        return observation - normal
    else:
        #month_day = pd.MultiIndex.from_arrays([observation['time.month'].values, observation['time.day'].values])
        #observation.coords['month_day'] = ('time', month_day)
        if freq == 'm':
            return xr.apply_ufunc(lambda x, norm: x - norm, observation.groupby('time.month'), normal)
        elif freq == 'w':
            return xr.apply_ufunc(lambda x, norm: x - norm, observation.groupby('time.week'), normal)
        else:
            return xr.apply_ufunc(lambda x, norm: x - norm, observation.groupby('time.day'), normal)


@timing
def weighted_anomaly(dt):
    '''
    Method to compute the weighted anomaly, by eliminating the bias along the latitude
    Args:
    - dt: an xarray Dataset

    Returns:
    - the weighted anomaly
    '''
    wgts = np.sqrt(np.cos(np.deg2rad(dt.latitude.values)).clip(0., 1.))
    wgts = wgts[np.newaxis, ..., np.newaxis]
    wgts = wgts.repeat(dt.to_array().shape[0], axis=0).repeat(dt.to_array().shape[-1], axis=-1)
    return dt * wgts


@timing
def build_data(normal='flat'):
    '''
    Method to build dataframe, including normal and anomaly for each timeframe
    Returns:
    - a pandas DataFrame anomaly
    '''
    if 'ERA-5_{}_Geopotential-500hPa_{}_{}_anomaly.nc'. \
            format(freq if freq != 'hourly' else 'daily', months, obs_years) not in os.listdir(READ_PATH):

        # read nc, eventually pass from hourly to daily, limit geography
        print("Reading nc file and converting to xarray Dataset")
        if freq == 'hourly':
            if 'ERA-5_{}_Geopotential-500hPa_{}_{}.nc'.format('daily', months, obs_years) not in os.listdir(READ_PATH):
                dt = read_nc(READ_PATH + '/ERA-5_{}_Geopotential-500hPa_{}_{}.nc'.format(freq, months, obs_years))
                dt = hourly_to_daily(dt)
            else:
                dt = read_nc(READ_PATH + '/ERA-5_{}_Geopotential-500hPa_{}_{}.nc'.format('daily', months, obs_years))
        else:
            dt = read_nc(READ_PATH + '/ERA-5_{}_Geopotential-500hPa_{}_{}.nc'.format(freq, months, obs_years))

        if 'expver' in dt.indexes:
            dt = xr.concat([dt.sel(time=slice("2021-07-31"), expver=1),
                            dt.sel(time=slice("2021-08-01", "2021-09-30"), expver=5)],
                           dim="time")

        dt = limit_geography(dt, LAT, LONG)
        dt = dt / G
        print("Evaluating the normal")

        if normal != 'flat':
            normal_dt = evaluate_normal(dt, domain='local', mode='dynamic', freq = 'm',start_year=1991)
            print("Evaluating the anomaly")
            anomaly_dt = evaluate_anomaly(dt, normal_dt, mode='dynamic')
        else:
            normal_dt = evaluate_normal(dt, domain='local', mode='flat', freq = 'm', start_year=1991)
            # anomaly_dt = xr.apply_ufunc(lambda x, normal: x - normal, dt.groupby('time'), normal_dt)
            print("Evaluating the anomaly")
            anomaly_dt = evaluate_anomaly(dt, normal_dt, mode='flat', freq = 'm')

        print("Pushing the normal to file")
        to_nc(normal_dt, variable='normal')

        print("Pushing the anomaly to file")
        to_nc(anomaly_dt, variable='anomaly')
        return anomaly_dt

    else:
        print("Reading anomaly from file")
        return read_nc(READ_PATH + '/ERA-5_{}_Geopotential-500hPa_{}_{}_anomaly.nc'. \
                       format(freq if freq != 'hourly' else 'daily', months, obs_years))


class Compresser(object):
    '''
    Class to manage compression of dataframes
    '''

    def __init__(self, latitude, longitude):
        self.enc_dicts = [{k: v for k, v in zip(latitude.unique(), range(len(latitude.unique())))},
                          {k: v for k, v in zip(longitude.unique(), range(len(longitude.unique())))}]

        self.dec_dicts = [{v: k for k, v in dd.items()} for dd in self.enc_dicts]

    @timing
    def encode_coords(self, df):
        '''
        Method to map float32 latitude, longitude coordinates in a more memory-conservative format (i.e. int16)
        Args:
        - df: a pandas DataFrame containing the coordinates to be converted
        '''

        df['latitude'] = df['latitude'].map(self.enc_dicts[0]).astype('int16')
        df['longitude'] = df['longitude'].map(self.enc_dicts[1]).astype('int16')
        return df

    @timing
    def decode_coords(self, df):
        '''
        Method to map int16-encoded latitude, longitude coordinates in the original float32 format
        Args:
        - df: a pandas DataFrame containing the coordinates to be converted
        '''
        df['latitude'] = df['latitude'].map(self.dec_dicts[0]).astype('float32')
        df['longitude'] = df['longitude'].map(self.dec_dicts[1]).astype('float32')
        return df


@timing
def flat_table(dt):
    '''
    Args:
    - dt: an xarray Dataset containing data to be flattened. The index remains the same as the input dataset

    Returns:
    - an xarray flattened
    '''
    return dt.stack(latlon=('latitude', 'longitude')).to_array().squeeze()


@timing
def eofs(dt, **kwargs):
    '''
    Args:
    - dt: an xarray Dataset

    Returns:
    - an xr.DataArray containing the EOFs, a numpy.array containing the PCs
    '''
    A, Lh, E = np.linalg.svd(dt, full_matrices=False)
    L = (Lh * Lh) / (len(dt) - 1)
    neofs = len(L)
    pcs = (A * Lh) / np.sqrt(L)
    eofs = E / np.sqrt(L)[..., np.newaxis]
    eofs = xr.DataArray(eofs, coords=dt.coords)

    return eofs, pcs

@timing
def reduce_dim(dt, reshape='latlon', method='PCA', **kwargs):
    '''
    Args:
    - df: an xarray Dataset
    - method: the method used to perform dimensionality reduction, if not specified PCA is used
    - **kwargs: a dictionary of further parameters, like the percentage of explained variance used to retain the components

    Returns:
    - a numpy.array reduced in the feature space
    '''

    if method == 'PCA':
        if dt.ndim != 2:
            dt = dt.squeeze()
        pca = wrap(PCA(kwargs["exp_variance"] if "exp_variance" in kwargs else .95), reshapes=reshape)
        reduced_dt = pca.fit_transform(dt)

    elif method == "VAE":
        time_idx = dt.coords['time']
        vae = ConvVAE(args = Args())
        vae.load_state_dict(torch.load('../models/'+kwargs['season']+'/sigma_vae_statedict_5', map_location='cpu'))
        dt = np.swapaxes(dt.to_array().values, 0, 1)
        dt = torch.from_numpy(dt).type(torch.FloatTensor)
        stack = []
        for i in range(0, len(dt), 256):
            print("\r", end="")
            print("Processing batch %d" % (i // 256 + 1), end="")

            batch = dt[i:i + 256, ...]
            with torch.no_grad():
                stack.append(vae(batch)[1])
        reduced_dt = xr.DataArray(torch.cat(stack).squeeze().detach().numpy(), coords=[time_idx, range(1, 6)])

    else:
        pass

    print("Number of days: %d, Density of the grid: %d cells" % (reduced_dt.shape[0], reduced_dt.shape[1]))
    return reduced_dt