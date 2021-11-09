import xarray as xr
import os
from modeling.utils.sigma_vae import *
from sklearn_xarray import wrap
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch
device = "cuda:0" if torch.cuda.is_available() else "cpu"

import numpy as np
from scipy.spatial import distance
from modeling.utils.tools import *
from functools import wraps, partial
from multiprocessing.dummy import Pool
import pickle
from copy import deepcopy

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state

from modeling.utils.data import eofs
from modeling.utils.plotting import plot_EOFS
from modeling.utils.models import performance_matrix
import pandas as pd
from modeling.utils.plotting import plot_regimes

def read_nc(path_to_file):
    '''
    Args:
    - path_to_file: string of the file path

    Returns:
    - an nc.Dataset variable
    '''
    # return nc.Dataset(path_to_file)
    return xr.open_dataset(path_to_file)

def to_nc(dt, variable='z'):
    '''
    Method to push a pandas DataFrame to a .nc file
    Args:
    - dt: an xarray Dataset
    '''
    dt.to_netcdf(READ_PATH + '/ERA-5_{}_SLP_{}_{}_{}.nc'. \
                 format(freq if freq != 'hourly' else 'daily', months, obs_years, variable),
                 engine="netcdf4")


def build_data(normal='flat'):
    '''
    Method to build dataframe, including normal and anomaly for each timeframe
    Returns:
    - a pandas DataFrame anomaly
    '''
    if 'ERA-5_{}_SLP_{}_{}_anomaly.nc'. \
            format(freq if freq != 'hourly' else 'daily', months, obs_years) not in os.listdir(READ_PATH):

        # read nc, eventually pass from hourly to daily, limit geography
        print("Reading nc file and converting to xarray Dataset")
        if freq == 'hourly':
            if 'ERA-5_{}_SLP_{}_{}.nc'.format('daily', months, obs_years) not in os.listdir(READ_PATH):
                dt = read_nc(READ_PATH + '/ERA-5_{}_SLP_{}_{}.nc'.format(freq, months, obs_years))
                dt = hourly_to_daily(dt)
            else:
                dt = read_nc(READ_PATH + '/ERA-5_{}_SLP_{}_{}.nc'.format('daily', months, obs_years))
        else:
            dt = read_nc(READ_PATH + '/ERA-5_{}_SLP_{}_{}.nc'.format(freq, months, obs_years))

        if 'expver' in dt.indexes:
            dt = xr.concat([dt.sel(time=slice("2021-07-31"), expver=1),
                            dt.sel(time=slice("2021-08-01", "2021-09-30"), expver=5)],
                           dim="time")

        dt = limit_geography(dt, LAT, LONG)
        print("Evaluating the normal")

        if normal != 'flat':
            normal_dt = evaluate_normal(dt, domain='local', mode='dynamic', freq = 'm',start_date="01-01-1991")
            print("Evaluating the anomaly")
            anomaly_dt = evaluate_anomaly(dt, normal_dt, mode='dynamic')
        else:
            normal_dt = evaluate_normal(dt, domain='local', mode='flat', freq = 'm', start_date="01-01-1991")
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
        return read_nc(READ_PATH + '/ERA-5_{}_SLP_{}_{}_anomaly.nc'. \
                       format(freq if freq != 'hourly' else 'daily', months, obs_years))

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
    ds.to_netcdf(READ_PATH + '/ERA-5_{}_SLP_{}_{}.nc'.format('daily', months, obs_years))
    return ds

def evaluate_normal(dt, domain='local', mode='flat', freq = 'm', start_date=None, end_date = None):
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

    if start_date is not None:
        dt = dt.loc[dict(time=slice(start_date, end_date))]

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

def flat_table(dt):
    '''
    Args:
    - dt: an xarray Dataset containing data to be flattened. The index remains the same as the input dataset

    Returns:
    - an xarray flattened
    '''
    return dt.stack(latlon=('latitude', 'longitude')).to_array().squeeze()

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


def cross_val(X, method="kmeans", scoring="score", season = "WINTER", verbose=True):
    '''
    Method to perform cross-validation of clustering methods
    Args
    - X: pandas DataFrame containing data
    - method: clustering method to be validated
    - scoring
    - season:
    - verbose
    '''

    if method == 'kmeans':
        estimator = KMeans(random_state=42)
        params = {"n_clusters": [4, 5, 6, 7], "init": ["k-means++", "random"],
                  "n_init": [10, 50], "max_iter": [100, 300, 1000], "tol": [1e-3, 1e-5, 1e-7]}

    elif method == 'bayesian_gmm':
        estimator = BayesianGaussianMixture(random_state=42)
        params = {"n_components": [4, 5, 6, 7], "covariance_type": ["full"],
                  "n_init": [5, 10], "max_iter": [100, 3000], "init_params": ["kmeans", "random"],
                  "tol": [1e-3, 1e-7],
                  "weight_concentration_prior_type": ["dirichlet_process", "dirichlet_distribution"]}

    elif method == 'gmm':
        estimator = GaussianMixture(random_state=42, warm_start=True)
        params = {"n_components": [4, 5, 6, 7], "covariance_type": ["full"],
                  "n_init": [5, 10], "max_iter": [100, 300, 1000], "init_params": ["kmeans", "random"],
                  "tol": [1e-3, 1e-5, 1e-7]}

    elif method == 'spectral':
        estimator = SpectralClustering(random_state=42)
        params = {"n_clusters": [4, 5, 6, 7], "affinity": ["nearest_neighbors", "rbf", "precomputed"],
                  "n_neighbors": [10, 50], "n_init": [10, 50]}

    else:
        estimator = None

    def make_generator(parameters):
        '''
        Method creating a generator on the fly returning all the combinations given the passed parameters
        Args:
        - parameters: a dictionary containing the parameters to be passed
        '''
        if not parameters:
            yield dict()
        else:
            key_to_iterate = list(parameters.keys())[0]
            next_round_parameters = {p: parameters[p]
                                     for p in parameters if p != key_to_iterate}
            for val in parameters[key_to_iterate]:
                for pars in make_generator(next_round_parameters):
                    temp_res = pars
                    temp_res[key_to_iterate] = val
                    yield temp_res

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    _best_estimator, _best_score, _best_params = None, -np.inf, None

    def get_score(indexes, estimator, X, scoring):
        train_X, val_X = X[indexes[0]], X[indexes[1]]
        try:
            estimator.fit(train_X)
            labels = _est.predict(val_X)

            if scoring == "score":
                scoring_fn = getattr(estimator, "score", None)
                if callable(scoring_fn):
                    score = estimator.score(val_X)
            elif scoring == "silhouette":
                score = silhouette_score(val_X, labels)

            elif scoring == "ch":
                score = calinski_harabasz_score(val_X, labels)

            elif scoring == "bic":
                scoring_fn = getattr(estimator, "bic", None)
                if callable(scoring_fn):
                    score = - estimator.bic(val_X)
                else:
                    score = bic_score(val_X, labels,
                                      estimator.cluster_centers_ if hasattr(estimator,
                                                                            "cluster_centers_") else estimator.means_)
        except:
            score = np.nan
        return score

    for i, param_grid in enumerate(make_generator(params)):
        _est = estimator.set_params(**param_grid)
        pool = Pool()
        scores = pool.map(partial(get_score, estimator=_est, X=X, scoring=scoring), kf.split(X))

        if verbose:
            print("\tCombination %d score: %.3f" % (i + 1, np.mean(scores)))

        if np.mean(scores) > _best_score:
            _best_estimator, _best_score, _best_params = deepcopy(_est), np.mean(scores), _est.get_params()

    with open("W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/models/" + season + "/SLP/" + method + '_model_' + scoring + '.pkl', 'wb') as f:
        pickle.dump(_best_estimator, f)

    print("Validation process ended with score {}\nBest parameters: {}".format(_best_score, _best_params))
    return _best_estimator

def bic_score(X, labels, centroids):
    '''
    Method to compute the BIC (Bayesian Information Criterion)
    Args:
    - X: the dataset on which evaluating the BIC
    - labels: labels associated to the samples in X
    - centroids: the centroids associated to X

    Returns:
    - the BIC score associated to the estimator
    '''
    eps = 1e-7
    m = centroids.shape[0]
    n = np.zeros((m,))
    hist = np.bincount(labels)
    n[:len(hist)] = hist

    N, D = X.shape

    const_term = 0.5 * m * np.log(N) * (D + 1)
    cl_var = (1. / (N - m) / D) * sum([
        sum(distance.cdist(X[np.where(labels == i)], [centroids[i]], 'euclidean') ** 2) for i in range(m)
    ])

    return np.sum([n[i] * np.log(n[i] + eps) -
                   n[i] * np.log(N) -
                   ((n[i] * D) / 2) * np.log(2 * np.pi * cl_var + eps) -
                   ((n[i] - 1) * D / 2) for i in range(m)]) - const_term

def extract_regimes(anomaly, method='kmeans', nb_regimes=5, **kwargs):
    '''
    Method clustering anomalies in different weather regimes
    Args:
    - anomaly: a pandas DataFrame containing an historical series of anomalies
    - clustering_algo: clustering algorithm to adopt. The class should expose the methods .fit(), .predict(), .fit_predict(), .fit_transform()
    - nb_regimes: number of different weather regimes to be identified, if 'estimator' in **kwargs, this parameter is ignored
    - **kwargs: a dictionary of further parameters, like the a pre_trained estimator, or the possibility to do directly inference

    Returns:
    - an array of regimes associated to each grid in the time series
    '''
    if method == 'kmeans':
        if 'estimator' not in kwargs:
            clustering_algo = KMeans(n_clusters=nb_regimes, random_state=42,
                                     tol=1e-5, n_init=50)
        else:
            clustering_algo = kwargs['estimator']
        clustering_algo.fit(anomaly)
        if 'test' in kwargs:
            test = kwargs['test']
            return clustering_algo.predict(test)
        return clustering_algo.labels_, clustering_algo.inertia_, clustering_algo

    elif method == 'bayesian_gmm':
        if 'estimator' not in kwargs:
            clustering_algo = BayesianGaussianMixture(n_components=nb_regimes, random_state=42, n_init=10,
                                                      covariance_type='full' if 'covariance_type' not in kwargs else
                                                      kwargs['covariance_type'],
                                                      reg_covar=1e-3, max_iter=1000)
        else:
            clustering_algo = kwargs['estimator']
        clustering_algo.fit(anomaly)
        probas = clustering_algo.predict_proba(anomaly)
        if 'test' in kwargs:
            test = kwargs['test']
            return clustering_algo.predict(test)
        return probas, clustering_algo.lower_bound_, clustering_algo.means_, clustering_algo.covariances_, clustering_algo

    elif method == 'gmm':
        if 'estimator' not in kwargs:
            clustering_algo = GaussianMixture(n_components=nb_regimes, random_state=42, n_init=10,
                                              covariance_type='full' if 'covariance_type' not in kwargs else kwargs[
                                                  'covariance_type'],
                                              reg_covar=1e-3, max_iter=1000)
        else:
            clustering_algo = kwargs['estimator']
        clustering_algo.fit(anomaly)
        probas = clustering_algo.predict_proba(anomaly)
        if 'test' in kwargs:
            test = kwargs['test']
            return clustering_algo.predict(test)
        return probas, clustering_algo.lower_bound_, clustering_algo.means_, clustering_algo.covariances_, clustering_algo

    elif method == 'spectral':
        if 'estimator' not in kwargs:
            clustering_algo = SpectralClustering(n_clusters=nb_regimes, random_state=42, n_init=50,
                                                 affinity='rbf' if 'affinity' not in kwargs else kwargs['affinity'])
        else:
            clustering_algo = kwargs['estimator']
        clustering_algo.fit(anomaly)
        if 'test' in kwargs:
            test = kwargs['test']
            return clustering_algo.predict(test)
        return clustering_algo.labels_, clustering_algo.affinity_matrix_, clustering_algo

    else:
        pass


def load_estimator(path_to_file):
    '''
    Method to load an estimator from the pickle file
    Args:
    - path_to_file: file path to the pickle file

    Returns:
    - the loaded estimator
    '''
    with open(path_to_file, 'rb') as pickle_file:
        estimator = pickle.load(pickle_file)

    return estimator
#############################################################################


READ_PATH = 'P:\CH\Weather Data\ERA-5\SLP'
freq = 'hourly' # 'monthly'
months = 'JunJulAug'#'DecJanFeb'  #'MayJunJulAugSep'
obs_years = '1979-2021'#'1979-2020'
LAT, LONG = (20.,80.), (-90., 30.)
reduction = "PCA"
model = "kmeans"
season = "SUMMER"

dt = read_nc(READ_PATH + '/ERA-5_{}_SLP_{}_{}.nc'.format(freq, months, obs_years))
dt = build_data()

#dt_daily = read_nc(READ_PATH + '/ERA-5_{}_SLP_{}_{}.nc'.format("daily", months, obs_years))

dt = weighted_anomaly(dt)

pivot_anomaly = flat_table(dt)   # only for PCA + training

if reduction == "PCA":
    reduced_anomaly = reduce_dim(pivot_anomaly, method='PCA', exp_variance=0.9)
    folder = 'pca'
else:
    reduced_anomaly = reduce_dim(dt, method='VAE', season = season)
    folder = 'vae'

#eofs, pcs = eofs(pivot_anomaly)
#plot_EOFS(eofs, savefig = False)

train_X, test_X, pivot_train, pivot_test = train_test_split(reduced_anomaly, pivot_anomaly, test_size = 0.2, random_state = 42)

'''
for scoring in ["score", "ch", "bic", "silhouette"]:
    estimator = cross_val(reduced_anomaly.values, method=model, scoring=scoring, season=season)

outputs = extract_regimes(train_X, method=model, nb_regimes = None, estimator = estimator)
'''

df_score = pd.DataFrame.from_dict({model:performance_matrix('W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/models/SUMMER/SLP/'+ model, train_X.values, test_X.values)\
                        for model in os.listdir('W:/UK/Research/Private/WEATHER/STAGE_ABALDO/scripts/models/SUMMER/SLP') if model.endswith('pkl')},
             orient='index')
print(df_score)

##
estimator = load_estimator('W:\\UK\\Research\\Private\\WEATHER\\STAGE_ABALDO\\scripts\\models\\SUMMER\\SLP\\kmeans_model_score.pkl')
outputs = extract_regimes(reduced_anomaly, method=model, nb_regimes = None, estimator = estimator)
plot_regimes(pivot_anomaly, outputs[0])