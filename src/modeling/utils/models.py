import numpy as np
import pandas as pd
from scipy.spatial import distance
from modeling.utils.tools import *
from functools import wraps, partial
from multiprocessing.dummy import Pool
import pickle
from copy import deepcopy
import os

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state



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


class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means

    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print("Converged at iteration", it + 1)
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)


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

    elif method == "kernel_kmeans":
        if 'estimator' not in kwargs:
            clustering_algo = KernelKMeans(n_clusters=nb_regimes, random_state=42,
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


def get_inertia(X, labels, centroids):
    '''
    Method to compute the Sum of Squared Errors (SSE)
    Args:
    - X: the dataset of clustered points
    - labels: the clustering assignments
    - centroids: the centroids associated to X
    '''
    m = centroids.shape[0]
    return np.sum(
        [np.sum(distance.cdist(X[np.where(labels == i)], [centroids[i]], 'euclidean') ** 2) for i in range(m)])


def performance_matrix(estimator, train_X, val_X):
    '''
    Method to evaluate an estimator, on different evaluation metrics
    Args:
    - estimator: the clustering algorithm to be tested, or the path to the pickle file to load the estimator

    Returns:
    - a pandas DataFrame comparing several estimators
    '''

    if isinstance(estimator, str):
        estimator = load_estimator(estimator)

    labels = estimator.fit(train_X).predict(val_X)

    return {
        # "Internal score": estimator.score(val_X, labels),
        "Number of Clusters": estimator.get_params()['n_clusters'
        if 'n_clusters' in estimator.get_params() else 'n_components'],
        "BIC": bic_score(val_X, labels,
                         estimator.cluster_centers_ if hasattr(estimator, "cluster_centers_") else estimator.means_),
        "Silhouette Score": silhouette_score(val_X, labels),
        "Calinski Harabsz Index": calinski_harabasz_score(val_X, labels),
        "Inertia": get_inertia(val_X, labels,
                               estimator.cluster_centers_ if hasattr(estimator,
                                                                     "cluster_centers_") else estimator.means_),

    }


@timing
def cross_val(X, method="kmeans", scoring="score", season = "WINTER", folder = '', verbose=True):
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

    elif method == "kernel_kmeans":
        estimator = KernelKMeans(random_state=42)
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

    try:
        os.makedirs("../models/" + season + "/" + folder)
    finally:
        with open("../models/" + season + "/" + folder + '/' + method + '_model_' + scoring + '.pkl', 'wb') as f:
            pickle.dump(_best_estimator, f)

    print("Validation process ended with score {}\nBest parameters: {}".format(_best_score, _best_params))
    return _best_estimator

@timing
def get_statistics(folder, train, test):
    '''
    Method to collect the statistics about the models' performances
    Args:
        folder: name of the folder where there are pickle files of the model
        train: the train dataset used to fit the model
        test: the test dataset used to evaluate the performance on

    Returns:
        A pandas DataFrame containing the statistics

    '''

    return pd.DataFrame.from_dict({model: performance_matrix(folder + '/' + model,
                                                      train.values, test.values) \
                            for model in os.listdir(folder) if
                            model.endswith('pkl')},
                           orient='index')
