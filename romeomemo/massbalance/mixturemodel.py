#!/usr/bin/env python
# coding=utf-8
import numpy as np

from scipy.stats import multivariate_normal

from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, Matern
from sklearn import preprocessing


class GMM:
    
    def __init__(self, n_clusters):
        
        self.n_clusters = n_clusters

    def set_training_values(self, x, y):
        """
        Set training data (values)

        Parameters
        ----------
        x : np.ndarray[nt, nx] or np.ndarray[nt]
            The input values for the nt training points
        y : np.ndarray[nt, ny] or np.ndarray[nt]
        """

        self.x = x
        self.y = y
        self.dim = x.shape[1]

    def train(self):
        """
        Gaussian mixture model cluster
        """

        ## Combine X and Y

        norm_y = preprocessing.normalize(self.y.reshape(1,-1))

        self.norm_y = np.squeeze(norm_y)

        self.train_points = np.c_[self.x,self.norm_y]

        self.gmm = GaussianMixture(self.n_clusters, tol=1e-3,
                                   covariance_type="full", n_init=100)
        self.gmm.fit(self.train_points)

        return


    def cluster_data(self):

        prob_cluster = self.gmm.predict_proba(self.train_points)
        hard_cluster = self.gmm.predict(self.train_points)

        return prob_cluster, hard_cluster

    def _create_cluster_distributions(self):
        """
        Create an array of frozen multivariate normal distributions (distribs)
        """

        distribs = []

        dim = self.dim
        means = self.gmm.means_
        cov = self.gmm.covariances_

        for k in range(self.n_clusters):
            meansk = means[k][0:dim]
            covk = cov[k][0:dim, 0:dim]
            mvn = multivariate_normal(meansk, covk)
            distribs.append(mvn)

        return distribs


    def _prob_cluster_one_sample(self, x, distribs):

        weights = self.gmm.weights_
        rvs = np.array([distribs[k].pdf(x) for k in range(len(weights))])

        probs = weights * rvs

        rad = np.sum(probs)

        if rad > 0:
            probs = probs / rad

        return probs


    def prob_membership(self, x):

        distribs = self._create_cluster_distributions()
        probs = np.array([self._prob_cluster_one_sample(x[i], distribs) for i
                          in range(len(x))])

        return probs

    def weights(self):
        return self.gmm.weights_

    def covariances(self):
        return self.gmm.covariances_




def sqexp_kernel(sigma_f, l, min_l, max_l):
    """
    Isotropic squared exponential covariance function kernel

    Parameters:
        sigma_f : float
             Initial guess for the vertical variation
        l : list
             Initial guess for the length scale
    Returns:
        sklearn.gaussian_process.kernels
    """
 
    con_term = ConstantKernel(constant_value = sigma_f, constant_value_bounds = (1e-3, 1e3))
    exp_term = RBF(length_scale=l, length_scale_bounds = (min_l, max_l))

    kernel = con_term * exp_term #Squared exponential covariance function

    return kernel


def matern_kernel(sigma_f, l, nu, min_l, max_l):
    """
    Isotropic squared exponential covariance function kernel

    Parameters:
        sigma_f : float
             Initial guess for the vertical variation
        l : list
             Initial guess for the length scale
    Returns:
        sklearn.gaussian_process.kernels
    """
 
    con_term = ConstantKernel(constant_value = sigma_f, constant_value_bounds = (1e-3, 1e3))
    exp_term = Matern(length_scale=l, nu=nu, length_scale_bounds = (min_l, max_l))

    kernel = con_term * exp_term #Squared exponential covariance function

    return kernel


def init_variance(obs_data, prob):

    weighted_data = obs_data * prob
    mean_data = np.mean(weighted_data)

    tmp_sigma_f = np.sum((weighted_data - mean_data)**2)
    sigma_f = tmp_sigma_f / np.sum(prob)

    return sigma_f

def init_lengthscale(dist, prob):
    """
    """

    weighted_l_scale = dist * prob
    mean_l_scale = np.mean(weighted_l_scale)

    l_scale = max(np.abs(weighted_l_scale - mean_l_scale))

    return l_scale

def EM_hyperparam(X, Y, Z, cov_kernel, sigma_n=0):
    """
    Parameters:
        X : 1D numpy.array

    """

    gp = GaussianProcessRegressor(kernel=cov_kernel, alpha=sigma_n,
                                  n_restarts_optimizer=100)

    points = list(zip(X,Y))
    obs_value = Z

    # pred_mesh = list(product(X_star, Y_star))

    gp.fit(points, obs_value)

    sigma_f = gp.kernel_.k1.get_params()["constant_value"]
    l1, l2 = gp.kernel_.k2.get_params()["length_scale"]

    # Z_star, Z_star_std = gp.predict(pred_mesh, return_std=True)

    # Z_star = Z_star.reshape(len(Y_star), len(X_star))
    # Z_star_std = Z_star_std.reshape(len(Y_star), len(X_star))

    return sigma_f, l1, l2


def EMC_hyperparam(X, Y, C, cov_kernel, sigma_n=0):
    """
    """
    gp = GaussianProcessClassifier(kernel=cov_kernel, n_restarts_optimizer=10)

    points = list(zip(X,Y))
    obs_value = C

    # pred_mesh = list(product(X_star, Y_star))

    gp.fit(points, obs_value)

    sigma_f = gp.kernel_.k1.get_params()["constant_value"]
    l1, l2 = gp.kernel_.k2.get_params()["length_scale"]

    # Z_star, Z_star_std = gp.predict(pred_mesh, return_std=True)

    # Z_star = Z_star.reshape(len(Y_star), len(X_star))
    # Z_star_std = Z_star_std.reshape(len(Y_star), len(X_star))

    return sigma_f, l1, l2
