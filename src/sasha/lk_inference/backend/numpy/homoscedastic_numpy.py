# @title Homoscedastic numba
# import numba.numpy as np
# from   numba import jit, grad, jacfwd, jacrev
import time
from collections import defaultdict
import numba
import numpy as np


def generate_samples(mean_spectrum, var, num_samples):
    SIGMA = var * np.eye(mean_spectrum.shape[1])
    samples = np.array(
        np.random.multivariate_normal(
            np.array(mean_spectrum).squeeze(),
            SIGMA, num_samples))
    return samples


def log_likelihood(params, inv_var, observed_spectra, forward_model):
    mean_spectra = forward_model(**params)
    diff = mean_spectra - observed_spectra
    mahalanobis_dist = inv_var * np.array([np.dot(row, row.T) for row in diff])
    log_likelihood_values = -0.5 * mahalanobis_dist
    return np.sum(log_likelihood_values)

def forward_model(params, forward_model):
    if not callable(forward_model):
        raise TypeError("forward_model must be callable")
    return forward_model(**params)

def jacob_dict_to_array(dict_data):
    arrays_list = [dict_data[key] for key in dict_data.keys()]
    final_array = np.stack(arrays_list, axis=-1)
    return final_array

def score_numerical():
    pass

def efim_numerical():
    pass    

def ofim_numerical():    
    pass
def model_jacob_numerical(param, eta):
  pass

def hessian_numerical():
  pass
