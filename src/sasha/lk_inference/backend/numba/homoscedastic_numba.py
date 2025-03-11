# @title Homoscedastic numba
# import numba.numpy as np
from   numba import jit, grad, jacfwd
import numba
import numpy as np
from numba import jit


def generate_samples(mean_spectrum, var, num_samples):
    SIGMA = var * np.eye(mean_spectrum.shape[1])
    samples = np.array(
        np.random.multivariate_normal(
            np.array(mean_spectrum).squeeze(), SIGMA, num_samples
        )
    )
    return samples


@jit(nopython=True)
def log_likelihood(params, inv_var, observed_spectra, forward_model):
    # Ensure that the inputs are of the correct type
    if (
        not isinstance(params, (np.ndarray, dict))
        or not isinstance(inv_var, np.ndarray)
        or not isinstance(observed_spectra, np.ndarray)
    ):
        raise TypeError("Invalid argument types")
    if not callable(forward_model):
        raise TypeError("forward_model must be callable")
    mean_spectra = forward_model(**params)
    diff = mean_spectra - observed_spectra
    mahalanobis_dist = inv_var * np.array([np.dot(row, row.T) for row in diff])
    log_likelihood_values = -0.5 * mahalanobis_dist
    return np.sum(log_likelihood_values)


# log_likelihood = numba.jit(log_likelihood, static_argnums=(3,))


@jit(nopython=True)
def forward_model(params, forward_model):
    if not callable(forward_model):
        raise TypeError("forward_model must be callable")
    return 1 * forward_model(**params)


forward_model_numba = numba.jit(forward_model, static_argnums=(1,))


def jacob_dict_to_array(dict_data):
    arrays_list = [dict_data[key] for key in dict_data.keys()]
    final_array = np.stack(arrays_list, axis=-1)
    return final_array


# jacob_dict_to_array = numba.jit(jacob_dict_to_array, static_argnums=(1,))


def model_jacob_numba(param, forward_model):
    jacob = jacfwd(forward_model_numba)(param, forward_model)
    return jacob


model_jacob_numba = numba.jit(model_jacob_numba, static_argnums=(1,))


def model_jacob_numerical(param, forward_model):
    model_jac_dict = model_jacob_numba(param, forward_model)
    model_jac_arr = jacob_dict_to_array(model_jac_dict)
    if model_jac_arr.shape[0] > 1:
        #  print('model jacob', model_jac_arr.shape)
        return np.sum(model_jac_arr, axis=1)
    else:
        return model_jac_arr


model_jacob_numerical = numba.jit(model_jacob_numerical, static_argnums=(1,))


def efim_numerical(param, inv_var, forward_model):
    model_jac_dict = model_jacob_numba(param, forward_model)
    model_jac_arr = jacob_dict_to_array(model_jac_dict)
    if model_jac_arr.shape[0] > 1:
        #  print('model jacob', model_jac_arr.shape)
        model_jac_arr = np.sum(model_jac_arr, axis=1)
        return ABA_product(model_jac_arr, inv_var)
    else:
        return ABA_product(model_jac_arr, inv_var)


@jit(nopython=True)
def score_numba(param, inv_var, observed_spectra, forward_model):
    model_jac = model_jacob_numerical(param, forward_model)
    mean_spectra = forward_model(**param)
    diff = mean_spectra - observed_spectra
    score = inv_var * diff @ np.squeeze(model_jac)
    return score


# @jit(nopython=True)
# score_numba   =   numba.jit(score_numba, static_argnums=(3,))


@jit(nopython=True)
def score_numerical(param, inv_var, observed_spectra, forward_model):
    score_dict = grad(log_likelihood)(param, inv_var, observed_spectra, forward_model)
    return np.array([score_dict[key] for key in score_dict.keys()])


# score_numerical   =   numba.jit(score_numerical, static_argnums=(3,))


def hessian_numba(param, inv_var, observed_spectra, forward_model):
    return jacfwd(grad(log_likelihood))(param, inv_var, observed_spectra, forward_model)


# hessian_numba      =   numba.jit(hessian_numba, static_argnums=(3,))


@jit(nopython=True)
def flatten_hessian_results(hessian_results):
    flattened_results = {}
    for main_key in hessian_results.keys():
        flattened_results[main_key] = {}
        for sub_key in hessian_results[main_key].keys():
            # Flatten the array to 1D
            hess = hessian_results[main_key][sub_key]
            # print('hess shape', hess.shape)
            try:
                hess_mainkey_key = np.diagonal(hess)
            except ValueError:
                hess_mainkey_key = np.array([hess])
            flattened_results[main_key][sub_key] = hess_mainkey_key
    return flattened_results


# flatten_hessian_results = numba.jit(flatten_hessian_results, static_argnums=(1,))


@jit(nopython=True)
def hessian_dict_to_array(hessian_dict):
    # Get the list of keys
    keys = list(hessian_dict.keys())
    # Determine the dimensions
    first_key = keys[0]
    n = len(hessian_dict[first_key][first_key])
    p = len(keys)
    # Initialize the resulting array with zeros
    result_array = np.zeros((n, p, p), dtype=np.float32)
    # Prepare slices
    slices = [
        (i, j, hessian_dict[key1].get(key2, None))
        for i, key1 in enumerate(keys)
        for j, key2 in enumerate(keys)
    ]
    # Fill in array
    for i, j, hessian_slice in slices:
        if hessian_slice is not None:
            if hessian_slice.ndim == 0:
                hessian_slice = np.array([hessian_slice] * n)
            result_array = result_array.at[:, i, j].set(hessian_slice)
    return result_array


# Sample u


@jit(nopython=True)
def hessian_numerical(param, inv_var, observed_spectra, forward_model):
    hess_numba = hessian_numba(param, inv_var, observed_spectra, forward_model)
    # print('hess_numba',hess_numba['chl']['chl'].shape)
    hess_flat = flatten_hessian_results(hess_numba)
    # print('hess_flat',hess_flat['chl']['chl'].shape)
    hess_arr = hessian_dict_to_array(hess_flat)
    # print('hess_arr',hess_arr.shape)
    return hess_arr


# hessian_numerical = numba.jit(hessian_numerical, static_argnums=(3,))


def ofim_numerical(param, inv_var, observed_spectra, forward_model):
    hess_numba = hessian_numba(param, inv_var, observed_spectra, forward_model)
    hess_flat = flatten_hessian_results(hess_numba)
    hess_arr = hessian_dict_to_array(hess_flat)
    return -hess_arr


def filter_hessian_dict(hessian_dict, nuis_param):
    """Remove excluded keys from hessian_dict"""
    keys_to_exclude = list(nuis_param.keys())
    return {
        key1: {
            key2: val for key2, val in inner_dict.items() if key2 not in keys_to_exclude
        }
        for key1, inner_dict in hessian_dict.items()
        if key1 not in keys_to_exclude
    }


@jit(nopython=True)
def score_analytical(observed_spectra, mean_spectra, model_jac, inv_var):
    n, m, p = model_jac.shape
    A_reshaped = model_jac.transpose(1, 0, 2).reshape(m, p * n)
    dist = np.sum((observed_spectra - mean_spectra[:, :, None]), axis=-1)
    B_tiled = np.repeat(dist, p, axis=0)
    # Perform the matrix multiplication to get a (n * p, 1) shape result
    Score = inv_var * np.sum(A_reshaped.T * B_tiled, axis=1).reshape(-1, 1)
    return Score


@jit(nopython=True)
def efim_analytical(model_jac, inv_var):
    return ABA_product(model_jac, inv_var)


@jit(nopython=True)
def ofim_analytical(observed_spectrum, mean_spectra, model_jac, model_hessian, inv_var):
    I = efim_analytical(model_jac, inv_var)
    dist = observed_spectrum.reshape(mean_spectra.shape) - mean_spectra
    aux = ABCBA_product(inv_var, model_hessian, dist.T)
    J = I - aux
    return J


@jit(nopython=True)
def ABA_product(A, B):
    C_vectorized = B * np.matmul(A.transpose(0, 2, 1), A)
    return C_vectorized


@jit(nopython=True)
def ABCBA_product(A, B, C):
    n, m, p, _ = A.shape
    results = []
    for i in range(n):
        A_block = A[i]
        result_block = np.dot(A_block.T, B * C[:, i])
        results.append(result_block)
    return np.stack(results, axis=0)


@jit(nopython=True)
def vectorized_outer_product(M, X):
    M_reshaped = M[:, :, np.newaxis]
    Y = M_reshaped * X[:, np.newaxis, :]
    return Y


@jit(nopython=True)
def generalized_product_scalar(A1, A2, B, C):
    # Ensure that A1, A2, and C have the expected shapes
    n, m, p = A1.shape
    temp1 = B * A1.transpose((0, 2, 1))  # Shape: (n, p, m)
    temp2 = np.matmul(temp1, C)  # Shape: (n, p, m)
    temp3 = B * temp2  # Shape: (n, p, m)
    result = np.matmul(temp3, A2)  # Shape: (n, p, p)
    return result
