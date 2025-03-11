# @title Homoscedastic module
from   ..lk_inference.models_registry.homoscedastic_registry import get_functions_registry
from   ..lk_inference.mvg_model import MVGStatisticalInference
from   ..config.backend import Backend
import numpy as np

def filter_hessian_array(hessian_array, param_dict, params_to_exclude):
    # Determine the order of parameters based on the sorted keys of the dictionary
    sorted_params = sorted(param_dict.keys())
    # Identify indices of parameters to exclude
    indices_to_exclude = [
        sorted_params.index(param)
        for param in params_to_exclude
        if param in sorted_params
    ]
    # Create a new array that excludes the rows and columns corresponding to the parameters to be excluded
    if indices_to_exclude:
        indices_to_keep = [
            i for i in range(hessian_array.shape[1]) if i not in indices_to_exclude
        ]
        filtered_array = hessian_array[:, indices_to_keep, :][:, :, indices_to_keep]
    else:
        filtered_array = (
            hessian_array  # No parameters to exclude, so return the original array
        )
    return filtered_array
def filter_jacobian_array(jacobian_array, param_dict, params_to_exclude):
    # Step 2: Determine the order of parameters based on the sorted keys of the dictionary
    sorted_params = sorted(param_dict.keys())
    # Step 3: Identify indices of parameters to exclude
    indices_to_exclude = [
        sorted_params.index(param)
        for param in params_to_exclude
        if param in sorted_params
    ]
    # Step 4: Create a new array that excludes the columns corresponding to the parameters to be excluded
    if indices_to_exclude:
        indices_to_keep = [
            i for i in range(jacobian_array.shape[2]) if i not in indices_to_exclude
        ]
        filtered_array = jacobian_array[:, :, indices_to_keep]
    else:
        filtered_array = (
            jacobian_array  # No parameters to exclude, so return the original array
        )
    return filtered_array
def determinant_rank(matrix):
    """
    Calculate the product of non-zero eigenvalues of a given square matrix.
    Parameters:
    - matrix (np.ndarray): The square matrix for which to calculate the product of non-zero eigenvalues.
    Returns:
    - float: The product of non-zero eigenvalues.
    """
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)
    # Filter out zero eigenvalues
    non_zero_eigenvalues = eigenvalues[np.abs(eigenvalues) < 1e-10]
    # Calculate and return product of non-zero eigenvalues
    return np.prod(non_zero_eigenvalues)



class HMOMVG(MVGStatisticalInference):

    def __init__(self, forward_model, variance, backend = Backend.JAX, **forward_modelargs):
        super().__init__(forward_model, variance, backend = backend, **forward_modelargs)
        self.variance = variance
        self.inv_var  = 1.0 / variance

    def get_function(self, function_name: str):
        FUNCTION_HOMOSCEDASTIC_REGISTRY = get_functions_registry(self.backend)
        return FUNCTION_HOMOSCEDASTIC_REGISTRY.get(function_name)

    def _forward_model(self, **kwargs):
        return self.get_function("forward_model")(kwargs, self.compute_forward_model)
        # return self.compute_forward_model(**kwargs)
    
    def model_jacob_(self, keys_to_exclude=[], **kwargs):
        jacobian =  self.get_function("model_jacob")(kwargs, self._forward_model)
        if not len(keys_to_exclude):
            return jacobian
        else:
            return filter_jacobian_array(jacobian, kwargs, keys_to_exclude)

    def log_likelihood(self, observed_spectra, **kwargs):
        return self.get_function("log_likelihood")(kwargs, self.inv_var, observed_spectra, self._forward_model)

    def score(self, observed_spectra, **kwargs):
        return self.get_function("score")(kwargs, self.inv_var, observed_spectra, self._forward_model)

    def efim(self, keys_to_exclude=[], **kwargs):
        efim = self.get_function("efim")(kwargs, self.inv_var, self._forward_model)
        if not len(keys_to_exclude):
            return efim
        else:
            return filter_hessian_array(efim, kwargs, keys_to_exclude)

    def ofim(self, observed_spectra, keys_to_exclude=[], **kwargs):
        ofim = self.get_function("ofim")(
            kwargs, self.inv_var, observed_spectra, self._forward_model
        )
        if not len(keys_to_exclude):
            return ofim
        else:
            return filter_hessian_array(ofim, kwargs, keys_to_exclude)

    def generate_samples(self, num_samples, **kwargs):
        mean_spectrum = self.forward_model(**kwargs)
        return self.get_function("generate_samples")(
            mean_spectrum, self.variance, num_samples
        )

    def adj_cox(
        self,
        observed_spectra,
        psi_key,
        psi_val,
        theta_hat_psi,
        rank_reduction=False,
        verbose=False,
    ):
        theta = {**{psi_key: psi_val}, **theta_hat_psi}
        j_nui_hat = self.ofim(observed_spectra, keys_to_exclude=[psi_key], **theta)
        acox = self.safe_log_det(
            j_nui_hat, rank_reduction=rank_reduction, verbose=verbose, label="Cox"
        )
        return acox if np.isscalar(acox) else acox[0]

    def adj_nui(
        self,
        observed_spectra,
        psi_key,
        psi_val,
        theta_hat_psi,
        mle,
        rank_reduction=False,
        verbose=False,
    ):
        theta_hat_psi = {**theta_hat_psi, **{psi_key: psi_val}}
        model_mle = self._forward_model(**mle)
        model_nuisance = self._forward_model(**theta_hat_psi)
        jacob_mle = self.model_jacob_(keys_to_exclude=[psi_key], **mle)
        jacob_nuisance = self.model_jacob_(keys_to_exclude=[psi_key], **theta_hat_psi)
        # Q             =    1/self.inv_var*cp.eye(jacob_mle.shape[0]) +(model_mle-model_nuisance).T@(model_mle-model_nuisance)
        I_nui = (self.inv_var) * jacob_nuisance[0, :, :].T @ jacob_nuisance[0, :, :]
        return self.safe_log_det(
            I_nui, rank_reduction=rank_reduction, verbose=verbose, label="Luigi"
        )

    def adj_sev(
        self,
        observed_spectra,
        psi_key,
        psi_val,
        theta_hat_psi,
        mle,
        rank_reduction=False,
        verbose=False,
    ):
        theta_hat_psi = {
            **theta_hat_psi,
            **{psi_key: psi_val},
        }
        # model_mle       =   self.forward_model(deriv=0, **mle)
        jacob_mle = self.forward_model(deriv=1, **mle)
        jacob_mle = self.model_jacob_(keys_to_exclude=[psi_key], **mle)
        jacob_nuisance = self.model_jacob_(keys_to_exclude=[psi_key], **theta_hat_psi)
        I_sev = (self.inv_var) * jacob_nuisance[0, :, :].T @ jacob_mle[0, :, :]
        return self.safe_log_det(
            I_sev, rank_reduction=rank_reduction, verbose=verbose, label="Severini"
        )

    def adj_all(
        self,
        observed_spectra,
        psi_key,
        psi_val,
        theta_hat_psi,
        mle,
        rank_reduction=False,
        verbose=False,
    ):
        adj_results = {}  # Initialize empty dictionary to hold all adjustments
        # Call adj_cox and store its result in the dictionary
        adj_results["Cox"] = self.adj_cox(
            observed_spectra, psi_key, psi_val, theta_hat_psi, verbose=verbose
        )
        # Call adj_nui and store its result in the dictionary
        adj_results["Luigi"] = self.adj_nui(
            observed_spectra, psi_key, psi_val, theta_hat_psi, mle, verbose=verbose
        )
        # Call adj_sev and store its result in the dictionary
        adj_results["Severini"] = self.adj_sev(
            observed_spectra, psi_key, psi_val, theta_hat_psi, mle, verbose=verbose
        )
        return adj_results  # Return the dictionary containing all adjustments

    def safe_log_det(self, matrix, rank_reduction=True, verbose=False, label=""):
        det_value = np.linalg.det(matrix)
        if det_value < 0 or np.isnan(det_value):
            if np.isnan(det_value):
                # sub_det = determinant_rank(matrix)
                if verbose:
                    print(
                        f"\033[91mDeterminant value: {det_value} replaced by nan \033[0m"
                    )
                return np.nan
            else:
                if np.abs(det_value) < 1e-10 and rank_reduction:
                    det_value = determinant_rank(matrix)
                if verbose:
                    print(
                        f"\033[91mError: Determinant non-positive in {label} adjustment with value {det_value}.\033[0m"
                    )
                return np.nan
        else:
            return np.log(det_value)
