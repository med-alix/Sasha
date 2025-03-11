import numpy as np
from scipy.optimize import minimize
import logging

class Optimizer:
    def __init__(
        self, parameter_manager, stat_inf_obj, method="trust-constr", log_level=logging.INFO
    ):
        self.parameter_manager = parameter_manager
        self.stat_inf_obj = stat_inf_obj
        self.method = method
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.handlers.clear()
        self.logger.propagate = False
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(log_level)

    def optimize(
        self,
        observed_spectrum,
        verbose=False,
        init_param={},
        bounds=None,
        param_confing=None,
        **kwargs,
    ):
        if init_param :
            self.parameter_manager.update_parameter_values(init_param)
        if not param_confing:
            free_params, fixed_params = (self.parameter_manager.get_optimization_parameters())
        else:
            free_params, fixed_params = param_confing
        
        free_params_keys = sorted(list(free_params.keys()))
        bounds = (
            [free_params[key]["bounds"] for key in free_params_keys] if bounds else None
        )

        if init_param :
            initial_guess = [init_param[key] for key in free_params_keys]
        else:
            initial_guess = [
                free_params[key]["initial_value"] for key in free_params_keys
            ]

        self.logger.info("Initial point")
        for i, key in enumerate(free_params_keys):
            self.logger.info(f"{key}: {initial_guess[i]:.2f}")

        result = minimize(
            self._objective_function,
            x0=np.array(initial_guess),
            args=(np.array(observed_spectrum), free_params_keys, fixed_params, verbose),
            method=self.method,
            bounds=bounds,
            **kwargs,
        )

        # Always log results at INFO level
        optimized_params = {
            key: val for key, val in zip(free_params_keys, result.x)
        }
        self.logger.info("Optimization Results:")
        for key, value in optimized_params.items():
            self.logger.info(f"{key}: {value:.4f}")
        self.logger.info(f"Optimized Function Value: {-result.fun:.4f}")
        self.logger.info(f"Optimization Success: {result.success}")

        return result

    def _objective_function(self, x, sample, free_params_keys, fixed_params, verbose):
        params = {key: val for key, val in zip(free_params_keys, x)}
        all_params = {
            **params,
            **{key: details["initial_value"] for key, details in fixed_params.items()},
        }

        log_likelihood = -np.array(
            self.stat_inf_obj.log_likelihood(sample, **all_params)
        ).squeeze()

        if verbose == 2:
            self.logger.debug("Current Optimization Step:")
            for key, value in params.items():
                self.logger.debug(f"{key}: {value:.4f}")
            self.logger.debug(f"Objective Function Value: {log_likelihood:.4f}")

        return log_likelihood


    
    def _objective_function(self, x, sample, free_params_keys, fixed_params, verbose):
        params = {key: val for key, val in zip(free_params_keys, x)}
        all_params = {
            **params,
            **{key: details["initial_value"] for key, details in fixed_params.items()},
        }
        
        log_likelihood = -np.array(
            self.stat_inf_obj.log_likelihood(sample, **all_params)
        ).squeeze()
        
        if verbose == 2:
            self.logger.section("Current Optimization Step")
            for key, value in params.items():
                self.logger.key_value(key, f"{value:.4f}")
            self.logger.key_value(
                "Objective Function Value", f"{log_likelihood:.4f}"
            )
        
        return log_likelihood


    def profile_parameter(
        self,
        observed_spectrum,
        interest_key,
        interest_val,
        init_param={},
        method="trust-constr",
        verbose=False,
        bounds=None,
        **kwargs,
    ):
        # Original parameter configuration retrieval
        free_params, fixed_params = self.parameter_manager.get_optimization_parameters()
        # for val in interest_val:
        # Update fixed_params with the profiling value
        # for key, value in zip(interest_key, [val]):
        fixed_params[interest_key] = {"initial_value": interest_val}
        free_params.pop(interest_key)
        # Define the initial guess, considering use_theta_true
        if init_param:
            kwargs["x0"] = [
                (
                    init_param[key]
                    if key in init_param
                    else free_params[key]["initial_value"]
                )
                for key in free_params
            ]
        else:
            kwargs["x0"] = [
                param["initial_value"] for param in free_params.values()
            ]

        # else :
        #     initial_guess = kwargs.get('x0')
        # Objective function wrapper to match signature
        def objective_wrapper(x):
            return self._objective_function(
                x,
                observed_spectrum,
                sorted(list(free_params.keys())),
                fixed_params,
                verbose,
            )

        # Set up bounds if not provided
        if bounds is None:
            bounds = [(None, None)] * len(kwargs["x0"])

        # Perform optimization
        result = minimize(objective_wrapper, method=method, bounds=bounds, **kwargs)

        # Extract and store profiling results
        optimized_params = dict(zip(sorted(list(free_params.keys())), result.x))
        profile_results = {
            "interest_val": interest_val,
            "lp_val": -result.fun,  # Log-likelihood value from optimization
            "nuisance_mle": optimized_params,
        }

        return profile_results

    