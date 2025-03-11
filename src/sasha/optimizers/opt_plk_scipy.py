# @title Optimizer ULT
from collections import defaultdict
from copy import deepcopy
from functools import partial

import numpy as np
from scipy.optimize import brentq, minimize, newton
from tqdm import tqdm

# Correcting the issue with the profile method by getting the MLE before switching the parameter to fixed.
INITIAL_STEP_SIZE = 0.5
MAX_ITERATIONS    = 7
ADAPTIVE_STEP_MULTIPLIER = 0.25
FALLBACK_STEP_MULTIPLIER = 0.5


# from src.utils.curve_utils import generate_sampling_points


def generate_sampling_points(x_existing, max_samples=30):
    """
    Generate additional sampling points based on existing x and y values.
    Parameters:
    - x_existing (array-like): The existing x-values that have already been sampled.
    - y_existing (array-like, optional): The existing y-values that correspond to x_existing.
    - max_samples (int): The maximum number of additional samples to generate.
    Returns:
    - new_sampling_points (array): The new x-values where sampling should be conducted.
    """
    # Sort the existing x-values
    x_existing = np.sort(x_existing)
    # Calculate the existing gaps between x-values
    gaps = np.diff(x_existing)
    # Normalize the gaps to their sum
    normalized_gaps = gaps / np.sum(gaps)
    # Calculate the number of points to insert in each gap
    points_to_insert = np.round(normalized_gaps * max_samples).astype(int)
    # Generate new sampling points
    new_points = []
    for x1, x2, n_points in zip(x_existing[:-1], x_existing[1:], points_to_insert):
        new_points.extend(
            np.linspace(x1, x2, n_points + 2)[1:-1]
        )  # Exclude the end points
    # Ensure that the number of new points does not exceed max_samples
    new_points = new_points[:max_samples]
    return np.array(new_points)


def res_opt_print(optimized_free_params, fun_value, success, verbose=False):
    if verbose:
        print("\033[1;36mOptimization Results\033[0m")
        print("\033[1;32m-------------------\033[0m")
        print("\033[1;34mOptimized Free Parameters:\033[0m")
        for key, value in optimized_free_params.items():
            print(f"\033[1;33m{key}: \033[1;35m{value}\033[0m")
        print(f"\033[1;34mOptimized Function Value: \033[1;35m{-fun_value}\033[0m")
        print(f"\033[1;34mOptimization Success: \033[1;35m{success}\033[0m")
        print("\033[1;32m-------------------\033[0m")


class LogLikelihoodOptimizer:

    def __init__(
        self,
        statistical_inference_obj,
        param_formatter,
        method="trust-constr",
        **solver_options,
    ):
        self.stat_inf_obj = statistical_inference_obj
        self.param_formatter = deepcopy(param_formatter)
        self.method = method
        self.solver_options = solver_options

    def _objective_function(
        self, free_params, observed_spectrum, free_params_keys, fixed_params, kwargs
    ):
        all_params = {**fixed_params, **dict(zip(free_params_keys, free_params))}
        log_likelihood = -np.array(
            self.stat_inf_obj.log_likelihood(observed_spectrum, **all_params).squeeze()
        )

        return -np.array(log_likelihood.squeeze())

    def optimize(self, observed_spectrum=None, bounded=False, verbose=False, **kwargs):
        param_formatter_clone = deepcopy(self.param_formatter)
        free_params_keys, fixed_params, initial_params, opt_bounds = (
            self.get_optimization_setup(param_formatter_clone, bounded)
        )
        opt_bounds = (
            [
                param_formatter_clone.free_parameters[key]["bounds"]
                for key in free_params_keys
            ]
            if bounded
            else None
        )
        result = minimize(
            self._objective_function,
            initial_params,
            args=(observed_spectrum, free_params_keys, fixed_params, kwargs),
            method=self.method,
            options=self.solver_options,
            bounds=opt_bounds,
        )
        print("fixed_params optimize", fixed_params)
        print("free_params_keys optimize", free_params_keys)
        print("initial_params", initial_params)
        optimized_free_params = dict(zip(free_params_keys, result.x))
        res_opt_print(
            optimized_free_params, result.fun, result.success, verbose=verbose
        )
        return optimized_free_params, -result.fun, result.success

    def run_minimization(
        self,
        initial_params,
        observed_spectrum,
        free_params_keys,
        fixed_params,
        opt_bounds,
        **kwargs,
    ):
        return minimize(
            self._objective_function,
            initial_params,
            args=(observed_spectrum, free_params_keys, fixed_params, kwargs),
            method=self.method,
            options=self.solver_options,
            bounds=opt_bounds,
        )

    def get_optimization_setup(self, param_formatter_clone, bounded):

        free_params_keys = list(param_formatter_clone.free_parameters.keys())
        fixed_params = {
            key: param_dict["actual_value"]
            for key, param_dict in param_formatter_clone.fixed_parameters.items()
        }
        initial_params = [
            param_formatter_clone.initial_values[key] for key in free_params_keys
        ]
        # initial_params  = [ _*(1+np.random.uniform(-.1,.1)) for _  in initial_params] # *(1+np.random.uniform(-.1,.1))
        # initial_params = [ _ for _  in initial_params] # *(1+np.random.uniform(-.1,.1))
        opt_bounds = (
            [
                param_formatter_clone.free_parameters[key]["bounds"]
                for key in free_params_keys
            ]
            if bounded
            else None
        )
        return free_params_keys, fixed_params, initial_params, opt_bounds

    def get_profiling_kwargs(self, **kwargs):
        observed_spectrum = kwargs.get("observed_spectrum", None)
        interest_key = kwargs.get("interest_key", None)
        profile_bounds = kwargs.get("profile_bounds", None)
        l_function = kwargs.get("l_function", "lp")
        mle = kwargs.get("mle", None)
        bounded = kwargs.get("bounded", False)
        verbose = kwargs.get("verbose", False)
        return (
            observed_spectrum,
            interest_key,
            profile_bounds,
            l_function,
            mle,
            bounded,
            verbose,
        )

    def lp(self, **kwargs):
        (
            observed_spectrum,
            interest_key,
            profile_bounds,
            l_function,
            mle,
            bounded,
            verbose,
        ) = self.get_profiling_kwargs(**kwargs)
        interest_val = kwargs.get("interest_val", 0)
        lp = kwargs.get("lp_val", None)
        nui_hat = kwargs.get("nui_hat", None)
        initial_values = kwargs.get("initial_values", None)

        if not lp or not nui_hat:  # if profiling results not provided recompute them
            param_formatter_clone = deepcopy(self.param_formatter)
            print(
                "fixed_params lp before substituion",
                param_formatter_clone.fixed_parameters.keys(),
            )
            print(
                "free_params_keys lp before substituion",
                param_formatter_clone.free_parameters.keys(),
            )
            param_formatter_clone.substitute_fixed_param(interest_key)
            print(
                "fixed_params lp post substituion",
                param_formatter_clone.fixed_parameters.keys(),
            )
            print(
                "free_params_keys lp post substituion",
                param_formatter_clone.free_parameters.keys(),
            )
            free_params_keys, fixed_params, initial_params, opt_bounds = (
                self.get_optimization_setup(param_formatter_clone, bounded)
            )
            print("free_params_keys lp postpost substituion", free_params_keys)
            print("fixed_params lp postpost substituion", fixed_params)
            print(opt_bounds)
            profile_bounds = param_formatter_clone.fixed_parameters[interest_key][
                "bounds"
            ]
            fixed_params[interest_key] = interest_val
            result = self.run_minimization(
                initial_params,
                observed_spectrum,
                free_params_keys,
                fixed_params,
                opt_bounds,
            )
            lp = -result.fun
            nui_hat = dict(zip(free_params_keys, result.x))
            print("nui_hat", nui_hat, fixed_params, opt_bounds)

        if l_function in ["lp"]:
            return nui_hat, lp, result.success
        elif l_function in ["la"]:
            la = lp + self.stat_inf_obj.adj_cox(
                observed_spectrum, interest_key, interest_val, nui_hat, verbose=verbose
            )
            return nui_hat, la, True
        elif l_function in ["lms"]:
            a_cox = self.stat_inf_obj.adj_cox(
                observed_spectrum, interest_key, interest_val, nui_hat, verbose=verbose
            )
            a_sev = self.stat_inf_obj.adj_sev(
                observed_spectrum,
                interest_key,
                interest_val,
                nui_hat,
                mle,
                verbose=verbose,
            )
            lm = lp + a_cox + a_sev
            return nui_hat, lm, True
        elif l_function in ["lmd"]:
            a_cox = self.stat_inf_obj.adj_cox(
                observed_spectrum, interest_key, interest_val, nui_hat, verbose=verbose
            )
            a_lui = self.stat_inf_obj.adj_nui(
                observed_spectrum,
                interest_key,
                interest_val,
                nui_hat,
                mle,
                verbose=verbose,
            )
            lm = lp + a_cox + a_lui
            return nui_hat, lm, True

    def profile_psi(
        self,
        interest_key,
        observed_spectrum=None,
        mle=None,
        bounded=False,
        verbose=False,
        l_function="lp",
        theta_true=None,
        n_samples=9,
        **kwargs,
    ):

        key_profile = defaultdict(list)
        key_psi_points = defaultdict(list)
        key_nuis_points = {}
        key_success_messages = defaultdict(list)
        param_formatter_clone = deepcopy(self.param_formatter)
        param_formatter_clone.substitute_fixed_param(interest_key)
        free_params_keys, fixed_params, initial_params, opt_bounds = (
            self.get_optimization_setup(param_formatter_clone, bounded)
        )
        print("fixed_params profile", fixed_params)
        print("free_params_keys profile", free_params_keys)
        profile_bounds = kwargs.get(
            "profile_bounds",
            param_formatter_clone.fixed_parameters[interest_key]["bounds"],
        )
        profiling_kwargs = {
            "observed_spectrum": observed_spectrum,
            "interest_key": interest_key,
            "profile_bounds": profile_bounds,
            "l_function": l_function,
            "mle": mle,
            "bounded": bounded,
            "verbose": verbose,
        }

        update_bool = False
        key_nuis_points[interest_key] = defaultdict(list)
        # Regular coarse profiling
        self.gridded_profiling(
            key_profile,
            key_psi_points,
            key_nuis_points,
            key_success_messages,
            n_samples=kwargs.get("n_samples", 9),
            **profiling_kwargs,
        )
        # Convergent binary profiling
        self.convergent_binary_profiling(
            key_profile,
            key_psi_points,
            key_nuis_points,
            key_success_messages,
            **profiling_kwargs,
        )

        # Complete gridded sampling
        psi_to_sample = generate_sampling_points(
            key_psi_points[interest_key], max_samples=15
        )
        self.gridded_profiling(
            key_profile,
            key_psi_points,
            key_nuis_points,
            key_success_messages,
            n_samples=15,
            **profiling_kwargs,
        )  # psi_to_sample = psi_to_sample ,
        # Sort profile
        (
            sorted_profiles,
            sorted_psi_points,
            sorted_nuis_points,
            sorted_success_messages,
        ) = self.post_process_results(
            key_profile, key_psi_points, key_nuis_points, key_success_messages
        )
        # Update the MLE based on profiling
        lmax = self.stat_inf_obj.log_likelihood(observed_spectrum, **mle)
        lc_max = np.nanmax(sorted_profiles[interest_key])
        ind_max = np.nanargmax(sorted_profiles[interest_key])
        mle_psi_updated = sorted_psi_points[interest_key][ind_max]
        mle_is_updated = True if lmax < lc_max else False
        mle_updated = {
            **{interest_key: mle_psi_updated},
            **{k: v[ind_max] for k, v in sorted_nuis_points[interest_key].items()},
        }

        return {
            "profiles": sorted_profiles,
            "psi_points": sorted_psi_points,
            "nuis_points": sorted_nuis_points,
            "success_messages": sorted_success_messages,
            "mle_is_updated": mle_is_updated,
            "mle": mle,
            "mle_updated": mle_updated,
            "lmax": lc_max if mle_updated else lmax,
        }

    def search_CI_bounds(self, l_max, **kwargs):
        # not implemented yet;
        lb, ub = kwargs.get("profile_bounds", 15)
        verbose = kwargs.get("verbose", 15)
        interest_key = kwargs.get("interest_key", None)

        def f(psi):
            nuis_optim, l_p, success = self.lp(interest_val=psi, **kwargs)
            return 2 * (l_max - l_p)

        root_brent1 = brentq(f, lb, ub)  # Searching in the interval [1, 2]
        return

    def gridded_profiling(
        self,
        profiles,
        psi_points,
        nuis_points,
        success_messages,
        psi_to_sample=[],
        **kwargs,
    ):
        n_samples = kwargs.get("n_samples", 15)
        lb, ub = kwargs.get("profile_bounds", [None, None])
        verbose = kwargs.get("verbose", False)
        interest_key = kwargs.get("interest_key", None)
        psi_arr = (
            psi_to_sample if any(psi_to_sample) else np.linspace(lb, ub, n_samples)
        )

        if verbose:
            print("\033[93m#" * 145 + "\033[0m")
            print(
                f"\033[1;33mGridded Profiling of  {len(psi_arr)} regularly spaced samples between bounds \033[0m: \033[0m \033[92m{lb,ub}\033[0m along the the interest parameter : \033[1;35;43m{interest_key}\033[0m."
            )
            print("\033[93m#" * 145 + "\033[0m")

        for i, interest_val in enumerate(psi_arr):
            if i == 0:
                nuis_optim, l_p, success = self.lp(interest_val=interest_val, **kwargs)
            else:
                initial_values = nuis_optim
                nuis_optim, l_p, success = self.lp(interest_val=interest_val, **kwargs)
            # nuis_optim, l_p, success = self.lp(interest_val=interest_val, **kwargs)

            profiles[interest_key].append(l_p)
            psi_points[interest_key].append(interest_val)
            success_messages[interest_key].append(success)
            for key in nuis_optim.keys():
                nuis_points[interest_key][key].append(nuis_optim[key])
            if verbose:
                print("\033[1;36;40m-" * 85 + "\033[0m")
                print(
                    f"\033[97mlp profiling  at : \033[0;35;47m{interest_key}={interest_val}  \033[90mlp({interest_key})  = \033[1;37;41m{l_p}\033[0m"
                )
        return

    def convergent_binary_profiling(
        self,
        profiles,
        psi_points,
        nuis_points,
        success_messages,
        epsilon=1e-3,
        **kwargs,
    ):
        (
            observed_spectrum,
            interest_key,
            profile_bounds,
            l_function,
            mle,
            bounded,
            verbose,
        ) = self.get_profiling_kwargs(**kwargs)
        #   Define your tolerance level
        lb, ub = profile_bounds
        #   Initialize a counter for the number of samples
        sample_count = 0
        if verbose:
            print("\033[96m#" * 145 + "\033[0m")
            print(
                f"\033[1;33;40mConvergent binary Profiling between bounds \033[0m: \033[0m \033[92m{profile_bounds} \033[0of the interest parameter : \033[1;35;43m{interest_key}\033[0m."
            )
            print("\033[96m#" * 145 + "\033[0m")

        while ub - lb > epsilon:
            mid1 = lb + (ub - lb) / 3
            mid2 = lb + 2 * (ub - lb) / 3
            l_p = [0, 0]
            for i, mid in enumerate([mid1, mid2]):
                nuis_optim, l_p[i], success = self.lp(interest_val=mid, **kwargs)
                profiles[interest_key].append(l_p[i])
                psi_points[interest_key].append(mid)
                success_messages[interest_key].append(success)
                for key in nuis_optim.keys():
                    nuis_points[interest_key][key].append(nuis_optim[key])
            sample_count = sample_count + 2
            if l_p[0] < l_p[1]:
                lb = mid1
            else:
                ub = mid2

            if verbose:
                # Using ANSI escape codes to colorize text (here, \033[91m is for red, \033[0m resets it)
                print(
                    f"\033[97mUpdating profiling between \033[93m{lb} \033[97mand \033[93m{ub} \033[97mlp update : \033[94m{max(l_p)}\033[0m"
                )
        # Print summary at the end of the while loop
        if verbose:
            print(
                f"\033[92mTotal convergent samples obtained for {interest_key}: {sample_count}\033[0m"
            )  # \033[92m is for green
            print(
                f"\033[94mFinal tolerance achieved: {ub - lb}\033[0m"
            )  # \033[94m is for blue

    def divergent_mle_profiling(
        self, profiles, psi_points, nuis_points, success_messages, **kwargs
    ):

        (
            observed_spectrum,
            interest_key,
            profile_bounds,
            l_function,
            mle,
            bounded,
            verbose,
        ) = self.get_profiling_kwargs(**kwargs)

        l_max = kwargs.get("l_max", None)
        lb, ub = profile_bounds
        sample_count = 0
        for direction in [-1, 1]:
            if verbose:
                d_str = "left" if direction == -1 else "right"

                print("\033[95m#" * 145 + "\033[0m")
                # print(f'profiling along the {d_str} half of the interest parameter : <<{psi_key}>>')
                print(
                    f"\033[1;33;40mDivergent Profiling from value \033[0m: \033[0m \033[92m{mle[interest_key]}\033[0m along the \033[1;33;40m{d_str} \033[0mhalf of the interest parameter : \033[1;35;43m{interest_key}\033[0m."
                )
                print("\033[95m#" * 145 + "\033[0m")

            proposed_psi = mle[interest_key] + INITIAL_STEP_SIZE * direction
            iteration = 0

            while lb < proposed_psi < ub and iteration < 1.5 * MAX_ITERATIONS:
                interest_val = proposed_psi
                nuis_optim, l_p, success = self.lp(interest_val=interest_val, **kwargs)
                profiles[interest_key].append(l_p)
                psi_points[interest_key].append(proposed_psi)
                success_messages[interest_key].append(success)

                for key in nuis_optim.keys():
                    nuis_points[interest_key][key].append(nuis_optim[key])

                if iteration < MAX_ITERATIONS:
                    step_size = ADAPTIVE_STEP_MULTIPLIER * (l_max - l_p)
                else:
                    step_size = FALLBACK_STEP_MULTIPLIER * np.abs(
                        mle[interest_key] - proposed_psi
                    )

                proposed_psi += step_size * direction
                iteration += 1
                sample_count = sample_count + 1
                if verbose:
                    print("\033[1;36;40m-" * 85 + "\033[0m")
                    print(
                        f"\033[97mlp profiling  at : \033[0;35;47m{interest_key}={proposed_psi}  \033[90mlp({interest_key})  = \033[1;37;41m{l_p}\033[0m"
                    )
        if verbose:
            print(
                f"\033[92mTotal divergent samples obtained for {interest_key}: {sample_count}\033[0m"
            )  # \033[92m is for green

    def adjust_profile_psi(
        self,
        interest_key,
        observed_spectrum,
        verbose=False,
        rank_reduction=False,
        **profile_results,
    ):

        profile = profile_results.get("profiles", None)
        psi_points = profile_results.get("psi_points", None)
        nuis_points = profile_results.get("nuis_points", None)
        mle = profile_results.get("mle_updated", None)
        success_messages = profile_results.get("sorted_success_messages", None)

        la_profiles = {}
        lms_profiles = {}
        lmd_profiles = {}

        la_profile = defaultdict(list)
        lms_profile = defaultdict(list)
        lmd_profile = defaultdict(list)

        for i, psi_val in enumerate(psi_points[interest_key]):
            lp_val_i = profile[interest_key][i]
            nui_hat_i = {
                key: val[i]
                for key, val in nuis_points[interest_key].items()
                if len(val) > i
            }
            a_cox = self.stat_inf_obj.adj_cox(
                observed_spectrum,
                interest_key,
                psi_val,
                nui_hat_i,
                rank_reduction=rank_reduction,
                verbose=verbose,
            )
            a_sev = self.stat_inf_obj.adj_sev(
                observed_spectrum,
                interest_key,
                psi_val,
                nui_hat_i,
                mle,
                rank_reduction=rank_reduction,
                verbose=verbose,
            )
            a_lui = self.stat_inf_obj.adj_nui(
                observed_spectrum,
                interest_key,
                psi_val,
                nui_hat_i,
                mle,
                rank_reduction=rank_reduction,
                verbose=verbose,
            )

            la_profile[interest_key].append(a_cox)
            lms_profile[interest_key].append(a_sev)
            lmd_profile[interest_key].append(a_lui)

            if verbose:
                # ANSI escape codes for styling
                ANSI_CYAN = "\033[96m"
                ANSI_GREEN = "\033[92m"
                ANSI_YELLOW = "\033[93m"
                ANSI_RESET = "\033[0m"
                ANSI_BOLD = "\033[1m"
                print(
                    f"{ANSI_BOLD}{ANSI_CYAN}{interest_key}={psi_val}{ANSI_RESET}: {ANSI_BOLD}{ANSI_GREEN}lp{ANSI_RESET}={ANSI_YELLOW}{lp_val_i}{ANSI_RESET}, {ANSI_BOLD}{ANSI_GREEN}COX{ANSI_RESET}={ANSI_YELLOW}{a_cox}{ANSI_RESET}, {ANSI_BOLD}{ANSI_GREEN}SEVERINI{ANSI_RESET}={ANSI_YELLOW}{a_sev}{ANSI_RESET}, {ANSI_BOLD}{ANSI_GREEN}LUI{ANSI_RESET}={ANSI_YELLOW}{a_lui}{ANSI_RESET}"
                )

        return {"acox": la_profile, "asev": lms_profile, "alui": lmd_profile}

    def post_process_results(self, profiles, psi_points, nuis_points, success_messages):
        sorted_profiles = defaultdict(list)
        sorted_psi_points = defaultdict(list)
        sorted_nuis_points = {}
        sorted_success_messages = defaultdict(list)

        for key in psi_points.keys():
            sorted_indices = np.argsort(psi_points[key])
            sorted_psi_points[key] = [psi_points[key][i] for i in sorted_indices]
            sorted_profiles[key] = [profiles[key][i] for i in sorted_indices]
            sorted_success_messages[key] = [
                success_messages[key][i] for i in sorted_indices
            ]
            sorted_nuis_points[key] = defaultdict(list)
            for nui_key in nuis_points[key].keys():
                sorted_nuis_points[key][nui_key] = [
                    nuis_points[key][nui_key][i] for i in sorted_indices
                ]

        return (
            sorted_profiles,
            sorted_psi_points,
            sorted_nuis_points,
            sorted_success_messages,
        )

        # No need to return anything since dictionaries are updated in place.
