"""
Base Profiler module implementing the main profiling functionality.
"""

import numpy as np
from typing import Dict, Any, List
from dataclasses import dataclass
import logging
from tqdm import tqdm
import sys

from .computation.adjustments import AdjustmentComputer
from .computation.intervals import IntervalComputer
from .computation.statistics import StatisticsComputer
from .computation.standard_errors import StandardErrorComputer
from .utils.initialization import ProfilerInitializer
from .utils.serialization import ResultSerializer
from .utils.value_generation import ValueGenerator
from .visualization.plotting import ProfilePlotter



@dataclass
class ProfileResults:
    """Container for profiling results."""
    mle_initial: Dict[str, float]
    mle_final: Dict[str, float]
    profiles: Dict[str, Any]
    interest_values: List[float]
    interest_key: str
    stats: Dict[str, Any]
    confidence_intervals: Dict[str, List[float]]
    standard_errors: Dict[str, float]




class Profiler:
    """Main profiler class for parameter profiling and analysis."""
    
    def __init__(self, optimizer, log_level=logging.INFO):
        """Initialize the profiler with required components and setup logging."""
        self.optimizer = optimizer
        self.stat_inf_obj = optimizer.stat_inf_obj
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        # Clear any existing handlers to prevent duplicate logging
        self.logger.handlers.clear()
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        # Prevent propagation to root logger to avoid duplicate logs
        self.logger.propagate = False
        self.logger.setLevel(log_level)
        
        # Initialize component handlers
        self.logger.debug("Initializing profiler components")
        self.initializer = ProfilerInitializer(optimizer)
        self.adjustment_computer = AdjustmentComputer(self.stat_inf_obj)
        self.interval_computer = IntervalComputer()
        self.statistics_computer = StatisticsComputer(self.stat_inf_obj)
        self.std_error_computer = StandardErrorComputer(self.stat_inf_obj)
        self.value_generator = ValueGenerator()
        self.result_serializer = ResultSerializer()
        
        # Get optimization parameters
        self.free_params, self.fixed_params = (
            optimizer.parameter_manager.get_optimization_parameters()
        )
        self.alpha = 0.05
        self.logger.debug("Profiler initialization complete")

    def profile(self, 
                interest_key: str, 
                sample: np.ndarray, 
                num_points: int = 10, 
                init_param: Dict[str,float] = {},
                width: float = 0.0,
                values: str = "uniform", 
                **kwargs) -> ProfileResults:
        """
        Perform uniform profiling for a parameter of interest.
        """
        self.logger.info(f"Starting profiling for parameter: {interest_key}")
        
        # Initial computation
        self.logger.debug("Computing overall MLE")
        mle = self.initializer._compute_overall_mle(sample, init_param = init_param, **kwargs)
        std_ofim = self.std_error_computer.compute_std_errors(sample, mle, method='ofim')[interest_key]
        
        self.logger.debug("Generating interest values")
        interest_vals = self.value_generator.generate_interest_values(
            mle[interest_key], values=values, num_points=num_points, 
            width=width, std_ofim=std_ofim
        )
        
        # Initialize results storage
        profiling_results = self.initializer.initialize_results_storage(
            mle, interest_key, interest_vals
        )
        
        # Profile each point with progress bar
        mle.pop("lp_val")
        initial_guess = mle
        self.logger.info(f"Profiling {num_points} points")
        
        with tqdm(total=len(interest_vals), desc="Profiling Progress") as pbar:
            for val in interest_vals:
                self._profile_single_point(profiling_results, sample, interest_key, val, initial_guess, **kwargs)
                pbar.update(1)
        
        # Post-process results
        self.logger.info("Post-processing results")
        return self.post_process_results(profiling_results, sample)

    def _profile_single_point(self, 
                            profiling_results: Dict[str, Any],
                            sample: np.ndarray,
                            interest_key: str,
                            interest_val: float,
                            init_param: Dict[str, float],
                            **kwargs) -> None:
        """Profile a single point in the parameter space."""
        self.logger.debug(f"Profiling point: {interest_key}={interest_val}")
                
        # Get profiling result
        result = self.optimizer.profile_parameter(
            sample, interest_key, interest_val, init_param=init_param, **kwargs
        )
        
        # Compute score
        theta_psi = {**{interest_key: interest_val}, **result["nuisance_mle"]}
        index_psi = sorted(list(theta_psi.keys())).index(interest_key)
        score = self.stat_inf_obj.score(sample, **theta_psi)[index_psi]
        
        # Update results
        profiling_results["profiles"]["lp"].append(result["lp_val"])
        profiling_results["profiles"]["score"].append(score)
        for key, value in result["nuisance_mle"].items():
            profiling_results["profiles"]["nuisance_mle"][key].append(value)

    def post_process_results(self, 
                           profiling_results: Dict[str, Any], 
                           sample: np.ndarray) -> ProfileResults:
        """
        Post-process profiling results using computation modules.
        """
        self.logger.info("Computing final MLE")
        self._compute_final_mle(profiling_results)
        
        self.logger.info("Computing profile statistics")
        stats, matrices, std_errors = self.statistics_computer.compute_stats(sample, profiling_results)

        self.logger.info("Computing confidence intervals")
        confidence_intervals = self.interval_computer.compute_ci(
            profiling_results["interest_values"],
            stats,
            self.alpha
        )
        
        self.logger.info("Profiling complete")
        # Package results
        return ProfileResults(
            mle_initial=profiling_results["mle_initial"],
            mle_final=profiling_results["mle_final"],
            interest_values=profiling_results["interest_values"],
            interest_key=profiling_results["interest_key"],
            profiles=profiling_results["profiles"],
            stats=stats,
            confidence_intervals=confidence_intervals,
            standard_errors=std_errors
        )

    def _compute_final_mle(self, profiling_results: Dict[str, Any]) -> None:
        """Compute final MLE and normalize log-probability values."""
        self.logger.debug("Computing final MLE and normalizing log-probability values")
        lp_values = np.array(profiling_results["profiles"]["lp"])
        lp_normalized = lp_values - np.nanmax(lp_values)
        max_lp_index = np.argmax(lp_values)
        
        interest_key = profiling_results["interest_key"]
        interest_val = profiling_results["interest_values"][max_lp_index]
        
        mle_final = {interest_key: interest_val}
        for key in profiling_results["profiles"]["nuisance_mle"]:
            mle_final[key] = profiling_results["profiles"]["nuisance_mle"][key][max_lp_index]
            
        profiling_results["mle_final"] = mle_final
        profiling_results["profiles"]["lp"] = lp_normalized
        profiling_results["profiles"]["score"] = np.array(
            profiling_results["profiles"]["score"]
        ).squeeze()

    def plot(profiling_results):
        ProfilePlotter.plot_profile(profiling_results=profiling_results)