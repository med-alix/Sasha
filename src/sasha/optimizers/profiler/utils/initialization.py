# shallow_sa/optimizers/profiler/utils/initialization.py
"""
Initialization utilities for the Profiler package.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

class ProfilerInitializer:
    """Handles initialization of profiling results and parameters."""
    
    def __init__(self, optimizer):
        """Initialize with optimizer instance."""
        self.optimizer = optimizer
        self.parameter_manager = optimizer.parameter_manager
        self.free_params, self.fixed_params = (
            optimizer.parameter_manager.get_optimization_parameters()
        )
        
    def get_optimization_parameters(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Get free and fixed parameters from the optimizer."""
        return self.parameter_manager.get_optimization_parameters()

    def initialize_results_storage(self, mle: Dict[str, float], interest_key: str, interest_vals: np.ndarray) -> Dict[str, Any]:
        """Initialize the storage structure for profiling results."""
        organized_results = {
            "mle_initial": mle,
            "interest_key": interest_key,
            "interest_values": interest_vals,
            "profiles": {
                "score": [],
                "lp": [],
                "nuisance_mle": {
                    key: [] for key in mle if key != interest_key and key != "lp_val"
                },
            },
        }
        return organized_results

    def _compute_overall_mle(self, sample: np.ndarray, init_param: Dict[str,float], **kwargs) -> Dict[str, float]:
        """Compute the overall maximum likelihood estimate."""
        kwargs["verbose"] = kwargs.get("verbose", 0)
        optimization_result = self.optimizer.optimize(
            sample, init_param=init_param, **kwargs
        )
        mle = dict(zip(sorted(self.free_params.keys()), optimization_result.x))
        mle["lp_val"] = -optimization_result.fun
        return mle