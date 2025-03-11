"""
Interval computations for profiling analysis.
"""

import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import chi2
from intersect import intersection

class IntervalComputer:
    """Handles computation of confidence intervals and related metrics."""
    
    def compute_intervals(self, psi_arr: np.ndarray, stat_arr: np.ndarray, 
                         alpha: float) -> Tuple[float, float]:
        """Compute confidence intervals for a given statistic."""
        z_alpha = chi2.ppf(1 - alpha, 1) ** 0.5
        
        psi_inf, _ = intersection(
            psi_arr, stat_arr, psi_arr, np.ones_like(psi_arr) * z_alpha
        )
        psi_sup, _ = intersection(
            psi_arr, stat_arr, psi_arr, -np.ones_like(psi_arr) * z_alpha
        )

        # Handle empty intersections
        if not len(psi_inf) or not len(psi_sup):
            psi_inf = [np.nan] if not len(psi_inf) else psi_inf
            psi_sup = [np.nan] if not len(psi_sup) else psi_sup

        return psi_inf[0], psi_sup[0]

    def compute_ci(self, psi_arr: np.ndarray, stats: Dict[str, np.ndarray], 
                  alpha: float) -> Dict[str, List[float]]:
        """Compute confidence intervals for all statistics."""
        ci = {}
        for stat_key, stat_vals in stats.items():
            inf, sup = self.compute_intervals(psi_arr, stat_vals, alpha)
            ci[stat_key] = [inf, sup]
        return ci