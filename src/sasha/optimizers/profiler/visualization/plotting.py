"""
Visualization utilities for profiling results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Dict, Any
from sasha.utils.plotting import plt_param, plt_style
plt.rcParams.update(plt_param)
plt.style.use(plt_style)


class ProfilePlotter:
    """Handles visualization of profiling results."""
    
    @staticmethod
    def plot_profile(profiling_results: Dict[str, Any], 
                    title: Optional[str] = None,
                    show_confidence: bool = True) -> None:
        """
        Plot profiling results.
        
        Args:
            profiling_results: Results to plot
            title: Optional plot title
            show_confidence: Whether to show confidence intervals
        """
        """Plot the profile results."""
        plt.figure(figsize=(12, 5))
        plt.plot(profiling_results.interest_values, profiling_results.profiles['lp'],
                  lw = '2', marker = '.',  markersize=12,)

        interest_key = profiling_results.interest_key
        d     = profiling_results.standard_errors['std_ofim'][interest_key]
        m     = profiling_results.mle_final[interest_key]
        title = f'{interest_key.capitalize()} Parameter Profile'
        plt.title(title or "Parameter Profile")
        plt.xlabel(f"Interest Parameter: {interest_key}")
        plt.ylabel("Log Profile likelihood")
        plt.ylim([-3,0])
        plt.xlim([m - 3*d, m + 3*d])
        plt.axvline(profiling_results.mle_initial[interest_key], color='r', linestyle='-.', label='Initial Optimal Value')
        plt.axvline(profiling_results.mle_final[interest_key], color='g', linestyle='-', label='Profile Optimal Value')
        
        
        if show_confidence:
            plt.axhline(-1.96**2/2, color='y', linestyle='--', label='95% Confidence level')

        plt.legend()
        plt.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.show()
