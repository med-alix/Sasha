"""
Standard error computations for profiling analysis.
"""

import numpy as np
from typing import Dict, Tuple, Any

class StandardErrorComputer:
    """Handles computation of standard errors using various methods."""
    
    def __init__(self, stat_inf_obj):
        self.stat_inf_obj = stat_inf_obj

    def compute_std_errors(self, sample: np.ndarray, mle: Dict[str, float], 
                          method: str = "efim") -> Tuple[Dict[str, float], np.ndarray]:
        """
        Compute standard errors using specified method.
        
        Args:
            sample: Data sample
            mle: Maximum likelihood estimates
            method: Method to use ('efim' or 'ofim')
            
        Returns:
            Tuple of (standard errors dict, flattened Fisher information matrix)
        """
        if method == "efim":
            fim = self.stat_inf_obj.efim(**mle)[0]
        else:
            fim = self.stat_inf_obj.ofim(sample, **mle)[0]
            
        inv_fim = np.linalg.pinv(fim)
        std_errors = np.sqrt(np.diag(inv_fim))
        
        return dict(zip(sorted(list(mle.keys())), std_errors))#, np.array(fim).flatten()