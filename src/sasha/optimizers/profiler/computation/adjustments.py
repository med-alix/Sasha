import numpy as np
from sasha.utils.interp_utils import interpolate_nans_with_method, replace_NaN

class AdjustmentComputer:
    """Handles computation of various statistical adjustments."""
    
    def __init__(self, stat_inf_obj):
        self.stat_inf_obj = stat_inf_obj

    def compute_adj(self, interest_key, interest_vals, profiling_results, sample, mle, **kwargs):
        """
        Compute adjustments following the original implementation exactly.
        """
        adj_inf, adj_nui_1, adj_nui_2 = [], [], []
        for i, interest_val in enumerate(interest_vals):
            # Get nuisance parameters 
            nuisance_mle = {
                key: profiling_results["nuisance_mle"][key][i]
                for key in sorted(list(profiling_results["nuisance_mle"].keys()))
            }
            # Compute adjustments 
            adj_inf.append(
                self.stat_inf_obj.adj_cox(
                    sample, interest_key, interest_val, nuisance_mle, mle, **kwargs
                )
            )
            adj_nui_1.append(
                self.stat_inf_obj.adj_sev(
                    sample, interest_key, interest_val, nuisance_mle, mle, **kwargs
                )
            )
            adj_nui_2.append(
                self.stat_inf_obj.adj_nui(
                    sample, interest_key, interest_val, nuisance_mle, mle, **kwargs
                )
            )

        # Process results exactly as in original
        lp = profiling_results["lp"]
        a_inf = interpolate_nans_with_method(replace_NaN(adj_inf))
        a_nui_1 = interpolate_nans_with_method(replace_NaN(adj_nui_1))
        a_nui_2 = interpolate_nans_with_method(replace_NaN(adj_nui_2))

        # Compute adjusted likelihoods
        la = lp + 0.5 * a_inf
        lm_1 = lp - 0.5 * a_inf + a_nui_1
        lm_2 = lp - 0.5 * a_inf + 0.5 * a_nui_2
        
        # Normalize exactly as in original
        la = la - np.nanmax(la)
        lm_1 = lm_1 - np.nanmax(lm_1)
        lm_2 = lm_2 - np.nanmax(lm_2)

        # Return same structure as original
        return {
            "adj_inf": a_inf,
            "adj_nui_1": a_nui_1,
            "adj_nui_2": a_nui_2,
            "adj_inf_old": adj_inf,
            "adj_nui_1_old": a_nui_1,
            "adj_nui_2_old": a_nui_2,
            "la": la,
            "lm_1": lm_1,
            "lm_2": lm_2,
            "lm_1_psi_max": interest_vals[np.argmax(lm_1)],
            "lm_2_psi_max": interest_vals[np.argmax(lm_2)],
            "la_psi_max": interest_vals[np.argmax(la)]
        }