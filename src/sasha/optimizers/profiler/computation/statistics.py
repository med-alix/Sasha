import numpy as np
from typing import Dict, Any, Tuple

class StatisticsComputer:
    """Handles computation of various statistical measures."""
    
    def __init__(self, stat_inf_obj):
        self.stat_inf_obj = stat_inf_obj

    def compute_stats(self, sample: np.ndarray,
                     profiling_results: Dict[str, Any],
                     **kwargs) -> Tuple[Dict[str, Any], 
                                                                    Dict[str, np.ndarray],
                                                                    Dict[str, float]]:
        """Compute various statistical measures for profiling analysis."""

        mle           = profiling_results["mle_final"]
        mle_initial   = profiling_results["mle_initial"]
        interest_vals = profiling_results["interest_values"]
        interest_key  = profiling_results["interest_key"]
        lp            = profiling_results["profiles"]["lp"]
        score         = profiling_results["profiles"]["score"] 

        # Convert any remaining JAX arrays to numpy
        def to_numpy(x):
            return np.array(x) if hasattr(x, 'device_buffer') else x

        try:
            # Convert JAX arrays to numpy if needed
            sample = to_numpy(sample)
            lp = np.array(lp).squeeze()
            score = np.array(score).squeeze()
            mle_interest = float(mle[interest_key])
            interest_vals = np.array(interest_vals)
            # Compute Wald standard errors with error handling
            try:
                efim_std, efim = self._compute_std_errors(sample, mle, method="efim")
                ofim_std, ofim = self._compute_std_errors(sample, mle, method="ofim")
                efim_std_old, efim_old = self._compute_std_errors(sample, mle_initial, method="efim")
                ofim_std_old, ofim_old = self._compute_std_errors(sample, mle_initial, method="ofim")
            except Exception as e:
                print(f"Warning: Standard error computation failed: {e}")
                # Create dummy standard errors
                dummy_std = {k: 1.0 for k in mle.keys()}
                dummy_fim = np.eye(len(mle))
                efim_std = ofim_std = efim_std_old = ofim_std_old = dummy_std
                efim = ofim = efim_old = ofim_old = dummy_fim

            # Compute statistics with error handling
            try:
                r   = np.sign(mle_interest-interest_vals)*np.sqrt(2*(np.nanmax(lp)-lp))
                t_o = np.array((mle_interest - interest_vals) / ofim_std[interest_key])
                t_e = np.array((mle_interest - interest_vals) / efim_std[interest_key])
                s   = np.array(ofim_std[interest_key] * score)

                # rs_1 = self._compute_rs_statistic(adjustments["lm_1"], 
                #                                 adjustments["lm_1_psi_max"], 
                #                                 interest_vals)
                # rs_2 = self._compute_rs_statistic(adjustments["lm_2"], 
                #                                 adjustments["lm_2_psi_max"], 
                #                                 interest_vals)
                # rs_a = self._compute_rs_statistic(adjustments["la"], 
                #                                 adjustments["la_psi_max"], 
                #                                 interest_vals)

            except Exception as e:
                print(f"Warning: Statistics computation failed: {e}")
                # Create dummy statistics
                r = t_o = t_e = s = np.zeros_like(interest_vals)

            return (
                {"r": r, "s": s, "t_o": t_o, "t_e": t_e},
                {
                    "efim": to_numpy(efim),
                    "ofim": to_numpy(ofim),
                    "efim_initial": to_numpy(efim_old),
                    "ofim_initial": to_numpy(ofim_old)
                },
                {
                    "std_efim": efim_std,
                    "std_ofim": ofim_std,
                    "std_efim_old": efim_std_old,
                    "std_ofim_old": ofim_std_old
                }
            )
            
        except Exception as e:
            print(f"Warning: Overall statistics computation failed: {e}")
            # Return dummy values
            dummy_stats = {
                "r": np.zeros_like(interest_vals),
                "s": np.zeros_like(interest_vals),
                "t_o": np.zeros_like(interest_vals),
                "t_e": np.zeros_like(interest_vals),
                "rs_1": np.zeros_like(interest_vals),
                "rs_2": np.zeros_like(interest_vals),
                "rs_a": np.zeros_like(interest_vals)
            }
            dummy_matrices = {
                "efim": np.eye(len(mle)),
                "ofim": np.eye(len(mle)),
                "efim_initial": np.eye(len(mle)),
                "ofim_initial": np.eye(len(mle))
            }
            dummy_stds = {
                "std_efim": {k: 1.0 for k in mle.keys()},
                "std_ofim": {k: 1.0 for k in mle.keys()},
                "std_efim_old": {k: 1.0 for k in mle_initial.keys()},
                "std_ofim_old": {k: 1.0 for k in mle_initial.keys()}
            }
            return dummy_stats, dummy_matrices, dummy_stds

    def _compute_std_errors(self, sample: np.ndarray, mle: Dict[str, float], 
                          method: str = "efim") -> Tuple[Dict[str, float], np.ndarray]:
        """Compute standard errors using either EFIM or OFIM method."""
        try:
            if method == "efim":
                fim = self.stat_inf_obj.efim(**mle)[0]
            else:
                fim = self.stat_inf_obj.ofim(sample, **mle)[0]
            fim = np.array(fim)  # Convert JAX array to numpy if needed
            inv_fim = np.linalg.pinv(fim)
            std_errors = np.sqrt(np.diag(inv_fim))
            # Convert to regular Python floats
            std_errors_dict = {
                key: float(err) for key, err in zip(sorted(list(mle.keys())), std_errors)
            }
            return std_errors_dict, fim
            
        except Exception as e:
            print(f"Warning: Standard error computation failed: {e}")
            # Return dummy values
            dummy_std = {k: 1.0 for k in mle.keys()}
            dummy_fim = np.eye(len(mle))
            return dummy_std, dummy_fim