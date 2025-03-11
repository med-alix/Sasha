from ..lk_inference.statmodel_base import StatisticalInference
from ..config.backend import Backend


class MVGStatisticalInference(StatisticalInference):
    def __init__(self, forward_model, covariance_matrix, backend = Backend.JAX, **forward_model_args):
        super().__init__(forward_model, backend = backend, **forward_model_args)
        self.cov_matrix  = covariance_matrix
