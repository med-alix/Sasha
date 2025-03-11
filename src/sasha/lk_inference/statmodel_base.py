# Generic Statistical Inference Class
from ..config.backend import Backend, get_cp




class StatisticalInference:

    def __init__(self, forward_model, backend = Backend.JAX, **forward_model_args):
        self.forward_model      = forward_model
        self.forward_model_args = forward_model_args
        self.backend            = backend
        self.cp                 = get_cp(backend)

    def generate_samples(self, num_samples, **kwargs):
        raise NotImplementedError("This method should be implemented by subclass")

    def log_likelihood(self, observed_spectrum, **kwargs):
        raise NotImplementedError("This method should be implemented by subclass")

    def compute_forward_model(self, **kwargs):
        all_args = {**self.forward_model_args, **kwargs}
        return self.forward_model(**all_args)

    def compute_score_analytical(self, observed_spectrum, **kwargs):
        raise NotImplementedError("This method should be implemented by subclass")

    def compute_fisher_exp_analytical(self, **kwargs):
        raise NotImplementedError("This method should be implemented by subclass")

    def compute_fisher_obs_analytical(self, observed_spectrum, **kwargs):
        raise NotImplementedError("This method should be implemented by subclass")

    def compute_first_order_tests(self, observed_spectrum, **kwargs):
        raise NotImplementedError("This method should be implemented by subclass")
