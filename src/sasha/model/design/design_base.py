from ...model.design.design_response import DesignResponse
from ...config.backend import Backend


class Design:
    def __init__(self, model_option, wvl, backend=Backend.JAX):
        self.model_option    = model_option
        self.response        = DesignResponse(model_option = model_option, wvl = wvl, backend = backend)
        self.wvl             = self.response.wvl
        if model_option not in "dirac":
            self.response_matrix = self.response.response_matrix
            self.bands   = self.response.retained_bands
            self.n_bands = len(self.bands)
        else:
            self.response_matrix = 1.0
            self.bands = self.wvl
            self.n_bands = self.wvl.shape[0]
        return
