import numpy as np
from shallow_sa.model.common.core import Spectral


class WaterBase(Spectral):
    def __init__(self, wvl, env_param, model_option="default"):
        super().__init__(wvl, env_param)
        self.model_option = model_option
        self.setup_properties(env_param)

    def model_factory(self):
        # This method should be overridden by the subclass to provide specific implementations
        raise NotImplementedError("This method should be overridden by subclass")

    def initialize_property(self, file_path):
        wvl_data, data = self._read_model_data(file_path)
        return self._interp_data(wvl_data, data)

    def setup_properties(self, env_param):
        # This method should be overridden by the subclass to provide specific implementations
        raise NotImplementedError("This method should be overridden by subclass")

    def get_property(self, deriv=False, **kwargs):
        # This method should be overridden by the subclass to provide specific implementations
        raise NotImplementedError("This method should be overridden by subclass")

    def _get_derivative(self, deriv, *args, prop_type):
        model_func = self.model_data[prop_type].get("model", None)
        deriv_func = self.model_data[prop_type].get("deriv", None)
        hessian_func = self.model_data[prop_type].get("hessian", None)

        if not deriv:
            return model_func(*args, self.wvl)
        elif deriv == 1:
            return deriv_func(*args, self.wvl)
        elif deriv == 2:
            return hessian_func(*args, self.wvl)


class AbsorptionModelBase(WaterBase):
    def __init__(self, wvl, env_param, model_option="default"):
        super().__init__(wvl, env_param, model_option=model_option)

    def model_factory(self):
        # Implement the model factory for absorption
        pass

    def setup_properties(self, env_param):
        # Implement property setup for absorption
        pass

    def get_property(self, deriv=False, **kwargs):
        # Implement property computation for absorption
        pass


class BackscatteringModelBase(WaterBase):
    def __init__(self, wvl, env_param, model_option="default"):
        super().__init__(wvl, env_param, model_option=model_option)

    def model_factory(self):
        # Implement the model factory for backscattering
        pass

    def setup_properties(self, env_param):
        # Implement property setup for backscattering
        pass

    def get_property(self, deriv=False, **kwargs):
        # Implement property computation for backscattering
        pass
