from pydantic import PrivateAttr
from ...model.common.core import SpectralComponentLoader

class BottomBase(SpectralComponentLoader):
    modeling_option: str
    _model_data: dict = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self.setup_properties()

    def model_factory(self):
        raise NotImplementedError("This method should be overridden by subclass")

    def initialize_property(self, file_path):
        wvl_data, data = self._read_model_data(file_path)
        return self._interp_data(wvl_data, data)

    def setup_properties(self):
        raise NotImplementedError("This method should be overridden by subclass")

    def get_property(self, **kwargs):
        raise NotImplementedError("This method should be overridden by subclass")
