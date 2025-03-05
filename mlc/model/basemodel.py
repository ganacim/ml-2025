from torch import nn


class BaseModel(nn.Module):
    _name = "base_model"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def name(cls):
        return cls._name

    @staticmethod
    def add_arguments(parser):
        raise NotImplementedError("BaseModel: Subclasses must implement add_arguments method")
