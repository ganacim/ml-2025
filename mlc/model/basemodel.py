import torch
from torch import nn


class BaseModel(nn.Module):
    _name = "base_model"

    def __init__(self, args):
        super().__init__()
        self._args = args

    @classmethod
    def name(cls):
        return cls._name

    def args(self):
        return self._args

    def get_optimizer(self, learning_rate):
        # this is the default optimizer for all models
        # define this method in subclasses to use different optimizers
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        raise NotImplementedError("BaseModel: Subclasses must implement evaluate_loss method")

    @staticmethod
    def add_arguments(parser):
        raise NotImplementedError("BaseModel: Subclasses must implement add_arguments method")

    # called  before each epoch
    def pre_epoch_hook(self, context):
        pass

    # called after each epoch
    def post_epoch_hook(self, context):
        pass

    # called before each batch
    def pre_batch_hook(self, context, X, Y):
        pass

    # called after each batch
    def post_batch_hook(self, context, X, Y, Y_pred, loss):
        pass
