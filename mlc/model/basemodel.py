import torch
from torch import nn


class BaseModel(nn.Module):
    _name = "base_model"

    def __init__(self, args):
        super().__init__()
        assert isinstance(args, dict), "args must be a dictionary"
        self._args = args

    @classmethod
    def name(cls):
        return cls._name

    @property
    def args(self):
        return self._args

    def get_optimizer(self, learning_rate, **kwargs):
        # this is the default optimizer for all models
        # define this method in subclasses to use different optimizers
        weight_decay = kwargs.get("weight_decay", 0.0)
        return torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

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

    # called before each training epoch
    def pre_train_hook(self, context):
        pass

    # called after each training epoch
    def post_train_hook(self, context):
        pass

    # called before each validation epoch
    def pre_validation_hook(self, context):
        pass

    # called after each validation epoch
    def post_validation_hook(self, context):
        pass

    # called before each batch
    def pre_train_batch_hook(self, context, X, Y):
        pass

    # called after each batch
    def post_train_batch_hook(self, context, X, Y, Y_pred, loss):
        pass

    # called before each validation batch
    def pre_validation_batch_hook(self, context, X, Y):
        pass

    # called after each batch
    def post_validation_batch_hook(self, context, X, Y, Y_pred, loss):
        pass
