class BaseDataset:
    """Base class for datasets"""

    _name = "base_dataset"

    def __init__(self, args):
        self._args = args

    @classmethod
    def name(cls):
        return cls._name

    def args(self):
        return self._args

    @staticmethod
    def add_arguments(parser):
        raise NotImplementedError("data.BaseDataset: Subclasses must implement add_arguments method")

    def get_fold(self, fold_name):
        raise NotImplementedError("data.BaseDataset: Subclasses must implement get_fold method")

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
