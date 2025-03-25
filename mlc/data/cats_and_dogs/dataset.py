import numpy as np
import torch

from ...util.resources import data_path
from ..basedataset import BaseDataset


class CatsAndDogs(BaseDataset):
    class DataFold(torch.utils.data.Dataset):
        def __init__(self, fold_name):
            super().__init__()
            self._data_path = data_path("cats_and_dogs")
            # read files
            fold_index = data_path("cats_and_dogs") / f"{fold_name}.txt"
            with open(fold_index, "r") as f:
                self.files = f.read().splitlines()
            # compute labels
            self.labels = [0 if "Cat" in f else 1 for f in self.files]

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            return (self.files[idx], self.labels[idx])

    def __init__(self, args):
        super().__init__(args)

        # read folder with images

    @classmethod
    def name(cls):
        return "cats_and_dogs"

    @staticmethod
    def add_arguments(parser):
        pass

    def get_fold(self, fold_name):
        return self.DataFold(fold_name)


def test(args):
    print("Testing CatsAndDogs dataset:", args)

    cats_and_dogs = CatsAndDogs(args)
    print(f"Dataset name: {cats_and_dogs.name()}")

    # get fold
    data = cats_and_dogs.get_fold("train")
    n_dogs = np.sum(np.array(data.labels))
    n_cats = len(data) - n_dogs
    # find number of cats and dogs
    print(f"Train data: {len(data)} samples[{n_cats} cats, {n_dogs} dogs]")

    data = cats_and_dogs.get_fold("validation")
    n_dogs = np.sum(np.array(data.labels))
    n_cats = len(data) - n_dogs
    print(f"Validation data: {len(data)} samples[{n_cats} cats, {n_dogs} dogs]")

    data = cats_and_dogs.get_fold("test")
    n_dogs = np.sum(np.array(data.labels))
    n_cats = len(data) - n_dogs
    print(f"Test data: {len(data)} samples[{n_cats} cats, {n_dogs} dogs]")

    # show some test data
    for i in range(5):
        print(data[i])


# Test the Spiral dataset
if __name__ == "__main__":
    pass
