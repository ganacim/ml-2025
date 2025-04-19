import argparse

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from ...util.resources import data_path
from ..basedataset import BaseDataset


class CatsAndDogs(BaseDataset):
    class DataFold(torch.utils.data.Dataset):
        def __init__(self, fold_name, scale=256):
            super().__init__()
            self.scale = scale
            self._data_path = data_path("cats_and_dogs")
            # read files
            fold_index = data_path("cats_and_dogs") / f"{fold_name}.txt"
            with open(fold_index, "r") as f:
                self.files = f.read().splitlines()
            # compute labels
            self.labels = [0 if "Cat" in f else 1 for f in self.files]

            if fold_name == "train":
                self.xform = v2.Compose(
                    [
                        v2.PILToTensor(),
                        v2.ToDtype(torch.float32, scale=True),  # to [0, 1]
                        v2.Resize((self.scale, self.scale)),
                        v2.RandomHorizontalFlip(p=0.5),
                        v2.RandomRotation((-45, 45)),
                        # v2.GaussianNoise(0,0.01)
                    ]
                )
            else:
                self.xform = v2.Compose(
                    [
                        v2.PILToTensor(),
                        v2.ToDtype(torch.float32, scale=True),  # to [0, 1]
                        v2.Resize((self.scale, self.scale)),
                    ]
                )

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            # open image with PIL
            img = Image.open(self._data_path / self.files[idx]).convert("RGB")
            # convert to tensor
            return (self.xform(img), self.labels[idx])

    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def name(cls):
        return "cats_and_dogs"

    @staticmethod
    def add_arguments(parser):
        # add rescale argument
        parser.add_argument("-s", "--scale", type=int, help="rescale image size", default=256)

    def get_fold(self, fold_name):
        return self.DataFold(fold_name, scale=self.args()["scale"])


def test(cmd_args):
    print("Testing CatsAndDogs dataset:", cmd_args)
    parser = argparse.ArgumentParser()
    CatsAndDogs.add_arguments(parser)
    args = parser.parse_args(cmd_args)

    cats_and_dogs = CatsAndDogs(vars(args))
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
    for i in range(1):
        print(data[i])
        print(data[i][0].shape)

    index = data_path("cats_and_dogs") / "train.txt"
    with open(index, "r") as f:
        files = f.read().splitlines()
    # open image with PIL
    for file in files:
        print(file)
        img = Image.open(data_path("cats_and_dogs") / file).convert("RGB")
        img.load()


# Test the Spiral dataset
if __name__ == "__main__":
    pass
