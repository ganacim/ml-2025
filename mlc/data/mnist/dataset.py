import argparse

import torch
from torchvision.datasets import MNIST as MNISTDataset
from torchvision.transforms import v2

from ...util.resources import data_path
from ..basedataset import BaseDataset


class MNIST(BaseDataset):
    class DataFold(torch.utils.data.Dataset):
        def __init__(self, fold_name, mnist):
            super().__init__()
            self._data_path = data_path("mnist")
            self._mnist = mnist
            # read files
            fold_index = self._data_path / f"{fold_name}.txt"
            with open(fold_index, "r") as f:
                self.idx = torch.tensor(list(map(int, f.read().splitlines())))

            self.xform = v2.Compose(
                [
                    v2.PILToTensor(),
                    v2.ToDtype(torch.float32, scale=True),  # to [0, 1]
                ]
            )

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, idx):
            # open image with PIL
            img = self._mnist.data[self.idx[idx]]
            img = self.xform(img)
            return (img, img)

        def get_label(self, idx):
            # open image with PIL
            label = self._mnist.targets[self.idx[idx]]
            return label

    def __init__(self, args):
        super().__init__(args)

        self._mnist = MNISTDataset(
            root=data_path("mnist"),
            train=True,
            download=True,
            transform=None,
        )

    @classmethod
    def name(cls):
        return "mnist"

    @staticmethod
    def add_arguments(parser):
        pass

    def get_fold(self, fold_name):
        return self.DataFold(fold_name, self._mnist)


def test(cmd_args):
    print("Testing MNIST dataset:", cmd_args)
    parser = argparse.ArgumentParser()
    MNIST.add_arguments(parser)
    args = parser.parse_args(cmd_args)

    mnist = MNIST(vars(args))
    print(f"Dataset name: {mnist.name()}")

    # get fold
    data = mnist.get_fold("train")

    img, _ = data[0]
    print(f"Dataset length: {len(data)}")
    print(f"Image shape: {img.shape}")
    print(f"Image dtype: {img.dtype}")
    print(f"Image min: {img.min()}")

    # index = data_path("cats_and_dogs") / "train.txt"
    # with open(index, "r") as f:
    #     files = f.read().splitlines()
    # # open image with PIL
    # for file in files:
    #     print(file)
    #     img = Image.open(data_path("cats_and_dogs") / file).convert("RGB")
    #     img.load()


# Test the Spiral dataset
if __name__ == "__main__":
    pass
