import argparse

import torch
from torchvision.datasets import FashionMNIST as FashionMNISTDataset
from torchvision.transforms import v2

from ...util.resources import data_path
from ..basedataset import BaseDataset


class FashionMNIST(BaseDataset):
    class DataFold(torch.utils.data.Dataset):
        def __init__(self, fold_name, mnist, noise=0, normalize=False):
            super().__init__()
            self._data_path = data_path("fashion_mnist")
            self._mnist = mnist
            # read files
            fold_index = self._data_path / f"{fold_name}.txt"
            with open(fold_index, "r") as f:
                self.idx = torch.tensor(list(map(int, f.read().splitlines())))

            self.xform = v2.Compose(
                [
                    v2.PILToTensor(),
                    v2.Lambda(lambda x: x.unsqueeze(0) if x.ndim == 2 else x),  # Ensure shape [1, H, W]
                    v2.ToDtype(torch.float32, scale=True),  # to [0, 1]
                    (
                        v2.Normalize(
                            mean=[0.5],  # FashionMNIST is grayscale
                            std=[0.5],
                            inplace=True,
                        )
                        if normalize
                        else v2.Identity()
                    ),  # to [-1, 1]
                ]
            )
            self.augmentation = [v2.Identity()]
            if noise > 0:
                self.augmentation.append(v2.Lambda(lambda x: x + torch.randn_like(x) * noise))
            self.augmentation = v2.Compose(self.augmentation)

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, idx):
            # open image with PIL
            img = self._mnist.data[self.idx[idx]]
            img = self.xform(img)
            return (self.augmentation(img), img)

        def get_label(self, idx):
            # open image with PIL
            label = self._mnist.targets[self.idx[idx]]
            return label

    def __init__(self, args):
        super().__init__(args)

        self._mnist = FashionMNISTDataset(
            root=data_path("fashion_mnist"),
            train=True,
            download=True,
            transform=None,
        )

    @classmethod
    def name(cls):
        return "fashion_mnist"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--noise", type=float, default=0, help="Add noise to the input")
        parser.add_argument("--normalize", action="store_true", help="Normalize the input")
        parser.set_defaults(normalize=False)

    def get_fold(self, fold_name):
        return self.DataFold(fold_name, self._mnist, self.args["noise"], self.args["normalize"])


def test(cmd_args):
    print("Testing FashionMNIST dataset:", cmd_args)
    parser = argparse.ArgumentParser()
    FashionMNIST.add_arguments(parser)
    args = parser.parse_args(cmd_args)

    mnist = FashionMNIST(vars(args))
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
