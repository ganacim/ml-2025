import argparse

import torch
from PIL import Image
from torchvision.transforms import v2

from ...util.resources import data_path
from ..basedataset import BaseDataset


class CelebaAlign(BaseDataset):
    class DataFold(torch.utils.data.Dataset):
        def __init__(self, fold_name, scale=256):
            super().__init__()
            self.scale = scale
            self._data_path = data_path("celeba_align")
            # read files
            fold_index = self._data_path / f"{fold_name}.txt"
            with open(fold_index, "r") as f:
                self.files = f.read().splitlines()

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
            xfimg = self.xform(img)
            return (xfimg, xfimg)

    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def name(cls):
        return "celeba_align"

    @staticmethod
    def add_arguments(parser):
        # add rescale argument
        parser.add_argument("-s", "--scale", type=int, help="rescale image size", default=64)

    def get_fold(self, fold_name):
        return self.DataFold(fold_name, scale=self.args["scale"])


def test(cmd_args):
    print("Testing Celeba dataset:", cmd_args)
    parser = argparse.ArgumentParser()
    CelebaAlign.add_arguments(parser)
    args = parser.parse_args(cmd_args)

    celeba = CelebaAlign(vars(args))
    print(f"Dataset name: {celeba.name()}")

    # get fold
    data = celeba.get_fold("train")

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
