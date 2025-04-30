import torch
from torchvision.datasets import MNIST as MNISTDataset

from ...command.base import Base
from ...util.resources import data_path


class CelebAAlignSplit(Base):
    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def name(cls):
        return "mnist.split"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("-t", "--train", type=float, help="train split", default=0.7)
        parser.add_argument("-v", "--val", type=float, help="validation split", default=0.2)
        parser.add_argument("-T", "--test", type=float, help="test split", default=0.1)
        parser.add_argument("--partial", type=float, help="partial split", default=1)

    def run(self):
        t = self.args["train"]
        v = self.args["val"]
        T = self.args["test"]
        # normalize splits
        s = t + v + T
        t /= s
        v /= s
        T /= s
        print(f"Splitting dataset into train:{t:0.02f}, validation:{v:0.02f} and test:{T:0.02f}")

        minst = MNISTDataset(
            root=data_path("mnist"),
            train=True,
            download=True,
            transform=None,
        )

        # save the index of the dataset for each fold
        data_len = len(minst.train_data)
        # better to use a seed for reproducibility
        # or even numpy for better randomization
        index = torch.randperm(data_len)

        if self.args["partial"] < 1:
            # partial split
            n = int(len(index) * self.args.partial)
            index = index[:n]

        # split files
        n = len(index)
        n_train = int(n * t)
        n_val = int(n * v)
        train_files = index[:n_train]
        val_files = index[n_train : n_train + n_val]
        test_files = index[n_train + n_val :]

        # save files
        with open(data_path("mnist") / "train.txt", "w") as f:
            f.write("\n".join([str(f.item()) for f in train_files]))

        with open(data_path("mnist") / "validation.txt", "w") as f:
            f.write("\n".join([str(f.item()) for f in val_files]))

        with open(data_path("mnist") / "test.txt", "w") as f:
            f.write("\n".join([str(f.item()) for f in test_files]))
