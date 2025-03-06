import pandas as pd
import torch

from ...util.resources import data_path
from ..basedataset import BaseDataset


class Spiral(BaseDataset):
    class DataFold(torch.utils.data.Dataset):
        def __init__(self, fold_name):
            super().__init__()
            # read csv
            csv_data = pd.read_csv(data_path("spiral") / f"{fold_name}.csv")
            # extract data
            self.X_data = csv_data[["x1", "x2"]].values.astype("float32")
            # one-hot encode the target
            self.Y_data = torch.zeros((len(csv_data), 3))
            self.Y_data[range(len(csv_data)), csv_data["y"]] = 1

        def __len__(self):
            return len(self.X_data)

        def __getitem__(self, idx):
            # return data as X, Y pair
            return (self.X_data[idx], self.Y_data[idx])

    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def name(cls):
        return "spiral"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--reshuffle", action="store_true", help="reshuffle the data")

    def get_fold(self, fold_name):
        return self.DataFold(fold_name)


# Test the Spiral dataset
if __name__ == "__main__":
    spiral = Spiral([])

    train_data = spiral.get_fold("train")
    print(f"Train data: {len(train_data)} samples")
    print(f"First sample: {train_data[0]}")
