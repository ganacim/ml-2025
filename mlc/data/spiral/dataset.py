import pandas as pd
import torch
from torch.utils.data import Dataset

from mlc.util.resources import data_path


class Spiral(Dataset):
    def __init__(self, fold_name):
        super().__init__()
        # read csv
        csv_data = pd.read_csv(data_path("spiral") / f"{fold_name}.csv")
        # extract data
        self.data = csv_data.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return data as X, Y pair
        return torch.tensor(self.data[idx, :-1]), torch.tensor(self.data[idx, -1], dtype=torch.int32)


# Test the Spiral dataset
if __name__ == "__main__":
    spiral = Spiral("train")

    print(spiral[0])
    print(spiral[1:10])
