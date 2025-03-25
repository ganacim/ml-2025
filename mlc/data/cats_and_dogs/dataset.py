import torch

from ..basedataset import BaseDataset


class CatsAndDogs(BaseDataset):
    class DataFold(torch.utils.data.Dataset):
        def __init__(self, fold_name):
            super().__init__()

        def __len__(self):
            return 0

        def __getitem__(self, idx):
            return None

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

    # # get train data
    # train_data = spiral.get_fold("train")
    # print(f"Train data: {len(train_data)} samples")
    # print(f"First sample: {train_data[0]}")


# Test the Spiral dataset
if __name__ == "__main__":
    pass
