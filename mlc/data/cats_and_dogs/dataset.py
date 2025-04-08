import pandas as pd
import torch
import torchvision
from glob import glob
import random
import numpy as np

from mlc.util.resources import data_path
from mlc.data.basedataset import BaseDataset

random.seed(10)

class CatsAndDogs(BaseDataset):
    class DataFold(torch.utils.data.Dataset):
        def __init__(self, fold_name):
            super().__init__()
            # read csv
            pets_folder_path = data_path("PetImages")

            all_pets_path_list = glob(str(pets_folder_path / "**/*.jpg"))
            self.pets_split = self.split_data(all_pets_path_list, fold_name)

        def split_data(self, data:list, fold_name:str) -> list:
            # shuffle data
            random.shuffle(data)
            train_proportion = 0.7
            val_proportion = 0.15

            train_size = int(train_proportion * len(data))
            val_size = int(val_proportion * len(data))
            train = data[:train_size]
            val = data[train_size:train_size + val_size]
            test = data[train_size + val_size:]
            if fold_name == "train":
                return train
            elif fold_name == "val":
                return val
            elif fold_name == "test":
                return test
            else:
                raise ValueError(f"{fold_name} invalid.")

        def __len__(self):
            return len(self.pets_split)

        def __getitem__(self, idx):
            image_path = self.pets_split[idx]
            is_cat_bool = 'Cat' in image_path

            # Load Image
            image = torchvision.io.read_image(image_path)
            # Resize Image
            image = torchvision.transforms.Resize((224, 224))(image)
            image = image.float() / 255.0
            if is_cat_bool:
                target = torch.tensor([0., 1.], dtype=torch.float32)
            else:
                target = torch.tensor([1., 0.], dtype=torch.float32)

            return image, target



    def __init__(self, args):
        super().__init__(args)

    @classmethod
    def name(cls):
        return "cats_and_dogs"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--reshuffle", action="store_true", help="reshuffle the data")

    def get_fold(self, fold_name):
        return self.DataFold(fold_name)



def test(args):
    # create CatsAndDogs dataset with no arguments
    cats_and_dogs = CatsAndDogs([])

    # get train data
    train_data = cats_and_dogs.get_fold("train")
    print(f"Train data: {len(train_data)} samples")
    print(f"First sample: {train_data[0]}")


# Test the CatsAndDogs dataset
if __name__ == "__main__":
    test(args=[])
