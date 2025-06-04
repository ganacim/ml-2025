import random

from ...command.base import Base
from ...util.resources import data_path


class CatsAndDogsSplit(Base):
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("-t", "--train", type=float, help="train split", default=0.7)
        parser.add_argument("-v", "--val", type=float, help="validation split", default=0.2)
        parser.add_argument("-T", "--test", type=float, help="test split", default=0.1)

    @classmethod
    def name(cls):
        return "cats_and_dogs.split"

    def run(self):
        t = self.args['train']
        v = self.args['val']
        T = self.args['test']
        # normalize splits
        s = t + v + T
        t /= s
        v /= s
        T /= s
        print(f"Splitting dataset into train:{t:0.02f}, validation:{v:0.02f} and test:{T:0.02f}")

        cats_folder = data_path("cats_and_dogs") / "Cat"
        cats_files = list(cats_folder.glob("*.jpg"))
        dogs_folder = data_path("cats_and_dogs") / "Dog"
        dogs_files = list(dogs_folder.glob("*.jpg"))

        # merge lists and shuffle
        files = cats_files + dogs_files
        # better to use a seed for reproducibility
        # or even numpy for better randomization
        for i in range(5):
            random.shuffle(files)

        # make paths relative to dataset folder
        files = [f.relative_to(data_path("cats_and_dogs")) for f in files]

        # split files
        n = len(files)
        n_train = int(n * t)
        n_val = int(n * v)
        train_files = files[:n_train]
        val_files = files[n_train : n_train + n_val]
        test_files = files[n_train + n_val :]

        # save files
        with open(data_path("cats_and_dogs") / "train.txt", "w") as f:
            f.write("\n".join([str(f) for f in train_files]))

        with open(data_path("cats_and_dogs") / "validation.txt", "w") as f:
            f.write("\n".join([str(f) for f in val_files]))

        with open(data_path("cats_and_dogs") / "test.txt", "w") as f:
            f.write("\n".join([str(f) for f in test_files]))
