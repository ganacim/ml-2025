from ..util.resources import data_path
from .base import Base


class Test(Base):
    name = "test"

    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_arguments(parser):
        pass

    def run(self):
        print("Running Test command")
        print("data path: ", data_path())
        for p in data_path().iterdir():
            print(p)
