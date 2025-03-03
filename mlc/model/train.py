from ..command.base import Base


class Train(Base):
    name = "model.train"

    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("-e", "--epochs", type=int, required=True)

    def run(self):
        print("Running train command")
