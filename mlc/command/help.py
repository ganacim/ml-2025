from .base import Base


class Help(Base):
    @staticmethod
    def add_arguments(parser):
        parser.add_argument("command", type=str, nargs="?")

    @classmethod
    def name(cls):
        return "help"

    def run(self):
        print("Running help command")
        print(self.args)
