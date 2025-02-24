from .base import Base

class Help(Base):
    name = "help"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("command", type=str, nargs="?")
        
    def run(self):
        print("Running help command")
        print(self.args)