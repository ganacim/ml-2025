import argparse

class Base:
    name = "Base"

    def __init__(self, args):
        self.args = args
        
    @staticmethod
    def add_arguments(parser):
        raise NotImplementedError("command.Base: Subclasses must implement add_arguments method")
    
    def run(self):
        raise NotImplementedError("command.Base: Subclasses must implement run method")
    
class Command(Base):
    name = "command"
    
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("-a", "--args", nargs=argparse.REMAINDER)
        
    def run(self):
        print("Running command")
        print(self.args)
