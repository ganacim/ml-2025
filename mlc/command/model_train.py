from .base import Base

class Train(Base):
    name = "model_train"
    
    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("-e", "--epochs", type=int)
        
    def run(self):
        print("Running train command")