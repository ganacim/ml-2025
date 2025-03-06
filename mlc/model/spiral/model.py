from torch import nn

from ..basemodel import BaseModel


class Spiral(BaseModel):
    _name = "spiral"

    def __init__(self, args):
        super().__init__(args)

        num_classes = 3
        self.layers = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes),
            nn.Softmax(dim=1),
        )

    @staticmethod
    def add_arguments(parser):
        pass

    def forward(self, x):
        return self.layers(x)
