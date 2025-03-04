from torch import nn


class Spiral(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.ReLU(),
            nn.Linear(10, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.layers(x)
