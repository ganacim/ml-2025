from torch import nn


class Spiral(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.fc1 = nn.Linear(2, 10)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(10, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
