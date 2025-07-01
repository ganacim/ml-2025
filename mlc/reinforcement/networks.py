from torch import nn
from .ResNet import ResNet, Bottleneck

class MLP(nn.Module):
    def __init__(self, dim_input=4, dim_output=4, dim_hidden=32):
        super().__init__()

        print(f"{dim_input=}")
        print(f"{dim_output=}")

        self.q = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_output),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.q(x)

class MLPBCE(nn.Module):
    def __init__(self, dim_input=4, dim_output=4, dim_hidden=8):
        super().__init__()

        print(f"{dim_input=}")
        print(f"{dim_output=}")

        self.q = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, x):
        return self.q(x)
    
class CNN(nn.Module):
    def __init__(self, dim_input=4, dim_output=4):
        super().__init__()
        self.q = ResNet(
            Bottleneck,
            [3,4,6,3],
            num_channels=dim_input,
            num_classes=dim_output,
        )
        
    def forward(self, x):
        return self.q(x)