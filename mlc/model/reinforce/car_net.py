from torch import nn


class CarNet(nn.Module):
    def __init__(self, dim_input=4, dim_output=4, dim_hidden=32):
        super().__init__()

        print(f"{dim_input=}")
        print(f"{dim_output=}")

        self.layers = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_output),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.layers(x)
