from torch import nn

class DQLNN(nn.Module):
    def __init__(self, dim_input=4, dim_output=4, dim_hidden=32):
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
