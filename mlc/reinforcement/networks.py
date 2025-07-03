from torch import nn

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
    def __init__(self, nframes=4, im_sz=84, nactions = 3):
        super().__init__()

        sz = (((im_sz-8) / 4) + 1 - 4) / 2 + 1
        self.q = nn.Sequential(
        nn.Conv2d(nframes, 16, 8, 4),
        nn.ReLU(),
        nn.Conv2d(16, 32, 4, 2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(sz*sz*32, 128),
        nn.ReLU(),
        nn.Linear(128, nactions),
        #nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.q(x)
