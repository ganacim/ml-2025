from torch import nn
import torch

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

    def __init__(self, obs_shape: tuple[int, int, int], dim_output: int):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, *obs_shape)).view(1, -1).shape[1]
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flat, 512), nn.ReLU(),
            nn.Linear(512, dim_output),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor):
        return self.fc(self.conv(x))
