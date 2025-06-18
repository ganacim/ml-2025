from torch import nn


class Modelo(nn.Module):
    def __init__(self, dim_input=4, dim_output=4, dim_hidden=32, init_ch=3):
        super().__init__()
        hidden_chs = [init_ch, 32, 64]
        print(f"{dim_input=}")
        print(f"{dim_output=}")
        layers = []
        for i, _ in enumerate(hidden_chs):
            layers.append(nn.Conv2d(hidden_chs[i], hidden_chs[i + 1], kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            if i < len(hidden_chs) - 1:
                layers.append(nn.BatchNorm2d(hidden_chs[i + 1]))

        self.layers = nn.Sequential(
            nn.Conv2d(hidden_chs[0], hidden_chs[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Batchnorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Linear(dim_input, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_output),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.layers(x)
