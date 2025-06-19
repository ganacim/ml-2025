from torch import nn


class Modelo(nn.Module):
    def __init__(self, dim_input=4, dim_output=4, dim_hidden=32, init_ch=3):
        super().__init__()
        hidden_chs = [init_ch] + [16, 32, 64, 64, 128]
        print(f"{dim_input=}")
        print(f"{dim_output=}")
        conv_layers = []
        for i, _ in enumerate(hidden_chs[:-1]):
            conv_layers += [nn.Conv2d(hidden_chs[i], hidden_chs[i + 1], kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),            
                            nn.BatchNorm2d(hidden_chs[i + 1]),
                            
                            nn.Conv2d(hidden_chs[i + 1], hidden_chs[i + 1], kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),            
                            nn.BatchNorm2d(hidden_chs[i + 1]),
                            nn.MaxPool2d(kernel_size=2, stride=2)]

        self.layers = nn.Sequential(
            *conv_layers,
            nn.Conv2d(hidden_chs[-1], hidden_chs[-1], kernel_size=3),
            nn.Flatten(start_dim=2),
            nn.Linear(hidden_chs[-1], 5),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.layers(x)
