from torch import nn


class Modelo(nn.Module):
    def __init__(self, dim_input=4, dim_output=4, dim_hidden=32, init_ch=3):
        super().__init__()
        hidden_chs = [init_ch] + [16, 32, 64, 64, 128, 256]
        # print(f"{dim_input=}")
        # print(f"{dim_output=}")
        conv_layers = []
        for i in range(1, len(hidden_chs) - 1):
            conv_layers += [nn.Conv2d(hidden_chs[i-1], hidden_chs[i], kernel_size=3, stride=1, padding=1),
                            nn.ReLU(),            
                            nn.BatchNorm2d(hidden_chs[i]),
                            
                            # nn.Conv2d(hidden_chs[i], hidden_chs[i], kernel_size=3, stride=1, padding=1),
                            # nn.ReLU(),            
                            # nn.BatchNorm2d(hidden_chs[i]),
                            nn.MaxPool2d(kernel_size=2, stride=2)]

        self.conv_layers = nn.Sequential(
            *conv_layers,
            nn.Conv2d(hidden_chs[-2], hidden_chs[-1], kernel_size=3),
            nn.Flatten(start_dim=1))
        self.linear_layers = nn.Sequential(
            nn.Linear(hidden_chs[-1], 5),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        x = self.conv_layers(x)

        x = self.linear_layers(x)

        return x
