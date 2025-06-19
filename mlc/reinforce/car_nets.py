from torch import nn


class Modelo(nn.Module):
    def __init__(self, modo = "discrete", dim_input=4, dim_outdis=5, dim_out_cont= 3, dim_hidden=64, init_ch=3):
        super().__init__()
        if modo == "discrete":
            self.dim_out = dim_outdis
        elif modo == "continuous":
            self.dim_out = dim_out_cont
        hidden_chs = [init_ch] + [8, 16, 32, 64, 128, 256]

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
            nn.Linear(hidden_chs[-1], dim_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden),
            nn.Linear(dim_hidden, self.dim_out),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.linear_layers(x)
        if self.modo == "discrete":
            x = nn.Softmax(dim=-1)(x)
        return x

# class ModeloCont(nn.Module):
#     def __init__(self):
#         super().__init__()
