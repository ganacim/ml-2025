from torch import nn
import torch
import torch.nn.functional as F
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=0.1)  # Optional dropout layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)        
        out = self.conv2(out)                
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Modelo(nn.Module):
    def __init__(self, mode = "discrete", dim_outdis=5, dim_out_cont= 3, dim_hidden=64, init_ch=3):
        super().__init__()
        self.modo = mode
        if self.modo == "discrete":
            self.dim_out = dim_outdis
        elif self.modo == "continuous":
            self.dim_out = dim_out_cont
        hidden_chs = [init_ch] + [24, 32, 32, 64, 128, 256]

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

class ModeloDQN(nn.Module):
    def __init__(self, dim_out=5, dim_hidden=256, init_ch=3*4):
        super().__init__()
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.init_ch = init_ch
        hidden_chs = [init_ch] + [16, 32, 64, 128] #[24, 32, 32, 64, 128, 256]
        conv_layers = []
        self.c_layers = nn.Sequential((nn.Conv2d(hidden_chs[0], hidden_chs[1], kernel_size=8, stride=4),
                    ResBlock(hidden_chs[1], hidden_chs[1], stride=1),
                    nn.Conv2d(hidden_chs[1], hidden_chs[2], kernel_size=4, stride=2),
                    ResBlock(hidden_chs[2], hidden_chs[2], stride=1),
                    nn.Conv2d(hidden_chs[2], hidden_chs[3], kernel_size=3),
                    nn.ReLU(),
                    nn.BatchNorm2d(hidden_chs[3]),
                    nn.Conv2d(hidden_chs[3], hidden_chs[4], kernel_size=3),
                    nn.ReLU(),
                    nn.Flatten(start_dim=1)
                    ))
                    

        for i in range(1, len(hidden_chs) - 1):
            conv_layers += [ResBlock(hidden_chs[i-1], hidden_chs[i-1], stride=1),
                            nn.Conv2d(hidden_chs[i-1], hidden_chs[i], kernel_size=3, stride=1, padding=1),                           
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
            nn.Dropout(p=0.1),  # Optional dropout layer
            nn.ReLU(),
            nn.BatchNorm1d(dim_hidden),
            nn.Linear(dim_hidden, self.dim_out),
        )
        self.v = nn.Sequential(
            nn.Linear(hidden_chs[-1], dim_hidden),
            nn.Dropout(p=0.2),  # Optional dropout layer
            nn.ReLU(),
            #nn.BatchNorm1d(dim_hidden),
            nn.Linear(dim_hidden, 1),)
        self.adv = nn.Sequential(
            nn.Linear(hidden_chs[-1], dim_hidden),
            nn.Dropout(p=0.2),  # Optional dropout layer
            nn.ReLU(),
            #nn.BatchNorm1d(dim_hidden),
            nn.Linear(dim_hidden, self.dim_out),)

    def forward(self, x):
        x = self.c_layers(x)
        x = self.linear_layers(x)
        duelingQ = self.adv(x) + self.v(x)
        return duelingQ
