from torch import nn
from torchsummary import summary

class CNN(nn.Module):
    def __init__(self, dim_input=4, dim_output=4, dim_hidden=8):
        super().__init__()

        print(f"{dim_input=}")
        print(f"{dim_output=}")
        
        in_channels = 3
        num_layers = 6

        layers = []

        for i in range(num_layers):
            layers.append(nn.Conv2d(in_channels, dim_hidden, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(dim_hidden))
            layers.append(nn.MaxPool2d(kernel_size=2))
            in_channels = dim_hidden
            if i % 2 == 0:
                dim_hidden *= 2

        self.q = nn.Sequential(
            *layers,
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(dim_hidden * 26 * 20, dim_output)

    def forward(self, x):
        return self.q(x)

    def name(self):
        return "CNN"

def test(args):
    print("Testing CNN model:")

    model = CNN(dim_input=3*210*160, dim_output=4)
    print(f"Model name: {model.name()}")
    summary(model, (3, 210, 160), device="cpu")

if __name__ == "__main__":
    pass
