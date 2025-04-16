import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class CNN(BaseModel):
    _name = "CNN"

    def __init__(self, kwargs):
        super().__init__(kwargs)

        # keep this here for clarity
        num_classes = 1  # kwargs.get("num_classes", 1)
        hidden_chs = kwargs.get("hidden_chs", [6, 6, 12, 12, 64, 64, 128, 128])
        kernel_size = kwargs.get("kernel_size", 3)
        stride = kwargs.get("stride", 1)
        padding = kwargs.get("padding", 1)
        dropout_rate = kwargs.get("dropout_rate", 0.0)

        layers = []
        prev_ch = 3  # input channels (RGB)
        for hidden_ch in hidden_chs:
            layers.append(nn.Conv2d(prev_ch, hidden_ch, kernel_size=kernel_size, stride=stride, padding=padding))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm2d(hidden_ch))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

            prev_ch = hidden_ch

        layers.append(nn.Flatten())
        layers.append(nn.Linear(prev_ch, num_classes))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--num-classes", type=int, default=1)
        parser.add_argument(
            "--hidden-chs",
            type=int,
            nargs="+",
            default=[6, 6, 12, 12, 64, 64, 128, 128],
            help="List of hidden layer channels",
        )
        parser.add_argument("--dropout-rate", type=float, default=0.0)  # Re-enable dropout rate argument
        parser.add_argument("--kernel-size", type=int, default=3, help="Kernel size for convolutional layers")
        parser.add_argument("--stride", type=int, default=1, help="Stride for convolutional layers")
        parser.add_argument("--padding", type=int, default=1, help="Padding for convolutional layers")

    def get_optimizer(self, learning_rate):
        return optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        # F.cross_entropy expects logits, not probabilities
        Y_pred = Y_pred.float().view(-1)
        Y = Y.float().view(-1)
        return F.binary_cross_entropy(Y_pred, Y)

    def forward(self, x):
        return self.layers(x)

    def pre_epoch_hook(self, context):
        print("pre_epoch_hook")


def test(kwargs):
    # create CNN model
    model = CNN(
        {
            "num_classes": kwargs.get("num_classes", 1),
            "hidden_chs": kwargs.get("hidden_chs", [6, 6, 12, 12, 64, 64, 128, 128]),
            "dropout_rate": kwargs.get("dropout_rate", 0.0),
        }
    )
    # create model summary
    summary(model, input_size=(3, 256, 256), device="cpu")
