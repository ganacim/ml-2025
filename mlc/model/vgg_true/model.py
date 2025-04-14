import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class VGG_p(BaseModel):
    _name = "vgg_paper"

    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=64, out_features=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--layers", type=int, default=1)
        parser.add_argument("--dropout-rate", type=float, default=0.0)

    def get_optimizer(self, learning_rate):
        return optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        # F.cross_entropy expects logits, not probabilities
        Y = Y.float().view(-1, 1)
        return F.binary_cross_entropy(Y_pred, Y)

    def forward(self, x):
        return self.layers(x)

    def pre_epoch_hook(self, context):
        print("pre_epoch_hook")


def test(args):
    # create SpiralParameterized model
    model = VGG_p([])
    # create model summary
    summary(model, input_size=(3, 256, 256), device="cpu")
