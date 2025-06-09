import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, num_convs, batchnorm, *args, **kwargs
    ):
        super().__init__()
        layers = []
        for _ in range(num_convs):
            layers.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            if batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = (
                out_channels  # after the first conv, in_channels is always out_channels
            )

        layers.append(nn.MaxPool2d(kernel_size=2))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class VGG(BaseModel):
    _name = "VGG"

    def __init__(self, args):
        self.epoch = 0
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dropout = args["dropout"]
        batchnorm = args["batchnorm"]
        super().__init__(args)
        self.layers = nn.Sequential(
            ConvBlock(3, 64, 2, dropout, batchnorm),  # 256x256
            ConvBlock(64, 128, 1, dropout, batchnorm),  # 128x128
            ConvBlock(128, 256, 1, dropout, batchnorm),  # 64x64
            ConvBlock(256, 512, 1, dropout, batchnorm),  # 32x32
            ConvBlock(512, 512, 1, dropout, batchnorm),  # 16x16
            nn.Flatten(),
            nn.Linear(in_features=512 * 8 * 8, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=1),
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--dropout",
            action="store_true",
            help="Enable dropout in fully connected layers",
        )
        parser.add_argument(
            "--batchnorm",
            action="store_true",
            help="Enable batch normalization after each conv layer",
        )

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        return F.binary_cross_entropy_with_logits(Y_pred, Y.view(-1, 1).float())

    def forward(self, x):
        return self.layers(x)

    def post_epoch_hook(self, context):
        # pq self.epoch
        self.epoch += 1
        validation_data_loader = context["validation_data_loader"]
        board = context["board"]
        acc = 0
        with torch.no_grad():
            for X, Y in validation_data_loader:
                X = X.to(self.device)
                Y = Y.to(self.device)
                Y_pred = (torch.sigmoid(self.forward(X)) > 0.5).float()
                acc += (Y == Y_pred).float().mean().item()

        board.log_scalar("accuracy", acc, self.epoch)


def test(args):
    # create SpiralParameterized model
    model = VGG()

    # create model summary
    summary(model, input_size=(3, 256, 256), device="cpu")
