import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class cnn_mod(BaseModel):
    _name = "cnn_mod"

    def __init__(self, args):
        super().__init__(args)
        self.layers = nn.Sequential(
            
            nn.BatchNorm2d(3),

            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),


            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
       

            nn.Flatten(start_dim=1, end_dim = -1),
            nn.Linear(in_features=256*4*4, out_features=16*4*4),
            nn.BatchNorm1d(16*4*4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),

            nn.Linear(in_features=16*4*4, out_features=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--layers", type=int, default=1)
        parser.add_argument("--dropout-rate", type=float, default=0.0)

    def get_optimizer(self, learning_rate):
        return optim.Adam(self.parameters(), lr=learning_rate, weight_decay = 1e-4)

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
    model = cnn_mod([])
    # create model summary
    summary(model, input_size=(3, 256, 256), device="cpu")
