import torch
import torch.nn.functional as F
import torch.optim as optim
import math

from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class vgg_inspired_smaller(BaseModel):
    _name = "vgg_inspired_smaller"
    def __init__(self, args):
        super().__init__(args)
        self.epoch = 0

        hidden_conv_dims = args["hidden_conv_dims"]
        hidden_dims = args["hidden_dims"]
        dropout_rate = args["dropout_rate"]

        input_shape = 256
        #adding convolution units
        prev_conv_dim = 3 #RGB
        layers = [
            nn.BatchNorm2d(prev_conv_dim)
        ]
        for hidden_conv_dim in hidden_conv_dims:
            layers += [
                nn.Conv2d(
                    in_channels=prev_conv_dim, 
                    out_channels=hidden_conv_dim, 
                    kernel_size=3,
                    padding=1
                    ),
                nn.BatchNorm2d(hidden_conv_dim),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
            ]
            prev_conv_dim = hidden_conv_dim
        
        #flattening and adding FC units
        layers.append(nn.Flatten( start_dim=1, end_dim=-1))
        prev_dim = int(prev_conv_dim * (input_shape / (2**len(hidden_conv_dims))) ** 2)

        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ]
            prev_dim = hidden_dim

        layers += [
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ]

        self.layers = nn.Sequential(*layers)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--hidden-conv-dims", type=int, default=[64, 128, 256, 256, 256, 256], help="List of Conv layer dimensions")
        parser.add_argument("--hidden-dims", type=int, default=[256], help="List of FC layer dimensions")
        parser.add_argument("--dropout-rate", type=float, default=0.35)

    def get_optimizer(self, learning_rate, weight_decay):
        return optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def evaluate_loss(self, Y_pred, Y):
        # F.cross_entropy expects logits, not probabilities
        Y = Y.float().view(-1, 1)
        return F.binary_cross_entropy(Y_pred, Y)

    def forward(self, x):
        return self.layers(x)

    def pre_epoch_hook(self, context):
        val_data_loader = context["validation_data_loader"]
        board = context["board"]
        acc = 0
        with torch.no_grad():
            for X, Y in val_data_loader:
                X = X.to("cuda")
                Y = Y.to("cuda")
                Y_pred = self.forward(X)
                Y_pred = torch.round(Y_pred).squeeze()
                acc += torch.sum(Y_pred == Y).item()

        acc /= len(val_data_loader.dataset)
        board.log_scalars("Accuracy", {"Val": acc}, self.epoch)
        self.epoch += 1


def test(args):
    model = vgg_inspired_smaller({"hidden_conv_dims": [64, 128, 256, 256, 256, 256], "hidden_dims": [256], "dropout_rate": 0.5})
    # create model summary
    summary(model, input_size=(3, 256, 256), device="cpu")
