import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class cnn(BaseModel):
    _name = "cnn"

    def __init__(self, args):
        super().__init__(args)

        # keep this here for clarity
        num_channels = args["num_channels"]
        hidden_dims = args["hidden_dims"]
        last_dim = hidden_dims[-1]
        number_layer = len(hidden_dims)
        dimension_reduction = 256 / (2**number_layer)

        layers = []
        num_channels = 3  # input dimension
        for hidden_dim in hidden_dims:
            layers.append(nn.Conv2d(in_channels=num_channels, out_channels=hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
            layers.append(nn.BatchNorm2d(hidden_dim))  
            num_channels = hidden_dim

        layers.append(nn.Flatten(start_dim=1))
        layers.append(nn.Linear(in_features=int(dimension_reduction * dimension_reduction * last_dim), out_features=1))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(in_features=1024, out_features=512))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(in_features=512, out_features=1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--num-channels", type=int, default=3)
        parser.add_argument(
            "--hidden-dims",
            type=int,
            nargs="+",
            default=[64, 64, 64, 64, 64, 64, 64],
            help="List of hidden layer dimensions",
        )

    def get_optimizer(self, learning_rate, weight_decay=0.0):
        return optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def evaluate_loss(self, Y_pred, Y):
        # F.cross_entropy expects logits, not probabilities
        # return nn.BCEWithLogitsLoss(Y_pred.squeeze().float(), Y.float())
        return F.binary_cross_entropy(Y_pred.squeeze(), Y.float())

    def correct_values(self, Y_pred, Y):
        # Y_pred is a probability
        Y_pred = (Y_pred.squeeze() > 0.5).float()
        correct = (Y_pred == Y.float()).float().sum()
        return correct

    def forward(self, x):
        x_out = self.layers(x)
        return x_out

    def pre_epoch_hook(self, context):
        print("pre_epoch_hook")


def test(args):
    # create cnn model
    model = cnn({"num_channels": 3, "hidden_dims": [64, 64, 64, 64, 64, 64, 64]})
    # create model summary
    summary(model, input_size=(3, 256, 256), device="cpu")
