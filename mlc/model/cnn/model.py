import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class CNN(BaseModel):
    _name = "CNN"

    def __init__(self, args):
        super().__init__(args)

        # keep this here for clarity
        num_classes = args["num_classes"]
        num_blocks = args["num_blocks"]
        max_pool = args["max_pool"]
        batch_norm = args["batch_norm"]

        layers = []

        prev_dim = args["channels"]  # output dim of the first conv layer
        layers.append(nn.Conv2d(3, prev_dim, 5))
        for _ in range(num_blocks):
            hidden_dim = prev_dim * 2
            layers.append(nn.Conv2d(prev_dim, prev_dim, 5))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm2d(prev_dim))
            if max_pool:
                layers.append(nn.MaxPool2d(2, 2))
            layers.append(nn.Conv2d(prev_dim, hidden_dim, 5))
            layers.append(nn.ReLU())
            if batch_norm:
                layers.append(nn.BatchNorm2d(hidden_dim))

            prev_dim = hidden_dim

        layers.append(nn.Conv2d(prev_dim, prev_dim, 1))
        layers.append(nn.AvgPool2d(52, 52))
        layers.append(nn.Conv2d(prev_dim, prev_dim, 1))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(prev_dim, num_classes, 1))
        layers.append(nn.Sigmoid())
        # layers.append(nn.Linear(16 * 5 * 5, 120))
        # layers.append(nn.ReLU())
        # layers.append(nn.Linear(120, num_classes))

        self.layers = nn.Sequential(*layers)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--num-classes", type=int, default=1)
        parser.add_argument("--num-blocks", type=int, default=2)
        parser.add_argument("--max_pool", type=bool, default=True)
        parser.add_argument("--batch_norm", type=bool, default=True)
        parser.add_argument("--channels", type=int, default=16)

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
        pass


def test(args):
    # create CNN model
    model = CNN({"num_classes": 1, "num_blocks": 2, "max_pool": True, "batch_norm": True})
    # create model summary
    summary(model, input_size=(3, 250, 250), device="cpu")
