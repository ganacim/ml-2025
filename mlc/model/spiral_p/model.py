import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class SpiralParameterized(BaseModel):
    _name = "spiral_parameterized"

    def __init__(self, args):
        super().__init__(args)

        # keep this here for clarity
        num_classes = args["num_classes"]
        hidden_dims = args["hidden_dims"]
        dropout_rate = args["dropout_rate"]

        layers = []
        prev_dim = 2  # input dimension
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--num-classes", type=int, default=3)
        parser.add_argument(
            "--hidden-dims", type=int, nargs="+", default=[100, 10], help="List of hidden layer dimensions"
        )
        parser.add_argument("--dropout-rate", type=float, default=0.0)

    def get_optimizer(self, learning_rate):
        return optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        # F.cross_entropy expects logits, not probabilities
        return F.cross_entropy(Y_pred, Y)

    def forward(self, x):
        return self.layers(x)

    def pre_validation_hook(self, context):
        # reset accuracy for each validation epoch
        self.accuracy = 0.0

    def post_validation_batch_hook(self, context, X, Y, Y_pred, loss):
        # compute accuracy for each batch and accumulate
        self.accuracy += (Y_pred.argmax(dim=1) == Y.argmax(dim=1)).float().sum().item()

    def post_validation_hook(self, context):
        # compute accuracy
        self.accuracy /= len(context["validation_data_loader"])
        context["board"].log_scalar("Curves/Accuracy", self.accuracy, context["epoch"])


def test(args):
    # create SpiralParameterized model
    model = SpiralParameterized({"num_classes": 3, "hidden_dims": [100, 10], "dropout_rate": 0.0})
    # create model summary
    summary(model, input_size=(2,), device="cpu")
