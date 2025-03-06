from torch import nn

from ..basemodel import BaseModel


class SpiralParameterized(BaseModel):
    _name = "spiral_parameterized"

    def __init__(self, args):
        super().__init__(args)

        # keep this here for clarity
        num_classes = args.num_classes
        hidden_dims = args.hidden_dims
        dropout_rate = args.dropout_rate

        layers = []
        prev_dim = 2  # input dimension
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        # keep softmax here for now, but we might want to remove it
        layers.append(nn.Softmax(dim=1))
        self.layers = nn.Sequential(*layers)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--num-classes", type=int, default=3)
        parser.add_argument(
            "--hidden-dims", type=int, nargs="+", default=[100, 10], help="List of hidden layer dimensions"
        )
        parser.add_argument("--dropout-rate", type=float, default=0.0)

    def forward(self, x):
        return self.layers(x)
