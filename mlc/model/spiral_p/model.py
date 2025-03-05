from torch import nn


class SpiralParametrized(nn.Module):

    def __init__(self, hidden_dims=[100, 10], num_classes=3, dropout_rate=0.5):
        super().__init__()

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

    def forward(self, x):
        return self.layers(x)
