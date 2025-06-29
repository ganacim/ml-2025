# networks_luna.py

from torch import nn

class MLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron that can be used for various
    reinforcement learning tasks. The dimensions are passed in during
    instantiation, making it reusable.
    """
    def __init__(self, dim_input=4, dim_output=4, dim_hidden=32):
        super().__init__()

        print(f"Initializing MLP with:")
        print(f"  Input Dimension:  {dim_input}")
        print(f"  Hidden Dimension: {dim_hidden}")
        print(f"  Output Dimension: {dim_output}")


        self.q = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.LeakyReLU(),
            nn.Linear(dim_hidden, dim_output),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        """Defines the forward pass of the model."""
        return self.q(x)

class MLPBCE(nn.Module):
    """
    An alternative MLP that outputs raw logits. This is useful for loss
    functions like nn.CrossEntropyLoss that have a softmax built-in.
    This is not used by the current train_luna.py script but is kept
    here for completeness.
    """
    def __init__(self, dim_input=4, dim_output=4, dim_hidden=8):
        super().__init__()

        print(f"{dim_input=}")
        print(f"{dim_output=}")

        self.q = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, x):
        return self.q(x)