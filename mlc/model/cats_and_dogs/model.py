import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchsummary import summary

from mlc.model.basemodel import BaseModel


class CatsAndDogs(BaseModel):
    _name = "cats_and_dogs"

    def __init__(self, args):
        super().__init__(args)

        # keep this here for clarity
        num_classes = args["num_classes"]

        conv_layers = []
        input_num_channels = 3 
        self.conv1 = nn.Conv2d(input_num_channels, 2*input_num_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool_1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(2*input_num_channels, 4*input_num_channels, kernel_size=3, stride=1, padding=1)
        self.max_pool_1 = nn.MaxPool2d(2,2)



        self.conv_layers = nn.Sequential(*conv_layers)

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
        x_out = self.conv1(x)
        x_out = self.max_pool_1(x_out)
        x_out = self.conv2(x_out)
        x_out = self.max_pool_1(x_out)
        x_out = x_out.view(x_out.size(0), -1)
        x_out = nn.Linear(x_out.size(1), 2)(x_out)
        return x_out



    def pre_epoch_hook(self, context):
        print("pre_epoch_hook")


def test(args):
    # create CatsAndDogs model
    model = CatsAndDogs({"num_classes": 2})
    # create model summary
    summary(model, input_size=(3,224,224), device="cpu")

if __name__ == "__main__":
    # Test the model
    test({})
