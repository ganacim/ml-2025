import torch
from torch import conv2d, nn
from torch.nn import Conv2d, ReLU, functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class CNN(BaseModel):
    _name = "cnn"

    def __init__(self, args):
        super().__init__(args)

        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2,padding=0),
            nn.ReLU(inplace=True),

            nn.Flatten(start_dim=1),
            nn.Linear(in_features=64,out_features=1),
            nn.Sigmoid()
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--layers",type=int,default=1)
      

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        Y=Y.float().view(-1,1)
        return F.binary_cross_entropy(Y_pred, Y)

    def forward(self, x):
        return self.layers(x)

def test(args):
    # create SpiralParameterized model
    model = CNN([])
    # create model summary
    summary(model, input_size=(3,256,256), device="cpu")


