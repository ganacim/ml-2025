import torch
from torch import conv2d, nn
from torch.nn import Conv2d, ReLU, functional as F
from torchsummary import summary
import torch.nn.init as init 

from ..basemodel import BaseModel


class CNN(BaseModel):
    _name = "cnn"

    def __init__(self, args):
        super().__init__(args)
        self.epoch = 0

        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2,padding=0),
            nn.ReLU(inplace=True),

            nn.Flatten(start_dim=1),
            nn.Linear(in_features=64,out_features=1),
            nn.Sigmoid()
        )

        # He initialization to Conv2d and Linear layers
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)

    @staticmethod
    def add_arguments(parser):
        #parser.add_argument("--layers",type=int,default=1)
        pass

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        Y=Y.float().view(-1,1)
        return F.binary_cross_entropy(Y_pred, Y)

    def forward(self, x):
        return self.layers(x)
    
    def post_epoch_hook(self, context):
        self.epoch += 1
        val_data_loader = context["validation_data_loader"]
        board = context["board"]
        acc = 0
        with torch.no_grad():
            for X, Y in val_data_loader:
                X = X.to("cuda")
                Y = Y.to("cuda")
                Y_pred = self.forward(X)
                predictions = (Y_pred >= 0.5).float()
                acc += torch.sum(predictions == Y).item()
        acc = acc / len(val_data_loader.dataset)
        board.log_scalar("accuracy", acc, self.epoch)


def test(args):
    model = CNN([])
    # create model summary
    summary(model, input_size=(3,256,256), device="cpu")
    print("test")