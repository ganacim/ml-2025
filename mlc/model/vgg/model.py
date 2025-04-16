import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class VGG(BaseModel):
    _name = "VGG"

    def __init__(self, *args, **kwargs):
        self.epoch = 0
        super().__init__(args)
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),  # 128x128
            #
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),  # 64x64
            #
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),  # 32 x 32
            #
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),  # 16 x 16
            #
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2),  # 8x8
            nn.Flatten(),
            nn.Linear(in_features=512 * 8 * 8, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=1),
        )

    @staticmethod
    def add_arguments(parser):
        pass
        # parser.add_argument("--threshold", type=float, default=0.5)

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        return F.binary_cross_entropy_with_logits(Y_pred, Y.view(-1, 1).float())

    def forward(self, x):
        return self.layers(x)

    def post_epoch_hook(self, context):
        # pq self.epoch
        self.epoch += 1
        validation_data_loader = context["validation_data_loader"]
        board = context["board"]
        acc = 0
        with torch.no_grad():
            for X, Y in validation_data_loader:
                X = X.to("cuda")
                Y = Y.to("cuda")
                Y_pred = (self.forward(X) > 0.5).float()
                acc += torch.abs(Y - Y_pred).mean().item()

        board.log_scalar("accuracy", acc, self.epoch)


def test(args):
    # create SpiralParameterized model
    model = VGG()

    # create model summary
    summary(model, input_size=(3, 256, 256), device="cpu")
