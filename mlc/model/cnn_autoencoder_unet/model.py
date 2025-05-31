
import torch
from torch import nn
from torch.nn import functional as F
from ..basemodel import BaseModel

class CNNAutoencoderUNet(BaseModel):
    _name = "cnn_autoencoder_unet"

    def __init__(self, args):
        super().__init__(args)
        self.epoch_counter = 0
        in_channels = args.get("in_channels", 3)
        self.input_size = args.get("input_size", 256)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 128x128
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 64x64
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32x32
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 128, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(128 + 64, 64, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64 + 32, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--in_channels", type=int, default=3, help="Input image channels (e.g. 1 or 3)")
        parser.add_argument("--input_size", type=int, default=256, help="Image height/width")
        parser.add_argument("--loss",type=str, choices=["mse", "bce"], default="bce", help="Loss function to use")

    def get_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, **kwargs)

    def evaluate_loss(self, Y_pred, Y):
        return F.binary_cross_entropy(Y_pred, Y)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bottleneck(e3)
        d3 = self.dec3(torch.cat([b, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        return d1
