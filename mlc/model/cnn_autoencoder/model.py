import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class CNNAutoencoder(BaseModel):
    _name = "cnn_autoencoder"

    def __init__(self, args):
        super().__init__(args)

        self.epoch_counter = 0
        in_channels = args.get("in_channels", 3)  
        self.input_size = args.get("input_size", 256)
        self.loss = args.get("loss", "bce")

        # ENCODER
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 128x128

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(4)  # 2x2x32
        )

        # DECODER
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=4),  # 8x8
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),  # 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 32x32
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  # 64x64
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),  # 128x128
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(16, in_channels, kernel_size=2, stride=2),  # 256x256
            nn.Sigmoid()  # assuming input is normalized to [0, 1]
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--in_channels", type=int, default=3, help="Input image channels (e.g. 1 or 3)")
        parser.add_argument("--input_size", type=int, default=256, help="Image height/width")
        parser.add_argument("--loss",type=str, choices=["mse", "bce"], default="bce", help="Loss function to use")

    def get_optimizer(self, learning_rate,**kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate,**kwargs)

    def evaluate_loss(self, Y_pred, Y):
        if self.loss == "mse":
            return F.mse_loss(Y_pred, Y)
        
        else: 
            return F.binary_cross_entropy(Y_pred, Y)
            

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def pre_epoch_hook(self, context):
        if self.epoch_counter == 0:
            self.log_images(context, self.epoch_counter, False)
            self.epoch_counter += 1

    def post_epoch_hook(self, context):
        if self.epoch_counter > 0:
            self.log_images(context, self.epoch_counter)
            self.epoch_counter += 1

    def log_images(self, context, epoch=None, use_model=True):
        validation_data_loader = context["validation_data_loader"]
        images = next(iter(validation_data_loader))
        imgs = images[0][0:8].to("cuda")
        with torch.no_grad():
            imgs_out = self(imgs) if use_model else imgs
        for i in range(8):
            img = imgs_out[i]
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing CNNAutoencoder model:", args)
    parser = argparse.ArgumentParser()
    CNNAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNAutoencoder(vars(args))
    print(f"Model name: {model.name()}")
    summary(model, (3, 256, 256), device="cpu") 


if __name__ == "__main__":
    pass
