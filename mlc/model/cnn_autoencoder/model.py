import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDown, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockUp, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class CNNAutoencoder(BaseModel):
    _name = "cnn_autoencoder"

    def __init__(self, args):
        super().__init__(args)

        self.epoch_counter = 0

        self.encoder = nn.Sequential(
            # down
            ConvBlockDown(3, 64),  # 256x256x3 -> 128x128x64
            ConvBlockDown(64, 128),  # 128x128x64 -> 64x64x128
            ConvBlockDown(128, 256),  # 64x64x128 -> 32x32x256
            ConvBlockDown(256, 512),  # 32x32x256 -> 16x16x512
            ConvBlockDown(512, 1024),  # 16x16x512 -> 8x8x1024
            ConvBlockDown(1024, 2048),  # 8x8x1024 -> 4x4x2048
        )

        self.decoder = nn.Sequential(
            ConvBlockUp(2048, 1024),  # 4x4x2048 -> 8x8x1024
            ConvBlockUp(1024, 512),  # 8x8x1024 -> 16x16x512
            ConvBlockUp(512, 256),  # 16x16x512 -> 32x32x256
            ConvBlockUp(256, 128),  # 32x32x256 -> 64x64x128
            ConvBlockUp(128, 64),  # 64x64x128 -> 128x128x64
            ConvBlockUp(64, 3),  # 128x128x64 -> 256x256x3
        )

    @staticmethod
    def add_arguments(parser):
        pass

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)
        # return torch.optim.SGD(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        return torch.linalg.norm(Y_pred - Y, ord=2, dim=0).mean()
        # return F.binary_cross_entropy(Y_pred, Y)

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
        # log some images to
        validation_data_loader = context["validation_data_loader"]
        # get a batch of 8 random images
        images = next(iter(validation_data_loader))
        batch_size = images[0].shape[0]
        imgs = images[0][0:batch_size]
        # get the model output
        with torch.no_grad():
            # move image to device
            imgs = imgs.to("cuda")
            # get the model output
            if use_model:
                imgs_out = self(imgs)
            else:
                imgs_out = imgs
            # save the image
        for i in range(batch_size):
            # print(imgs_out[i].shape)
            img = imgs_out[i].view(3, 256, 256)
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
