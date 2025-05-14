import argparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class cnn(BaseModel):
    _name = "cnn"

    def __init__(self, args):
        super().__init__(args)

        self.epoch_counter = 0

        num_channels = 3  # input dimension
        hidden_dims = args["hidden_dims"]

        layers = []

        for hidden_dim in hidden_dims:
            layers.append(nn.Conv2d(in_channels=num_channels, out_channels=hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2))
            num_channels = hidden_dim
        self.encoder = nn.Sequential(
            *layers,
        )

        layers_decoder = []
        decoder_dim = np.flip(hidden_dims)
        for hidden_dim in decoder_dim:
            layers_decoder.append(
                nn.ConvTranspose2d(in_channels=num_channels, out_channels=hidden_dim, kernel_size=2, stride=2)
            )
            layers_decoder.append(nn.ReLU())
            layers_decoder.append(nn.BatchNorm2d(hidden_dim))
            num_channels = hidden_dim
        layers_decoder.append(nn.Conv2d(in_channels=num_channels, out_channels=3, kernel_size=3, padding=1))

        self.decoder = nn.Sequential(
            *layers_decoder,
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--hidden_dims", type=int, nargs="+", default=[32, 64, 128, 256, 512], help="Hidden dimensions"
        )

    def get_optimizer(self, learning_rate, weight_decay=0.0, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def evaluate_loss(self, Y_pred, Y):
        return F.mse_loss(Y_pred, Y)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
        # return z

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
        imgs = images[0][0:8]
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
        for i in range(8):
            img = imgs_out[i].view(3, 256, 256)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing cnn model:", args)

    parser = argparse.ArgumentParser()

    cnn.add_arguments(parser)
    args = parser.parse_args(args)

    model = cnn(vars(args))
    print(f"Model name: {model.name()}")
    summary(model, (3, 256, 256), device="cpu")


if __name__ == "__main__":
    pass
