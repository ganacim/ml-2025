import argparse
import math

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

        init_dim = args["init_dim"]
        neck_dim = args["neck_dim"]
        start_channel_dim = args["start_nchannels"]
        channel_max_dim = args["max_nchannels"]
        conv_neck_dim = args["conv_neck_dim"]

        Normalization1d = nn.BatchNorm1d if not args["nobatchnorm"] else nn.Identity
        Normalization2d = nn.BatchNorm2d if not args["nobatchnorm"] else nn.Identity
        self.Loss = F.mse_loss if not args["bce_loss"] else F.binary_cross_entropy
        bias = False if not args["nobatchnorm"] else True
        FullyConnected = True if not args["noFC"] else False
        Activation = nn.ReLU() if not args["leaky_relu"] else nn.LeakyReLU()
        enc_layers = [
            Normalization2d(3),
            nn.Conv2d(3, start_channel_dim, 3, padding=1, bias=bias),
            Activation,
        ]
        current_channel_dim = start_channel_dim
        next_channel_dim_try = 2 *start_channel_dim
        encoder_channels = [current_channel_dim]
        for i in range(int(math.log2(init_dim // conv_neck_dim))):

            next_channel_dim = min(next_channel_dim_try, channel_max_dim)

            enc_layers += [
                nn.Conv2d(current_channel_dim, next_channel_dim, 3, padding=1, bias=bias),
                Normalization2d(next_channel_dim),
                Activation,
                nn.MaxPool2d(2)
            ]
            encoder_channels.append(next_channel_dim)
            current_channel_dim = next_channel_dim
            next_channel_dim_try *= 2 
        dec_layers = []

        if FullyConnected:
            enc_layers += [
                nn.Flatten(),
                nn.Linear(next_channel_dim * conv_neck_dim**2, neck_dim, bias=bias),
                Normalization1d(neck_dim),
                Activation,
            ]
            dec_layers += [
                nn.Linear(neck_dim, next_channel_dim * conv_neck_dim**2, bias=bias),
                Normalization1d(next_channel_dim * conv_neck_dim**2),
                Activation,
                nn.Unflatten(1, (next_channel_dim, conv_neck_dim, conv_neck_dim)),
            ]
        print(encoder_channels)
        for i in range(int(math.log2(init_dim // conv_neck_dim)), 0, -1):
            dec_layers += [
                nn.ConvTranspose2d(encoder_channels[i], encoder_channels[i-1], 2, stride=2, bias=bias),
                Normalization2d(encoder_channels[i-1]),
                Activation,
            ]

        dec_layers += [
                nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1, bias=bias),
                Normalization2d(encoder_channels[0]),
                Activation,
                nn.Conv2d(encoder_channels[0], 3, 3, padding=1, bias=bias),
                nn.Sigmoid(),
            ]
    
        self.encoder = nn.Sequential(
            # down
            *enc_layers,
        )
        self.decoder = nn.Sequential(
            *dec_layers,
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init_dim", type=int, default=256, help="Initial input H, W")
        parser.add_argument("--neck_dim", type=int, default=256, help="Neck dimension for fully connected unit")
        parser.add_argument("--conv_neck_dim", type=int, default=4, help="Neck dimension for convolutional unit")
        parser.add_argument("--nobatchnorm", action="store_true", help="Disable batch normalization")
        parser.add_argument("--start_nchannels", type = int, default = 32, help= "Number of convolution channels to begin Conv network")
        parser.add_argument("--max_nchannels", type = int, default = 256, help= "Maximum number of convolution channels in Conv network")
        parser.add_argument("--noFC", action="store_true", help= "Disable fully connected unit")
        parser.add_argument("--leaky_relu", action="store_true", help= "Use leaky relu activation")
        parser.add_argument("--bce_loss", action="store_true", help= "Use BCE loss")

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        return self.Loss(Y_pred, Y)

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
    print("Testing CNNAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    CNNAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNAutoencoder(vars(args))
    print(f"Model name: {model.name()}")
    summary(model, (3, 256, 256), device="cpu")


if __name__ == "__main__":
    pass
