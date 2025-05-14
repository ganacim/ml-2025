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
        layer_dim = init_dim
        neck_dim = args["neck_dim"]

        num_channels = 3
        rate_channels = 1.6
        new_num_channels = int(num_channels * rate_channels )
        channels = [num_channels]

        enc_layers = []
        for i in range(int(math.log2(layer_dim // neck_dim))):
            enc_layers += [
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=new_num_channels,
                    kernel_size=3,
                    padding=1
                ),
                nn.BatchNorm2d(
                    num_features=new_num_channels
                ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ]
            num_channels = new_num_channels
            channels.append(num_channels)
            new_num_channels = int(num_channels * rate_channels )

        self.encoder = nn.Sequential(
            *enc_layers,
        )

        dec_layers = []
        for i in range(1,len(channels)):
            dec_layers += [
                nn.ConvTranspose2d(
                    in_channels=channels[-i],
                    out_channels=channels[-(i+1)],
                    kernel_size=2,
                    stride=2
                ),
                nn.BatchNorm2d(num_features=channels[-(i+1)]),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=channels[-(i+1)],
                    out_channels=channels[-(i+1)],
                    kernel_size=3,
                    padding=1
                ),
            ]
            layer_dim = layer_dim * 2
        print(init_dim)
        self.decoder = nn.Sequential(
            *dec_layers,
            nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                padding=1
            ),
            nn.Sigmoid(),
            # nn.Unflatten(1, (28, 28)),
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init_dim", type=int, default=256, help="First hidden layer dimension")
        parser.add_argument("--neck_dim", type=int, default=2, help="Neck dimension")

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        return F.binary_cross_entropy(Y_pred, Y)

    def forward(self, x):
        z = self.encoder(x)
        # return z
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
