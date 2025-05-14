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

        init_dim = args["init_dim"]
        layer_dim = init_dim
        num_blocks = args["num_blocks"]
        image_channels = 1
        kernel_size = 3

        print(f"CNNAutoencoder: init_dim={init_dim}, layer_dim={layer_dim}, num_blocks={num_blocks}")

        Normalization = nn.BatchNorm2d if args["batchnorm"] else nn.Identity
        bias = False if args["batchnorm"] else True

        enc_layers = []
        for i in range(num_blocks):
            enc_layers += [
                nn.Conv2d(layer_dim, layer_dim * 2, kernel_size, bias=bias, stride=2, padding="same"),
                Normalization(layer_dim * 2),
                nn.ReLU(),
            ]
            layer_dim = layer_dim * 2

        self.encoder = nn.Sequential(
            # down
            nn.Conv2d(image_channels, init_dim, kernel_size, bias=bias, padding="same"),
            Normalization(init_dim),
            nn.ReLU(),
            *enc_layers,
        )

        dec_layers = []
        for i in range(num_blocks):
            dec_layers += [
                nn.Conv2d(layer_dim, layer_dim // 2, bias=bias, padding="same"),
                Normalization(layer_dim // 2),
                nn.ReLU(),
            ]
            layer_dim = layer_dim // 2
        self.decoder = nn.Sequential(
            *dec_layers,
            nn.Conv2d(init_dim, image_channels, padding="same"),
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init-dim", type=int, default=32, help="First Conv2d number of channels")
        parser.add_argument("--num-blocks", type=int, default=2, help="Number of Encoding blocks")
        parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
        parser.set_defaults(batchnorm=False)

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
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
            img = imgs_out[i].view(1, 28, 28)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing CNNAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    CNNAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNAutoencoder(vars(args))
    print(f"Model name: {model.name()}")

    print(summary(model,(28,28,),device="cpu"))


if __name__ == "__main__":
    pass
