import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class MLPAutoencoder(BaseModel):
    _name = "mlp_autoencoder"

    def __init__(self, args):
        super().__init__(args)

        self.epoch_counter = 0

        init_dim = args["init_dim"]
        layer_dim = init_dim
        neck_dim = args["neck_dim"]

        print(f"MLPAutoencoder: init_dim={init_dim}, layer_dim={layer_dim}, neck_dim={neck_dim}")

        Normalization = nn.BatchNorm1d if args["batchnorm"] else nn.Identity
        bias = False if args["batchnorm"] else True

        enc_layers = []
        for i in range(int(math.log2(layer_dim // neck_dim))):
            enc_layers += [
                nn.Linear(layer_dim, layer_dim // 2, bias=bias),
                Normalization(layer_dim // 2),
                nn.ReLU(),
            ]
            layer_dim = layer_dim // 2

        self.encoder = nn.Sequential(
            # down
            nn.Flatten(),
            nn.Linear(28 * 28, init_dim, bias=bias),
            Normalization(init_dim),
            nn.ReLU(),
            *enc_layers,
        )

        dec_layers = []
        for i in range(int(math.log2(init_dim // layer_dim))):
            dec_layers += [
                nn.Linear(layer_dim, layer_dim * 2, bias=bias),
                Normalization(layer_dim * 2),
                nn.ReLU(),
            ]
            layer_dim = layer_dim * 2
        self.decoder = nn.Sequential(
            *dec_layers,
            nn.Linear(init_dim, 28 * 28),
            nn.Sigmoid(),
            nn.Unflatten(1, (28, 28)),
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init-dim", type=int, default=32, help="First hidden layer dimension")
        parser.add_argument("--neck-dim", type=int, default=16, help="Neck dimension")
        parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
        parser.set_defaults(batchnorm=False)

    def get_optimizer(self, learning_rate, **kwargs):
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
    print("Testing MPLAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    MLPAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = MLPAutoencoder(vars(args))
    print(f"Model name: {model.name()}")
    summary(model, (28, 28), device="cpu")


if __name__ == "__main__":
    pass