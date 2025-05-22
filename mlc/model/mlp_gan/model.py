import argparse
import math

import torch
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class MLPGAN(BaseModel):
    _name = "mlp_gan"

    def __init__(self, args):
        super().__init__(args)

        self.epoch = 0
        self.x_dim = 28 * 28
        self.z_dim = args["latent_dim"]

        # draw a random sample from the latent space
        # will send to device later
        self.z_samples = torch.randn(16, self.z_dim)

        last_dim = args["last_dim"]
        use_batchnorm = args["batchnorm"]
        leakyness = args["leakyness"]

        Normalization = nn.Identity
        bias = True
        if use_batchnorm == "generator" or use_batchnorm == "both":
            Normalization = nn.BatchNorm1d
            bias = False

        print(f"MLPGAN: latent_dim={self.z_dim}, last_dim={last_dim}")

        gen_layers = []
        layer_dim = self.z_dim
        for i in range(int(math.log2(last_dim // layer_dim))):
            gen_layers += [
                nn.Linear(layer_dim, layer_dim * 2, bias=bias),
                Normalization(layer_dim * 2),
                nn.LeakyReLU(leakyness),
                # nn.Dropout(0.3),
            ]
            layer_dim = layer_dim * 2
        self.generator = nn.Sequential(
            *gen_layers,
            nn.Linear(last_dim, self.x_dim, bias=False),
            nn.Tanh(),
        )

        Normalization = nn.Identity
        bias = True
        if use_batchnorm == "discriminator" or use_batchnorm == "both":
            Normalization = nn.BatchNorm1d
            bias = False

        dis_layers = []
        for i in range(int(math.log2(layer_dim // self.z_dim))):
            dis_layers += [
                nn.Linear(layer_dim, layer_dim // 2, bias=bias),
                Normalization(layer_dim // 2),
                nn.LeakyReLU(leakyness),
                # nn.Dropout(0.3),
            ]
            layer_dim = layer_dim // 2

        self.discriminator = nn.Sequential(
            # down
            nn.Linear(self.x_dim, last_dim, bias=True),
            # do not use batchnorm on the first layer
            # nn.BatchNorm1d(last_dim),
            nn.LeakyReLU(leakyness),
            *dis_layers,
            # classifier
            nn.Linear(layer_dim, 1, bias=True),
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--last-dim", type=int, default=64, help="First hidden layer dimension")
        parser.add_argument("--latent-dim", type=int, default=16, help="Latent space dimension")
        parser.add_argument("--batchnorm", choices=["generator", "discriminator", "both", "none"], default="none")
        parser.add_argument("--leakyness", type=float, default=0.01, help="LeakyReLU leakyness")

    def latent_dimension(self):
        return self.z_dim

    def get_discriminator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            # betas=(0.5, 0.999)
        )

    def get_generator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            # betas=(0.5, 0.999)
        )

    def pre_epoch_hook(self, context):
        pass

    def post_epoch_hook(self, context):
        # self._log_images(context, context["epoch"])
        pass

    def post_train_batch_hook(self, context, X, Y, Y_pred, loss):
        self._log_images(context, context["round"])

    def pre_train_batch_hook(self, context, X, Y):
        pass

    def pre_validation_batch_hook(self, context, X, Y):
        pass

    def pre_train_hook(self, context):
        pass

    def pre_validation_hook(self, context):
        pass

    def post_train_hook(self, context):
        pass

    def post_validation_hook(self, context):
        pass

    def _log_images(self, context, epoch=None):
        self.z_samples = self.z_samples.to(self.device)
        with torch.no_grad():
            self.generator.eval()
            imgs_out = self.generator(self.z_samples)
            # unnormlize the images
            imgs_out = (imgs_out + 1.0) / 2.0
        for i in range(self.z_samples.size(0)):
            img = imgs_out[i].view(1, 28, 28)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing MPLGAN model:", args)

    parser = argparse.ArgumentParser()
    MLPGAN.add_arguments(parser)
    args = parser.parse_args(args)

    model = MLPGAN(vars(args))
    print(f"Model name: {model.name()}")
    summary(model.generator, (16,), device="cpu")
    summary(model.discriminator, (28 * 28,), device="cpu")


if __name__ == "__main__":
    pass
