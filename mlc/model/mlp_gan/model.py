import argparse

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

        gen_dims = args["generator_dims"]
        dis_dims = args["discriminator_dims"]
        use_batchnorm = args["batchnorm"]
        leakyness = args["leakyness"]

        self.z_dim = gen_dims[0]

        # draw a random sample from the latent space
        # will send to device later
        self.z_samples = torch.randn(16, self.z_dim)

        Normalization = nn.Identity
        bias = True
        if use_batchnorm == "generator" or use_batchnorm == "both":
            Normalization = nn.BatchNorm1d
            bias = False

        print(f"MLPGAN: generator_dims={gen_dims}, discriminator_dims={dis_dims}")

        gen_layers = []
        for i in range(len(gen_dims) - 1):
            gen_layers += [
                nn.Linear(gen_dims[i], gen_dims[i + 1], bias=bias),
                Normalization(gen_dims[i + 1]),
                nn.LeakyReLU(leakyness),
            ]
        self.generator = nn.Sequential(
            *gen_layers,
            nn.Linear(gen_dims[-1], self.x_dim, bias=False),
            nn.Tanh(),
        )

        Normalization = nn.Identity
        bias = True
        if use_batchnorm == "discriminator" or use_batchnorm == "both":
            Normalization = nn.BatchNorm1d
            bias = False

        dis_layers = []
        for i in range(len(dis_dims) - 1):
            dis_layers += [
                nn.Linear(dis_dims[i], dis_dims[i + 1], bias=bias),
                Normalization(dis_dims[i + 1]),
                nn.LeakyReLU(leakyness),
            ]

        self.discriminator = nn.Sequential(
            # down
            nn.Dropout(args["discriminator_dropout"]),
            nn.Linear(self.x_dim, dis_dims[0], bias=True),
            # do not use batchnorm on the first layer
            nn.LeakyReLU(leakyness),
            *dis_layers,
            # classifier
            nn.Linear(dis_dims[-1], 1, bias=True),
        )

    @staticmethod
    def add_arguments(parser):
        # generator dimensions is a list
        parser.add_argument("--generator-dims", type=int, nargs="+", default=[128, 256])
        # discriminator dimensions is a list
        parser.add_argument("--discriminator-dims", type=int, nargs="+", default=[256, 128])
        parser.add_argument("--discriminator-dropout", type=float, default=0)
        parser.add_argument("--batchnorm", choices=["generator", "discriminator", "both", "none"], default="none")
        parser.add_argument("--leakyness", type=float, default=0.01, help="LeakyReLU leakyness")
        parser.add_argument("--init", choices=["both", "none", "discriminator", "generator"], default="generator")

    def latent_dimension(self):
        return self.z_dim

    def initialize(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0)

        if self.args["init"] == "both" or self.args["init"] == "generator":
            self.generator.apply(weights_init)
        if self.args["init"] == "both" or self.args["init"] == "discriminator":
            self.discriminator.apply(weights_init)

    def get_discriminator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate,
            # betas=(0.5, 0.9)
        )

    def get_generator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate,
            # betas=(0.5, 0.9)
        )

    def pre_epoch_hook(self, context):
        pass

    def post_epoch_hook(self, context):
        self._log_images(context, context["epoch"])

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
    summary(model.generator, (model.latent_dimension(),), device="cpu")
    summary(model.discriminator, (28 * 28,), device="cpu")


if __name__ == "__main__":
    pass
