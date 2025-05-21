import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F
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
        self.trainig_discriminator = None

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
            nn.Linear(self.x_dim, last_dim, bias=False),
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

    def get_losses(self):
        return self._dis_loss, self._gen_loss

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

    def evaluate_discriminator_loss(self, Y_pred, Y_labels):
        loss = F.binary_cross_entropy_with_logits(Y_pred, Y_labels)
        return loss

    def evaluate_generator_loss(self, Y_pred):
        Y = torch.ones_like(Y_pred)
        # Y = torch.zeros_like(Y_pred)
        # alternative loss (see Goodfellow et al. 2016)
        loss = F.binary_cross_entropy_with_logits(Y_pred, Y)
        return loss

    def discriminator_forward(self, x):
        return self.discriminator(x.view(x.size(0), -1))
        # then with generated samples
        # z = torch.randn(x.size(0), self.z_dim).to(self.device)
        # d_sample = self.discriminator(self.generator(z))

        # return torch.cat([d_data, d_sample], dim=0)

    def generator_forward(self, batch_size):
        z = torch.randn(batch_size, self.z_dim).to(self.device)
        return self.discriminator(self.generator(z))

    def pre_epoch_hook(self, context):
        # boot = 2
        self.epoch = context["epoch"]
        # if self.epoch <= boot or (self.epoch - boot) % 5 == 0:
        #     self._set_training_discriminator(True)
        # else:
        #     self._set_training_discriminator(False)
        self._reset_losses()

    def post_epoch_hook(self, context):
        self._log_images(context, context["epoch"])

    def pre_train_batch_hook(self, context, X, Y):
        # if self.trainig_discriminator:
        #     self._set_training_discriminator(False)
        # else:
        #     self._set_training_discriminator(True)
        # batch_number = context["batch_number"]
        # if batch_number % 4 == 0:
        #     self._set_training_discriminator(True)
        # else:
        #     self._set_training_discriminator(False)
        # self._log_images(context, context["epoch"])
        pass

    def post_train_batch_hook(self, context, X, Y, Y_pred, loss):
        # self._log_images(context, context["epoch"])
        pass

    def pre_validation_batch_hook(self, context, X, Y):
        # if self.trainig_discriminator:
        #     self._set_training_discriminator(False)
        # else:
        #     self._set_training_discriminator(True)
        pass

    def pre_train_hook(self, context):
        self._reset_losses()

    def pre_validation_hook(self, context):
        self._reset_losses()

    def post_train_hook(self, context):
        # self._log_losses(context, context["train_data_loader"], "Train")
        pass

    def post_validation_hook(self, context):
        # self._log_losses(context, context["validation_data_loader"], "Validation")
        # self._log_images(context, context["epoch"])
        pass

    def _set_training_discriminator(self, value):
        self.trainig_discriminator = value
        if value:
            self.discriminator.train(True)
            self.discriminator.requires_grad_(True)
            self.generator.train(False)
            self.generator.requires_grad_(False)
        else:
            self.discriminator.train(False)
            self.discriminator.requires_grad_(False)
            self.generator.train(True)
            self.generator.requires_grad_(True)

    def _reset_losses(self):
        self._dis_loss = 0
        self._gen_loss = 0

    def _log_losses(self, context, data_loader, name):
        board = context["board"]
        dis_loss = self._dis_loss / len(data_loader.dataset)
        gen_loss = self._gen_loss / len(data_loader.dataset)
        board.log_scalars(
            "Curves/Loss",
            {"D(x)": dis_loss, f"Generator_{name}": gen_loss},
            context["epoch"],
        )

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
