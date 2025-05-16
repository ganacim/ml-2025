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
        use_batchnorm = not args["no_batchnorm"]
        Normalization = nn.BatchNorm1d if use_batchnorm else nn.Identity
        bias = False if use_batchnorm else True

        print(f"MLPAutoencoder: latent_dim={self.z_dim}, last_dim={last_dim}")

        gen_layers = []
        layer_dim = self.z_dim
        for i in range(int(math.log2(last_dim // layer_dim))):
            gen_layers += [
                nn.Linear(layer_dim, layer_dim * 2, bias=bias),
                Normalization(layer_dim * 2),
                nn.ReLU(),
            ]
            layer_dim = layer_dim * 2
        self.generator = nn.Sequential(
            *gen_layers,
            nn.Linear(last_dim, self.x_dim),
            nn.Sigmoid(),
        )

        dis_layers = []
        for i in range(int(math.log2(layer_dim // self.z_dim))):
            dis_layers += [
                nn.Linear(layer_dim, layer_dim // 2, bias=bias),
                Normalization(layer_dim // 2),
                nn.ReLU(),
            ]
            layer_dim = layer_dim // 2

        self.discriminator = nn.Sequential(
            # down
            nn.Linear(self.x_dim, last_dim, bias=bias),
            Normalization(last_dim),
            nn.ReLU(),
            *dis_layers,
            # classifier
            nn.Linear(layer_dim, 1, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--last-dim", type=int, default=64, help="First hidden layer dimension")
        parser.add_argument("--latent-dim", type=int, default=16, help="Latent space dimension")
        parser.add_argument("--no-batchnorm", action="store_true", help="Do NOT use batch normalization")
        parser.set_defaults(no_batchnorm=False)

    def get_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y_label):
        if self.trainig_discriminator:
            Y = torch.cat(
                (torch.ones(Y_label.size(0), 1).to(self.device), torch.zeros(Y_label.size(0), 1).to(self.device)), dim=0
            )
            loss = F.binary_cross_entropy(Y_pred, Y, reduction="sum") / len(Y_label)
            self._dis_loss += loss.item()
            with torch.no_grad():
                Y = torch.zeros(Y_label.size(0), 1).to(self.device)
                self._gen_loss += -F.binary_cross_entropy(Y_pred[len(Y_pred) // 2 :], Y).item()

        else:
            Y = torch.zeros(Y_label.size(0), 1).to(self.device)
            loss = -F.binary_cross_entropy(Y_pred, Y)
            self._gen_loss += loss.item()

            with torch.no_grad():
                Y = torch.zeros(Y_label.size(0), 1).to(self.device)
                self._dis_loss += F.binary_cross_entropy(Y_pred, Y, reduction="sum").item() / len(Y_label)
        return loss

    def forward(self, x):
        z = torch.randn(x.size(0), self.z_dim).to(self.device)
        if self.trainig_discriminator:
            X = torch.cat(
                (
                    x.view(x.size(0), -1),
                    self.generator(z),
                ),
                dim=0,
            )
        else:
            X = self.generator(z)

        return self.discriminator(X)

    def pre_epoch_hook(self, context):
        # boot = 2
        self.epoch = context["epoch"]
        # if self.epoch <= boot or (self.epoch - boot) % 5 == 0:
        #     self._set_training_discriminator(True)
        # else:
        #     self._set_training_discriminator(False)

    def post_epoch_hook(self, context):
        pass

    def pre_train_batch_hook(self, context, X, Y):
        # if self.trainig_discriminator:
        #     self._set_training_discriminator(False)
        # else:
        #     self._set_training_discriminator(True)
        batch_number = context["batch_number"]
        if batch_number % 4 == 0:
            self._set_training_discriminator(True)
        else:
            self._set_training_discriminator(False)

    def pre_validation_batch_hook(self, context, X, Y):
        if self.trainig_discriminator:
            self._set_training_discriminator(False)
        else:
            self._set_training_discriminator(True)

    def pre_train_hook(self, context):
        self._reset_losses()

    def pre_validation_hook(self, context):
        self._reset_losses()

    def post_train_hook(self, context):
        self._log_losses(context, context["train_data_loader"], "Train")

    def post_validation_hook(self, context):
        self._log_losses(context, context["validation_data_loader"], "Validation")
        self._log_images(context, context["epoch"])

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
            {f"Discriminator_{name}": dis_loss, f"Generator_{name}": gen_loss},
            context["epoch"],
        )

    def _log_images(self, context, epoch=None):
        self.z_samples = self.z_samples.to(self.device)
        with torch.no_grad():
            imgs_out = self.generator(self.z_samples)
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
