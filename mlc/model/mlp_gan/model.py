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
        last_dim = args["last_dim"]
        self.z_dim = args["latent_dim"]
        print(f"MLPAutoencoder: latent_dim={self.z_dim}, last_dim={last_dim}")

        use_batchnorm = not args["no_batchnorm"]
        Normalization = nn.BatchNorm1d if use_batchnorm else nn.Identity
        bias = False if use_batchnorm else True

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
            return F.binary_cross_entropy(Y_pred, Y)
        else:
            Y = torch.ones(Y_pred.size(0), 1).to(self.device)
            return F.binary_cross_entropy(Y_pred, Y)

    def forward(self, x):
        # train the discriminator
        if self.trainig_discriminator:
            # batch of real images
            x = x.view(x.size(0), -1)
            X = torch.cat((x, torch.randn(x.size(0), self.x_dim).to(self.device)), dim=0)
            # get the model output
            Y_pred = self.discriminator(X)
            return Y_pred
        else:
            # train the generator
            z = torch.randn(x.size(0), self.z_dim).to(self.device)
            return self.discriminator(self.generator(z))

    def pre_epoch_hook(self, context):
        self.epoch = context["epoch"]
        if self.epoch > 2 and self.epoch % 2 == 1:
            print("Switching to generator training", self.epoch)
            self.trainig_discriminator = False
            self.discriminator.train(False)
            self.discriminator.requires_grad__(False)
        else:
            print("Switching to discriminator training", self.epoch)
            self.trainig_discriminator = True
            self.generator.train(False)
            self.generator.requires_grad__(False)

    def post_epoch_hook(self, context):
        pass

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
