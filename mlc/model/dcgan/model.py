import argparse
import math

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel
from ...util.model import load_model_from_path

#TODO: Test without MHA on discriminator

class DCGAN(BaseModel):
    _name = "dcgan"

    def __init__(self, args):
        super().__init__(args)

        init_dim = args["init_dim"]
        layer_dim = init_dim

        self.z_dim = args["neck_dim"]
        self.x_dim = 28 * 28
        self.x_sigma = args["sigma"]
        self.load_encoder = args["load_encoder"]

        self.z_samples = torch.randn(16, 100, 1, 1)

        # used for logging
        self._rec_loss = 0
        self._kl_loss = 0

        print(f"DCGAN: init_dim={init_dim}, layer_dim={layer_dim}, neck_dim={self.z_dim}")

        dim = 64

        # self.discriminator = Discriminator(dim)

        self.discriminator = nn.Sequential(
            nn.Dropout(0.30),
            nn.Conv2d(3, dim, (4, 4), (2, 2), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # State size. 64 x 32 x 32
            nn.Conv2d(dim, 2*dim, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(2*dim),
            nn.LeakyReLU(0.2, True),
            # State size. 128 x 16 x 16
            # MultiHeadAttention(embed_dim=2*dim, num_heads=8),
            nn.Conv2d(2*dim, 4*dim, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(4*dim),
            nn.LeakyReLU(0.2, True),
            # State size. 256 x 8 x 8
            nn.Conv2d(4*dim, 8*dim, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(8*dim),
            nn.LeakyReLU(0.2, True),
            # State size. 512 x 4 x 4
            nn.Conv2d(8*dim, 1, (4, 4), (1, 1), (0, 0), bias=True),
        )

        self.generator = nn.Sequential(
             # Input is 100, going into a convolution.
            nn.ConvTranspose2d(100, 512, (4, 4), (1, 1), (0, 0), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # state size. 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 16 x 16
            MultiHeadAttention(embed_dim=128, num_heads=8),
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # state size. 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, (4, 4), (2, 2), (1, 1), bias=True),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh()
            # state size. 1 x 64 x 64
        )



    @classmethod
    def name(cls):
        return "dcgan"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init-dim", type=int, default=256, help="First hidden layer dimension")
        parser.add_argument("--num-channels", type=int, default=32, help="Number of channels in the Neck dimension")
        parser.add_argument("--neck-dim", type=int, default=2, help="Neck dimension")
        parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
        parser.add_argument("--sigma", type=float, default=1, help="\\sigma for P(x|z) = N(x|z, \\sigma)")
        parser.set_defaults(batchnorm=True)
        parser.add_argument("--log-zpca", action="store_true", help="Log z_\\mu PCA")
        parser.set_defaults(log_zpca=False)
        parser.add_argument("--load-encoder", action="store_true", help="Load encoder from a checkpoint")
        parser.add_argument("--encoder-path", type=str, default="models/cnn_autoencoder/latest", help="Path to the encoder checkpoint")
        parser.add_argument("--init", choices=["both", "none", "discriminator", "generator"], default="generator")


    def latent_dimension(self):
        return (100,1,1)

    def forward(self, x):
        x = self.generator(x)
        return self.discriminator(x)

    def _reset_losses(self):
        self._rec_loss = 0
        self._kl_loss = 0

    def _log_images(self, context, epoch=None, use_model=True):
        self.z_samples = self.z_samples.to(self.device)
        with torch.no_grad():
            self.generator.eval()
            imgs_out = self.generator(self.z_samples)
            # unnormlize the images
            imgs_out = (imgs_out + 1.0) / 2.0
        for i in range(self.z_samples.size(0)):
            img = imgs_out[i].view(3, 64, 64)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)

    def initialize(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0)

        if self.args["init"] == "both" or self.args["init"] == "generator":
            self.generator.apply(weights_init)
        if self.args["init"] == "both" or self.args["init"] == "discriminator":
            self.discriminator.apply(weights_init)

    def pre_epoch_hook(self, context):
        epoch = context["epoch"]
        if epoch == 1:
            self._log_images(context, 0, False)

    def pre_train_hook(self, context):
        self._reset_losses()

    def post_train_hook(self, context):
        self._log_losses(context, context["train_data_loader"], "Train")

    def post_epoch_hook(self, context):
        self._log_images(context, context["epoch"])

    def post_train_batch_hook(self, context, X, Y, Y_pred, loss):
        self._log_images(context, context["round"])

    def get_discriminator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    def get_generator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    def lower_dropout(self):
        # lower the dropout rate of the discriminator
        for layer in self.discriminator:
            if isinstance(layer, nn.Dropout):
                layer.p /= 2


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)
        x, _ = self.attn(x, x, x)  # Self-attention
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x


def test(args):
    print("Testing DCGAN model:", args)

    parser = argparse.ArgumentParser()
    DCGAN.add_arguments(parser)
    args = parser.parse_args(args)

    model = DCGAN(vars(args))
    print(f"Model name: {model.name()}")
    summary(model.discriminator, (3,64,64), device="cpu")
    summary(model.generator, (100,1,1), device="cpu")


if __name__ == "__main__":
    pass
