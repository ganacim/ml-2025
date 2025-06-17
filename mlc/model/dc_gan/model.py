import argparse

import torch
import numpy as np
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel
from .model_parts import Generator, Discriminator 

class DCGAN(BaseModel):
    _name = "dc_gan"

    def __init__(self, args):
        super().__init__(args)


        self.ngf = args["generator_channels"]
        self.ndf = args["discriminator_channels"]
        self.z_dim = args["z_dim"]
        self.epoch_counter = 0
        

        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.scale = args["scale"]


        # draw a random sample from the latent space
        # will send to device later
        self.z_samples = torch.randn(16, self.z_dim)

        self.generator = Generator(
            ngpu=1,  # number of GPUs
            nz=self.z_dim,  # latent vector size
            ngf=args["generator_channels"],  # generator feature maps
            nc=3  # number of channels in the output image (RGB)
        )
        self.discriminator = Discriminator(
            ngpu=1,  # number of GPUs
            nc=3,  # number of channels in the input image (RGB)
            ndf=args["discriminator_channels"],  # discriminator feature maps
            dropout=args["discriminator_dropout"]  # dropout rate for the discriminator
        )



        print(f"DCGAN: ")

     

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--generator-channels", "-ngf", type=int, default=64, help="Number of feature maps in the generator")
        parser.add_argument("--discriminator-dropout","-dd" ,type=float, default=0)
        parser.add_argument("--discriminator-channels", "-ndf", type=int, default=64, help="Number of feature maps in the discriminator")
        parser.add_argument("--z-dim", "-z", type=int, default=100, help="Dimension of the latent vector (default: 100)")

        parser.add_argument("--init", choices=["both", "none", "discriminator", "generator"], default="generator")
        parser.add_argument("--beta1", type=float, default=0.1, help="Adam optimizer beta1")
        parser.add_argument("--beta2", type=float, default=0.99, help="Adam optimizer beta2")
        parser.add_argument("--scale", "-s", type=int, default=256, help="Image scale (default: 256)")
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

    def lower_dropout(self):
        # lower the dropout rate of the discriminator
        for layer in self.discriminator:
            if isinstance(layer, nn.Dropout):
                layer.p /= 2

    def get_discriminator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(self.beta1, self.beta2))

    def get_generator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(self.beta1,self.beta2))

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
            img = imgs_out[i].view(3, self.scale, self.scale)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing DCGAN model:", args)

    parser = argparse.ArgumentParser()
    DCGAN.add_arguments(parser)
    args = parser.parse_args(args)

    model = DCGAN(vars(args))
    print(f"Model name: {model.name()}")
    summary(model.generator, (model.latent_dimension(),), device="cpu")
    summary(model.discriminator, (3* model.scale*model.scale,), device="cpu")


if __name__ == "__main__":
    pass
