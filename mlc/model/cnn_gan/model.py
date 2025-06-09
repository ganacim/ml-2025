import argparse

import torch
import numpy as np
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class CNNGAN(BaseModel):
    _name = "cnn_gan"

    def __init__(self, args):
        super().__init__(args)


        gen_dims = args["generator_dims"]
        dis_dims = args["discriminator_dims"]
        use_batchnorm = args["batchnorm"]
        discriminator_dropout = args["discriminator_dropout"]
        leakyness = args["leakyness"]
        self.beta1 = args["beta1"]
        self.beta2 = args["beta2"]
        self.scale = args["scale"]
        num_channels = 3  # input channels, RGB images
        # use_pretrained = args["use_pretrained"]
        # loss = args["loss"]

        self.epoch = 0
        self.x_dim = self.scale * self.scale * 3  # input dimension
        self.z_x_dim = (self.scale*2) // (2 ** len(gen_dims))
        self.channels_dis =  (self.scale // (2 ** len(dis_dims)))*  (self.scale // (2 ** len(dis_dims))) * dis_dims[-1]


        self.z_channels = gen_dims[0]
        self.z_dim = self.z_channels * self.z_x_dim * self.z_x_dim

        # draw a random sample from the latent space
        # will send to device later
        self.z_samples = torch.randn(16, self.z_dim)

        Normalization = nn.Identity
        bias = True
        if use_batchnorm == "generator" or use_batchnorm == "both":
            Normalization = nn.BatchNorm2d
            bias = False

        print(f"CNNGAN: generator_dims={gen_dims}, discriminator_dims={dis_dims}, latent_dim={self.z_dim} ")

        layers_generator = []
        layers_generator.append(nn.Unflatten(1, (self.z_channels, self.z_x_dim, self.z_x_dim)))
        init_gen = gen_dims[0]
        for hidden_dim in gen_dims[1:]:
            layers_generator.append(
                nn.ConvTranspose2d(in_channels=init_gen, out_channels=hidden_dim, kernel_size=2, stride=2)
            )
            layers_generator.append(nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1))
            # if dropout_prob > 0.0:
            #     layers_generator.append(nn.Dropout2d(p=dropout_prob))
            layers_generator.append(Normalization(hidden_dim))
            layers_generator.append(nn.LeakyReLU(leakyness))
            init_gen = hidden_dim
        layers_generator.append(nn.Conv2d(in_channels=init_gen, out_channels=3, kernel_size=3, padding=1))
        # flatten the output to a vector
        layers_generator.append(nn.Flatten())


        layers_generator.append(nn.Tanh())
        self.generator = nn.Sequential(
            *layers_generator,
        )


        Normalization = nn.Identity
        bias = True
        if use_batchnorm == "discriminator" or use_batchnorm == "both":
            Normalization = nn.BatchNorm2d
            bias = False

        layers_discriminator = []

        # encoder
        layers_discriminator.append(nn.Unflatten(1, (num_channels, self.scale, self.scale)))

        for hidden_dim in dis_dims:
            layers_discriminator.append(nn.Conv2d(in_channels=num_channels, out_channels=hidden_dim, kernel_size=3, padding=1))
            if discriminator_dropout > 0.0:
                layers_discriminator.append(nn.Dropout2d(p=discriminator_dropout))

            layers_discriminator.append(Normalization(hidden_dim))

            layers_discriminator.append(nn.LeakyReLU(leakyness))
            layers_discriminator.append(nn.MaxPool2d(kernel_size=2))
            num_channels = hidden_dim
        # VAE needs to output mu and logsigma, so we duplicate the output
        layers_discriminator.append(nn.Flatten()) # Vai ser um  self.z_dim
        layers_discriminator.append(nn.Linear(self.channels_dis , 1, bias=bias))

        self.discriminator = nn.Sequential(
            *layers_discriminator,
        )

    @staticmethod
    def add_arguments(parser):
        # generator dimensions is a list
        parser.add_argument("--generator-dims", type=int, nargs="+", default=[512, 256, 128, 64, 32])
        # discriminator dimensions is a list
        parser.add_argument("--discriminator-dims", type=int, nargs="+", default=[16,32, 64, 128, 256, 512])
        parser.add_argument("--discriminator-dropout", type=float, default=0)
        parser.add_argument("--batchnorm", choices=["generator", "discriminator", "both", "none"], default="none")
        parser.add_argument("--leakyness", type=float, default=0.01, help="LeakyReLU leakyness")
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
    print("Testing CNNGAN model:", args)

    parser = argparse.ArgumentParser()
    CNNGAN.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNGAN(vars(args))
    print(f"Model name: {model.name()}")
    summary(model.generator, (model.latent_dimension(),), device="cpu")
    summary(model.discriminator, (3* model.scale*model.scale,), device="cpu")


if __name__ == "__main__":
    pass
