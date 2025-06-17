import argparse

import torch
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class CNNGAN(BaseModel):
    _name = "cnn_gan"

    def __init__(self, args):
        super().__init__(args)

        self.epoch = 0

        img_channels = args["image_channels"]
        img_dim = args["image_dim"]

        self.x_dim = img_dim**2 * img_channels

        num_blocks = args["num_blocks"]
        use_batchnorm = args["batchnorm"]
        leakyness = args["leakyness"]

        kernel_size = 3
        self.z_dim = args["z_dim"]
        layer_dim = args["layer_channels"]

        # draw a random sample from the latent space
        # will send to device later
        self.z_samples = torch.randn(16, self.z_dim)

        reshape_dim = img_dim // (2 ** num_blocks)
        
        Normalization = nn.Identity
        bias = True
        if use_batchnorm == "generator" or use_batchnorm == "both":
            Normalization = nn.BatchNorm2d
            bias = False

        print(f"CNNGAN: Image channels: {img_channels}, Image dim: {img_dim}, Latent dim: {self.z_dim}, Layer channels: {layer_dim}, Use batchnorm: {use_batchnorm}")

        gen_layers = []
        for i in range(num_blocks):
            gen_layers += [
                nn.ConvTranspose2d(layer_dim, layer_dim, kernel_size=3, stride=2, bias=bias, padding=1, output_padding=1),
                Normalization(layer_dim),
                nn.LeakyReLU(leakyness),

                nn.Conv2d(layer_dim, layer_dim, kernel_size, padding=kernel_size//2),
                Normalization(layer_dim),
                nn.LeakyReLU(leakyness),
            ]
        self.generator = nn.Sequential(
            # input layer
            nn.Linear(self.z_dim, layer_dim * reshape_dim * reshape_dim, bias=bias),
            nn.BatchNorm1d(layer_dim * reshape_dim * reshape_dim),
            nn.LeakyReLU(leakyness),

            # reshape to 4D tensor
            nn.Unflatten(1, (layer_dim, reshape_dim, reshape_dim)),

            *gen_layers,
            nn.Conv2d(layer_dim, img_channels, 1, padding="same"),
            nn.Tanh(),
        )

        Normalization = nn.Identity
        bias = True
        if use_batchnorm == "discriminator" or use_batchnorm == "both":
            Normalization = nn.BatchNorm2d
            bias = False

        dis_layers = []
        for i in range(num_blocks):
            dis_layers += [
                nn.Conv2d(layer_dim, layer_dim, kernel_size, stride = 2, bias=bias, padding= kernel_size//2),
                Normalization(layer_dim),
                nn.LeakyReLU(leakyness),
            ]

        self.discriminator = nn.Sequential(
            # down
            nn.Conv2d(img_channels, layer_dim, kernel_size, stride = 1, bias=bias, padding= kernel_size//2),
            # do not use batchnorm on the first layer
            nn.LeakyReLU(leakyness),
            *dis_layers,
            # classifier
            #nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(layer_dim * reshape_dim * reshape_dim, 1, bias=bias),
        )

    @staticmethod
    def add_arguments(parser):
        

        parser.add_argument("--batchnorm", choices=["generator", "discriminator", "both", "none"], default="both")
        parser.add_argument("--leakyness", type=float, default=0.01, help="LeakyReLU leakyness")
        parser.add_argument("--num-blocks", type=int, default=4, help="Number of Encoding blocks")
        parser.add_argument("--init", choices=["both", "none", "discriminator", "generator"], default="generator")

        parser.add_argument("--image-channels", type=int, default=3, help="Number of image channels")
        parser.add_argument("--layer-channels", type=int, default=24, help="Number of channels for convolutional layers")
        parser.add_argument("--image-dim", type=int, default=64, help="Image size (height and width)")
        parser.add_argument("--z-dim", type=int, default=64, help="Latent space dimension")

    def latent_dimension(self):
        return self.z_dim

    def initialize(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Linear") != -1 or classname.find("Conv") != -1:
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
        return torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.1, 0.5))

    def get_generator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.1, 0.5))

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
            img = imgs_out[i]#.view(self.img_channels, self.img_dims, self.img_dims)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing CNNGAN model:", args)

    parser = argparse.ArgumentParser()
    CNNGAN.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNGAN(vars(args))
    print(f"Model name: {model.name()}")
    print("Generator summary:")
    summary(model.generator, (model.latent_dimension(),), device="cpu")
    print("Discriminator summary:")
    summary(model.discriminator, (3, 64, 64), device="cpu")


if __name__ == "__main__":
    pass
