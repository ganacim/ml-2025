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
        use_batchnorm = args["batchnorm"]
        self.init_dim = args["init_dim"]
        self.z_dim = args["neck_dim"]
        self.z_samples = torch.randn(16, self.z_dim)

        #generator parameters
        gen_channels = args["gen_channels"]
        #generator batchnorm
        Normalization = nn.Identity
        bias = True
        if use_batchnorm == "generator" or use_batchnorm == "both":
            Normalization = nn.BatchNorm2d
            bias = False

        #creating generator
        gen_layers = []
        for i in range(len(gen_channels) - 1):
            gen_layers += [
                nn.ConvTranspose2d(gen_channels[i], gen_channels[i+1], 4, stride=2, padding=1,bias=bias),
                Normalization(gen_channels[i+1]),
                nn.ReLU(),
            ]
        self.generator = nn.Sequential(
            nn.Unflatten(1, (self.z_dim, 1, 1)),
            nn.ConvTranspose2d(self.z_dim, gen_channels[0], kernel_size=4, stride=1 ,bias=bias),
            Normalization(gen_channels[0]),
            nn.ReLU(),
            *gen_layers,
            nn.ConvTranspose2d(gen_channels[-1], 3, 4, stride=2, padding=1,bias=True),
            nn.Tanh(),
            nn.Flatten()
        )

        #discriminator parameters
        leakyness = args["leakyness"]
        hidden_conv_dims = args["hidden_conv_dims"]
        start_dropout_r = args["dropout"]
        #discriminator batchnorm
        Normalization = nn.Identity
        bias = True
        if use_batchnorm == "discriminator" or use_batchnorm == "both":
            Normalization = nn.BatchNorm2d
            bias = False
        
        #creating discriminator
        dis_layers = []
        prev_conv_dim = hidden_conv_dims[0] #RGB
        for hidden_conv_dim in hidden_conv_dims[1:]:
            dis_layers += [
                nn.Conv2d(
                    in_channels=prev_conv_dim, 
                    out_channels=hidden_conv_dim, 
                    kernel_size=4,
                    padding=1,
                    stride = 2,
                    bias=bias
                    ),
                Normalization(hidden_conv_dim),
                nn.LeakyReLU(leakyness),
            ]
            prev_conv_dim = hidden_conv_dim
        
        self.discriminator = nn.Sequential(
            nn.Unflatten(1, (3, self.init_dim, self.init_dim)),
            nn.Dropout2d(start_dropout_r),
            nn.Conv2d(
                    in_channels=3, 
                    out_channels=hidden_conv_dims[0],
                    kernel_size=4,
                    padding=1,
                    stride = 2,
                    bias=True
                    ),
                nn.LeakyReLU(leakyness),
            *dis_layers,
            nn.Conv2d(
                    in_channels=hidden_conv_dim, 
                    out_channels=1,
                    kernel_size=4,
                    stride = 1,
                    bias=False
                    ),
            )
    @staticmethod
    def add_arguments(parser):
        # generator dimensions is a list
        parser.add_argument("--init-dim", type=int, default=64)
        parser.add_argument("--gen-channels", type=int, nargs="+", default=[512, 256, 128, 64])
        parser.add_argument("--neck-dim", type=int, default=100)
        # discriminator dimensions is a list
        parser.add_argument("--hidden_conv_dims", type=int, default=[ 64, 128, 256])
        parser.add_argument("--batchnorm", choices=["generator", "discriminator", "both", "none"], default="both")
        parser.add_argument("--leakyness", type=float, default=0.2, help="LeakyReLU leakyness")
        parser.add_argument("--dropout", type=float, default=0.1, help="Discriminator first layer dropout")
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

    def lower_dropout(self):
        # lower the dropout rate of the discriminator
        for layer in self.discriminator:
            if isinstance(layer, nn.Dropout):
                layer.p /= 2

    def get_discriminator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    def get_generator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

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
            img = imgs_out[i].view(3, self.init_dim, self.init_dim)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing CNNGAN model:", args)

    parser = argparse.ArgumentParser()
    CNNGAN.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNGAN(vars(args))
    print(f"Model name: {model.name()}")
    print(f"Model Generator:")
    summary(model.generator, (model.latent_dimension(),), device="cpu")
    print(f"Model Discriminator:")
    summary(model.discriminator, (3*model.init_dim*model.init_dim,), device="cpu")


if __name__ == "__main__":
    pass
