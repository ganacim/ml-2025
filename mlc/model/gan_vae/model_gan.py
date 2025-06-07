import argparse

import torch
from torch import nn
from torchsummary import summary

from ..basemodel import BaseModel


class GANVAE(BaseModel):
    _name = "gan_vae"

    def __init__(self, args):
        super().__init__(args)

        self.epoch = 0
        self.x_dim = 256 * 256

        gen_dims = args["generator_dims"]
        dis_dims = args["discriminator_dims"]
        use_batchnorm = args["batchnorm"]
        leakyness = args["leakyness"]

        self.z_dim = gen_dims[0]

        init_ch = [args.get("init_ch", 3)]
        hidden_chs = init_ch + [8, 16, 32, 32, 64, 128, 256, 512]

        # draw a random sample from the latent space
        # will send to device later
        self.z_samples = torch.randn(16, self.z_dim)


        print(f"GANVAE: generator_dims={gen_dims}, discriminator_dims={dis_dims}")

        # generator

        g = nn.LeakyReLU(leakyness)  # ReLu() #PReLU() #nn.Sigmoid()
        gen_layers = []
        for i in range(len(hidden_chs) - 1, 1, -1):
            gen_layers += [
                        nn.ConvTranspose2d(hidden_chs[i], hidden_chs[i - 1], kernel_size=2, stride=2, padding=0),
                        nn.Conv2d(hidden_chs[i - 1], hidden_chs[i - 1], kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(hidden_chs[i - 1]),
                        nn.LeakyReLU(leakyness),
                    ]

        self.generator = nn.Sequential(
            *gen_layers,
            nn.Conv2d(hidden_chs[-1], hidden_chs[-1], kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )
  
        
        # discriminator
        dis_layers = []
        g = nn.LeakyReLU(leakyness)  # PReLU() #nn.Sigmoid()
        for i in range(1, len(hidden_chs)):
            # if i%2==0:
            #     g = nn.ReLU()
            # else:
            dis_layers += [
                nn.Conv2d(hidden_chs[i - 1], hidden_chs[i], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_chs[i]),
                g,
                nn.Conv2d(hidden_chs[i], hidden_chs[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_chs[i]),
                g,
            ]
        # VAE needs to output mu and logsigma

        
        # deixar 2x2 com 16 canais
        self.discriminator = nn.Sequential(
            # down
            nn.Dropout(args["discriminator_dropout"]),
            nn.Conv2d(hidden_chs[0], hidden_chs[0], kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(leakyness),
            *dis_layers,
            nn.Conv2d(hidden_chs[-1], hidden_chs[-1], kernel_size=3, stride=1, padding=1),
            nn.Flatten(),
        )

    @staticmethod
    def add_arguments(parser):
        # generator dimensions is a list
        parser.add_argument("--generator-dims", type=int, nargs="+", default=[128, 256])
        # discriminator dimensions is a list
        parser.add_argument("--discriminator-dims", type=int, nargs="+", default=[256, 128])
        parser.add_argument("--discriminator-dropout", type=float, default=0.1)
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
            img = imgs_out[i].view(1, 28, 28)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing GANVAE model:", args)

    parser = argparse.ArgumentParser()
    GANVAE.add_arguments(parser)
    args = parser.parse_args(args)

    model = GANVAE(vars(args))
    print(f"Model name: {model.name()}")
    summary(model.generator, (model.latent_dimension(),), device="cpu")
    summary(model.discriminator, (28 * 28,), device="cpu")


if __name__ == "__main__":
    pass
