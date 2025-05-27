import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class ConvAutoencoder(BaseModel):
    _name = "cnn_autoencoder"

    def __init__(self, args):
        super().__init__(args)

        self.epoch_counter = 0

        # init_dim = args["init_dim"]

        init_ch = [args.get("init_ch", 3)]
        hidden_chs = init_ch + [ 16, 32, 32, 64, 128, 512]

        enc_layers = []
        for i in range(1, len(hidden_chs)):
            # if i%2==0:
            #     g = nn.ReLU()
            # else:
            g = nn.Sigmoid()
            enc_layers += [
                nn.Conv2d(hidden_chs[i - 1], hidden_chs[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_chs[i]),
                
                g,
                nn.Conv2d(hidden_chs[i], hidden_chs[i], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_chs[i]),
                g,
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            ]

        # deixar 2x2 com 16 canais
        self.encoder = nn.Sequential(
            # down
            *enc_layers,
        )

        dec_layers = []
        for i in range(len(hidden_chs) - 1, 0, -1):
            dec_layers += [
                nn.ConvTranspose2d(hidden_chs[i], hidden_chs[i - 1], kernel_size=2, stride=2, padding=0),
                nn.Conv2d(hidden_chs[i - 1], hidden_chs[i - 1], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(hidden_chs[i - 1]),
                nn.Sigmoid(),
            ]

        self.decoder = nn.Sequential(
            *dec_layers, nn.Conv2d(hidden_chs[0], init_ch[0], kernel_size=3, stride=1, padding=1), nn.Sigmoid()
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init_dim", type=int, default=32, help="First hidden layer dimension")
        parser.add_argument("--neck_dim", type=int, default=16, help="Neck dimension")
        parser.add_argument("--init_ch", type=int, default=3, help="First number of channels")
        parser.add_argument("--hidden_chs", type=int, default=[3, 6, 6, 8, 16, 32, 32, 64], help="Hidden channels")

    def get_optimizer(self, learning_rate, weight_decay=0.0):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def evaluate_loss(self, Y_pred, Y):
        return F.mse_loss(Y_pred, Y)  # F.binary_cross_entropy(Y_pred, Y)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def pre_epoch_hook(self, context):
        if self.epoch_counter == 0:
            self.log_images(context, self.epoch_counter, False)
            self.epoch_counter += 1

    def post_epoch_hook(self, context):
        if self.epoch_counter > 0:
            self.log_images(context, self.epoch_counter)
            self.epoch_counter += 1

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
            img = imgs_out[i]  # .view(1, 28, 28)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing CNNAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    ConvAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = ConvAutoencoder(vars(args))
    print(f"Model name: {model.name()}")
    summary(model, (3, 256, 256), device="cpu")


if __name__ == "__main__":
    pass
