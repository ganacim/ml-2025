import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class CNNAutoencoder(BaseModel):
    _name = "cnn_autoencoder"

    def __init__(self, args):
        super().__init__(args)

        self.epoch_counter = 0

        init_dim = args["init_dim"]
        layer_dim = init_dim
        neck_dim = args["neck_dim"]

        #enc_layers = []
        #for i in range(int(math.log2(layer_dim // neck_dim))):
            #enc_layers += [
            #    nn.Linear(layer_dim, layer_dim // 2),
           #     nn.ReLU(),
          #  ]
         #   layer_dim = layer_dim // 2

        self.encoder = nn.Sequential(
            # down
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
                            )

        #dec_layers = []
        #for i in range(int(math.log2(init_dim // layer_dim))):
         #   dec_layers += [
                
          #      nn.Linear(layer_dim, layer_dim * 2),
           #     nn.ReLU(),
           # ]
            #layer_dim = layer_dim * 2


        self.decoder = nn.Sequential(

            
            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=4, stride = 2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=4, stride = 2,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride = 2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride = 2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride = 2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=4, stride = 2,padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=4, stride = 2,padding=1),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,padding=1),

            nn.Sigmoid()

        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init_dim", type=int, default=32, help="First hidden layer dimension")
        parser.add_argument("--neck_dim", type=int, default=16, help="Neck dimension")

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y, **kwargs):
        loss = kwargs.get("loss")
        if loss == "mse":
            return F.mse_loss(
                Y_pred.view(Y_pred.size(0), -1),
                Y.view(Y.size(0), -1)
            )
        elif loss == "bce":
            return F.binary_cross_entropy( 
        Y_pred.view(Y_pred.size(0), -1),
        Y.view(Y.size(0), -1)
    )


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
            img = imgs_out[i]
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing MPLAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    CNNAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNAutoencoder(vars(args))
    print(f"Model name: {model.name()}")
    summary(model, (3,256, 256), device="cpu")


if __name__ == "__main__":
    pass
