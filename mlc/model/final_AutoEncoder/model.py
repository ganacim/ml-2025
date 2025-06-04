import argparse
import math

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class ConvAutoencoder(BaseModel):
    _name = "CONV_AE_bce"

    def __init__(self, args):
        super().__init__(args)

        self.epoch_counter = 0

        init_dim = 3
        layer_dim = 4

        enc_layers = []

        n = 6
        for i in range(n):
            enc_layers += [
                nn.Conv2d(3 if i == 0 else layer_dim//2, layer_dim, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(layer_dim),
                
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #??? não comuta???
                ]
                
            layer_dim = layer_dim * 2

        self.encoder = nn.Sequential(
            # down
            *enc_layers,
            nn.Conv2d(layer_dim//2, layer_dim, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(layer_dim),
            nn.ReLU(),
            nn.Flatten(1, -1),  # Flatten the output to a vector
            nn.Linear(256,128),
            #unflatten
            nn.Unflatten(1, (128, 1, 1)),  # Unflatten to (batch_size, layer_dim, 1, 1)
            nn.ReLU()
        )

        # layer_dim = layer_dim // 2  # Adjust layer_dim for the decoder
        
        # dec_layers = []
        # for i in range(n+1):
        #     dec_layers += [
        #         nn.ConvTranspose2d(layer_dim, layer_dim//2, kernel_size=2, stride=2, bias=False),
        #         nn.Conv2d(layer_dim//2, layer_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
        #         nn.BatchNorm2d(layer_dim//2),
        #         nn.ReLU()
        #         ]
        #     layer_dim = layer_dim // 2
            
        # dec_layers += [
        #     nn.ConvTranspose2d(layer_dim, 3, kernel_size=2, stride=2),
        #     nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
        # ]
        # self.decoder = nn.Sequential(
        #     *dec_layers,
        # )

        dec_layers = []

        layer_dim = 128
        
        for i in range(n+1):
            if i%2:
                dec_layers += [
                    nn.ConvTranspose2d(layer_dim, layer_dim, kernel_size=2, stride=2, bias=False),
                    nn.Conv2d(layer_dim, layer_dim, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(layer_dim),
                    nn.ReLU()
                    ]
                layer_dim = layer_dim
            else:
                dec_layers += [
                    nn.ConvTranspose2d(layer_dim, layer_dim//2, kernel_size=2, stride=2, bias=False),
                    nn.Conv2d(layer_dim//2, layer_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(layer_dim//2),
                    nn.ReLU()
                    ]
                layer_dim = layer_dim//2
            
        dec_layers += [
            nn.ConvTranspose2d(layer_dim, 3, kernel_size=2, stride=2),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
        ]
        self.decoder = nn.Sequential(
            *dec_layers,
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init_dim", type=int, default=32, help="First hidden layer dimension")
        parser.add_argument("--neck_dim", type=int, default=16, help="Neck dimension")

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        return F.binary_cross_entropy(torch.sigmoid(Y_pred), Y) #BCE
        # return F.mse_loss(Y_pred,Y) #MSE

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
                imgs_out = torch.sigmoid(self(imgs))
            else:
                imgs_out = imgs
            # save the image
        for i in range(8):
            img = imgs_out[i].view(3, 256, 256)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing CONV_AE_bce model:", args)

    parser = argparse.ArgumentParser()
    ConvAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = ConvAutoencoder(vars(args))
    print(f"Model name: {model.name()}")
    summary(model, (3,256,256), device="cpu")


if __name__ == "__main__":
    pass
