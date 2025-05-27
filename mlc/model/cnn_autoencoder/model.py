import argparse

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

        loss_func = "BCE"
        if loss_func == "BCE":
            self.loss_function = F.binary_cross_entropy
        elif loss_func == "MSE":
            self.loss_function = F.mse_loss
        else:
            raise ValueError(f"Unknown loss function: {args['loss_function']}")

        init_dim = args["init_dim"]
        layer_dim = init_dim
        num_blocks = args["num_blocks"]
        image_channels = args["image_channels"]
        kernel_size = 3

        print(f"CNNAutoencoder: init_dim={init_dim}, layer_dim={layer_dim}, num_blocks={num_blocks}")

        Normalization = nn.BatchNorm2d if args["batchnorm"] else nn.Identity
        bias = False if args["batchnorm"] else True
        sigmoid = nn.Sigmoid() if args["sigmoid"] else nn.Identity()

        enc_layers = []
        for i in range(num_blocks):
            enc_layers += [
                nn.MaxPool2d(2, stride = 2, padding = 0),
                nn.Conv2d(layer_dim, layer_dim, kernel_size, bias=bias, padding= kernel_size//2),
                nn.ReLU(inplace=True),
                Normalization(layer_dim),
            ]
            #layer_dim = layer_dim * 2

        self.encoder = nn.Sequential(
            # down
            nn.Conv2d(image_channels, init_dim, kernel_size, bias=bias, padding= kernel_size//2),
            nn.ReLU(inplace=True),
            Normalization(init_dim),
            *enc_layers,
        )

        dec_layers = []
        for i in range(num_blocks):
            dec_layers += [
                #nn.Upsample(scale_factor = 2),
                #nn.Conv2d(layer_dim, layer_dim // 2, kernel_size, bias=bias, padding="same"),
                nn.ConvTranspose2d(layer_dim, layer_dim, kernel_size=3, stride=2, bias=bias, padding=1, output_padding=1),
                nn.ReLU(inplace=True),
                Normalization(layer_dim),
            ]
            #layer_dim = layer_dim // 2
        self.decoder = nn.Sequential(
            *dec_layers,
            nn.Conv2d(init_dim, image_channels, 1, padding="same"),
            sigmoid,
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init-dim", type=int, default=6, help="First Conv2d number of channels")
        parser.add_argument("--num-blocks", type=int, default=2, help="Number of Encoding blocks")
        parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
        parser.add_argument("--sigmoid", action="store_true", help="Use sigmoid activation in the last layer")
        parser.add_argument("--image-channels", type=int, default=3, help="Number of image channels")
        #parser.add_argument("--loss-funtion", type=str, default="MSE", help="Use MSE or BCE loss function")
        parser.set_defaults(batchnorm = True)
        parser.set_defaults(sigmoid = True)

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        return self.loss_function(Y_pred, Y)

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
        shape = imgs.shape

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
            img = imgs_out[i].view(shape[1], shape[2], shape[3])
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing CNNAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    CNNAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNAutoencoder(vars(args))
    print(f"Model name: {model.name()}")

    print(summary(model,(3,256,256),device="cpu"))


if __name__ == "__main__":
    pass
