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


        self.init_dim = args["init_dim"]
        self.layer_dim = self.init_dim
        self.num_blocks = args["num_blocks"]
        self.image_channels = args["image_channels"]
        self.image_size = args.get("image_size", 64)
        self.kernel_size = 3

        print(f"CNNAutoencoder: init_dim={self.init_dim}, layer_dim={self.layer_dim}, num_blocks={self.num_blocks}")

        Normalization = nn.BatchNorm2d if args["batchnorm"] else nn.Identity
        bias = False if args["batchnorm"] else True
        bias = True
        sigmoid = nn.Sigmoid() if args["sigmoid"] else nn.Identity()
        use_maxpool = args.get("maxpool", False)
        
        enc_layers = []
        for i in range(self.num_blocks):
            if use_maxpool:
                enc_layers += [
                    nn.MaxPool2d(2, stride = 2, padding = 0),
                    nn.Conv2d(self.layer_dim, self.layer_dim, self.kernel_size, bias=bias, padding= self.kernel_size//2),
                    nn.Conv2d(self.layer_dim, self.layer_dim, self.kernel_size, bias=bias, padding= self.kernel_size//2),
                    nn.ReLU(inplace=True),
                    Normalization(self.layer_dim),
                ]
            else:
                enc_layers += [
                    nn.Conv2d(self.layer_dim, self.layer_dim, self.kernel_size, stride= 2, bias=bias, padding= self.kernel_size//2),
                    nn.Conv2d(self.layer_dim, self.layer_dim, self.kernel_size, bias=bias, padding= self.kernel_size//2),
                    nn.ReLU(inplace=True),
                    Normalization(self.layer_dim),
                ]
            #layer_dim = layer_dim * 2

        self.encoder = nn.Sequential(
            # down
            nn.Conv2d(self.image_channels, self.init_dim, self.kernel_size, bias=bias, padding= self.kernel_size//2),
            nn.ReLU(inplace=True),
            Normalization(self.init_dim),
            *enc_layers,
        )

        self.z_dim = args["z_dim"]
        self.layer_mult = (self.image_size // (2 ** self.num_blocks))
        
        self.z_encode = nn.Linear(self.layer_dim * self.layer_mult ** 2, self.z_dim, bias=bias)
        self.z_decode = nn.Linear(self.z_dim, self.layer_dim * self.layer_mult ** 2, bias=bias)

        dec_layers = []
        for i in range(self.num_blocks):
            dec_layers += [
                #nn.Upsample(scale_factor = 2),
                #nn.Conv2d(layer_dim, layer_dim, kernel_size, bias=bias, padding= kernel_size//2),
                #nn.Conv2d(layer_dim, layer_dim, kernel_size, bias=bias, padding= kernel_size//2),
                nn.ConvTranspose2d(self.layer_dim, self.layer_dim, kernel_size=3, stride=2, bias=bias, padding=1, output_padding=1),
                nn.Conv2d(self.layer_dim, self.layer_dim, self.kernel_size, bias=bias, padding= self.kernel_size//2),
                nn.ReLU(inplace=True),
                Normalization(self.layer_dim),
            ]
            #layer_dim = layer_dim // 2
        self.decoder = nn.Sequential(
            *dec_layers,
            nn.Conv2d(self.layer_dim, self.init_dim, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.layer_dim, self.image_channels, 1, padding="same"),
            sigmoid,
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init-dim", type=int, default=24, help="First Conv2d number of channels")
        parser.add_argument("--image-dim", type=int, default=64, help="Image size (height and width)")
        parser.add_argument("--z-dim", type=int, default=128, help="Size of the latent space (z)")
        parser.add_argument("--num-blocks", type=int, default=4, help="Number of Encoding blocks")
        parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
        parser.add_argument("--sigmoid", action="store_true", help="Use sigmoid activation in the last layer")
        parser.add_argument("--image-channels", type=int, default=3, help="Number of image channels")
        parser.add_argument("--maxpool", action="store_true", help="Use maxpooling inside encoder")
        #parser.add_argument("--loss-funtion", type=str, default="MSE", help="Use MSE or BCE loss function")
        parser.set_defaults(batchnorm = True)
        parser.set_defaults(sigmoid = True)
        parser.set_defaults(maxpool = False)

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def evaluate_loss(self, Y_pred, Y):
        Y = torch.clip(Y, 0.0, 1.0)  # ensure Y is in [0, 1] range
        return self.loss_function(Y_pred, Y)

    def forward(self, x):
        z = self.encoder(x)
        z = torch.flatten(z, start_dim=1)  # flatten the output for encoding
        z = self.z_encode(z)
        z = self.z_decode(z)
        z = z.view(z.size(0), self.layer_dim, self.layer_mult, self.layer_mult)
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

    def print_summary(self):
        print(f"Model: {self.name()}")
        print(summary(self, (3, 64, 64), device="cpu"))


def test(args):
    print("Testing CNNAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    CNNAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNAutoencoder(vars(args))
    print(f"Model name: {model.name()}")

    print(summary(model,(3,64,64),device="cpu"))


if __name__ == "__main__":
    pass
