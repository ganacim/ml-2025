import argparse

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from torchvision.models import VGG16_Weights, vgg16
from torchvision.transforms.functional import resize

from ..basemodel import BaseModel


class VGGFeatureLoss(nn.Module):
    def __init__(self, layer_indices=None):
        super().__init__()

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        vgg.eval()

        if layer_indices is None:
            layer_indices = [3, 8, 15, 22, 29]
        self.layer_indices = layer_indices

        self.vgg_layers = nn.Sequential()
        for i, layer in enumerate(vgg):
            self.vgg_layers.add_module(str(i), layer)

        for param in self.vgg_layers.parameters():
            param.requires_grad = False
        self.vgg_layers.eval()

    def forward(self, input, target):
        input = resize(input, (224, 224), antialias=True)
        target = resize(target, (224, 224), antialias=True)

        input_features = self.get_features(input)
        target_features = self.get_features(target)

        loss = 0
        for i, (input_feature, target_feature) in enumerate(
            zip(input_features, target_features)
        ):
            loss += nn.functional.mse_loss(input_feature, target_feature)
        return loss

    def get_features(self, x):
        features = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
        return features


class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockDown, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlockUp, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),  # add to create interaction between pixels
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class CNNAutoencoder(BaseModel):
    _name = "cnn_autoencoder"

    def __init__(self, args: dict):
        super().__init__(args)

        assert args.get("loss_type") in [
            "VGG",
            "BCE",
            "L2",
        ], f"Invalid loss type: {self.loss_type}. Choose from 'VGG', 'BCE', or 'L2'."

        self.loss_type = args.get("loss_type", "BCE")

        # Latent sparsity
        self.latent_sparsity = args.get("latent_sparsity", False)
        self.sparsity_weight = args.get("sparsity_weight", 0.1)

        self.denoising = args.get("denoising", False)
        self.noise_level = args.get("noise_level", 0.1)

        self.masking = args.get("masking", False)
        self.last_mask = None

        self.epoch_counter = 0

        self.loss_fn = None
        self.last_z: torch.Tensor = None

        match self.loss_type:
            case "VGG":
                vgg = VGGFeatureLoss().to(self.device)
                self.loss_fn = lambda y_pred, y: 0.4 * vgg(
                    F.sigmoid(y_pred), y
                ) + 0.6 * F.mse_loss(F.sigmoid(y_pred), y)
            case "BCE":
                self.loss_fn = F.binary_cross_entropy_with_logits
                # self.loss_fn = F.binary_cross_entropy
            case "L2":
                self.loss_fn = lambda y_pred, y: F.mse_loss(F.sigmoid(y_pred), y)

        self.encoder = nn.Sequential(
            ConvBlockDown(3, 32),  # 3x256x256 -> 64x128x128
            ConvBlockDown(32, 64),  # 64x128x128 -> 128x64x64
            ConvBlockDown(64, 64),  # 128x64x64 -> 128x32x32
            ConvBlockDown(64, 64),  # 128x64x64 -> 128x32x32
            ConvBlockDown(64, 64),  # 128x64x64 -> 128x32x32
            ConvBlockDown(64, 64),  # 128x64x64 -> 128x32x32
            ConvBlockDown(64, 64),  # 128x64x64 -> 128x32x32
        )

        self.decoder = nn.Sequential(
            ConvBlockUp(64, 64),  # 128x64x64 -> 128x32x32
            ConvBlockUp(64, 64),  # 128x64x64 -> 128x32x32
            ConvBlockUp(64, 64),  # 128x64x64 -> 128x32x32
            ConvBlockUp(64, 64),  # 128x64x64 -> 128x32x32
            ConvBlockUp(64, 64),  # 128x64x64 -> 128x32x32
            ConvBlockUp(64, 32),  # 64x64x64 -> 32x128x128
            ConvBlockUp(32, 3),  # 32x128x128 -> 256x256x3
            nn.Conv2d(3, 3, kernel_size=3, padding=1),  # final conv to adjust channels
        )

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--loss_type", type=str, default="BCE")
        parser.add_argument("--latent_sparsity", action="store_true", default=False)
        parser.add_argument("--denoising", action="store_true", default=False)
        parser.add_argument("--masking", action="store_true", default=False)

    def get_optimizer(self, learning_rate, weight_decay):
        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def evaluate_loss(self, Y_pred, Y):
        if self.masking:
            # compute the loss only on the masked pixels
            Y_pred = Y_pred * self.mask
            Y = Y * self.mask
        loss = self.loss_fn(Y_pred, Y)

        if self.latent_sparsity:
            # compute l1 norm of the latent space to enforce sparsity
            loss += self.sparsity_weight * F.l1_loss(
                self.last_z, torch.zeros_like(self.last_z)
            )
        return loss

    def forward(self, x: torch.Tensor):
        if self.denoising:  # add noise to the input
            noise = torch.randn_like(x, device=x.device) * self.noise_level
            x = x + noise
            x = torch.clamp(x, 0, 1)
        if self.masking:
            self.mask = torch.randint(0, 2, size=x.shape, device=x.device)
            x = x * self.mask
        z = self.encoder(x)
        self.last_z = z if self.latent_sparsity else None
        return self.decoder(z)

    def pre_epoch_hook(self, context):
        if self.epoch_counter == 0:
            self.log_images(context, self.epoch_counter, False)
            print("entra aqui")
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
        img_shape = images[0].shape[1:]
        batch_size = images[0].shape[0]
        total_images = 8 if batch_size > 8 else batch_size
        imgs = images[0][0:total_images]
        # get the model output
        with torch.no_grad():
            # move image to device
            imgs = imgs.to(device=context["device"])
            # get the model output
            if use_model:
                imgs_out = F.sigmoid(self(imgs))
            else:
                imgs_out = imgs
            # save the image
        for i in range(total_images):
            img = imgs_out[i].view(*img_shape)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing CNNAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    CNNAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNAutoencoder(vars(args))
    print(f"Model name: {model.name()}")
    summary(model, (3, 256, 256), device="cpu")


if __name__ == "__main__":
    pass
