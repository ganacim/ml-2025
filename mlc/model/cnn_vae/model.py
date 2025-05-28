import argparse
import math

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class ConvBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(ConvBlockDown, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.conv(x)


class ConvBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, batchnorm=True):
        super(ConvBlockUp, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1
            ),  # add to create interaction between pixels
            nn.BatchNorm2d(out_channels) if batchnorm else nn.Identity(),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class ConvolutionalVAE(BaseModel):
    def __init__(self, args: dict):
        super().__init__(args)

        init_dim = args["init_dim"]
        layer_dim = init_dim
        batch_norm = args.get("batchnorm", False)
        self.z_dim = args["neck_dim"]
        self.x_dim = 256 * 256 * 3  # 256x256x3 images
        self.x_sigma = args["sigma"]

        # used for logging
        self._rec_loss = 0
        self._kl_loss = 0

        print(
            f"ConvolutionalVAE: init_dim={init_dim}, layer_dim={layer_dim}, neck_dim={self.z_dim}"
        )

        self.encoder = nn.Sequential(
            ConvBlockDown(3, 64, batchnorm=batch_norm),  # 256x256x3 -> 64x128x128
            ConvBlockDown(64, 128, batchnorm=batch_norm),  # 64x128x128 -> 128x64x64
            ConvBlockDown(128, 256, batchnorm=batch_norm),  # 128x64x64 -> 256x32x32
            ConvBlockDown(256, 512, batchnorm=batch_norm),  # 256x32x32 -> 512x16x16
            nn.Flatten(),
            nn.Linear(
                512 * 16 * 16, 2 * self.z_dim
            ),  # 16x16x512 -> (mean, logvar) \in (z_dim, z_dim)
        )
        self.decoder_pre_conv = nn.Linear(self.z_dim, 1024 * 16 * 16)
        self.decoder = nn.Sequential(
            ConvBlockUp(1024, 512, batchnorm=batch_norm),  # 1024x16x16 -> 512x32x32
            ConvBlockUp(512, 256, batchnorm=batch_norm),  # 512x32x32 -> 256x64x64
            ConvBlockUp(256, 128, batchnorm=batch_norm),  # 256x64x64 -> 128x128x128
            ConvBlockUp(128, 64, batchnorm=batch_norm),  # 128x128x128 -> 64x256x256
            nn.Conv2d(64, 3, kernel_size=3, padding=1),  # 64x256x256 -> 3x256x256
            nn.Flatten(),
            nn.Sigmoid(),  # output pixel values in [0, 1]
        )

    @classmethod
    def name(cls):
        return "conv_vae"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument(
            "--init-dim", type=int, default=128, help="First hidden layer dimension"
        )
        parser.add_argument("--neck-dim", type=int, default=16, help="Neck dimension")
        parser.add_argument(
            "--batchnorm", action="store_true", help="Use batch normalization"
        )
        parser.add_argument(
            "--sigma",
            type=float,
            default=1,
            help="\\sigma for P(x|z) = N(x|z, \\sigma)",
        )
        parser.set_defaults(batchnorm=False)
        parser.add_argument("--log-zpca", action="store_true", help="Log z_\\mu PCA")
        parser.set_defaults(log_zpca=False)

    def get_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, **kwargs)

    def kl_divergence(self, z_mu, z_sigma2):
        # Assuming sigma is a vector of the diagonal covariance matrix
        tr_sigma = torch.sum(z_sigma2, dim=1)
        muT_mu = (z_mu * z_mu).sum(dim=1)
        # det_sigma = torch.prod(z_sigma2, dim=1) + 1e-10 * torch.ones_like(z_sigma2[:, 0])
        log_det_sigma = torch.sum(torch.log(z_sigma2), dim=1)
        # kl_loss = 0.5 * (tr_sigma + muT_mu - torch.log(det_sigma) - self.z_dim)
        kl_loss = 0.5 * (tr_sigma + muT_mu - log_det_sigma - self.z_dim)
        # print shapes
        # print(f"tr_sigma: {tr_sigma.shape},
        # muT_mu: {muT_mu.shape}, det_sigma: {det_sigma.shape}, kl_loss: {kl_loss.shape}")
        # test if KL is negative
        if torch.any(kl_loss < 0):
            # get the index of the negative KL loss
            neg_kl_indices = torch.where(kl_loss < 0)[0]
            print("KL loss is negative")
            print(f"kl_loss: {kl_loss[neg_kl_indices]}")
            print("--")
            print(f"trace: {tr_sigma[neg_kl_indices]}")
            print(f"muT_mu: {muT_mu[neg_kl_indices]}")
            # print(f"-log det: {-torch.log(det_sigma)[neg_kl_indices]}")
            print(f"-log det: {-log_det_sigma[neg_kl_indices]}")
            print(f"-z_dim: {-self.z_dim}")
            print("--")
            print(f"sigma: {z_sigma2[neg_kl_indices]}")
            # print(f"det_sigma: {det_sigma[neg_kl_indices]}")
            print(f"log_det_sigma_alt: {log_det_sigma[neg_kl_indices]}")
            # print(f"det: {det_sigma[neg_kl_indices]}")
            print(f"mu: {z_mu[neg_kl_indices]}")

        return kl_loss

    def reconstruction_loss(self, Y_pred, Y, x_sigma, x_dim):
        s2_inv = 1.0 / (2.0 * x_sigma * x_sigma)
        loss = -s2_inv * F.mse_loss(
            Y_pred, Y.flatten(start_dim=1), reduction="none"
        ).sum(dim=1)
        # loss += -0.5 * x_dim * math.log(2 * x_sigma * x_sigma * math.pi) * torch.ones_like(loss)
        # print(f"> {-0.5 * x_dim * math.log(2 * x_sigma * x_sigma * math.pi)}")
        return loss

    def evaluate_loss(self, Y_pred, Y):
        rec_loss = self.reconstruction_loss(Y_pred, Y, self.x_sigma, self.x_dim).mean()
        self._rec_loss += -rec_loss.item() * len(Y)
        kl_loss = self.kl_divergence(self._z_mu, self._z_sigma2).mean()
        self._kl_loss += kl_loss.item() * len(Y)
        return -1.0 * (rec_loss - kl_loss)

    def forward(self, x: torch.Tensor):
        # q_mu_logsigma has the form (mu, logsigma)
        q_mu_logsigma2 = self.encoder(x)
        self._z_mu = q_mu_logsigma2[:, : self.z_dim]
        self._z_sigma2 = torch.exp(q_mu_logsigma2[:, self.z_dim :])
        z_sigma = torch.sqrt(self._z_sigma2)
        # reparameterization trick

        eps = torch.randn_like(self._z_mu)
        # print(f"mu: {mu.shape}, sigma: {sigma.shape}, eps: {eps.shape}")
        z = self._z_mu + eps * z_sigma  # dim: (batch_size, z_dim)

        # z_dim -> (batch_size, 2*z_dim, 4, 4)
        z = self.decoder_pre_conv(z)  # (batch_size, 1024*4*4)
        z = z.view(-1, 1024, 16, 16)  # reshape to (batch_size, 1024, 4, 4)
        # return self.decoder(z)
        # decoder_input_flat = self.fc_decoder_input(z)
        # decoder_input_conv_shape = (-1, self.encoder_out_channels, self.encoded_spatial_dim, self.encoded_spatial_dim)
        # decoder_input_conv = decoder_input_flat.view(decoder_input_conv_shape)

        # reconstructed_features = self.decoder_conv(decoder_input_conv)
        # reconstructed_image_logits = self.decoder_final_conv(reconstructed_features)
        # reconstructed_image = self.decoder_output_activation(reconstructed_image_logits)

        return self.decoder(z)

    def _reset_losses(self):
        self._rec_loss = 0
        self._kl_loss = 0

    def _log_images(self, context, epoch=None, use_model=True):
        # log some images to
        validation_data_loader = context["validation_data_loader"]
        # get a batch of 8 random images
        images = next(iter(validation_data_loader))
        batch_size = images[0].shape[0]
        img_shape = images[0].shape[1:]  # (3, 28, 28) for MNIST-like datasets
        imgs = images[0][0:batch_size]
        # get the model output
        with torch.no_grad():
            # move image to device
            imgs = imgs.to(context["device"])
            # get the model output
            if use_model:
                imgs_out = self(imgs)
            else:
                imgs_out = imgs
            # save the image
        for i in range(batch_size):
            img = imgs_out[i].view(*img_shape)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)

    def _log_losses(self, context, data_loader, name):
        board = context["board"]
        rec_loss = self._rec_loss / len(data_loader.dataset)
        kl_loss = self._kl_loss / len(data_loader.dataset)
        board.log_scalars(
            "Curves/Loss",
            {f"Reconstruction_{name}": rec_loss, f"KLDivergence_{name}": kl_loss},
            context["epoch"],
        )

    def pre_epoch_hook(self, context):
        epoch = context["epoch"]
        if epoch == 1:
            self._log_images(context, 0, False)

    def pre_validation_hook(self, context):
        self._reset_losses()
        if self.args["log_zpca"]:
            self._z_mu_list = list()

    def post_validation_batch_hook(self, context, X, Y, Y_pred, loss):
        # log the z_mu
        if self.args["log_zpca"]:
            self._z_mu_list.append(self._z_mu)

    def pre_train_hook(self, context):
        self._reset_losses()

    def post_train_hook(self, context):
        self._log_losses(context, context["train_data_loader"], "Train")

    def post_validation_hook(self, context):
        self._log_losses(context, context["validation_data_loader"], "Validation")
        if context["epoch"] >= 0:
            self._log_images(context, context["epoch"], True)
            if self.args["log_zpca"]:
                z_mu = torch.cat(self._z_mu_list, dim=0)
                # compute PCA using sklearn
                pca = PCA()  # get all components
                z_pca = pca.fit_transform(z_mu.cpu().numpy())
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                ax[0].scatter(z_pca[:, 0], z_pca[:, 1], s=1)
                ax[0].set_title("PCA of z_mu")
                ax[0].set_xlabel("PCA 1")
                ax[0].set_ylabel("PCA 2")
                ax[0].set_xlim(-3, 3)
                ax[0].set_ylim(-3, 3)
                ax[0].grid()
                # plot the explained variance
                ax[1].bar(
                    range(1, len(pca.explained_variance_ratio_) + 1),
                    pca.explained_variance_ratio_,
                )
                ax[1].set_title("Explained variance")
                context["board"].log_figure("PCA/z_mu", fig, context["epoch"])
                plt.close(fig)


def test(args):
    print("Testing CNNVariationalAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    ConvolutionalVAE.add_arguments(parser)
    args = parser.parse_args(args)

    model = ConvolutionalVAE(vars(args))
    print(f"Model name: {model.name()}")
    # summary(model.encoder, (28, 28), device="cpu")
    summary(model, (3, 256, 256), device="cpu")
    # summary(model.decoder, (16,), device="cpu")


if __name__ == "__main__":
    pass
