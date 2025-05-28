import argparse

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class CNNVAE(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        num_channels = 3  # input dimension
        hidden_dims = args["hidden_dims"]
        activation = args["activation"]
        dropout_prob = args["dropout_prob"]

        self.x_dim = 256 * 256 * 3  # input dimension
        self.z_x_dim = 256 // (2 ** len(hidden_dims))

        self.last_dim = hidden_dims[-1]
        self.z_dim = self.z_x_dim * self.z_x_dim * self.last_dim  # neck dimension

        self.x_sigma = args["sigma"]

        # check if activation is valid
        if activation not in ["relu", "leaky_relu"]:
            raise ValueError(f"Activation function {activation} is not supported")
        if activation == "relu":
            activation = nn.ReLU()
        elif activation == "leaky_relu":
            activation = nn.LeakyReLU(0.2, inplace=True)

        # used for logging
        self._rec_loss = 0
        self._kl_loss = 0

        print(f"CNNVAE: init_dim={hidden_dims[0]}, neck_dim={self.z_dim}")

        Normalization = nn.BatchNorm2d if args["batchnorm"] else nn.Identity

        layers = []

        # encoder

        for hidden_dim in hidden_dims:
            layers.append(nn.Conv2d(in_channels=num_channels, out_channels=hidden_dim, kernel_size=3, padding=1))
            if dropout_prob > 0.0:
                layers.append(nn.Dropout2d(p=dropout_prob))

            layers.append(Normalization(hidden_dim))

            layers.append(activation)
            layers.append(nn.MaxPool2d(kernel_size=2))
            num_channels = hidden_dim
        # VAE needs to output mu and logsigma, so we duplicate the output
        layers.append(nn.Flatten())
        layers.append(nn.Linear(self.z_dim, 2 * self.z_dim))

        self.encoder = nn.Sequential(
            *layers,
        )

        layers_decoder = []
        layers_decoder.append(nn.Unflatten(1, (self.last_dim, self.z_x_dim, self.z_x_dim)))
        decoder_dim = np.flip(hidden_dims)
        for hidden_dim in decoder_dim:
            layers_decoder.append(
                nn.ConvTranspose2d(in_channels=num_channels, out_channels=hidden_dim, kernel_size=2, stride=2)
            )
            layers_decoder.append(nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1))
            if dropout_prob > 0.0:
                layers.append(nn.Dropout2d(p=dropout_prob))
            layers.append(Normalization(hidden_dim))
            num_channels = hidden_dim
        layers_decoder.append(nn.Conv2d(in_channels=num_channels, out_channels=3, kernel_size=3, padding=1))

        self.decoder = nn.Sequential(
            *layers_decoder,
        )
        layers_decoder = []

    @classmethod
    def name(cls):
        return "cnn_vae"

    @staticmethod
    def add_arguments(parser):
        # parser.add_argument("--init-dim", type=int, default=128, help="First hidden layer dimension")
        # parser.add_argument("--neck-dim", type=int, default=16, help="Neck dimension")
        parser.add_argument(
            "--hidden_dims", type=int, nargs="+", default=[32, 64, 128, 256, 512], help="Hidden dimensions"
        )
        parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
        parser.add_argument(
            "--activation", "-a", type=str, default="relu", choices=["relu", "leaky_relu"], help="Activation function"
        )
        parser.add_argument(
            "--dropout", type=float, default=0.0, help="Dropout probability (0.0 = no dropout)", dest="dropout_prob"
        )
        parser.add_argument("--sigma", type=float, default=1, help="\\sigma for P(x|z) = N(x|z, \\sigma)")
        parser.set_defaults(batchnorm=False)
        parser.add_argument("--log-zpca", action="store_true", help="Log z_\\mu PCA")
        parser.set_defaults(log_zpca=False)

    def get_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

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
        loss = -s2_inv * F.mse_loss(Y_pred.flatten(start_dim=1), Y.flatten(start_dim=1), reduction="none").sum(dim=1)
        # loss += -0.5 * x_dim * math.log(2 * x_sigma * x_sigma * math.pi) * torch.ones_like(loss)
        # print(f"> {-0.5 * x_dim * math.log(2 * x_sigma * x_sigma * math.pi)}")
        return loss

    def evaluate_loss(self, Y_pred, Y):
        rec_loss = self.reconstruction_loss(Y_pred, Y, self.x_sigma, self.x_dim).mean()
        self._rec_loss += -rec_loss.item() * len(Y)
        kl_loss = self.kl_divergence(self._z_mu, self._z_sigma2).mean()
        self._kl_loss += kl_loss.item() * len(Y)
        return -1.0 * (rec_loss - kl_loss)

    def forward(self, x):
        # q_mu_logsigma has the form (mu, logsigma)
        q_mu_logsigma2 = self.encoder(x)
        self._z_mu = q_mu_logsigma2[:, : self.z_dim]
        self._z_sigma2 = torch.exp(q_mu_logsigma2[:, self.z_dim :])
        z_sigma = torch.sqrt(self._z_sigma2)
        # reparameterization trick
        eps = torch.randn_like(self._z_mu)
        # print(f"mu: {mu.shape}, sigma: {sigma.shape}, eps: {eps.shape}")
        z = self._z_mu + eps * z_sigma
        # reshape to (batch_size, z_dim, x_z_dim, x_z_dim) for decoder
        return self.decoder(z)

    def _reset_losses(self):
        self._rec_loss = 0
        self._kl_loss = 0

    def _log_images(self, context, epoch=None, use_model=True):
        # log some images to
        validation_data_loader = context["validation_data_loader"]
        # get a batch of 8 random images
        images = next(iter(validation_data_loader))
        imgs = images[0][0:8]
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
        for i in range(8):
            img = imgs_out[i].view(3, 256, 256)
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
                ax[1].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
                ax[1].set_title("Explained variance")
                context["board"].log_figure("PCA/z_mu", fig, context["epoch"])
                plt.close(fig)


def test(args):
    print("Testing MPLAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    CNNVAE.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNVAE(vars(args))
    print(f"Model name: {model.name()}")
    summary(model.encoder, (3, 256, 256), device="cpu")
    summary(model.decoder, (model.last_dim, model.z_x_dim, model.z_x_dim), device="cpu")


if __name__ == "__main__":
    pass
