import math
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from ..basemodel import BaseModel


class NewVAEAutoencoder(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.epoch_counter = 0

        self.in_channels = args.get("in_channels", 3)
        self.latent_dim = args.get("latent_dim", 128)
        self.x_sigma = args.get("sigma", 1.0)
        self.x_dim = 256 * 256 * self.in_channels

        self._rec_loss = 0
        self._kl_loss = 0

        # Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128, 2 * self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (32, 2, 2)),

            nn.ConvTranspose2d(32, 64, kernel_size=4, stride=4),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, self.in_channels, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    @classmethod
    def name(cls):
        return "new_vae_autoencoder"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--latent-dim", type=int, default=128)
        parser.add_argument("--in-channels", type=int, default=3)
        parser.add_argument("--sigma", type=float, default=1.0)
        parser.add_argument("--log-zpca", action="store_true")
        parser.set_defaults(log_zpca=False)

    def get_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def kl_divergence(self, z_mu, z_sigma2):
        tr_sigma = torch.sum(z_sigma2, dim=1)
        muT_mu = (z_mu * z_mu).sum(dim=1)
        log_det_sigma = torch.sum(torch.log(z_sigma2 + 1e-10), dim=1)
        kl_loss = 0.5 * (tr_sigma + muT_mu - log_det_sigma - self.latent_dim)

        if torch.any(kl_loss < 0):
            idx = torch.where(kl_loss < 0)[0]
            print("KL loss is negative!")
            print(f"kl_loss: {kl_loss[idx]}")
            print("--")
            print(f"trace: {tr_sigma[idx]}")
            print(f"muT_mu: {muT_mu[idx]}")
            print(f"-log det: {-log_det_sigma[idx]}")
            print(f"-z_dim: {-self.latent_dim}")
            print("--")
            print(f"sigma2: {z_sigma2[idx]}")
            print(f"mu: {z_mu[idx]}")

        return kl_loss

    def reconstruction_loss(self, Y_pred, Y, x_sigma, x_dim):
        s2_inv = 1.0 / (2.0 * x_sigma * x_sigma)
        loss = -s2_inv * F.mse_loss(Y_pred, Y, reduction="none").flatten(start_dim=1).sum(dim=1)
        return loss

    def evaluate_loss(self, Y_pred, Y):
        rec_loss = self.reconstruction_loss(Y_pred, Y, self.x_sigma, self.x_dim).mean()
        self._rec_loss += -rec_loss.item() * len(Y)

        kl_loss = self.kl_divergence(self._z_mu, self._z_sigma2).mean()
        self._kl_loss += kl_loss.item() * len(Y)

        kl_weight = 0.0 if self.epoch_counter <= 100 else 1.0

        return -1.0 * (rec_loss - (kl_weight*kl_loss))

    def forward(self, x):
        q = self.encoder_cnn(x)
        self._z_mu = q[:, : self.latent_dim]
        self._z_sigma2 = torch.exp(q[:, self.latent_dim:])
        z_sigma = torch.sqrt(self._z_sigma2)
        eps = torch.randn_like(self._z_mu)
        z = self._z_mu + eps * z_sigma
        return self.decoder(z)

    def _reset_losses(self):
        self._rec_loss = 0
        self._kl_loss = 0

    def _log_images(self, context, epoch=None, use_model=True):
        validation_data_loader = context["validation_data_loader"]
        images = next(iter(validation_data_loader))
        imgs = images[0][0:8].to(context["device"])

        with torch.no_grad():
            imgs_out = self(imgs) if use_model else imgs

        for i in range(8):
            img = imgs_out[i]
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
        if context["epoch"] == 1:
            self._log_images(context, 0, False)

    def pre_validation_hook(self, context):
        self._reset_losses()
        if self.args["log_zpca"]:
            self._z_mu_list = []

    def post_validation_batch_hook(self, context, X, Y, Y_pred, loss):
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
                pca = PCA()
                z_pca = pca.fit_transform(z_mu.cpu().numpy())
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                ax[0].scatter(z_pca[:, 0], z_pca[:, 1], s=1)
                ax[0].set_title("PCA of z_mu")
                ax[1].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
                ax[1].set_title("Explained variance")
                context["board"].log_figure("PCA/z_mu", fig, context["epoch"])
                plt.close(fig)


def test(args=None):
    import argparse
    parser = argparse.ArgumentParser()
    NewVAEAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)
    model = NewVAEAutoencoder(vars(args))
    print("Model name:", model.name())
    summary(model.encoder_cnn, (3, 256, 256), device="cpu")
    summary(model.decoder, (128,), device="cpu")


if __name__ == "__main__":
    test([])
