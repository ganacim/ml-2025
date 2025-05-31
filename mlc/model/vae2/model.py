import argparse
import math

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel

class VAE2(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        init_dim = args["init_dim"]
        layer_dim = init_dim
        self.z_dim = 64
        self.x_dim = 3 * 256 * 256
        self.x_sigma = args["sigma"]

        self._rec_loss = 0
        self._kl_loss = 0
        self._recon_loss_history = []
        self._beta = 0.0
        self._beta_max = 1.0
        self._beta_threshold = 0.005
        self._beta_min_epoch = 10
        self._beta_schedule = []

        print(f"VAE2: init_dim={init_dim}, layer_dim={layer_dim}, neck_dim={self.z_dim}")

        Normalization = nn.BatchNorm1d if args["batchnorm"] else nn.Identity
        bias = False if args["batchnorm"] else True

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Flatten(),
        )
        sample_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            self.flattened_size = self.encoder_conv(sample_input).shape[1]
        self.fc_mu = nn.Linear(self.flattened_size, self.z_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, self.z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.z_dim, 128 * 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (128, 32, 32)),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    @classmethod
    def name(cls):
        return "vae2"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init-dim", type=int, default=128)
        parser.add_argument("--neck-dim", type=int, default=16)
        parser.add_argument("--batchnorm", action="store_true")
        parser.add_argument("--sigma", type=float, default=1)
        parser.set_defaults(batchnorm=False)
        parser.add_argument("--log-zpca", action="store_true")
        parser.set_defaults(log_zpca=False)

    def get_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def kl_divergence(self, z_mu, z_sigma2):
        tr_sigma = torch.sum(z_sigma2, dim=1)
        muT_mu = (z_mu * z_mu).sum(dim=1)
        log_det_sigma = torch.sum(torch.log(z_sigma2), dim=1)
        kl_loss = 0.5 * (tr_sigma + muT_mu - log_det_sigma - self.z_dim)
        return kl_loss

    def reconstruction_loss(self, Y_pred, Y, x_sigma, x_dim):
        s2_inv = 1.0 / (2.0 * x_sigma * x_sigma)
        loss = -s2_inv * F.mse_loss(Y_pred, Y, reduction="none").flatten(start_dim=1).sum(dim=1)
        return loss

    def evaluate_loss(self, Y_pred, Y, **kwargs):
        rec_loss = self.reconstruction_loss(Y_pred, Y, self.x_sigma, self.x_dim).mean()
        self._rec_loss += -rec_loss.item() * len(Y)
        kl_loss = self.kl_divergence(self._z_mu, self._z_sigma2).mean()
        self._kl_loss += kl_loss.item() * len(Y)
        self._recon_loss_history.append(rec_loss.item())
        epoch = kwargs.get("epoch", 0)
    
        if self.args["log_zpca"] and hasattr(self, "_last_explained_var") and epoch >= self._beta_min_epoch:
            self._beta = min(self._beta_max, self._last_explained_var * 5)
            self._beta_schedule.append(self._beta)
    
        total_loss = -1.0 * (rec_loss - self._beta * kl_loss)
        return total_loss

    def encode(self, x):
        x = self.encoder_conv(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def forward(self, x):
        z_mu, logvar = self.encode(x)
        self._z_mu = z_mu
        self._z_sigma2 = torch.exp(logvar)
        self._z_sigma2 = torch.clamp(self._z_sigma2, min=1e-5, max=1e5)
        z_sigma = torch.sqrt(self._z_sigma2)
        eps = torch.randn_like(z_mu)
        z = z_mu + eps * z_sigma
        x_recon = self.decoder(z)
        return x_recon, z_mu

    def _reset_losses(self):
        self._rec_loss = 0
        self._kl_loss = 0

    def _log_images(self, context, epoch=None, use_model=True):
        validation_data_loader = context["validation_data_loader"]
        images = next(iter(validation_data_loader))
        imgs = images[0][0:8]
        with torch.no_grad():
            imgs = imgs.to(context["device"])
            if use_model:
                imgs_out, _ = self(imgs)
            else:
                imgs_out = imgs
        for i in range(min(8, imgs_out.size(0))):
            img = imgs_out[i].view_as(imgs[i])
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

    def post_validation_batch_hook(self, context, X, Y, Y_pred, loss, z_mu_batch=None):
        if self.args["log_zpca"] and z_mu_batch is not None:
            self._z_mu_list.append(z_mu_batch.detach().cpu())

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
                if torch.isnan(z_mu).any():
                    print("⚠️ Atenção: z_mu contém NaNs!")
                    print(z_mu)
                    return
                z_pca = pca.fit_transform(z_mu.cpu().numpy())
                self._last_explained_var = pca.explained_variance_ratio_[:10].sum()
                fig, ax = plt.subplots(1, 2, figsize=(16, 8))
                ax[0].scatter(z_pca[:, 0], z_pca[:, 1], s=1)
                ax[0].set_title("PCA of z_mu")
                ax[0].set_xlabel("PCA 1")
                ax[0].set_ylabel("PCA 2")
                ax[0].set_xlim(-3, 3)
                ax[0].set_ylim(-3, 3)
                ax[0].grid()
                ax[1].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
                ax[1].set_title("Explained variance")
                #fig.savefig(f"z_mu_PCA_epoch_latest2.png")
                plt.close(fig)

                if len(self._beta_schedule) > 1:
                    fig2 = plt.figure()
                    plt.plot(self._beta_schedule)
                    plt.title("Beta Schedule")
                    plt.xlabel("Epoch")
                    plt.ylabel("Beta")
                    plt.grid()
                    fig2.savefig("beta_schedule.png")
                    plt.close(fig2)



def test(args):
    print("Testing MPLAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    VAE2.add_arguments(parser)
    args = parser.parse_args(args)

    model = VAE2(vars(args))
    print(f"Model name: {model.name()}")
    summary(model.encoder_conv, (3, 256, 256), device="cpu")
    summary(model.decoder, (self.z_dim,), device="cpu")



if __name__ == "__main__":
    pass