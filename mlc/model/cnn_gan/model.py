import argparse
import math

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel
from ...util.model import load_model_from_path


class CNNVAEGAN(BaseModel):
    _name = "cnn_gan"

    def __init__(self, args):
        super().__init__(args)

        init_dim = args["init_dim"]
        layer_dim = init_dim

        self.z_dim = args["neck_dim"]
        self.x_dim = 28 * 28
        self.x_sigma = args["sigma"]
        self.load_encoder = args["load_encoder"]

        self.z_samples = torch.randn(16, 42, 2, 2)

        # used for logging
        self._rec_loss = 0
        self._kl_loss = 0

        num_channels = 3
        rate_channels = 1.5
        new_num_channels = int(num_channels * rate_channels )
        channels = [num_channels]

        print(f"CNNVAE: init_dim={init_dim}, layer_dim={layer_dim}, neck_dim={self.z_dim}")

        Normalization = nn.BatchNorm2d if args["batchnorm"] else nn.Identity
        bias = False if args["batchnorm"] else True

        enc_layers = []
        for i in range(int(math.log2(layer_dim // self.z_dim)) ):
            enc_layers += [
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=new_num_channels,
                    kernel_size=3,
                    padding=1
                ),
                Normalization(new_num_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ]
            num_channels = new_num_channels
            channels.append(num_channels)
            new_num_channels = int(num_channels * rate_channels )

            layer_dim = layer_dim // 2

        # layer_dim = layer_dim // 2

        dec_layers = []
        self.num_channels = num_channels
        self.layer_dim = layer_dim
        self.neck_size = layer_dim**2 * num_channels // 2
        # layer_dim = self.z_dim
        for i in range(1,len(channels)):
            dec_layers += [
                nn.ConvTranspose2d(
                    in_channels=channels[-i],
                    out_channels=channels[-(i+1)],
                    kernel_size=2,
                    stride=2
                ),
                nn.Conv2d(
                    in_channels=channels[-(i+1)],
                    out_channels=channels[-(i+1)],
                    kernel_size=3,
                    padding=1
                ),
                Normalization(channels[-(i+1)]),
                nn.ReLU(),
                # nn.Dropout(0.5),
            ]
            layer_dim = layer_dim * 2

        d_layer_dim = init_dim
        d_num_channels = 3
        d_new_num_channels = int(d_num_channels * rate_channels )
        discriminator_layers = []
        for j in range(int(math.log2(d_layer_dim // self.z_dim)) ):
            discriminator_layers += [
                nn.Conv2d(
                    in_channels=d_num_channels,
                    out_channels=d_new_num_channels,
                    kernel_size=3,
                    padding=1,
                    stride=2
                ),
                Normalization(d_new_num_channels),
                nn.LeakyReLU(0.05, inplace=True),
                # nn.Dropout(0.5),
                # nn.MaxPool2d(kernel_size=2),
            ]
            d_num_channels = d_new_num_channels
            channels.append(d_num_channels)
            d_new_num_channels = int(d_num_channels * rate_channels )

            d_layer_dim = d_layer_dim // 2

        self.discriminator = nn.Sequential(
            # down
            nn.Dropout(0.2),
            *discriminator_layers,
            nn.Flatten(),
            nn.Linear(168, 1, bias=True),
        )

        if args["load_encoder"] and args["encoder_path"] is not None:
            # load encoder from a checkpoint
            self.model = load_model_from_path(args["encoder_path"])
            self.encoder = self.model[0].encoder
            self.generator = self.model[0].decoder
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=223,
                    out_features=2 * 222,
                )
            )

        else:
            self.encoder = nn.Sequential(
                # down
                *enc_layers,
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=2*num_channels, # Multiply the number of channels by two so we can divide half for mu and half for sigma
                    kernel_size=3,
                    padding=1
                ),
            )
            self.generator = nn.Sequential(
                *dec_layers,
                # nn.ConvTranspose2d(
                #     in_channels=channels[-(i+1)],
                #     out_channels=channels[-(i+1)],
                #     kernel_size=2,
                #     stride=2
                # ),
                nn.Conv2d(
                    in_channels=channels[0],
                    out_channels=channels[0],
                    kernel_size=3,
                    padding=1
                ),
                nn.Tanh()
            )
            layer_dim = layer_dim * 2

    @classmethod
    def name(cls):
        return "cnn_gan"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init-dim", type=int, default=256, help="First hidden layer dimension")
        parser.add_argument("--num-channels", type=int, default=32, help="Number of channels in the Neck dimension")
        parser.add_argument("--neck-dim", type=int, default=2, help="Neck dimension")
        parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
        parser.add_argument("--sigma", type=float, default=1, help="\\sigma for P(x|z) = N(x|z, \\sigma)")
        parser.set_defaults(batchnorm=True)
        parser.add_argument("--log-zpca", action="store_true", help="Log z_\\mu PCA")
        parser.set_defaults(log_zpca=False)
        parser.add_argument("--load-encoder", action="store_true", help="Load encoder from a checkpoint")
        parser.add_argument("--encoder-path", type=str, default="models/cnn_autoencoder/latest", help="Path to the encoder checkpoint")
        parser.add_argument("--init", choices=["both", "none", "discriminator", "generator"], default="generator")

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

    def latent_dimension(self):
        return (42,2,2)

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
        q_mu_logsigma2 = self.encoder(x).flatten(start_dim=1)
        if self.load_encoder:
            q_mu_logsigma2 = self.fc(q_mu_logsigma2)

        # Divide half for mu and half for sigma
        half = len(q_mu_logsigma2[0]) // 2
        self._z_mu = q_mu_logsigma2[:, : half]
        self._z_sigma2 = torch.exp(q_mu_logsigma2[:, half :])
        z_sigma = torch.sqrt(self._z_sigma2)
        # reparameterization trick
        eps = torch.randn_like(self._z_mu)
        # print(f"mu: {mu.shape}, sigma: {sigma.shape}, eps: {eps.shape}")
        z = self._z_mu + eps * z_sigma
        z = z.reshape(-1, 42,2,2)

        return self.generator(z)

    def _reset_losses(self):
        self._rec_loss = 0
        self._kl_loss = 0

    def _log_images(self, context, epoch=None, use_model=True):
        self.z_samples = self.z_samples.to(self.device)
        with torch.no_grad():
            self.generator.eval()
            imgs_out = self.generator(self.z_samples)
            # unnormlize the images
            imgs_out = (imgs_out + 1.0) / 2.0
        for i in range(self.z_samples.size(0)):
            img = imgs_out[i].view(3, 256, 256)
            context["board"].log_image(f"Images/Image_{i}", img, epoch)

    def initialize(self):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find("Linear") != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(m.bias.data, 0)

        if self.args["init"] == "both" or self.args["init"] == "generator":
            self.generator.apply(weights_init)
        if self.args["init"] == "both" or self.args["init"] == "discriminator":
            self.discriminator.apply(weights_init)

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

    def post_epoch_hook(self, context):
        self._log_images(context, context["epoch"])

    def post_train_batch_hook(self, context, X, Y, Y_pred, loss):
        self._log_images(context, context["round"])

    def post_validation_hook(self, context):
        print("Logging validation losses")
        self._log_losses(context, context["validation_data_loader"], "Validation")
        if context["epoch"] >= 0:
            print("Logging images after validation")
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

    def get_discriminator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.9))

    def get_generator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.generator.parameters(), lr=learning_rate, betas=(0.5, 0.9))


def test(args):
    print("Testing CNNAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    CNNVAEGAN.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNVAEGAN(vars(args))
    print(f"Model name: {model.name()}")
    # summary(model.encoder, (3, 256, 256), device="cpu")
    # summary(model.generator, (384,1,1), device="cpu")
    summary(model, (3, 256, 256), device="cpu")


if __name__ == "__main__":
    pass
