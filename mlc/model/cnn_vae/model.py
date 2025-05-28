import argparse
import math

import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel
from mlc.util.model import load_model_from_path

class CNNVAE(BaseModel):
    def __init__(self, args):
        super().__init__(args)

        self.epoch_counter = 0

        init_dim = args["init_dim"]
        self.z_dim = args["neck_dim"]
        start_channel_dim = args["start_nchannels"]
        channel_max_dim = args["max_nchannels"]
        conv_neck_dim = args["conv_neck_dim"]
        self.x_sigma = args["sigma"]
        model_path = args["model_path"]
        self.kl_loss_factor = args["kl_loss_factor"]
        self.x_dim = 3 * 256 * 256
        if args["use_pretrain"]:
            print("Loading pretrained model")
            pretrained_model, _, _, _, pretrained_metadata = load_model_from_path(model_path)
            pretrained_args = pretrained_metadata["model"]["args"]
            conv_neck_dim = pretrained_args["conv_neck_dim"]
            max_nchannels = pretrained_args["max_nchannels"]
            Normalization1d = nn.BatchNorm1d if not pretrained_args["nobatchnorm"] else nn.Identity
            self.Loss = F.mse_loss if not pretrained_args["bce_loss"] else F.binary_cross_entropy
            bias = False if not pretrained_args["nobatchnorm"] else True
            Activation = nn.ReLU() if not pretrained_args["leaky_relu"] else nn.LeakyReLU()
            enc_layers = pretrained_model.encoder
            dec_layers = pretrained_model.decoder
            self.encoder = nn.Sequential(
                *enc_layers,
                nn.Flatten(),
                nn.Linear(max_nchannels * conv_neck_dim**2, self.z_dim * 2, bias=bias),
            )
            self.decoder = nn.Sequential(
                nn.Linear(self.z_dim, max_nchannels * conv_neck_dim**2, bias=bias),
                Normalization1d(max_nchannels * conv_neck_dim**2),
                Activation,
                nn.Unflatten(1, (max_nchannels, conv_neck_dim, conv_neck_dim)),
                *dec_layers
            )
        else:
            Normalization1d = nn.BatchNorm1d if not args["nobatchnorm"] else nn.Identity
            Normalization2d = nn.BatchNorm2d if not args["nobatchnorm"] else nn.Identity
            self.Loss = F.mse_loss if not args["bce_loss"] else F.binary_cross_entropy
            bias = False if not args["nobatchnorm"] else True
            Activation = nn.ReLU() if not args["leaky_relu"] else nn.LeakyReLU()
            enc_layers = [
                Normalization2d(3),
                nn.Conv2d(3, start_channel_dim, 3, padding=1, bias=bias),
                Activation,
            ]
            current_channel_dim = start_channel_dim
            next_channel_dim_try = 2 *start_channel_dim
            encoder_channels = [current_channel_dim]
            for i in range(int(math.log2(init_dim // conv_neck_dim))):

                next_channel_dim = min(next_channel_dim_try, channel_max_dim)

                enc_layers += [
                    nn.Conv2d(current_channel_dim, next_channel_dim, 3, padding=1, bias=bias),
                    Normalization2d(next_channel_dim),
                    Activation,
                    nn.MaxPool2d(2)
                ]
                encoder_channels.append(next_channel_dim)
                current_channel_dim = next_channel_dim
                next_channel_dim_try *= 2 
            dec_layers = []

            enc_layers += [
                nn.Flatten(),
                nn.Linear(next_channel_dim * conv_neck_dim**2, self.z_dim * 2, bias=bias),
            ]
            dec_layers += [
                nn.Linear(self.z_dim, next_channel_dim * conv_neck_dim**2, bias=bias),
                Normalization1d(next_channel_dim * conv_neck_dim**2),
                Activation,
                nn.Unflatten(1, (next_channel_dim, conv_neck_dim, conv_neck_dim)),
            ]

            for i in range(int(math.log2(init_dim // conv_neck_dim)), 0, -1):
                dec_layers += [
                    nn.ConvTranspose2d(encoder_channels[i], encoder_channels[i-1], 2, stride=2, bias=bias),
                    Normalization2d(encoder_channels[i-1]),
                    Activation,
                ]

            dec_layers += [
                    nn.Conv2d(encoder_channels[0], encoder_channels[0], 3, padding=1, bias=bias),
                    Normalization2d(encoder_channels[0]),
                    Activation,
                    nn.Conv2d(encoder_channels[0], 3, 3, padding=1, bias=bias),
                    nn.Sigmoid(),
                ]
        
            self.encoder = nn.Sequential(
                # down
                *enc_layers,
            )
            self.decoder = nn.Sequential(
                *dec_layers,
            )


    @classmethod
    def name(cls):
        return "cnn_vae"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init_dim", type=int, default=256, help="Initial input H, W")
        parser.add_argument("--neck_dim", type=int, default=128, help="Neck dimension for fully connected unit")
        parser.add_argument("--conv_neck_dim", type=int, default=4, help="Neck dimension for convolutional unit")
        parser.add_argument("--nobatchnorm", action="store_true", help="Disable batch normalization")
        parser.add_argument("--start_nchannels", type = int, default = 32, help= "Number of convolution channels to begin Conv network")
        parser.add_argument("--max_nchannels", type = int, default = 256, help= "Maximum number of convolution channels in Conv network")
        parser.add_argument("--leaky_relu", action="store_true", help= "Use leaky relu activation")
        parser.add_argument("--bce_loss", action="store_true", help= "Use BCE loss")
        parser.add_argument("--sigma", type=float, default=1, help="\\sigma for P(x|z) = N(x|z, \\sigma)")
        parser.set_defaults(batchnorm=False)
        parser.add_argument("--log_zpca", action="store_true", help="Log z_\\mu PCA")
        parser.add_argument("--use_pretrain", action = "store_true", help="If pretrained autoencoder will be used")
        parser.add_argument("--model_path",type = str, help="Model name")
        parser.add_argument("--kl_loss_factor",type = float, default = 2,help="Reconstruction loss reduction factor")

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
        kl_loss = self.kl_divergence(self._z_mu, self._z_sigma2).mean() * self.kl_loss_factor
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
    print("Testing CNNVAE model:", args)

    parser = argparse.ArgumentParser()
    CNNVAE.add_arguments(parser)
    args = parser.parse_args(args)

    model = CNNVAE(vars(args))
    print(f"Model name: {model.name()}")
    summary(model, (3,256, 256), device="cpu")


if __name__ == "__main__":
    pass
