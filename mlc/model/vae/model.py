import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class VAE(BaseModel):
    _name = "vae"

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

        # used for logging
        self._rec_loss = 0
        self._kl_loss = 0

        self.x_sigma = args["sigma"]
        self.z_dim = args["z_dim"]

        init_dim = args["init_dim"]
        layer_dim = init_dim
        num_blocks = args["num_blocks"]
        image_channels = args["image_channels"]
        kernel_size = 3

        print(f"C-VAE: init_dim={init_dim}, layer_dim={layer_dim}, num_blocks={num_blocks}")

        Normalization = nn.BatchNorm2d if args["batchnorm"] else nn.Identity
        bias = False if args["batchnorm"] else True
        bias = True # using bias before relu
        sigmoid = nn.Sigmoid() if args["sigmoid"] else nn.Identity()

        enc_layers = []
        for i in range(num_blocks):
            enc_layers += [
                #nn.MaxPool2d(2, stride = 2, padding = 0),
                nn.Conv2d(layer_dim, layer_dim, kernel_size, stride = 2, bias=bias, padding= kernel_size//2),
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

        self.image_size = args["image_dim"]
        self.layer_mult = (self.image_size // (2 ** num_blocks))
        self.layer_channels = layer_dim
        self.z_encode = nn.Linear(layer_dim * self.layer_mult ** 2, self.z_dim, bias=bias)
        self.z_sigma_encode = nn.Linear(layer_dim * self.layer_mult ** 2, self.z_dim, bias=bias)
        self.z_sigma_encode.weight.data.fill_(0.001)  # initialize sigma to near 0
        self.z_sigma_encode.bias.data.fill_(0.001)    # initialize sigma to near 0
        self.z_decode = nn.Linear(self.z_dim, layer_dim * self.layer_mult ** 2, bias=bias)
        
        dec_layers = []
        for i in range(num_blocks):
            dec_layers += [
                #nn.Upsample(scale_factor = 2),
                #nn.Conv2d(layer_dim, layer_dim, kernel_size, bias=bias, padding= kernel_size//2),
                #nn.Conv2d(layer_dim, layer_dim, kernel_size, bias=bias, padding= kernel_size//2),
                nn.ConvTranspose2d(layer_dim, layer_dim, kernel_size=3, stride=2, bias=bias, padding=1, output_padding=1),
                nn.Conv2d(layer_dim, layer_dim, kernel_size, bias=bias, padding= kernel_size//2),
                nn.ReLU(inplace=True),
                Normalization(layer_dim),
            ]
            #layer_dim = layer_dim // 2
        self.decoder = nn.Sequential(
            *dec_layers,
            nn.Conv2d(layer_dim, init_dim, 3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(layer_dim, image_channels, 1, padding="same"),
            sigmoid,
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init-dim", type=int, default=24, help="First Conv2d number of channels")
        parser.add_argument("--image-dim", type=int, default=64, help="Image size (height and width)")
        parser.add_argument("--num-blocks", type=int, default=4, help="Number of Encoding blocks")
        parser.add_argument("--z-dim", type=int, default=128, help="Latent space dimension")
        parser.add_argument("--sigma", type=float, default=1, help="\\sigma for P(x|z) = N(x|z, \\sigma)")
        parser.add_argument("--image-channels", type=int, default=3, help="Number of image channels")
        #parser.add_argument("--loss-funtion", type=str, default="MSE", help="Use MSE or BCE loss function")
        parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
        parser.add_argument("--sigmoid", action="store_true", help="Use sigmoid activation in the last layer")
        parser.set_defaults(batchnorm = True)
        parser.set_defaults(sigmoid = True)
        parser.add_argument("--log-zpca", action="store_true", help="Log z_\\mu PCA")
        parser.set_defaults(log_zpca=True)

    def get_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate)

    def kl_divergence(self, z_mu, z_sigma2):
        # Assuming sigma is a vector of the diagonal covariance matrix
        tr_sigma = torch.sum(z_sigma2, dim=(1))
        muT_mu = (z_mu * z_mu).sum(dim=(1))
        # det_sigma = torch.prod(z_sigma2, dim=1) + 1e-10 * torch.ones_like(z_sigma2[:, 0])
        log_det_sigma = torch.sum(torch.log(z_sigma2), dim=(1))
        # kl_loss = 0.5 * (tr_sigma + muT_mu - torch.log(det_sigma) - self.z_dim)
        #z_dim = z_mu.shape[1] * z_mu.shape[2] * z_mu.shape[3]
        z_dim = self.z_dim
        kl_loss = 0.5 * (tr_sigma + muT_mu - log_det_sigma - z_dim)
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
        shape = Y_pred.shape
        dim = shape[1] * shape[2] * shape[3]
        rec_loss = self.reconstruction_loss(Y_pred, Y, self.x_sigma, dim).mean()
        self._rec_loss += -rec_loss.item() * len(Y)
        kl_loss = self.kl_divergence(self._z_mu, self._z_sigma2).mean()
        self._kl_loss += kl_loss.item() * len(Y)
        return -1.0 * (rec_loss - kl_loss)

    def forward(self, x):
        # q_mu_logsigma has the form (mu, logsigma)
        q_mu_logsigma2 = self.encoder(x)
        #self._z_mu = self.z_mu_transform(q_mu_logsigma2)
        #self._z_sigma2 = torch.exp(self.z_sigma_transform(q_mu_logsigma2))

        self._z_mu  = self.z_encode(q_mu_logsigma2.flatten(start_dim=1))
        self._z_sigma2 = torch.exp(self.z_sigma_encode(q_mu_logsigma2.flatten(start_dim=1)))
        
        #print(f"q_mu_logsigma2: {q_mu_logsigma2.shape}, mu: {self._z_mu.shape}, sigma2: {self._z_sigma2.shape}")
        
        z_sigma = torch.sqrt(self._z_sigma2)
        # reparameterization trick
        eps = torch.randn_like(self._z_mu)
        # print(f"mu: {mu.shape}, sigma: {sigma.shape}, eps: {eps.shape}")
        z = self._z_mu + eps * z_sigma
        
        z = self.z_decode(z)
        z = z.view(z.shape[0], self.layer_channels, self.layer_mult, self.layer_mult)
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
                z_pca = pca.fit_transform(z_mu.cpu().flatten(start_dim = 1).numpy())
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
    print("Testing CNN VAE model:", args)

    parser = argparse.ArgumentParser()
    VAE.add_arguments(parser)
    args = parser.parse_args(args)

    model = VAE(vars(args))
    print(f"Model name: {model.name()}")

    print(summary(model,(3,64,64),device="cpu"))


if __name__ == "__main__":
    pass
