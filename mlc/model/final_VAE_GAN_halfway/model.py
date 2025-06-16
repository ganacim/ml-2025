import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

from ..basemodel import BaseModel


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layer_dim = 4
        enc_layers = []

        n = 6
        for i in range(n):
            enc_layers += [
                nn.Conv2d(3 if i == 0 else layer_dim//2, layer_dim, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(layer_dim),
                
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #??? não comuta???
                ]
                
            layer_dim = layer_dim * 2

        self.decision = nn.Sequential(
            *enc_layers,
            nn.Conv2d(layer_dim//2, layer_dim, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(layer_dim),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256,256),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decision(x).view(x.size(0), -1).mean(dim=1)


class VAE_GAN(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        
        init_dim = args["init_dim"]
        layer_dim = 4
        self.z_dim = args["neck_dim"]
        self.x_sigma = args["sigma"]

        # used for logging
        self._rec_loss = 0
        self._kl_loss = 0

        print(f"CONV_VAE_GAN: init_dim={init_dim}, layer_dim={layer_dim}, neck_dim={self.z_dim}")


        enc_layers = []

        n = 6
        for i in range(n):
            enc_layers += [
                nn.Conv2d(3 if i == 0 else layer_dim//2, layer_dim, kernel_size=3, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(layer_dim),
                
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), #??? não comuta???
                ]
                
            layer_dim = layer_dim * 2

        self.encoder = nn.Sequential(
            # down
            *enc_layers,
            nn.Conv2d(layer_dim//2, layer_dim, kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(layer_dim),
            nn.ReLU(),
        )


        dec_layers = []

        layer_dim = 128
        
        for i in range(n+1):
            if i%2:
                dec_layers += [
                    nn.ConvTranspose2d(layer_dim, layer_dim, kernel_size=2, stride=2, bias=False),
                    nn.Conv2d(layer_dim, layer_dim, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(layer_dim),
                    nn.ReLU()
                    ]
                layer_dim = layer_dim
            else:
                dec_layers += [
                    nn.ConvTranspose2d(layer_dim, layer_dim//2, kernel_size=2, stride=2, bias=False),
                    nn.Conv2d(layer_dim//2, layer_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(layer_dim//2),
                    nn.ReLU()
                    ]
                layer_dim = layer_dim//2
            
        dec_layers += [
            nn.ConvTranspose2d(layer_dim, 3, kernel_size=2, stride=2),
            nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
        ]
        self.decoder = nn.Sequential(
            *dec_layers,
        )

        self.discriminator = Discriminator()#.to('cuda')

    @classmethod
    def name(cls):
        return "final_VAE_GAN_halfway"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--init-dim", type=int, default=128, help="First hidden layer dimension")
        parser.add_argument("--neck-dim", type=int, default=16, help="Neck dimension")
        parser.add_argument("--batchnorm", action="store_true", help="Use batch normalization")
        parser.add_argument("--sigma", type=float, default=1, help="\\sigma for P(x|z) = N(x|z, \\sigma)")
        parser.set_defaults(batchnorm=False)
        parser.add_argument("--log-zpca", action="store_true", help="Log z_\\mu PCA")
        parser.set_defaults(log_zpca=False)

    def get_optimizer(self, learning_rate):
        return torch.optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters())
                                , lr=learning_rate)

    def get_discriminator_optimizer(self, learning_rate):
        return torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate)

    def kl_divergence(self, z_mu, z_sigma2):
        # Assuming sigma is a vector of the diagonal covariance matrix
        tr_sigma = torch.sum(z_sigma2, dim=1)
        muT_mu = (z_mu * z_mu).sum(dim=1)
        # det_sigma = torch.prod(z_sigma2, dim=1) + 1e-10 * torch.ones_like(z_sigma2[:, 0])
        log_det_sigma = torch.sum(torch.log(z_sigma2), dim=1)
        # kl_loss = 0.5 * (tr_sigma + muT_mu - torch.log(det_sigma) - self.z_dim)
        kl_loss = 0.5 * (tr_sigma + muT_mu - log_det_sigma - self.z_dim)
        ###kl_loss = 0.5 * torch.sum(1 + self._z_logvar - self._z_mu.pow(2) - self._z_logvar.exp())
        # print shapes
        # print(f"tr_sigma: {tr_sigma.shape},
        # muT_mu: {muT_mu.shape}, det_sigma: {det_sigma.shape}, kl_loss: {kl_loss.shape}")
        # test if KL is negative
        # print(kl_loss)
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
        return loss

    def evaluate_loss(self, Y_pred, Y, lbd = 1):
        rec_loss = self.reconstruction_loss(Y_pred, Y, self.x_sigma, 256 * 256).mean()

        kl_loss = self.kl_divergence(self._z_mu, self._z_sigma2).mean()
        adv_loss = F.binary_cross_entropy(self.discriminator(Y_pred), torch.ones(Y.size(0), device=Y.device))

        self._rec_loss += -rec_loss.item() * len(Y)
        self._kl_loss += kl_loss.item() * len(Y)
        self.adv_loss += adv_loss.item() * len(Y)
        

        return -1.0 * (rec_loss - kl_loss) + lbd * adv_loss, rec_loss, kl_loss, adv_loss 

    def discriminator_loss(self, real, recon):
        real_loss = F.binary_cross_entropy(self.discriminator(real), torch.ones(real.size(0), device=real.device))
        fake_loss = F.binary_cross_entropy(self.discriminator(recon.detach()), torch.zeros(real.size(0), device=real.device))
        return real_loss + fake_loss

    def forward(self, x):
        # q_mu_logsigma has the form (mu, logsigma)
        q_mu_logsigma = self.encoder(x)
        mean, logvar = q_mu_logsigma[:,:128], q_mu_logsigma[:,128:]
        self._z_mu = mean
        self._z_sigma2 = (torch.exp(logvar)**2)
        self._z_logvar = logvar
        
        z = mean + torch.exp(logvar/2)*torch.randn_like(mean)
        z = z.view(z.shape[0], 128, 1, 1)
        return self.decoder(z)

    def _reset_losses(self):
        self._rec_loss = 0
        self._kl_loss = 0
        self.adv_loss = 0

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
        adv_loss = self.adv_loss / len(data_loader.dataset)
        board.log_scalars(
            "Curves/Loss",
            {f"Reconstruction_{name}": rec_loss, f"KLDivergence_{name}": kl_loss, f"Adversarial_{name}": adv_loss}, 
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
    print("Testing final_VAE_GAN_halfway model:", args)

    parser = argparse.ArgumentParser()
    VAE_GAN.add_arguments(parser)
    args = parser.parse_args(args)

    model = VAE_GAN(vars(args))
    print(f"Model name: {model.name()}")
    summary(model.encoder, (3, 256, 256), device="cpu")
    summary(model.decoder, (128,1,1), device="cpu")
    summary(model.discriminator, (3, 256, 256), device="cpu")


if __name__ == "__main__":
    pass