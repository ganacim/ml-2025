import torch
from torch import nn
from torch.nn import functional as F
import argparse
from torchsummary import summary

from mlc.model.basemodel import BaseModel

class VAEAutoencoder(BaseModel):
    _name = "vae_autoencoder"

    def __init__(self, args):
        super().__init__(args)
        
        self.epoch_counter = 0

        self.latent_dim = args.get("latent_dim", 128)
        in_channels = args.get("in_channels", 3)
        self.x_sigma = args.get("sigma", 1.0)  

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Flatten()
        )

        sample_input = torch.randn(1, in_channels, 256, 256)
        with torch.no_grad():
            self.flattened_size = self.encoder(sample_input).shape[1]
        
        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim)

        self.decoder_input = nn.Sequential(
            nn.Linear(self.latent_dim, 32*2*2),
            nn.LeakyReLU(0.2)
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
            nn.Conv2d(16, 16, kernel_size=3, padding=1),  # 16 → 16
            nn.LeakyReLU(0.2),

            #nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2, padding=1, output_padding=1),  # up to 256x256
            #nn.Conv2d(16, in_channels, kernel_size=3, padding=1),
            #nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, in_channels, kernel_size=2, stride=2),  # Final conv to match input channels
            nn.Sigmoid()

        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--in_channels", type=int, default=3, help="Input image channels")
        parser.add_argument("--latent_dim", type=int, default=128, help="Latent space dimension")
        parser.add_argument("--sigma", type=float, default=1.0, help="Sigma for MSE scaling")

    def get_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, **kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        logvar = torch.clamp(logvar, min=-20.0, max=20.0)
        #z = self.reparameterize(mu, logvar)
        var = torch.exp(logvar)
        eps = torch.randn_like(var)
        z = mu + eps * var.sqrt()  # Reparameterization trick
        x_decoded = self.decoder(self.decoder_input(z))
        return x_decoded, mu, var

    def evaluate_loss(self, Y_pred, Y):
        Y_hat, mu, var = Y_pred
    
    # 1. σ-scaled MSE reconstruction loss (changed from BCE)
        recon_loss = -1 * F.mse_loss(Y_hat, Y, reduction='mean') / (2 * self.x_sigma**2) 
        
        # (2 * self.x_sigma**2) I changed the denumerator to Y.numel() because loss was getting to small
    
    # 2. Numerically stable KL divergence calculation
        var = torch.clamp(var, min=1e-8, max=1e8)  # Prevent extreme logvar
    
        tr_var = torch.sum(var, dim=1)  # Trace of the diagonal covariance matrix
        norm_mu = (mu*mu).sum(dim=1) # L2 norm of mu   
        log_det_var = torch.sum(torch.log(var), dim=1)  # Log determinant of the diagonal covariance matrix
        kl_loss = 0.5 * (tr_var + norm_mu - log_det_var - self.latent_dim) # KL divergence term
        
        var_reg = 0.01 * (1/var).mean()
        kl_loss = kl_loss.mean() + var_reg

        if torch.any(kl_loss < 0):  # Small tolerance for numerical errors because I was getting some annoying warnings 
            problematic = kl_loss <0 
            print(f"KL violation! mu: {mu[problematic].detach().cpu().numpy()}, "
                f"logvar: {var[problematic].detach().cpu().numpy()}, "
                f"term: {kl_loss[problematic].detach().cpu().numpy()}")

    
        kl_weight = min(1.0, 0.01 + self.epoch_counter/1000)
    
        self._last_recon_loss = -1 * recon_loss.item()
        self._last_kl_div = kl_loss.item()
        self._last_kl_weight = kl_weight

        return -1.0 * (recon_loss - (kl_weight*kl_loss))  

    def pre_epoch_hook(self, context):
        if self.epoch_counter == 0:
            self.log_images(context, self.epoch_counter, False)
            self.epoch_counter += 1

    def post_epoch_hook(self, context):
        if self.epoch_counter > 0:
            self.log_images(context, self.epoch_counter)

            board = context["board"]
            board.log_scalar("Loss/Reconstruction", self._last_recon_loss, self.epoch_counter)
            board.log_scalar("Loss/KL Divergence", self._last_kl_div, self.epoch_counter)
            board.log_scalar("Loss/KL Weight", self._last_kl_weight, self.epoch_counter)

            self.epoch_counter += 1

    def log_images(self, context, epoch=None, use_model=True):
        validation_data_loader = context["validation_data_loader"]
        images = next(iter(validation_data_loader))
        imgs = images[0][0:8].to("cuda")
    
        with torch.no_grad():
            if use_model:
                imgs_out, _, _ = self(imgs)
            else:
                imgs_out = imgs
            
        for i in range(8):
            img = imgs_out[i]
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing VAEAutoencoder model:", args)
    parser = argparse.ArgumentParser()
    VAEAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = VAEAutoencoder(vars(args))
    summary(model, (3, 256, 256), device="cpu")

if __name__ == "__main__":
    pass
