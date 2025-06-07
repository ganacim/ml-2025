# mlc/model/vaegan.py

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary

from ..basemodel import BaseModel


class VAEGAN(BaseModel):
    """
    VAE/GAN model updated to more closely match the Larsen et al. implementation.
    Includes Batch Norm, ReLU, a redesigned discriminator, and configurable recon_depth.
    """

    def __init__(self, args):
        super().__init__(args)
        self.epoch_counter = 0

        # === Hyperparameters ===
        self.in_channels = args.get("in_channels", 3)
        self.latent_dim  = args.get("latent_dim", 128)
        self.gamma       = args.get("gamma", 0.99)
        self.recon_depth = args.get("recon_depth", 8)
        self.equilibrium = args.get("equilibrium", 0.68)
        self.margin      = args.get("margin", 0.4)
        self.kl_anneal_steps = args.get("kl_anneal_steps", 0)


        # === Encoder q(z|x) ===
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 2 * self.latent_dim)
        )

        # === Decoder G(z) ===
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.latent_dim, 256 * 16 * 16),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 16, 16)),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, self.in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() #Had to change CelebA normalization to [-1, 1] for Tanh activation as indicated in the slides
        )

        # === Discriminator ===
        disc_layers = [
            nn.Conv2d(self.in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 1),
            nn.Sigmoid()
        ]
        
        if self.recon_depth > 0:
            self.discriminator_feat = nn.Sequential(*disc_layers[:self.recon_depth])
            self.discriminator_clf = nn.Sequential(*disc_layers[self.recon_depth:])
        else:
            self.discriminator_feat = nn.Identity()
            self.discriminator_clf = nn.Sequential(*disc_layers)
        
        self.discriminator = nn.Sequential(*disc_layers)

        self.bce = nn.BCELoss()
        self.pixel_mse = nn.MSELoss()

    @classmethod
    def name(cls):
        return "vaegan"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--latent-dim", type=int, default=128)
        parser.add_argument("--in-channels", type=int, default=3)
        parser.add_argument("--gamma", type=float, default=0.5,
                            help="Weight for reconstruction loss vs GAN loss for the decoder.")
        parser.add_argument("--recon-depth", type=int, default=8,
                            help="Layer depth for feature-matching loss. Set to 0 for pixel-wise loss only.")
        parser.add_argument("--margin", type=float, default=0.4,
                            help="Margin for equilibrium gating.")
        parser.add_argument("--equilibrium", type=float, default=0.68,
                            help="Equilibrium point for gating.")
        parser.add_argument("--kl-anneal-steps", type=int, default=100000,
                            help="Number of training steps to anneal KL loss over. Set to 0 to disable.")

    def initialize(self):
        pass

    def get_encoder_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.encoder_cnn.parameters(), lr=learning_rate, betas=(0.5, 0.99))

    def get_decoder_optimizer(self, learning_rate, **kwargs):
        # The decoder now has two parts, fc and conv
        params = list(self.decoder_fc.parameters()) + list(self.decoder.parameters())
        return torch.optim.Adam(params, lr=learning_rate, betas=(0.5, 0.99))

    def get_discriminator_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    def encode(self, x):
        q = self.encoder_cnn(x)
        z_mu = q[:, :self.latent_dim]
        z_logvar = q[:, self.latent_dim:]
        return z_mu, z_logvar

    def reparameterize(self, z_mu, z_logvar):
        z_sigma = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(z_mu)
        return z_mu + eps * z_sigma

    def decode(self, z):
        h = self.decoder_fc(z)
        return self.decoder(h)

    def kl_divergence(self, z_mu, z_logvar):
        z_sigma2 = torch.exp(z_logvar)
        tr_sigma = torch.sum(z_sigma2, dim=1)
        mu_sq = (z_mu * z_mu).sum(dim=1)
        log_det = torch.sum(z_logvar, dim=1)
        return 0.5 * (tr_sigma + mu_sq - log_det - self.latent_dim)

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_recon = self.decode(z)
        return x_recon