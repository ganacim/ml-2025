import torch
from torch import nn
from torch.nn import functional as F
import argparse
from torchsummary import summary

from mlc.model.basemodel import BaseModel

class OldVAEAutoencoder(BaseModel):
    _name = "old_vae_autoencoder"

    def __init__(self, args):
        super().__init__(args)
        
        self.epoch_counter = 0

        self.latent_dim = args.get("latent_dim", 128)
        in_channels = args.get("in_channels", 3)

        # Encoder
        self.encoder = self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
    
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.2),
            nn.ReLU(),
    
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
    
            
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.2), 
            nn.ReLU(),
    
            nn.Flatten()
        )

        
        sample_input = torch.randn(1, in_channels, 256, 256)
        with torch.no_grad():
            self.flattened_size = self.encoder(sample_input).shape[1]
        
        self.fc_mu = nn.Linear(self.flattened_size, self.latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_size, self.latent_dim)

        # Decoder
        self.decoder_input = nn.Sequential(
        nn.Linear(self.latent_dim, self.flattened_size),
        nn.Dropout(0.2),  
        )

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.Dropout2d(0.2),  # Dropout 20% of channels
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--in_channels", type=int, default=3, help="Input image channels")
        parser.add_argument("--latent_dim", type=int, default=128, help="Latent space dimension")

    def get_optimizer(self, learning_rate, **kwargs):
        return torch.optim.Adam(self.parameters(), lr=learning_rate, **kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu = self.fc_mu(x_encoded)
        logvar = self.fc_logvar(x_encoded)
        z = self.reparameterize(mu, logvar)
        x_decoded = self.decoder(self.decoder_input(z))
        return x_decoded, mu, logvar

    def evaluate_loss(self, Y_pred, Y): #ELBO loss
        Y_hat, mu, logvar = Y_pred
        recon_loss = F.mse_loss(Y_hat, Y, reduction='mean')

        kl_weight = min(1.5, 0.1 + self.epoch_counter/100) #this weight is so that we can enforce a normal distribution later in training
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        self._last_recon_loss = recon_loss.item()
        self._last_kl_div = kl_div.item()
        self._last_kl_weight = kl_weight

        return recon_loss 

    def pre_epoch_hook(self, context):
        if self.epoch_counter == 0:
            self.log_images(context, self.epoch_counter, False)
            self.epoch_counter += 1

    def post_epoch_hook(self, context):
        if self.epoch_counter > 0:
            self.log_images(context, self.epoch_counter)

            board = context["board"] #Adding these so I can see the different components of the loss 
            board.log_scalar("Loss/Reconstruction", self._last_recon_loss, self.epoch_counter)
            board.log_scalar("Loss/KL Divergence", self._last_kl_div, self.epoch_counter)
            board.log_scalar("Loss/KL Weight", self._last_kl_weight, self.epoch_counter)

            self.epoch_counter += 1

    def log_images(self, context, epoch=None, use_model=True):
        validation_data_loader = context["validation_data_loader"]
        images = next(iter(validation_data_loader))
        imgs = images[0][0:8].to("cuda")  # Shape: [8, 3, 256, 256]
    
        with torch.no_grad():
            if use_model:
                imgs_out, _, _ = self(imgs)  # Shape: [8, 3, 256, 256]
            else:
                imgs_out = imgs
            
        for i in range(8):
            img = imgs_out[i]
            context["board"].log_image(f"Images/Image_{i}", img, epoch)


def test(args):
    print("Testing OldVAEAutoencoder model:", args)
    parser = argparse.ArgumentParser()
    OldVAEAutoencoder.add_arguments(parser)
    args = parser.parse_args(args)

    model = OldVAEAutoencoder(vars(args))  # Convert Namespace to dict
    summary(model, (3, 256, 256), device="cpu")
