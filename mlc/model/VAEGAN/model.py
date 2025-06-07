import torch
from torch import nn
from torch.nn import functional as F
from ..basemodel import BaseModel
import argparse
from torchvision.utils import make_grid

class VAEGAN(BaseModel):
    _name = "vae_gan"

    def __init__(self, args):
        super().__init__(args)

        self.z_dim = args.get("neck_dim", 128)  
        self.gamma = args.get("gamma", 1.0)  
        image_channels = 3
        
        self.encoder = nn.Sequential(
    nn.Conv2d(image_channels, 64, 4, 2, 1), nn.ReLU(True),            # 256 → 128
    nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),  # 128 → 64
    nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True), # 64 → 32
    nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True), # 32 → 16
    nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True), # 16 → 8
)

        self.flatten = nn.Flatten()

        self.fc_mu = nn.Linear(512 * 8 * 8, self.z_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, self.z_dim)



        self.fc = nn.Linear(self.z_dim, 1024 * 4 * 4)

        self.generator = nn.Sequential(
    nn.ConvTranspose2d(1024, 512, 4, 2, 1),  # 4→8
    nn.BatchNorm2d(512), nn.ReLU(True),

    nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8→16
    nn.BatchNorm2d(256), nn.ReLU(True),

    nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16→32
    nn.BatchNorm2d(128), nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 32→64
    nn.BatchNorm2d(64), nn.ReLU(True),

    nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 64→128
    nn.BatchNorm2d(32), nn.ReLU(True),

    nn.ConvTranspose2d(32, 3, 4, 2, 1),     # 128→256
    nn.Tanh()
)


        self.discriminator = nn.Sequential(
    nn.Conv2d(3, 32, 4, 2, 1), nn.LeakyReLU(0.2), nn.Dropout(0.3),   # 256→128
    nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2), nn.Dropout(0.3), # 128→64
    nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),               # 64→32
    nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),              # 32→16
    nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2), nn.Dropout(0.3), # 16→8
    nn.Conv2d(512, 1024, 4, 2, 1),       # 8→4
    nn.Conv2d(1024, 1, 4, 1, 0),         # 4→1
    nn.Flatten()
)

        self.encoder_params = list(self.encoder.parameters()) + list(self.fc_mu.parameters()) + list(self.fc_logvar.parameters())
        self.decoder_params = list(self.fc.parameters()) + list(self.generator.parameters())
        self.discriminator_params = list(self.discriminator.parameters())

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--neck-dim", type=int, default=64, help="Dimensão do espaço latente (z)")
        parser.add_argument("--gamma", type=float, default=1.0, help="Peso para a loss adversarial")

    def encode(self, x):
        h = self.encoder(x)          
        h = self.flatten(h)           
        mu = self.fc_mu(h)           
        logvar = self.fc_logvar(h)    
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.fc(z).view(-1, 1024, 4, 4)
        return self.generator(x)


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def reconstruction_loss(self, x_recon, x):
        return F.l1_loss(x_recon, x, reduction="mean") 

    def kl_divergence(self, mu, logvar):
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    def evaluate_loss(self, preds_real=None, preds_fake=None, mode="D"):
        crit = nn.BCEWithLogitsLoss()

        if mode == "D":
            real_lbl = torch.full_like(preds_real, 0.9) 
            fake_lbl = torch.zeros_like(preds_fake)
            return crit(preds_real, real_lbl) + crit(preds_fake, fake_lbl)

        elif mode == "G":
            target = torch.ones_like(preds_fake)
            return crit(preds_fake, target)
            
        else:
            raise ValueError(f"Modo desconhecido na função de loss: {mode}")
            
    def generate(self, z): 
        return self.decode(z)

    def get_optimizers(self, lr_g=0.0002, lr_d=0.0001):
        opt_Enc = torch.optim.Adam(self.encoder_params, lr=lr_g, betas=(0.5, 0.999))
        opt_Dec = torch.optim.Adam(self.decoder_params, lr=lr_g, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.discriminator_params, lr=lr_d, betas=(0.5, 0.999))
        return opt_Enc, opt_Dec, opt_D

    def post_train_batch_hook(self, context, X, Y, Y_pred, loss):
        pass 

    def post_epoch_hook(self, context):
        epoch = context["epoch"]
        device = context["device"]
        board = context["board"]
        
        batch_losses = context.get("batch_losses", [])
        if batch_losses:
            d_loss = sum(b.get("D", 0) for b in batch_losses) / len(batch_losses)
            g_loss = sum(b.get("G", 0) for b in batch_losses) / len(batch_losses)
            recon_loss = sum(b.get("Recon", 0) for b in batch_losses) / len(batch_losses)
            kl_loss = sum(b.get("KL", 0) for b in batch_losses) / len(batch_losses)

            board.log_scalars("VAEGAN/Loss_Epoch", {
                "Discriminator": d_loss,
                "Generator_Adversarial": g_loss,
                "Reconstruction": recon_loss,
                "KL_Divergence": kl_loss
            }, epoch)
            
            batch_losses.clear()

        with torch.no_grad():
            fixed_noise = torch.randn(16, self.z_dim, device=device) * 1.5
            fake_images = self.generate(fixed_noise)
            z_var = fixed_noise.var().item()
            board.log_scalar("VAEGAN/z_variance", z_var, epoch)
            grid = make_grid(fake_images, nrow=4, normalize=True, value_range=(-1, 1))
            board.log_image("VAEGAN/Generated", grid, epoch)

        val_loader = context["validation_data_loader"]
        real_images, _ = next(iter(val_loader))
        real_images = real_images.to(device)
        
        with torch.no_grad():
            recons_images, _, _ = self.forward(real_images)
            comparison = torch.cat([real_images[:8], recons_images[:8]])
            grid = make_grid(comparison, nrow=8, normalize=True, value_range=(-1, 1))
            board.log_image("VAEGAN/Reconstruction", grid, epoch)


def test(args):
    print("Testing MPLAutoencoder model:", args)

    parser = argparse.ArgumentParser()
    VAEGAN.add_arguments(parser)
    args = parser.parse_args(args)

    model = VAEGAN(vars(args))
    print(f"Model name: {model.name()}")
    summary(model.encoder_conv, (3, 256, 256), device="cpu")
    summary(model.decoder, (model.z_dim,), device="cpu")

if __name__ == "__main__":
    pass
