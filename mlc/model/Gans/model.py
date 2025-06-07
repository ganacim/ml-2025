import argparse
import math

import torch
import matplotlib.pyplot as plt
import os
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from torchvision.utils import make_grid

from ..basemodel import BaseModel

class GANs(BaseModel):
    _name = "gans"

    def __init__(self, args):
        super().__init__(args)

        self.epoch_counter = 0

        self.z_dim = 64                    
        self.init_channels = 1024             

        self.fc = nn.Linear(self.z_dim, self.init_channels * 4 * 4)

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # 4→8
            nn.BatchNorm2d(512), nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),   # 8→16
            nn.BatchNorm2d(256), nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),   # 16→32
            nn.BatchNorm2d(128), nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),    # 32→64
            nn.BatchNorm2d(64), nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),   # 64 →128 
            nn.BatchNorm2d(32), nn.ReLU(True),

            nn.ConvTranspose2d(32, 3, 4, 2, 1, bias=False),      # 128 →256
            nn.Tanh()                                            
        )

        self.discriminator = nn.Sequential(

                    nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   #256 -> 128
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),

                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  #128 -> 64     
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),


                    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  #64 -> 32
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),

                    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  #32 -> 16
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2),

                    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  #16 -> 8
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(0.3),


                    nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),  #8 -> 4
                    
                    nn.Conv2d(1024, 1, 4, 1, 0),  # 4 → 1
                    nn.Flatten()
        )

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--z_dim", type=int, default=128, help="Dimensão do vetor latente")
        parser.add_argument("--init_dim", type=int, default=32, help="First hidden layer dimension")
        parser.add_argument("--neck_dim", type=int, default=16, help="Neck dimension")

    def pre_epoch_hook(self, context):
        if context["epoch"] == 1:
            self._log_images(context, 1)

    def post_epoch_hook(self, context):
        epoch = context["epoch"]
        self._log_images(context, epoch)

        bl = context.get("batch_losses", [])
        if bl:
            d = sum(b["D"] for b in bl)/len(bl)
            g = sum(b["G"] for b in bl)/len(bl)
            context["board"].log_scalars("GAN/Loss", {"D": d, "G": g}, epoch)
            bl.clear()

    def evaluate_loss(self,
                      preds_real=None,
                      preds_fake=None,
                      mode="D"):
        if mode == "D":
            crit = nn.BCEWithLogitsLoss()
            real_lbl = torch.full_like(preds_real, 0.9)  # label smoothing
            fake_lbl = torch.zeros_like(preds_fake)
            loss_real = crit(preds_real, real_lbl)
            loss_fake = crit(preds_fake, fake_lbl)
            return loss_real + loss_fake

        elif mode == "G":
            crit = nn.BCEWithLogitsLoss()
            target = torch.ones_like(preds_fake)
            return crit(preds_fake, target)

        else:
            raise ValueError(f"Modo desconhecido: {mode}")

    def get_optimizers(self, lr_g, lr_d):
        opt_G = torch.optim.Adam(self.generator.parameters(),
                                 lr=lr_g, betas=(0.5, 0.999))
        opt_D = torch.optim.Adam(self.discriminator.parameters(),
                                 lr=lr_d, betas=(0.5, 0.999))
        return opt_G, opt_D
    def post_epoch_hook(self, context):
        epoch = context["epoch"]
        board = context["board"]

        # Log de imagens fixas
        self._log_images(context, epoch)

        # Acessa losses por batch
        bl = context.get("batch_losses", [])
        if bl:
            d_losses = [b["D"] for b in bl]
            g_losses = [b["G"] for b in bl]

            d_mean = sum(d_losses) / len(d_losses)
            g_mean = sum(g_losses) / len(g_losses)
            board.log_scalars("GAN/Loss", {"D": d_mean, "G": g_mean}, epoch)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(d_losses, label="Discriminator Loss", alpha=0.8)
            ax.plot(g_losses, label="Generator Loss", alpha=0.8)
            ax.set_title(f"Losses por batch — Época {epoch}")
            ax.set_xlabel("Batch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)

            fig.savefig(f"losses_por_batch_epoch_latest.png")
            plt.close(fig)
            bl.clear()



    def forward(self, z):
        x = self.fc(z).view(-1, self.init_channels, 4, 4)
        return self.generator(x)             

    def _log_images(self, context, epoch, n_samples=16):
        board  = context["board"]
        device = context["device"]
        if not hasattr(self, "_fixed_noise"):
            self._fixed_noise = torch.randn(n_samples, self.z_dim, device=device)

        with torch.no_grad():
            imgs = self(self._fixed_noise)
            grid = make_grid(imgs, nrow=int(n_samples**0.5),
                             normalize=True, value_range=(-1,1))
            board.log_image("GAN/Samples", grid, epoch)


if __name__ == "__main__":
    pass
