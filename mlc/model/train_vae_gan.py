# mlc/model/train_vae_gan.py

import argparse
import re
import nvtx
import torch
import torch.nn.functional as F
import torch.optim.lr_scheduler as lrs
from tqdm import tqdm

from ..command.base import Base
from ..util.board import Board
from ..util.model import load_model_from_path, save_checkpoint, save_metadata
from ..util.resources import get_available_datasets, get_available_models


class TrainVAEGAN(Base):
    """
    Train VAE/GAN with separated optimizers, equilibrium gating, and hybrid loss.
    """

    def __init__(self, args):
        super().__init__(args)

        # Device setup
        self.device = "cpu"
        if args["device"].startswith("cuda"):
            if torch.cuda.is_available():
                self.device = torch.device(args["device"])
            else:
                raise RuntimeError("CUDA is not available")

        self.learning_rate = args["learning_rate"]
        self.batch_size    = args["batch_size"]
        self.weight_decay  = args["weight_decay"]

    @classmethod
    def name(cls):
        return "model.train_vae_gan"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("-s", "--seed", type=int, default=42)
        parser.add_argument("-e", "--epochs", type=int, required=True)
        parser.add_argument("-d", "--device", type=str, default="cuda")
        parser.add_argument("-l", "--learning-rate", type=float, default=0.0001)
        parser.add_argument("-b", "--batch-size", type=int, default=32)
        parser.add_argument("-c", "--check-point", type=int, default=0,
                            help="Checkpoint every n epochs. Set to 0 to disable (default: 0).")
        parser.add_argument("-t", "--tensorboard", action="store_true")
        parser.add_argument("-p", "--personal", action="store_true")
        parser.add_argument("-n", "--name", type=str, default=None)
        parser.add_argument("-w", "--weight-decay", type=float, default=0.0)
        
        datasets = list(get_available_datasets().keys())
        model_subparsers = parser.add_subparsers(dest="model", help="Model to train")
        for model_name, model_class in get_available_models().items():
            mparser = model_subparsers.add_parser(model_name, help=model_class.__doc__)
            model_class.add_arguments(mparser)
            mparser.add_argument("--continue-from", type=str)
            mparser.add_argument("dataset", choices=datasets)
            mparser.add_argument("dataset_args", nargs=argparse.REMAINDER)

    def run(self):
        nvtx.push_range("Training Session")

        # --- Dataset setup ---
        dataset_class = get_available_datasets()[self.args["dataset"]]
        ds_parser = argparse.ArgumentParser()
        dataset_class.add_arguments(ds_parser)
        ds_args = ds_parser.parse_args(self.args["dataset_args"])
        dataset = dataset_class(vars(ds_args))

        train_loader = torch.utils.data.DataLoader(dataset.get_fold("train"), batch_size=self.batch_size, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(dataset.get_fold("validation"), batch_size=self.batch_size, num_workers=4)

        # --- Model setup ---
        model_class = get_available_models()[self.args["model"]]
        model = model_class(self.args)
        model.to(self.device)
        model.initialize()

        # --- Optimizers & Schedulers ---
        d_optimizer = model.get_discriminator_optimizer(self.learning_rate, weight_decay=self.weight_decay)
        enc_optimizer = model.get_encoder_optimizer(self.learning_rate, weight_decay=self.weight_decay)
        dec_optimizer = model.get_decoder_optimizer(self.learning_rate, weight_decay=self.weight_decay)
        
        d_scheduler = lrs.StepLR(d_optimizer, step_size=100, gamma=0.5)
        enc_scheduler = lrs.StepLR(enc_optimizer, step_size=100, gamma=0.5)
        dec_scheduler = lrs.StepLR(dec_optimizer, step_size=100, gamma=0.5)

        save_metadata(model, dataset, use_personal_folder=self.args["personal"], name=self.args["name"])

        board = Board(self.args["model"], enabled=self.args["tensorboard"])

    
        print("Fetching a fixed batch of validation images for logging...")
        fixed_val_images, _ = next(iter(val_loader))
        fixed_val_images = fixed_val_images.to(self.device)

        if self.args["tensorboard"]:
            print("Logging original validation images to TensorBoard...")
            num_to_log = min(8, fixed_val_images.size(0))
            for i in range(num_to_log):
                board.log_image(f"Validation_Images/Comparison_{i}", fixed_val_images[i], 0)

        try:
            global_step = 0
            pbar_epochs = tqdm(range(1, self.args["epochs"] + 1), desc="Epochs")
            for epoch in pbar_epochs:
                nvtx.push_range(f"Epoch {epoch}")
                model.train()
                
                # Initialize accumulators for epoch losses
                total_kl, total_advg, total_advd = 0.0, 0.0, 0.0
                total_recon_loss, total_feat_loss, total_pix_loss, total_loss_scale = 0.0, 0.0, 0.0, 0.0
                total_dx_perf, total_dgz_perf = 0.0, 0.0
                # ADDED: Initialize counters for gating summary
                decoder_updates_skipped = 0
                discriminator_updates_skipped = 0
                
                pbar_batch = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False)
                for i, (X_train, _) in enumerate(pbar_batch):
                    nvtx.push_range("Batch")
                    X_train = X_train.to(self.device)
                    
                    z_mu, z_logvar = model.encode(X_train)
                    z = model.reparameterize(z_mu, z_logvar)
                    X_recon = model.decode(z)

                    kl_loss = model.kl_divergence(z_mu, z_logvar).mean()
                    
                    pixel_loss = model.pixel_mse(X_recon, X_train)
                    if model.recon_depth > 0:
                        with torch.no_grad():
                            feat_real = model.discriminator_feat(X_train)
                        feat_fake = model.discriminator_feat(X_recon)
                        feature_loss = F.mse_loss(feat_fake, feat_real)
                        
                        loss_scale = feature_loss.detach() / (pixel_loss.detach() + 1e-8)
                        reconstruction_loss = feature_loss + loss_scale * pixel_loss
                    else:
                        feature_loss = torch.tensor(0.0, device=self.device)
                        loss_scale = torch.tensor(0.0, device=self.device)
                        reconstruction_loss = pixel_loss

                    d_fake_for_g = model.discriminator(X_recon)
                    g_adv_loss = model.bce(d_fake_for_g, torch.ones_like(d_fake_for_g))

                    enc_optimizer.zero_grad()
                    dec_optimizer.zero_grad()
                    
                    if model.kl_anneal_steps > 0:
                        beta = min(1.0, global_step / model.kl_anneal_steps)
                    else:
                        beta = 1.0
                    
                    loss_enc = (beta * kl_loss) + reconstruction_loss
                    loss_enc.backward(retain_graph=True)
                    
                    gamma = model.gamma
                    loss_dec = (gamma * reconstruction_loss) + ((1 - gamma) * g_adv_loss)
                    loss_dec.backward()

                    enc_optimizer.step()

                    with torch.no_grad():
                        d_real_perf = model.discriminator(X_train).mean().item()
                        d_fake_perf = model.discriminator(X_recon).mean().item()
                    
                    dec_update, dis_update = True, True

                    if d_real_perf < model.equilibrium - model.margin or d_fake_perf > model.equilibrium + model.margin:
                        dec_update = False
                        decoder_updates_skipped += 1
                    
                    if d_real_perf > model.equilibrium + model.margin or d_fake_perf < model.equilibrium - model.margin:
                        dis_update = False
                        discriminator_updates_skipped += 1
                    
                    if not (dec_update or dis_update):
                        dec_update, dis_update = True, True

                    if dec_update:
                        dec_optimizer.step()

                    if dis_update:
                        d_optimizer.zero_grad()
                        d_real_out = model.discriminator(X_train)
                        d_loss_real = model.bce(d_real_out, torch.ones_like(d_real_out) * 0.9)
                        d_fake_out = model.discriminator(X_recon.detach())
                        d_loss_fake = model.bce(d_fake_out, torch.ones_like(d_fake_out) * 0.1)
                        d_loss = (d_loss_real + d_loss_fake) / 2
                        d_loss.backward()
                        d_optimizer.step()
                    else:
                        with torch.no_grad():
                            d_real_out = model.discriminator(X_train)
                            d_loss_real = model.bce(d_real_out, torch.ones_like(d_real_out) * 0.9)
                            d_fake_out = model.discriminator(X_recon.detach())
                            d_loss_fake = model.bce(d_fake_out, torch.ones_like(d_fake_out) * 0.1)
                            d_loss = (d_loss_real + d_loss_fake) / 2
                    
                    batch_size = X_train.size(0)
                    total_kl += kl_loss.item() * batch_size
                    total_advg += g_adv_loss.item() * batch_size
                    total_advd += d_loss.item() * batch_size
                    total_recon_loss += reconstruction_loss.item() * batch_size
                    total_feat_loss += feature_loss.item() * batch_size
                    total_pix_loss += pixel_loss.item() * batch_size
                    total_loss_scale += loss_scale.item() * batch_size
                    total_dx_perf += d_real_perf * batch_size
                    total_dgz_perf += d_fake_perf * batch_size
                    
                    global_step += 1
                    nvtx.pop_range() # Batch

                # === End of Training Batch Loop ===
                
                
                if decoder_updates_skipped > 0 or discriminator_updates_skipped > 0:
                    total_batches = len(train_loader)
                    print(f"  [Epoch {epoch} Gating Summary] "
                          f"Decoder updates skipped: {decoder_updates_skipped}/{total_batches} batches. | "
                          f"Discriminator updates skipped: {discriminator_updates_skipped}/{total_batches} batches.")
                
                N_train = len(train_loader.dataset)
                avg_recon_train = total_recon_loss / N_train
                avg_kl_train = total_kl / N_train
                avg_advg_train = total_advg / N_train
                vae_gen_train_loss = avg_recon_train + avg_kl_train + avg_advg_train
                discriminator_train_loss = total_advd / N_train
                D_x = total_dx_perf / N_train
                DG_z1 = total_dgz_perf / N_train
                avg_feat_loss = total_feat_loss / N_train
                avg_pix_loss = total_pix_loss / N_train
                avg_loss_scale = total_loss_scale / len(train_loader)

                # Validation pass
                model.eval()
                val_recon, val_kl, val_advg = 0.0, 0.0, 0.0
                with torch.no_grad():
                    for X_val, _ in val_loader:
                        X_val = X_val.to(self.device)
                        z_mu_val, z_logvar_val = model.encode(X_val)
                        z_val = model.reparameterize(z_mu_val, z_logvar_val)
                        X_recon_val = model.decode(z_val)

                        val_kl += model.kl_divergence(z_mu_val, z_logvar_val).sum().item()
                        
                        val_pixel_loss = model.pixel_mse(X_recon_val, X_val)
                        if model.recon_depth > 0:
                            feat_real_v = model.discriminator_feat(X_val)
                            feat_fake_v = model.discriminator_feat(X_recon_val)
                            val_feature_loss = F.mse_loss(feat_fake_v, feat_real_v)
                            val_loss_scale = val_feature_loss / (val_pixel_loss + 1e-8)
                            val_recon += (val_feature_loss + val_loss_scale * val_pixel_loss).item() * X_val.size(0)
                        else:
                            val_recon += val_pixel_loss.item() * X_val.size(0)

                        d_fake_g_v = model.discriminator(X_recon_val)
                        val_advg += model.bce(d_fake_g_v, torch.ones_like(d_fake_g_v)).item() * X_val.size(0)

                N_val = len(val_loader.dataset)
                avg_recon_v = val_recon / N_val
                avg_kl_v = val_kl / N_val
                avg_advg_v = val_advg / N_val
                vae_gen_val_loss = avg_recon_v + avg_kl_v + avg_advg_v

                # === LOGGING, CHECKPOINTING, AND PBAR UPDATE ===

                pbar_epochs.set_description(f"Epoch {epoch}, VAE/Gen Loss [train/val]: {vae_gen_train_loss:.5f}/{vae_gen_val_loss:.5f}")

                board.log_scalars("Curves/VAE_Generator_Loss", {"Train": vae_gen_train_loss, "Validation": vae_gen_val_loss}, epoch)
                
                board.log_scalars(
                    "Curves/Epoch_Losses",
                    {
                        "Discriminator_Train": discriminator_train_loss,
                        "VAE_Generator_Train": vae_gen_train_loss,
                        "KL_Divergence": avg_kl_train,
                        "D(x)": D_x,
                        "DG(z)_1": DG_z1,
                    },
                    epoch
                )

                board.log_scalars(
                    "Curves/Reconstruction_Loss_Components",
                    {
                        "Feature_Loss_Raw": avg_feat_loss,
                        "Pixel_Loss_Raw": avg_pix_loss,
                        "Adaptive_Scale_Factor": avg_loss_scale,
                    },
                    epoch
                )

                board.log_scalars("Curves/Annealing", {"KL_Beta": beta}, epoch)

                board.log_layer_gradients(model, epoch)
                
                with torch.no_grad():
                    recon_images = model(fixed_val_images)
                    num_to_log = min(8, recon_images.size(0))
                    for i in range(num_to_log):
                        board.log_image(f"Validation_Images/Comparison_{i}", recon_images[i], epoch)
                
                if self.args["check_point"] > 0:
                    if (epoch % self.args["check_point"] == 0) or epoch == self.args["epochs"]:
                        save_checkpoint(model, epoch, use_personal_folder=self.args["personal"])

                nvtx.pop_range() # Epoch
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user.")

        finally:
            board.close()
            print("Training finished or interrupted. Board closed.")
            nvtx.pop_range() # Training Session