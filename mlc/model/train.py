import argparse
import nvtx
import torch
from tqdm import tqdm
from ..command.base import Base
from ..util.board import Board
from ..util.model import load_model_from_path, save_checkpoint, save_metadata
from ..util.resources import get_available_datasets, get_available_models
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


class Train(Base):
    def __init__(self, args):
        super().__init__(args)

        self.device = "cpu"
        if args["device"] == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError("CUDA is not available")

        self.learning_rate = args["learning_rate"]
        self.batch_size = args["batch_size"]
        self.loss = args["loss"]
        self.gan = (args["gan"] == "gan")

    @classmethod
    def name(cls):
        return "model.train"

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("-s", "--seed", type=int, default=42)
        parser.add_argument("-e", "--epochs", type=int, required=True)
        parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cuda")
        parser.add_argument("-o", "--loss", choices=["mse", "bce"], default="mse", help="Loss function")
        parser.add_argument("-l", "--learning-rate", type=float, default=0.0001)
        parser.add_argument("-b", "--batch-size", type=int, default=32)
        parser.add_argument("-c", "--check-point", type=int, default=10, help="Check point every n epochs")
        parser.add_argument("-t", "--tensorboard", action="store_true", help="Enable tensorboard logging")
        parser.add_argument("-g", "--gan", choices=["gan", "n"], default="n", help="GAN training mode")
        parser.set_defaults(tensorboard=False)
        parser.add_argument("-p", "--personal", action="store_true", help="Enable personal folder")
        parser.set_defaults(personal=False)

        datasets = list(get_available_datasets().keys())
        model_subparsers = parser.add_subparsers(dest="model", help="Model to train")

        for model_name, model_class in get_available_models().items():
            model_parser = model_subparsers.add_parser(model_name, help=model_class.__doc__)
            model_parser.add_argument("--continue-from", type=str, help="Continue training from checkpoint")
            model_class.add_arguments(model_parser)
            model_parser.add_argument("dataset", choices=datasets, help="Dataset name")
            model_parser.add_argument("dataset_args", nargs=argparse.REMAINDER, help="Arguments to the dataset")

    def run(self):
        nvtx.push_range("Training Session")
        #torch.autograd.set_detect_anomaly(True) 
        
        dataset_class = get_available_datasets()[self.args["dataset"]]
        dataset_parser = argparse.ArgumentParser(usage="... [dataset options]")
        dataset_class.add_arguments(dataset_parser)
        dataset_args = dataset_parser.parse_args(self.args["dataset_args"])
        dataset = dataset_class(vars(dataset_args))

        train_data = dataset.get_fold("train")
        validation_data = dataset.get_fold("validation")

        train_data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        validation_data_loader = torch.utils.data.DataLoader(
            validation_data, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )

        model_class = get_available_models()[self.args["model"]]
        model = None

        if self.args["continue_from"] is not None:
            print("Loading model from checkpoint:", self.args["continue_from"])
            model = load_model_from_path(self.args["continue_from"], self.args["personal"])
        else:
            model = model_class(self.args)

        model.to(self.device)

        context = {
            "model": model,
            "dataset": dataset,
            "train_data_loader": train_data_loader,
            "validation_data_loader": validation_data_loader,
            "board": Board(self.args["model"], use_personal_folder=self.args["personal"], enabled=self.args["tensorboard"]),
            "epoch": 0,
            "device": self.device,
            "loss": self.args["loss"],
            "gan": self.gan
        }

        if hasattr(context["board"].writer, "log_dir"):
            print("TB dir:", context["board"].writer.log_dir)
        else:
            print("Este writer não grava. Flag -t chegou como False.")

        if self.gan:
            if model.__class__.__name__.lower() == "vaegan":
                opt_Enc, opt_Dec, opt_D = model.get_optimizers(lr_g=self.learning_rate, lr_d=self.learning_rate)
                context["optimizer_Enc"] = opt_Enc
                context["optimizer_Dec"] = opt_Dec
                context["optimizer_D"] = opt_D
            else:
                opt_G, opt_D = model.get_optimizers(lr_g=self.learning_rate, lr_d=self.learning_rate)
                context["optimizer_G"] = opt_G
                context["optimizer_D"] = opt_D
        else:
            optimizer = model.get_optimizer(learning_rate=self.learning_rate)
            context["optimizer"] = optimizer

        try:
            pbar = tqdm(range(1, self.args["epochs"] + 1), desc="Epoch")
            for epoch in pbar:
                nvtx.push_range("Epoch")
                context["epoch"] = epoch
                model.pre_epoch_hook(context)
                model.train()
                total_train_loss = 0
                context["batch_losses"] = []
                batch_idx = 0

                for X_train, Y_train in tqdm(train_data_loader, leave=False, desc="Train"):
                    nvtx.push_range("Batch")
                    X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
                    model.pre_train_batch_hook(context, X_train, Y_train)

                    if context["gan"]:
                        is_vaegan = model.__class__.__name__.lower() == "vaegan"
                        B = X_train.size(0)
                        z = torch.randn(B, model.z_dim, device=self.device)
                        X_train = X_train.clone() + 0.05 * torch.randn_like(X_train)

                        if is_vaegan:
                            gamma = model.gamma  

                            context["optimizer_Enc"].zero_grad()
                            context["optimizer_Dec"].zero_grad()

                            x_recon, mu, logvar = model(X_train)
                            z = model.reparameterize(mu, logvar)

                            with torch.no_grad():
                                z_var = z.var().item()
                            context["board"].log_scalar("VAEGAN/z_variance", z_var, epoch)

                            
                            recon_loss = model.reconstruction_loss(x_recon, X_train)
                            kl_loss = model.kl_divergence(mu, logvar)
                            
                            pred_fake_recon = model.discriminator(x_recon)
                            adv_loss = model.evaluate_loss(preds_fake=pred_fake_recon, mode="G")
                            
                            if epoch < 10:
                                beta = 0.0
                            elif epoch < 30:
                                beta = (epoch - 10) / 20
                            else:
                                beta = 1.0

                            gamma = min(1.0, epoch / 10) * model.gamma 
                            z_norm = z.pow(2).mean()
                            loss_G_total = recon_loss + beta * kl_loss + gamma * adv_loss + 1e-4 * z_norm
                            loss_G_total.backward()
                            total_train_loss += loss_G_total.item() * B


                            context["optimizer_Enc"].step()
                            context["optimizer_Dec"].step()

                            context["optimizer_D"].zero_grad()

                            z_p = torch.randn(B, model.z_dim, device=self.device)
                            x_fake_noise = model.generate(z_p).detach()

                            pred_real = model.discriminator(X_train)
                            pred_fake = model.discriminator(x_fake_noise)

                            loss_D = model.evaluate_loss(preds_real=pred_real, preds_fake=pred_fake, mode="D")
                            loss_D.backward()
                            context["optimizer_D"].step()
                            
                            context["batch_losses"].append({
                                "D": loss_D.item(),
                                "G": adv_loss.item(),
                                "Recon": recon_loss.item(),
                                "KL": kl_loss.item()
                            })

                            model.post_train_batch_hook(context, X_train, None, None, {})
                            nvtx.pop_range()
                            batch_idx += 1
                            continue

                        else:
                            if batch_idx % 2 == 0:
                                with torch.no_grad():
                                    fake_imgs = model(z)
                                context["optimizer_D"].zero_grad()
                                pred_real = model.discriminator(X_train)
                                pred_fake = model.discriminator(fake_imgs)
                                loss_D = model.evaluate_loss(pred_real, pred_fake, mode="D")
                                loss_D.backward()
                                context["optimizer_D"].step()
                            else:
                                loss_D = torch.tensor(0.0)

                            z = torch.randn(B, model.z_dim, device=self.device)
                            fake_imgs = model(z)
                            context["optimizer_G"].zero_grad()
                            pred_fake_G = model.discriminator(fake_imgs)
                            loss_G = model.evaluate_loss(preds_fake=pred_fake_G, mode="G")
                            loss_G.backward()
                            context["optimizer_G"].step()

                            total_train_loss += loss_G.item() * B
                            context["batch_losses"].append({"D": loss_D.item(), "G": loss_G.item()})
                            model.post_train_batch_hook(context, X_train, Y_train, None, None)
                            nvtx.pop_range()
                            batch_idx += 1
                            continue

                    else:
                        optimizer.zero_grad()
                        Y_train_pred = model(X_train)
                        train_loss = model.evaluate_loss(Y_train_pred, Y_train, loss=self.loss)
                        train_loss.backward()
                        optimizer.step()

                        model.post_train_batch_hook(context, X_train, Y_train, Y_train_pred, train_loss)
                        total_train_loss += train_loss.item() * len(X_train)
                        nvtx.pop_range()

                if not context["gan"]:
                    total_train_loss /= len(train_data_loader.dataset)

                model.post_train_hook(context)
                nvtx.pop_range()

                model.eval()
                total_validation_loss = 0

                with torch.no_grad():
                    nvtx.push_range("Validation")
                    model.pre_validation_hook(context)

                    if not context["gan"]:
                        for X_val, Y_val in tqdm(validation_data_loader, leave=False, desc="Validation"):
                            nvtx.push_range("Batch")
                            model.pre_validation_batch_hook(context, X_val, Y_val)
                            X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
                            Y_val_pred = model(X_val)
                            val_loss = model.evaluate_loss(Y_val_pred, Y_val, loss=self.loss)
                            total_validation_loss += val_loss.item() * len(X_val)
                            model.post_validation_batch_hook(context, X_val, Y_val, Y_val_pred, val_loss)
                            nvtx.pop_range()
                        total_validation_loss /= len(validation_data_loader.dataset)
                    elif model.__class__.__name__.lower() == "vaegan":
                        total_validation_loss = 0
                        with torch.no_grad():
                            for X_val, _ in tqdm(validation_data_loader, leave=False, desc="Validation"):
                                X_val = X_val.to(self.device)
                                x_recon, mu, logvar = model(X_val)
                                pred_fake_recon = model.discriminator(x_recon)
                                recon_loss = model.reconstruction_loss(x_recon, X_val)
                                kl_loss = model.kl_divergence(mu, logvar)
                                adv_loss = model.evaluate_loss(preds_fake=pred_fake_recon, mode="G")
                                total_validation_loss += (recon_loss + kl_loss + adv_loss).item() * len(X_val)
                        total_validation_loss /= len(validation_data_loader.dataset)


                    model.post_validation_hook(context)
                    nvtx.pop_range()

                pbar.set_description(f"Epoch {epoch}, loss [t/v]: {total_train_loss:.5f}/{total_validation_loss:.5f}")
                model.post_epoch_hook(context)

                if (epoch % self.args["check_point"] == 0) or epoch == self.args["epochs"]:
                    save_checkpoint(model, epoch, use_personal_folder=self.args["personal"])

                if not self.gan:
                    context["board"].log_scalars("Curves/Loss", {"Train": total_train_loss, "Validation": total_validation_loss}, epoch)
                context["board"].log_layer_gradients(model, epoch)


        except KeyboardInterrupt:
            print("Training interrupted")
        finally:
            context["board"].close()
            nvtx.pop_range()
