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


class TrainGANIntro(Base):

    def __init__(self, args):
        super().__init__(args)

        # try to use the device specified in the arguments
        self.device = "cpu"
        if args["device"].startswith("cuda"):
            if torch.cuda.is_available():
                self.device = torch.device(args["device"])
            else:
                raise RuntimeError("CUDA is not available")

        self.learning_rate = args["learning_rate"]
        self.batch_size = args["batch_size"]
        self.weight_decay = args["weight_decay"]

    @classmethod
    def name(cls):
        return "model.traingan_intro"

    @staticmethod
    def add_arguments(parser):
        def _parse_device_arg(arg_value):
            pattern = re.compile(r"(cpu|cuda|cuda:\d+)")
            if not pattern.match(arg_value):
                raise argparse.ArgumentTypeError("invalid value")
            return arg_value

        parser.add_argument("-s", "--seed", type=int, default=42)  # TODO: use seed
        parser.add_argument("-e", "--epochs", type=int, required=True)
        parser.add_argument("-d", "--device", type=_parse_device_arg, default="cuda", help="Device to use for training")
        parser.add_argument("-l", "--learning-rate", type=float, default=0.00001)
        parser.add_argument("-b", "--batch-size", type=int, default=32)
        parser.add_argument("-c", "--check-point", type=int, default=10, help="Check point every n epochs")
        parser.add_argument("-t", "--tensorboard", action="store_true", help="Enable tensorboard logging")
        parser.set_defaults(tensorboard=False)
        parser.add_argument("-p", "--personal", action="store_true", help="Enable personal folder")
        parser.set_defaults(personal=False)
        parser.add_argument("-n", "--name", type=str, default=None, help="Name this run")
        parser.add_argument("-w", "--weight-decay", type=float, default=0.0, help="Weight decay for optimizer")
        parser.add_argument("--noise", type=float, default=0.0, help="Noise to add to the input")

        # get dataset names
        datasets = list(get_available_datasets().keys())
        # add param for model name
        model_subparsers = parser.add_subparsers(dest="model", help="Model to train")
        for model_name, model_class in get_available_models().items():
            model_parser = model_subparsers.add_parser(model_name, help=model_class.__doc__)
            model_parser.add_argument("--continue-from", type=str, help="Continue training from checkpoint")
            model_class.add_arguments(model_parser)
            model_parser.add_argument("dataset", choices=datasets, help="Dataset name")
            # collect all remaining arguments for use by the dataset parser
            model_parser.add_argument("dataset_args", nargs=argparse.REMAINDER, help="Arguments to the dataset")


    def run(self):
        nvtx.push_range("Training Session")

        # process dataset arguments
        dataset_class = get_available_datasets()[self.args["dataset"]]
        dataset_parser = argparse.ArgumentParser(usage="... [dataset options]")
        dataset_class.add_arguments(dataset_parser)
        dataset_args = dataset_parser.parse_args(self.args["dataset_args"])

        # create dataset instance
        dataset_args_dict = vars(dataset_args)  # convert arguments to dictionary
        dataset = dataset_class(dataset_args_dict)

        # load data
        train_data = dataset.get_fold("train")
        validation_data = dataset.get_fold("validation")

        # create torch dataloaders
        train_data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True
        )
        validation_data_loader = torch.utils.data.DataLoader(
            validation_data, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )

        # create model
        model_class = get_available_models()[self.args["model"]]

        # create model instance
        model = None
        # load model from checkpoint if specified
        delta_e = 0
        if self.args["continue_from"] is not None:
            print("Loading model from checkpoint:", self.args["continue_from"])
            model, _, _, _, metadata = load_model_from_path(self.args["continue_from"], self.args["personal"])
            delta_e = metadata["model"]["args"]["epochs"]
        else:
            model = model_class(self.args)

        # send model to device
        model.to(self.device)

        # initialize model
        model.initialize()

        # create optimizer
        discriminator_optimizer = model.get_discriminator_optimizer(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )
        discriminator_scheduler = lrs.StepLR(discriminator_optimizer, step_size=1000, gamma=0.1)
        generator_optimizer = model.get_generator_optimizer(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )
        generetor_scheduler = lrs.StepLR(generator_optimizer, step_size=1000, gamma=0.1)

        # save session metadata
        save_metadata(model, dataset, use_personal_folder=self.args["personal"], name=self.args["name"])

        # initialize tensorboard
        board = Board(self.args["model"], use_personal_folder=self.args["personal"], enabled=self.args["tensorboard"])

        # create context dict for hooks
        context = {
            "model": model,
            "dataset": dataset,
            "train_data_loader": train_data_loader,
            "validation_data_loader": validation_data_loader,
            "optimizer": {"discriminator": discriminator_optimizer, "generator": generator_optimizer},
            "board": board,
            "epoch": 0,
            "epochs": self.args["epochs"],
            "batch_size": self.batch_size,
            "batch_number": 0,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "device": self.device,
        }

        def lerp(Data, Noise, t, T):
            a = min(1, max(0, t / T))
            return (1.0 - a) * Noise + a * Data
            # return Data
            


        try:  # let's catch keyboard interrupt
            pbar = tqdm(range(1 + delta_e, self.args["epochs"] + 1 + delta_e))
            pbar.set_description("Epoch")
            # T = self.args["epochs"] * len(train_data_loader)
            # T = int(0.5*self.args["epochs"] * len(train_data_loader))
            T = 200 * len(train_data_loader)
            # T = 1
            
            for epoch in pbar:
                nvtx.push_range("Epoch")
                context["epoch"] = epoch
                # call pre_epoch_hook
                model.pre_epoch_hook(context)
                # set model for training
                model.train()
                
                pbar_train = tqdm(train_data_loader, leave=False)
                pbar_train.set_description("Train")
                nvtx.push_range("Train")
                model.pre_train_hook(context)
                total_discriminator_train_loss = 0
                total_generator_train_loss = 0
                # D_x = 0
                # DG_z1 = 0
                # DG_z2 = 0
                Loss_d = 0
                Loss_g = 0
                for b, (X_train, Y_train) in enumerate(pbar_train):
                    nvtx.push_range("Batch")
                    context["batch_number"] = b
                    t = (epoch - 1) * len(train_data_loader) + b
                    context["round"] = t

                    # send data to device in batches
                    # this is suboptimal, we should send the whole dataset to the device if possible
                    X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)

                    # call pre_batch_hook
                    model.pre_train_batch_hook(context, X_train, Y_train)

                    # Set both models to train mode during training
                    model.discriminator.train()
                    model.generator.train()

                    # discriminator is training...
                    discriminator_optimizer.zero_grad()
                    generator_optimizer.zero_grad()


                    u, sigma = torch.tensor_split(model.discriminator(X_train), 2,dim =1)
                    u = u.to(model.device)
                    sigma = sigma.to(model.device)
                    z = u + sigma * torch.randn_like(u, device=model.device)
                    z.requires_grad_(True)
                    z.retain_grad()

                    zp = torch.randn_like(z, device=model.device)
                    Xr,Xp = model.generator(z), model.generator(zp)
                    Lae = model.Lae(Xr, X_train)

                    Zr,Zpp = model.discriminator(Xr), model.discriminator(Xp)
                    ur,sigmar = torch.tensor_split(Zr, 2, dim=1)
                    up, sigmap = torch.tensor_split(Zpp, 2, dim=1)
                    
                    Le = model.evaluate_loss_disc(u, sigma, ur, sigmar, up, sigmap)
                    Le += model.beta * Lae.mean()
                    Lg = model.evaluate_loss_gen(ur, sigmar, up, sigmap)
                    Lg += model.beta * Lae.mean()
                    Le.backward(retain_graph=True)
                    Lg.backward()
                    print("1")
                    
                    discriminator_optimizer.step()                  
                    generator_optimizer.step()
                    
                    Loss_d += Le.item()
                    Loss_g += Lg.item()
                    
                
                    # # train generator
                    # generator_optimizer.zero_grad()
                    # # lets use the allready generated data
                    # Z = torch.randn(X_train.size(0), model.latent_dimension(), device=self.device)
                    # Z.requires_grad_(True)
                    # X_fake = model.generator(Z)
                    # X_fake.requires_grad_(True)
                    # X_fake.retain_grad()
                    # Y_pred = torch.sigmoid(model.discriminator(X_fake))

                    # g_train_loss = F.binary_cross_entropy_with_logits(Y_pred, torch.ones_like(Y_pred))
                    # g_train_loss.backward()
                    # dg_z2 = torch.sum(Y_pred).item()
                    # DG_z2 += dg_z2  # torch.sum(Y_pred).item()
                    # z_grad = torch.norm(Z.grad, p=1, dim=1).mean().item()
                    # x_fake_grad = torch.norm(X_fake.grad, p=1, dim=1).mean().item()


                    if epoch%100==0:
                        model.lower_dropout()

                    board.log_scalars(
                        "Curves/Loss_Batch",
                        {
                            "Discriminator_Loss": (Loss_d / len(X_train)),
                            "Generator_Loss": (Loss_g / len(X_train)),
                            "Lae": (Lae.mean().item() / len(X_train)),
                    #         "D(x)": d_x / len(X_train),
                    #         "DG(z)_1": dg_z1 / len(X_train),
                    #         "DG(z)_2": dg_z2 / len(X_train),
                    #         "Z_grad_l1": z_grad,
                    #         "Xf_grad_l1": x_fake_grad,
                         },
                         t,
                 )
                    # print(epoch*len(train_data_loader) + b)

                    # call post_batch_hook
                    # model.post_train_batch_hook(context, X_train, Y_pred, Y_train, None)
                    

                    nvtx.pop_range()  # Batch
                discriminator_scheduler.step()
                generetor_scheduler.step()
                # normalize loss
                total_discriminator_train_loss /= len(train_data_loader.dataset)
                total_generator_train_loss /= len(train_data_loader.dataset)
                # D_x /= len(train_data_loader.dataset)
                # DG_z1 /= len(train_data_loader.dataset)
                # DG_z2 /= len(train_data_loader.dataset)

                # model.post_train_hook(context)
                nvtx.pop_range()  # Train

                nvtx.pop_range()  # Epoch

                # call post_epoch_hook
                model.post_epoch_hook(context)

                # save model if checkpoint or last epoch
                if (epoch % self.args["check_point"] == 0) or epoch == self.args["epochs"]:
                    save_checkpoint(model, epoch, use_personal_folder=self.args["personal"])

                #log to tensorboard
                board.log_scalars(
                    "Curves/Loss_Epoch",
                    {
                        "Discriminator_Loss": total_discriminator_train_loss,
                        "Generator_Loss": total_generator_train_loss,
                        "Lae": (Lae.mean().item() / len(train_data_loader.dataset)),
                #         "D(x)": D_x,
                #         "DG(z)_1": DG_z1,
                #         "DG(z)_2": DG_z2,
                    },
                    epoch,
                )
                board.log_layer_gradients(model.generator, epoch)

        except KeyboardInterrupt:
            print("Training interrupted")
        finally:
            board.close()

        nvtx.pop_range()  # Training
