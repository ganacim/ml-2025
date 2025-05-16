import argparse
import re

import nvtx
import torch
from tqdm import tqdm

from ..command.base import Base
from ..util.board import Board
from ..util.model import load_model_from_path, save_checkpoint, save_metadata
from ..util.resources import get_available_datasets, get_available_models


class TrainGAN(Base):

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
        return "model.traingan"

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
        parser.add_argument("-l", "--learning-rate", type=float, default=0.0001)
        parser.add_argument("-b", "--batch-size", type=int, default=32)
        parser.add_argument("-c", "--check-point", type=int, default=10, help="Check point every n epochs")
        parser.add_argument("-t", "--tensorboard", action="store_true", help="Enable tensorboard logging")
        parser.set_defaults(tensorboard=False)
        parser.add_argument("-p", "--personal", action="store_true", help="Enable personal folder")
        parser.set_defaults(personal=False)
        parser.add_argument("-n", "--name", type=str, default=None, help="Name this run")
        parser.add_argument("-w", "--weight-decay", type=float, default=0.0, help="Weight decay for optimizer")

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
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
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

        # create optimizer
        discriminator_optimizer = model.get_discriminator_optimizer(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )
        generator_optimizer = model.get_generator_optimizer(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay
        )

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

        try:  # let's catch keyboard interrupt
            pbar = tqdm(range(1 + delta_e, self.args["epochs"] + 1 + delta_e))
            pbar.set_description("Epoch")
            for epoch in pbar:
                nvtx.push_range("Epoch")
                context["epoch"] = epoch
                # call pre_epoch_hook
                model.pre_epoch_hook(context)
                # set model for training
                model.train()
                total_discriminator_train_loss = 0
                total_generator_train_loss = 0
                pbar_train = tqdm(train_data_loader, leave=False)
                pbar_train.set_description("Train")
                nvtx.push_range("Train")
                model.pre_train_hook(context)
                for b, (X_train, Y_train) in enumerate(pbar_train):
                    nvtx.push_range("Batch")
                    context["batch_number"] = b
                    # send data to device in batches
                    # this is suboptimal, we should send the whole dataset to the device if possible
                    X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)

                    # call pre_batch_hook
                    model.pre_train_batch_hook(context, X_train, Y_train)

                    discriminator_optimizer.zero_grad()
                    Y_train_discriminator_pred = model.discriminator_forward(X_train)
                    discriminator_train_loss = model.evaluate_discriminator_loss(
                        Y_train_discriminator_pred, batch_size=len(X_train)
                    )
                    discriminator_train_loss.backward()
                    discriminator_optimizer.step()

                    generator_optimizer.zero_grad()
                    Y_train_generator_pred = model.generator_forward(batch_size=len(X_train))
                    generator_train_loss = model.evaluate_generator_loss(Y_train_generator_pred)
                    generator_train_loss.backward()
                    generator_optimizer.step()

                    # call post_batch_hook
                    # model.post_train_batch_hook(context, X_train, Y_train_pred, Y_train, train_loss)
                    #
                    total_discriminator_train_loss += discriminator_train_loss.item() * len(X_train)
                    total_generator_train_loss += generator_train_loss.item() * len(X_train)
                    #
                    nvtx.pop_range()  # Batch
                # normalize loss
                total_discriminator_train_loss /= len(train_data_loader.dataset)
                total_generator_train_loss /= len(train_data_loader.dataset)
                model.post_train_hook(context)
                nvtx.pop_range()  # Train

                model.eval()
                total_discriminator_validation_loss = 0
                total_generator_validation_loss = 0
                with torch.no_grad():
                    nvtx.push_range("Validation")
                    model.pre_validation_hook(context)
                    pbar_validation = tqdm(validation_data_loader, leave=False)
                    pbar_validation.set_description("Validation")
                    for b, (X_val, Y_val) in enumerate(pbar_validation):
                        nvtx.push_range("Batch")
                        context["batch_number"] = b
                        model.pre_validation_batch_hook(context, X_val, Y_val)
                        X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)

                        Y_val_discriminator_pred = model.discriminator_forward(X_val)
                        discriminator_val_loss = model.evaluate_discriminator_loss(
                            Y_val_discriminator_pred, batch_size=len(X_val)
                        )

                        Y_val_generator_pred = model.generator_forward(batch_size=len(X_val))
                        generator_val_loss = model.evaluate_generator_loss(Y_val_generator_pred)

                        total_discriminator_validation_loss += discriminator_val_loss.item() * len(X_val)
                        total_generator_validation_loss += generator_val_loss.item() * len(X_val)

                        # model.post_validation_batch_hook(context, X_val, Y_val_pred, Y_val, val_loss)
                        nvtx.pop_range()  # Batch

                    model.post_validation_hook(context)
                    # normalize loss
                    total_discriminator_validation_loss /= len(validation_data_loader.dataset)
                    total_generator_validation_loss /= len(validation_data_loader.dataset)
                    nvtx.pop_range()  # Validation

                nvtx.pop_range()  # Epoch

                # call post_epoch_hook
                model.post_epoch_hook(context)

                # save model if checkpoint or last epoch
                if (epoch % self.args["check_point"] == 0) or epoch == self.args["epochs"]:
                    save_checkpoint(model, epoch, use_personal_folder=self.args["personal"])

                # log to tensorboard
                board.log_scalars(
                    "Curves/Loss",
                    {
                        "Discriminator_Train": total_discriminator_train_loss,
                        "Discriminator_Validation": total_discriminator_validation_loss,
                        "Generator_Train": total_generator_train_loss,
                        "Generator_Validation": total_generator_validation_loss,
                    },
                    epoch,
                )
                board.log_layer_gradients(model, epoch)

        except KeyboardInterrupt:
            print("Training interrupted")
        finally:
            board.close()

        nvtx.pop_range()  # Training
