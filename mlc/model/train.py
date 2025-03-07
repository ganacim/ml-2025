import argparse

import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..command.base import Base
from ..data import get_available_datasets
from ..util.model import save_checkpoint, save_metadata
from ..util.plot import plot_2d_model_ax
from . import get_available_models


class Train(Base):
    name = "model.train"

    def __init__(self, args):
        super().__init__(args)

        # try to use the device specified in the arguments
        self.device = "cpu"
        if args.device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError("CUDA is not available")

        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("-s", "--seed", type=int, default=42)  # TODO: use seed
        parser.add_argument("-e", "--epochs", type=int, required=True)
        parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cuda")
        parser.add_argument("-l", "--learning-rate", type=float, default=0.0001)
        parser.add_argument("-b", "--batch-size", type=int, default=32)
        parser.add_argument("-c", "--check-point", type=int, default=10, help="Check point every n epochs")
        # get dataset names
        datasets = list(get_available_datasets().keys())
        # add param for model name
        model_subparsers = parser.add_subparsers(dest="model", help="Model to train")
        for model_name, model_class in get_available_models().items():
            model_parser = model_subparsers.add_parser(model_name, help=model_class.__doc__)
            model_class.add_arguments(model_parser)
            model_parser.add_argument("dataset", choices=datasets, help="Dataset name")
            # collect all remaining arguments for use by the dataset parser
            model_parser.add_argument("dataset_args", nargs=argparse.REMAINDER, help="Arguments to the dataset")

    def run(self):
        # process dataset arguments
        dataset_class = get_available_datasets()[self.args.dataset]
        dataset_parser = argparse.ArgumentParser(usage="... [dataset options]")
        dataset_class.add_arguments(dataset_parser)
        dataset_args = dataset_parser.parse_args(self.args.dataset_args)

        # create dataset instance
        dataset_args_dict = vars(dataset_args)  # convert arguments to dictionary
        dataset = dataset_class(dataset_args_dict)

        # load data
        train_data = dataset.get_fold("train")
        validation_data = dataset.get_fold("validation")

        # create torch dataloaders
        train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        validation_data_loader = torch.utils.data.DataLoader(validation_data, batch_size=len(validation_data))

        # create model
        model_class = get_available_models()[self.args.model]
        args_dict = vars(self.args)  # convert arguments to dictionary
        model = model_class(args_dict).to(self.device)

        # create optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        # create loss function
        loss_fn = torch.nn.BCELoss()

        # training loop
        train_losses = []
        validation_losses = []

        # save session metadata
        save_metadata(model, dataset)

        pbar = tqdm(range(self.args.epochs))
        for epoch in pbar:
            # set model for training
            model.train()
            total_train_loss = 0
            for X_train, Y_train in train_data_loader:
                # send data to device in batches
                # this is suboptimal, we should send the whole dataset to the device if possible
                X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)

                optimizer.zero_grad()
                Y_train_pred = model(X_train)
                train_loss = loss_fn(Y_train_pred, Y_train)
                train_loss.backward()
                optimizer.step()

                total_train_loss += train_loss.item() * len(X_train)

            train_losses.append(total_train_loss / len(train_data))

            model.eval()
            with torch.no_grad():
                X_val, Y_val = next(iter(validation_data_loader))
                X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)

                Y_val_pred = model(X_val)
                loss = loss_fn(Y_val_pred, Y_val)

                validation_losses.append(loss.item())

            pbar.set_description(f"Epoch {epoch}, loss [t/v]: {train_losses[-1]:0.5f}/{validation_losses[-1]:0.5f}")

            # save model if checkpoint or last epoch
            if ((epoch + 1) % self.args.check_point == 0) or (epoch == self.args.epochs - 1):
                save_checkpoint(model, epoch)

        plt.figure()
        plt.plot(train_losses, label="train")
        plt.plot(validation_losses, label="validation")
        plt.legend()
        plt.savefig("training_and_validation_loss.png")
        plt.close()

        X_val, Y_val = next(iter(validation_data_loader))
        X_val, Y_val = X_val, torch.argmax(Y_val, dim=1)
        fig, ax = plt.subplots(figsize=(10, 10))
        plot_2d_model_ax(ax, X_val, Y_val, model)
        fig.savefig("model.png")
        plt.close()

        # save model
