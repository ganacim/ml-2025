import argparse

import nvtx
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix as sk_confusion_matrix

from ..command.base import Base
from ..util.board import Board
from ..util.model import save_checkpoint, save_metadata
from ..util.resources import get_available_datasets, get_available_models


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
        parser.add_argument("-t", "--tensorboard", action="store_true", help="Enable tensorboard logging")
        parser.set_defaults(tensorboard=False)
        parser.add_argument("-p", "--personal", action="store_true", help="Enable personal folder")
        parser.set_defaults(personal=False)
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
        nvtx.push_range("Training Session")
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
        train_data_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        validation_data_loader = torch.utils.data.DataLoader(
            validation_data, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )

        # create model
        model_class = get_available_models()[self.args.model]
        args_dict = vars(self.args)  # convert arguments to dictionary
        model = model_class(args_dict).to(self.device)

        # create optimizer
        optimizer = model.get_optimizer(learning_rate=self.learning_rate)

        # # create loss function
        # loss_fn = torch.nn.BCELoss()

        # training loop
        train_losses = []
        validation_losses = []

        train_confusion_matrix = torch.zeros(2, 2)

        # save session metadata
        save_metadata(model, dataset, use_personal_folder=self.args.personal)

        # initialize tensorboard
        board = Board(self.args.model, use_personal_folder=self.args.personal, enabled=self.args.tensorboard)

        # create context dict for hooks
        context = {
            "model": model,
            "dataset": dataset,
            "train_data_loader": train_data_loader,
            "validation_data_loader": validation_data_loader,
            "optimizer": optimizer,
            "board": board,
        }

        try:  # let's catch keyboard interrupt
            pbar = tqdm(range(self.args.epochs))
            pbar.set_description("Epoch")
            for epoch in pbar:
                nvtx.push_range("Epoch")
                # call pre_epoch_hook
                model.pre_epoch_hook(context)
                # set model for training
                model.train()
                total_train_loss = 0
                pbar_train = tqdm(train_data_loader, leave=False)
                pbar_train.set_description("Train")
                nvtx.push_range("Train")
                for X_train, Y_train in pbar_train:
                    nvtx.push_range("Batch")
                    # send data to device in batches
                    # this is suboptimal, we should send the whole dataset to the device if possible
                    X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)

                    # call pre_batch_hook
                    model.pre_batch_hook(context, X_train, Y_train)

                    optimizer.zero_grad()
                    Y_train_pred = model(X_train)
                    train_loss = model.evaluate_loss(Y_train_pred, Y_train)
                    train_loss.backward()
                    optimizer.step()

                    # Update confusion matrix
                    Y_pred_binary = (Y_train_pred.detach().cpu().numpy() > 0.5).astype(int)
                    train_confusion_matrix += sk_confusion_matrix(Y_train.detach().cpu().numpy(), Y_pred_binary)

                    # call post_batch_hook
                    model.post_batch_hook(context, X_train, Y_train, Y_train_pred, train_loss)
                    #
                    total_train_loss += train_loss.item() * len(X_train)
                    #
                    nvtx.pop_range()  # Batch
                nvtx.pop_range()  # Train

                train_losses.append(total_train_loss / len(train_data))


                val_confusion_matrix = torch.zeros(2, 2)

                model.eval()
                total_validation_loss = 0
                with torch.no_grad():
                    nvtx.push_range("Validation")
                    pbar_validation = tqdm(validation_data_loader, leave=False)
                    pbar_validation.set_description("Validation")
                    for X_val, Y_val in pbar_validation:
                        nvtx.push_range("Batch")
                        X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)

                        Y_val_pred = model(X_val)
                        loss = model.evaluate_loss(Y_val_pred, Y_val)

                        # Update confusion matrix
                        Y_pred_binary = (Y_val_pred.detach().cpu().numpy() > 0.5).astype(int)
                        val_confusion_matrix += sk_confusion_matrix(Y_val.detach().cpu().numpy(), Y_pred_binary)

                        total_validation_loss += loss.item() * len(X_val)

                        nvtx.pop_range()  # Batch

                    validation_losses.append(total_validation_loss / len(validation_data))
                    nvtx.pop_range()  # Validation

                nvtx.pop_range()  # Epoch

                pbar.set_description(f"Epoch {epoch}, loss [t/v]: {train_losses[-1]:0.5f}/{validation_losses[-1]:0.5f}")

                # call post_epoch_hook
                model.post_epoch_hook(context)

                # save model if checkpoint or last epoch
                if ((epoch + 1) % self.args.check_point == 0) or (epoch == self.args.epochs - 1):
                    save_checkpoint(model, epoch, use_personal_folder=self.args.personal)

                # log to tensorboard
                board.log_scalars(
                    "Curves/Loss", {"Train": train_losses[-1], "Validation": validation_losses[-1]}, epoch
                )
                board.log_layer_gradients(model, epoch)

                # Calculate metrics
                train_accuracy = (train_confusion_matrix[0, 0] + train_confusion_matrix[1, 1]) / train_confusion_matrix.sum()
                train_precision = train_confusion_matrix[1, 1] / (train_confusion_matrix[1, 1] + train_confusion_matrix[0, 1])
                train_recall = train_confusion_matrix[1, 1] / (train_confusion_matrix[1, 1] + train_confusion_matrix[1, 0])
                train_f1_score = 2 * (train_precision * train_recall) / (train_precision + train_recall)
                board.log_scalars(
                    "Metrics/Train",
                    {
                        "Accuracy": train_accuracy,
                        "Precision": train_precision,
                        "Recall": train_recall,
                        "F1 Score": train_f1_score,
                    },
                    epoch,
                )

                val_accuracy = (val_confusion_matrix[0, 0] + val_confusion_matrix[1, 1]) / val_confusion_matrix.sum()
                val_precision = val_confusion_matrix[1, 1] / (val_confusion_matrix[1, 1] + val_confusion_matrix[0, 1])
                val_recall = val_confusion_matrix[1, 1] / (val_confusion_matrix[1, 1] + val_confusion_matrix[1, 0])
                val_f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall)
                board.log_scalars(
                    "Metrics/Validation",
                    {
                        "Accuracy": val_accuracy,
                        "Precision": val_precision,
                        "Recall": val_recall,
                        "F1 Score": val_f1_score,
                    },
                    epoch,
                )
                

        except KeyboardInterrupt:
            print("Training interrupted")
        finally:
            board.close()

        nvtx.pop_range()  # Training

        # X_val, Y_val = next(iter(validation_data_loader))
        # X_val, Y_val = X_val, torch.argmax(Y_val, dim=1)
        # fig, ax = plt.subplots(figsize=(10, 10))
        # plot_2d_model_ax(ax, X_val, Y_val, model)
        # fig.savefig("model.png")
        # plt.close()
