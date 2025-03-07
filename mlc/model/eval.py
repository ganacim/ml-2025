import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..command.base import Base
from ..data import get_available_datasets
from ..util.model import load_checkpoint, load_metadata
from ..util.resources import model_path
from . import get_available_models


class Eval(Base):
    name = "model.eval"

    def __init__(self, args):
        super().__init__(args)

        self.device = "cpu"
        if args.device == "cuda":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError("CUDA is not available")

    @staticmethod
    def add_arguments(parser):
        models = [m for m in get_available_models().keys()]
        model_help = "Model name or path to model version. Choices of model name: {" ""
        for m in models:
            model_help += f"{m}, "
        model_help = model_help[:-2] + "}"
        parser.add_argument("model", type=str, help=model_help)
        parser.add_argument("-v", "--version", help="Model version to evaluate", type=str, default="latest")
        parser.add_argument("-c", "--checkpoint", help="Checkpoint to evaluate", type=str, default="latest")
        parser.add_argument("-d", "--device", choices=["cpu", "cuda"], default="cuda")

    def run(self):
        model_name = None
        model_version = None
        model_checkpoint = None

        # is self.args.model a model name?
        if self.args.model in get_available_models():
            model_name = self.args.model
            model_version = self.args.version
            model_checkpoint = self.args.checkpoint
        else:
            # remove trailing slash
            self.args.model = self.args.model.rstrip("/")

            # is self.args.model a model path?
            if (Path(self.args.model) / "model.txt").exists():
                # self.args.model is a model path
                model_name = self.args.model.split("/")[-1]
                model_version = self.args.version
                model_checkpoint = self.args.checkpoint
            elif (Path(self.args.model) / "metadata.json").exists():
                # self.args.model is a path, if metadata.json exists, it is a model version
                model_name = self.args.model.split("/")[-2]
                model_version = self.args.model.split("/")[-1]
                model_checkpoint = self.args.checkpoint
            elif (Path(self.args.model) / "model_state.pt").exists():
                # self.args.model is a path, if model_state.pt exists, it is a model checkpoint
                model_name = self.args.model.split("/")[-3]
                model_version = self.args.model.split("/")[-2]
                model_checkpoint = self.args.model.split("/")[-1]
            else:
                raise ValueError(f"Model name of folder {self.args.model} not found")

        print(f"Evaluating model: {model_name}, version: {model_version}, checkpoint: {model_checkpoint}")
        # load training session metadata
        metadata = load_metadata(model_name, model_version)
        # load model
        model_args = metadata["model"]["args"]
        model = load_checkpoint(model_name, model_args, model_version, model_checkpoint)
        model = model.to(self.device)
        # load dataset
        dataset_args = metadata["dataset"]["args"]
        dataset = get_available_datasets()[metadata["dataset"]["name"]](dataset_args)

        # create loss function
        loss_fn = torch.nn.BCELoss(reduction="none")

        # evaluate model
        model.eval()
        results = dict()
        with torch.no_grad():
            for fold in ["train", "validation", "test"]:
                results[fold] = dict()
                partial_result = list()
                data = DataLoader(dataset.get_fold(fold), batch_size=32, shuffle=False)
                for X, Y in tqdm(data):
                    X = X.to(self.device)
                    Y = Y.to(self.device)
                    Y_pred = model(X)
                    loss = torch.mean(loss_fn(Y_pred, Y), dim=1)
                    partial_result.append(loss)
                results[fold]["loss"] = list(map(float, torch.cat(partial_result).cpu().numpy()))

        # save results
        checkpoint_path = model_path(model_name) / model_version / model_checkpoint
        with open(checkpoint_path / "eval.json", "w") as f:
            json.dump(results, f, indent=4)
