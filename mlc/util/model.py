import json
import sys
from pathlib import Path

import torch

from .resources import get_available_models, get_time_as_str, model_path


def save_checkpoint(model, epoch, use_personal_folder=False):
    # get model path
    m_path = model_path(model.name(), use_personal_folder=use_personal_folder)  # path to model, can be absolute
    m_version = get_time_as_str()  # version of the model
    cp_name = f"{epoch:04d}"  # checkpoint name
    cp_path = m_path / m_version / cp_name  # full path to checkpoint
    # check if model path exists
    if not cp_path.exists():
        cp_path.mkdir(parents=True)

        # create symlink to latest checkpoint
        latest_cp_path = m_path / m_version / "latest"
        if latest_cp_path.exists():
            latest_cp_path.unlink()
        latest_cp_path.symlink_to(cp_name)

    # load metadata
    with open(m_path / m_version / "metadata.json", "r") as f:
        metadata = json.load(f)
        # save number of epochs trained
        if "training" not in metadata:
            metadata["training"] = {}
        metadata["training"]["epochs"] = epoch
    # save metadata
    with open(m_path / m_version / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    # save model
    torch.save(model.state_dict(), cp_path / "model_state.pt")


def save_metadata(model, dataset, use_personal_folder=False, name=None):
    # get model path
    m_path = model_path(model.name(), use_personal_folder=use_personal_folder)  # path to model, can be absolute
    m_version = get_time_as_str()  # version of the model
    m_version_path = m_path / m_version  # full path to model version

    # check if model/version path exists
    if not m_version_path.exists():
        m_version_path.mkdir(parents=True)

    # create symlink to latest model
    latest_model_path = m_path / "latest"
    if latest_model_path.exists():
        latest_model_path.unlink()
    # this is a symlink to the latest model version
    latest_model_path.symlink_to(m_version)

    if name is not None:
        # create symlink to latest model version
        latest_model_name = m_path / name
        latest_named_model_name = m_path / "named" / name
        if latest_model_name.exists():
            i = 1
            while latest_model_name.with_name(f"{name}_{str(i)}").exists():
                i += 1
            latest_model_name = latest_model_name.with_name(f"{name}_{str(i)}")
            latest_named_model_name = latest_named_model_name.with_name(f"{name}_{str(i)}")
        # this is a symlink to the latest model version
        latest_model_name.symlink_to(m_version)
        # create a folder "named" for named models
        named_model_path = m_path / "named"
        if not named_model_path.exists():
            named_model_path.mkdir(parents=True)
        if latest_named_model_name.exists():
            latest_named_model_name.unlink()
        # this is a symlink to the latest model version
        latest_named_model_name.symlink_to(Path("..") / m_version)

    # create flag model folder
    m_flag = model_path(model.name(), use_personal_folder=use_personal_folder) / "model.txt"
    if not m_flag.exists():
        m_flag.touch()

    metadata = {
        "command_line": " ".join(sys.argv[1:]),
        "model": {
            "name": model.name(),
            "args": model.args,
        },
        "dataset": {
            "name": dataset.name(),
            "args": dataset.args,
        },
        "training": {
            "epochs": 0,  # number of epochs trained
        },
    }

    with open(m_version_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def load_metadata(model_name, model_version, use_personal_folder=False):
    # get model path
    m_path = model_path(model_name, use_personal_folder=use_personal_folder) / model_version
    # load metadata
    with open(m_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    return metadata


def load_checkpoint(model_name, model_version, checkpoint, use_personal_folder=False):
    # get model path
    m_path = model_path(model_name, use_personal_folder=use_personal_folder) / model_version / checkpoint
    # load metatadata
    metadata = load_metadata(model_name, model_version, use_personal_folder=use_personal_folder)

    # load model
    model = get_available_models()[model_name](metadata["model"]["args"])
    model.load_state_dict(torch.load(m_path / "model_state.pt", weights_only=True))
    return model


def load_model_from_path(path, from_personal_folder=False):
    # remove trailing slash
    path = Path(path)
    model_name = None
    model_version = "latest"
    model_checkpoint = "latest"
    # is path a model path?
    if (path / "model.txt").exists():
        # path is a model path
        model_name = path.parts[-1]
    elif (path / "metadata.json").exists():
        # path is a path, if metadata.json exists, it is a model version
        model_name = path.parts[-2]
        model_version = path.parts[-1]
    elif (path / "model_state.pt").exists():
        # path is a path, if model_state.pt exists, it is a model checkpoint
        model_name = path.parts[-3]
        model_version = path.parts[-2]
        model_checkpoint = path.parts[-1]
    else:
        raise ValueError(f"Model name of folder {str(path)} not found")

    metadata = load_metadata(model_name, model_version, use_personal_folder=from_personal_folder)
    # load model
    model_args = metadata["model"]["args"]
    model = load_checkpoint(
        model_name, model_args, model_version, model_checkpoint, use_personal_folder=from_personal_folder
    )
    return model, model_name, model_version, model_checkpoint, metadata
