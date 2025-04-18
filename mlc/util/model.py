import json
import sys

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

        # create symlink to latest model
        latest_model_path = m_path / "latest"
        if latest_model_path.exists():
            latest_model_path.unlink()
        # this is a symlink to the latest model version
        latest_model_path.symlink_to(m_version)

        # create symlink to latest checkpoint
        latest_cp_path = m_path / m_version / "latest"
        if latest_cp_path.exists():
            latest_cp_path.unlink()
        latest_cp_path.symlink_to(cp_name)

    # save model
    torch.save(model.state_dict(), cp_path / "model_state.pt")


def save_metadata(model, dataset, use_personal_folder=False):
    # get model path
    m_path = model_path(model.name(), use_personal_folder=use_personal_folder) / get_time_as_str()
    # check if model path exists
    if not m_path.exists():
        m_path.mkdir(parents=True)

    # create flag model folder
    m_flag = model_path(model.name(), use_personal_folder=use_personal_folder) / "model.txt"
    if not m_flag.exists():
        m_flag.touch()

    metadata = {
        "command_line": " ".join(sys.argv[1:]),
        "model": {
            "name": model.name(),
            "args": model.args(),
        },
        "dataset": {
            "name": dataset.name(),
            "args": dataset.args(),
        },
    }

    with open(m_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)


def load_metadata(model_name, model_version, use_personal_folder=False):
    # get model path
    m_path = model_path(model_name, use_personal_folder=use_personal_folder) / model_version
    # load metadata
    with open(m_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    return metadata


def load_checkpoint(model_name, model_args, model_version, checkpoint, use_personal_folder=False):
    # get model path
    m_path = model_path(model_name, use_personal_folder=use_personal_folder) / model_version / checkpoint
    # load model
    model = get_available_models()[model_name](model_args)
    model.load_state_dict(torch.load(m_path / "model_state.pt", weights_only=True))
    return model
