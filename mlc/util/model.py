import json
import sys

import torch

from .resources import get_time_as_str, model_path


def save_checkpoint(model, epoch):
    # get model path
    m_path = model_path(model.name())  # path to model, can be absolute
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


def save_metadata(model, dataset):
    # get model path
    m_path = model_path(model.name()) / get_time_as_str()
    # check if model path exists
    if not m_path.exists():
        m_path.mkdir(parents=True)

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
