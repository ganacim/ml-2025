import torch

from .resources import get_time_as_str, model_path


def save_checkpoint(model, epoch):
    # get model path
    m_path = model_path(model.name()) / get_time_as_str() / f"{epoch:04d}"
    # check if model path exists
    if not m_path.exists():
        m_path.mkdir(parents=True)

    # save model
    torch.save(model.state_dict(), m_path / "model_state.pt")
