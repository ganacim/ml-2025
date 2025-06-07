from torch.utils.tensorboard import SummaryWriter

from .resources import get_time_as_str, model_path


class Board:
    def __init__(self, model_name, use_personal_folder=False, enabled=False):
        if not enabled:
            self.writer = None
        else:
            log_dir = (
                model_path(model_name, use_personal_folder=use_personal_folder) / get_time_as_str() / "tensorboard"
            )
            if not log_dir.exists():
                log_dir.mkdir(parents=True)
            self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, tag, value, step):
        if self.writer:
            self.writer.add_scalars(tag, value, step)

    def log_image(self, tag, image, step):
        if self.writer:
            self.writer.add_image(tag, image, step)

    def log_text(self, tag, text, step):
        if self.writer:
            self.writer.add_text(tag, text, step)

    def log_graph(self, model, input_to_model):
        if self.writer:
            self.writer.add_graph(model, input_to_model)

    def log_layer_gradients(self, model, epoch):
        if self.writer:
            for name, params in model.named_parameters():
                if params.grad is None:
                    continue
                self.writer.add_histogram(f"Layer Gradients/{name}", params.grad, epoch)

    def log_figure(self, tag, figure, step, close=True):
        if self.writer:
            self.writer.add_figure(tag, figure, step, close=close)

    def close(self):
        if self.writer:
            self.writer.close()
