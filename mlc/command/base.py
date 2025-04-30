class Base:
    def __init__(self, args):
        # args should be a dictionary
        assert isinstance(args, dict), "args must be a dictionary"
        self._args = args

    @classmethod
    def name(cls):
        return "base_command"

    @property
    def args(self):
        return self._args

    @staticmethod
    def add_arguments(parser):
        raise NotImplementedError("command.Base: Subclasses must implement add_arguments method")

    def run(self):
        raise NotImplementedError("command.Base: Subclasses must implement run method")
