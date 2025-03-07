import argparse
import importlib

from .base import Base


class Test(Base):
    name = "test"

    def __init__(self, args):
        super().__init__(args)

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("module", type=str, help="python module to test")
        # collect all remaining arguments
        parser.add_argument("test_args", nargs=argparse.REMAINDER, help="arguments to pass to the module")

    def run(self):
        # try to find module in self.args.module
        try:
            module = importlib.import_module(self.args.module)
        except ImportError:
            print(f"Could not import module {self.args.module}")
            return 1

        print(f"Running {self.args.module}.test({self.args.test_args})")

        return module.test(self.args.test_args)
