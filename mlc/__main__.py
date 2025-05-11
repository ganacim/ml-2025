import argparse
import sys

import torch

# import mlc.command as cmds
from .util.resources import get_available_commands


def main():
    # avaliable_commands
    available_commands = get_available_commands()

    # create parser
    parser = argparse.ArgumentParser(description="Machine Learning Command Line Interface")
    parser.add_argument("-D", "--debug", action="store_true", help="Enable debug mode")
    parser.set_defaults(debug=False)
    parser.add_argument("-A", "--detect-anomaly", action="store_true", help="Enable anomaly detection")
    parser.set_defaults(detect_anomaly=False)
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    for name, cmd_type in available_commands.items():
        subparser = subparsers.add_parser(name, help=cmd_type.__doc__)
        cmd_type.add_arguments(subparser)
    args = parser.parse_args()

    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        print("Anomaly detection enabled")

    # run command
    try:
        for name, cmd_type in available_commands.items():
            if name == args.command:
                cmd = cmd_type(vars(args))
                cmd.run()
                return 0
        else:
            if args.command is None:
                parser.print_help()
            else:
                print(f"Command {args.command} not found")

    except RuntimeError as e:
        print(f"RuntimeError: {e}")
        if args.debug:
            raise e

    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            raise e

    else:
        # return success
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
