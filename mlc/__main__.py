import argparse
import sys

import mlc.command as cmds


def main():
    # avaliable_commands
    available_commands = cmds.get_available_commands()

    # create parser
    parser = argparse.ArgumentParser(description="Machine Learning Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    for cmd_type in available_commands:
        subparser = subparsers.add_parser(cmd_type.name, help=cmd_type.__doc__)
        cmd_type.add_arguments(subparser)
    args = parser.parse_args()

    # run command
    try:
        for cmd_type in available_commands:
            if cmd_type.name == args.command:
                cmd = cmd_type(args)
                cmd.run()
                return 0
        else:
            if args.command is None:
                parser.print_help()
            else:
                print(f"Command {args.command} not found")

    except RuntimeError as e:
        print(f"RuntimeError: {e}")

    except Exception as e:
        print(f"Error: {e}")

    else:
        # return success
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
