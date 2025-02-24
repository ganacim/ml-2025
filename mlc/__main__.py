import sys
import argparse

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
    for cmd_type in available_commands:
        if cmd_type.name == args.command:
            cmd = cmd_type(args)
            cmd.run()
            return 0
    else:
        print(f"Command {args.command} not found")

    return 0
    

if __name__ == '__main__':
    sys.exit(main())