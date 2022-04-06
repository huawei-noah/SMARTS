import click
from cli.cli import scl
from pathlib import Path

# Recursively go through each command's --help and write it to cli.rst
def recursive_help(cmd, file, parent=None):
    ctx = click.core.Context(cmd, info_name=cmd.name, parent=parent)
    line = "=" * len(cmd.name)
    file.write(line + "\n" + cmd.name + "\n" + line + "\n\n")
    file.write(cmd.get_help(ctx) + "\n\n")
    commands = getattr(cmd, 'commands', {})
    for sub in commands.values():
        recursive_help(sub, file, ctx)

if __name__ == "__main__":
    directory = str(Path(__file__).parent.parent)
    f = open(directory + "/docs/sim/cli.rst", "w")
    recursive_help(scl, f)
    f.close