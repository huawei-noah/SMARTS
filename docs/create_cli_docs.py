import click
from cli.cli import scl
from pathlib import Path


# Recursively go through each command's --help and write it to cli.rst
def recursive_help(cmd, file, level, parent=None):
    ctx = click.core.Context(cmd, info_name=cmd.name, parent=parent)
    line = "=" * len(cmd.name) if level == 0 else "-" * len(cmd.name)
    if level <= 1:
        file.write(line + "\n")
    file.write(cmd.name + "\n" + line + "\n\n")
    file.write(cmd.get_help(ctx) + "\n\n")
    commands = getattr(cmd, 'commands', {})
    for sub in commands.values():
        recursive_help(sub, file, level + 1, ctx)

if __name__ == "__main__":
    directory = str(Path(__file__).parent.parent)
    f = open(directory + "/docs/sim/cli.rst", "w")
    f.write(".. _cli: \n\n")
    title = "Command Line Interface" 
    underline = "=" * len(title)
    f.write(title + "\n" + underline + "\n\n")
    recursive_help(scl, f, 0)
    f.close