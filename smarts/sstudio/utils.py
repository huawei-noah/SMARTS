import sys
from enum import Enum


class PromptFormat(Enum):
    YesNo = 0
    """Require an answer."""
    YesNoDefaultYes = 1
    """Default answer is yes."""
    YesNoDefaultNo = 2
    """Default answer is no."""


def prompt_yes_no(query, format=PromptFormat.YesNoDefaultYes):
    """Ask a yes/no query and return their answer.

    "query" is the prompt question to be addressed to the user.
    "format" is how the presumtion when the user hits <Enter>.

    Returns 'True' for 'yes' and 'False' for 'no'.
    """
    selection_group = {"y": True, "n": False}
    prompt_group = {
        PromptFormat.YesNo: (" [y/n] ", ""),
        PromptFormat.YesNoDefaultYes: (" [Y/n] ", "y"),
        PromptFormat.YesNoDefaultNo: (" [y/N] ", "n"),
    }
    try:
        prompt, default = prompt_group[format]
    except KeyError:
        raise ValueError(f"`format` must be of type: {PromptFormat.__name__}")

    while True:
        choice = input(query + prompt).lower()
        if default and choice == "":
            return selection_group[default]
        elif choice and choice[0] in selection_group:
            return selection_group[choice[0]]
        else:
            sys.stdout.write("Please respond with 'y' or 'n'.\n")
