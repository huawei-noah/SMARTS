# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
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
