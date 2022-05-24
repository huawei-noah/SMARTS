# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple


class SingleAttributeOverride(NamedTuple):
    """Options for filtering out attributes."""

    enabled: bool
    """If the stream value is enabled."""
    default: Any
    """The default value for the stream if not enabled."""
    max_count: int = sys.maxsize
    """The maximum number of elements an iterable attribute can contain."""


@dataclass(frozen=True)
class EnvisionStateFilter:
    """A state filtering tool."""

    actor_data_filter: Dict[str, SingleAttributeOverride]
    """Actor filtering."""
    simulation_data_filter: Dict[str, SingleAttributeOverride]
    """Simulation filtering."""

    @classmethod
    def default(cls):
        """Give a new default filter."""

        def default_override():
            return SingleAttributeOverride(True, None)

        return cls(defaultdict(default_override), defaultdict(default_override))
