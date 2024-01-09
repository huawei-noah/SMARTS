# MIT License
#
# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


from dataclasses import dataclass

from smarts.sstudio.sstypes.constants import MAX


@dataclass(frozen=True)
class BubbleLimits:
    """Defines the capture limits of a bubble."""

    hijack_limit: int = MAX
    """The maximum number of vehicles the bubble can hijack"""
    shadow_limit: int = MAX
    """The maximum number of vehicles the bubble can shadow"""

    def __post_init__(self):
        if self.shadow_limit is None:
            raise ValueError("Shadow limit must be a non-negative real number")
        if self.hijack_limit is None or self.shadow_limit < self.hijack_limit:
            raise ValueError("Shadow limit must be >= hijack limit")
