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


import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Distribution:
    """A gaussian distribution used for randomized parameters."""

    mean: float
    """The mean value of the gaussian distribution."""
    sigma: float
    """The sigma value of the gaussian distribution."""

    def sample(self):
        """The next sample from the distribution."""
        return random.gauss(self.mean, self.sigma)


@dataclass
class UniformDistribution:
    """A uniform distribution, return a random number N
    such that a <= N <= b for a <= b and b <= N <= a for b < a.
    """

    a: float
    b: float

    def __post_init__(self):
        if self.b < self.a:
            self.a, self.b = self.b, self.a

    def sample(self):
        """Get the next sample."""
        return random.uniform(self.a, self.b)
