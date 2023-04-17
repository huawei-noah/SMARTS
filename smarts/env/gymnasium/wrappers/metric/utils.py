# MIT License

# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from collections import deque
from dataclasses import fields
from typing import Callable, TypeVar, Union

from smarts.env.gymnasium.wrappers.metric.costs import Costs
from smarts.env.gymnasium.wrappers.metric.counts import Counts

T = TypeVar("T", Costs, Counts)


def add_dataclass(first: T, second: T) -> T:
    """Sums the fields of two dataclass objects.

    Args:
        first (T): First dataclass object.
        second (T): Second dataclass object.

    Returns:
        T: New summed dataclass object.
    """
    assert type(first) is type(second)
    new = {}
    for field in fields(first):
        new[field.name] = getattr(first, field.name) + getattr(second, field.name)
    output = first.__class__(**new)

    return output


def op_dataclass(
    first: T,
    second: Union[int, float],
    op: Callable[[Union[int, float], Union[int, float]], float],
) -> T:
    """Performs operation `op` on the fields of a dataclass object.

    Args:
        first (T): Dataclass object.
        second (Union[int, float]): Value input for the operator.
        op (Callable[[Union[int, float], Union[int, float]], float]): Operation to be performed.

    Returns:
        T: New dataclass object with operation performed on all of its fields.
    """
    new = {}
    for field in fields(first):
        new[field.name] = op(getattr(first, field.name), second)
    output = first.__class__(**new)

    return output


def divide(value: Union[int, float], divider: Union[int, float]) -> float:
    """Division operation.

    Args:
        value (Union[int, float]): Numerator
        divider (Union[int, float]): Denominator

    Returns:
        float: Numerator / Denominator
    """
    return float(value / divider)


def multiply(value: Union[int, float], multiplier: Union[int, float]) -> float:
    """Multiplication operation.

    Args:
        value (Union[int, float]): Value
        multiplier (Union[int, float]): Multiplier

    Returns:
        float: Value x Multiplier
    """
    return float(value * multiplier)


class SlidingWindow:
    def __init__(self, size:int):
        self._values = deque(maxlen=size)
        self._max_candidates = deque(maxlen=size)
        self._size = size
        self._time = -1

    def move(self, x):
        self._time += 1

        # Remove head element if deque is full.
        # Append new element to deque tail.
        if len(self._values) == self._size:
            if self._values[0][1] == self._max_candidates[0][1]:
                self._max_candidates.popleft()
        self._values.append((self._time,x))

		# Remove all elements from deque's head which are less than x.
        # Append new element to deque tail.
        while self._max_candidates and self._max_candidates[0][1] < x:
            self._max_candidates.popleft()
        self._max_candidates.append((self._time,x))
    
    def max(self):
	    # Max element is at the deque's head 
        return self._max_candidates[0][1]