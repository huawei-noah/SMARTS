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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import numpy as np

from smarts.core.utils.math import (
    combination_pairs_with_unique_indices,
    position_to_ego_frame,
    world_position_from_ego_frame,
)


def test_egocentric_conversion():
    p_start = [1, 2, 3]
    pe = [1, -5, 2]
    he = -3

    pec = position_to_ego_frame(p_start, pe, he)

    assert np.allclose([-0.9878400564190705, -6.929947476203118, 1.0], pec)

    p_end = world_position_from_ego_frame(pec, pe, he)

    assert np.allclose(p_end, p_start)


def test_combination_pairs_with_unique_indices():

    assert not tuple(combination_pairs_with_unique_indices("", ""))
    assert tuple(combination_pairs_with_unique_indices("a", "")) == (("a", None),)
    assert tuple(
        combination_pairs_with_unique_indices("abc", "12", second_group_default=4)
    ) == (
        (("a", "1"), ("b", "2"), ("c", 4)),
        (("a", "1"), ("b", 4), ("c", "2")),
        (("a", "2"), ("b", "1"), ("c", 4)),
        (("a", "2"), ("b", 4), ("c", "1")),
        (("a", 4), ("b", "1"), ("c", "2")),
        (("a", 4), ("b", "2"), ("c", "1")),
    )
    assert tuple(combination_pairs_with_unique_indices("ab", "123")) == (
        (("a", "1"), ("b", "2")),
        (("a", "1"), ("b", "3")),
        (("a", "2"), ("b", "1")),
        (("a", "2"), ("b", "3")),
        (("a", "3"), ("b", "1")),
        (("a", "3"), ("b", "2")),
    )
