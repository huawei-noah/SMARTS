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

import numpy as np


def truncate_pad_li_2d(li, ref, null_value):
    """Truncate or pad a 2-dimensional list li to shape (ref[0], ref[1]). Padding uses null_value.
    Args:
        li:
            A nested 2 dimensional list.
        ref:
            A tuple(int, int) which indicates the desired lengths in each dimension.
        null_value:
            A tuple(type, type) which provides the value for padding in each dimension.
    """

    temp_li = truncate_pad_li(li, ref[0], null_value[0])
    new_li = list(
        map(lambda elem: truncate_pad_li(elem, ref[1], null_value[1]), temp_li)
    )
    return new_li


def truncate_pad_li(li, ref, null_value):
    """Truncate or pad a 1-dimensional list li to shape (ref,). Padding uses `null_value`.
    Args:
        li:
            A 1 dimensional list.
        ref:
            Desired length of list.
        null_value:
            Value used for padding.
    """
    if len(li) == ref:
        new_li = li
    elif len(li) < ref:
        new_li = li + [null_value] * (ref - len(li))
    else:  # len(li) > ref
        new_li = li[:ref]

    return new_li


def truncate_pad_arr(arr, ref, null_value):
    """Truncate or pad a 1-dimensional np.ndarray arr to shape (ref,). Padding usses `null_value`.
    Args:
        arr:
            A 1 dimensional numpy array.
        ref:
            Desired length of array.
        null_value:
            Value used for padding.
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array([arr])

    if len(arr) == ref:
        new_arr = arr
    elif len(arr) < ref:
        new_arr = np.pad(
            arr, (0, ref - len(arr)), "constant", constant_values=(null_value)
        )
    else:  # len(arr) > ref
        new_arr = arr[:ref]

    return new_arr
