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
from abc import ABC
from typing import Callable, Tuple

import numpy as np


class HeightField(ABC):
    def __init__(self, data: np.ndarray, size: Tuple[int, int]) -> None:
        assert data.dtype == np.uint8 and (
            len(data.shape) == 2 or (len(data.shape) == 3 and data.shape[-1] == 1)
        ), f"Image is not greyscale format."
        if len(data.shape) == 3:
            data = np.squeeze(data, axis=2)
        self._data = data
        self._size = size
        self._resolution = data.shape

    @property
    def data(self):
        return self._data

    @property
    def size(self):
        return self._size

    @property
    def resolution(self):
        return self._resolution

    def apply_kernel(
        self, kernel: np.ndarray, min_val=-np.inf, max_val=np.inf, pad_mode="edge"
    ):
        # kernel can be asymmetric but still needs to be odd
        assert len(kernel.shape) == 2 and np.all(
            [k % 2 for k in kernel.shape]
        ), "Kernel shape must be 2D and shape dimension values must be odd"
        k_height, k_width = kernel.shape
        m_height, m_width = self.data.shape
        k_size = max(k_height, k_width)
        padded = np.pad(self.data, (int(k_size / 2), int(k_size / 2)), mode=pad_mode)

        if k_size > 1:
            if k_height == 1:
                padded = padded[1:-1, :]
            elif k_width == 1:
                padded = padded[:, 1:-1]

        # iterate through matrix, apply kernel, and sum
        output = np.empty_like(self.data)
        for v in range(m_height):
            for u in range(m_width):
                between = padded[v : k_height + v, u : k_width + u] * kernel
                output[v][u] = min(max(np.sum(between), min_val), max_val)

        return HeightField(output, self.size)

    def apply_function(
        self,
        fn: Callable[[np.ndarray, int, int], np.uint8],
        min_val=-np.inf,
        max_val=np.inf,
    ):
        output = np.empty_like(self.data)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                output[i][j] = min(max(fn(self.data, i, j), min_val), max_val)

        return HeightField(output, self.size)

    def write_image(self, file):
        from PIL import Image

        a = self.data.astype(np.uint8)
        im = Image.fromarray(a, "L")
        im.save(file)

    @classmethod
    def load_image(cls, file):
        from PIL import Image

        with Image.open(file) as im:
            data = np.asarray(im)
            assert len(data.shape) == 2
        return cls(data, data.shape[:2])
