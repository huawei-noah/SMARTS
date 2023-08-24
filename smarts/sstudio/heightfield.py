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
from typing import Tuple

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

    def apply_kernel(self, kernel: np.ndarray, output_dtype=np.uint8, min_val=-np.inf, max_val=np.inf)):
        # kernel can be asymmetric but still needs to be odd
        k_height, k_width = kernel.shape
        m_height, m_width = self.data.shape
        k_size = max(k_height, k_width)
        padded = np.pad(self.data, (int(k_size / 2), int(k_size / 2)))

        if k_size > 1:
            if k_height == 1:
                padded = padded[1:-1, :]
            elif k_width == 1:
                padded = padded[:, 1:-1]

        # iterates through matrix, applies kernel, and sums
        output = []
        for i in range(m_height):
            for j in range(m_width):
                between = padded[i : k_height + i, j : k_width + j] * kernel
                output.append(min(max(np.sum(between), min_val), max_val))

        output = np.array(output, dtype=output_dtype).reshape((m_height, m_width))
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
            
