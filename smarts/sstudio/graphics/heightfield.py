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
from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Optional, Tuple, Union

import numpy as np


class HeightField(ABC):
    """A utility for working with greyscale values."""

    def __init__(
        self,
        data: np.ndarray,
        size: Union[Tuple[int, int], np.ndarray],
        metadata: Optional[Dict] = None,
    ) -> None:
        assert isinstance(data, np.ndarray), "Image must be a numpy array."
        assert data.dtype == np.uint8 and (
            len(data.shape) == 2 or (len(data.shape) == 3 and data.shape[-1] == 1)
        ), f"Image with {data.dtype} and shape {data.shape} is not greyscale format."
        if len(data.shape) == 3:
            data = np.squeeze(data, axis=2)
        self._data = data
        self._size = np.array(size, dtype=np.uint64)
        self._resolution = np.array(list(reversed(data.shape)), dtype=np.int64)
        self._reciprocal_resolution = np.reciprocal(self._resolution)
        self._inverse_size = np.reciprocal(self._size, dtype=np.float64)
        self._metadata = metadata or {}

    @property
    def data(self):
        """The raw underlying data."""
        return self._data

    @property
    def dtype(self):
        """The per element data type."""
        return self._data.dtype

    @property
    def size(self):
        """The width and height."""
        return self._size

    @property
    def resolution(self):
        """Resolution of this height field."""
        return self._resolution

    @property
    def metadata(self):
        """Additional metadata."""
        return self._metadata

    def _check_match(self, other: HeightField):
        return np.all(self._resolution == other._resolution) and np.all(
            self._size == other._size
        )

    def add(self, other: HeightField) -> HeightField:
        """Add by element."""
        assert self._check_match(other)
        return HeightField(np.add(self._data, other._data), self._size)

    def subtract(self, other: HeightField) -> HeightField:
        """Subtract by element."""
        assert self._check_match(other)
        data = np.subtract(self._data, other._data)
        return HeightField(data, self.size)

    def scale_by(self, other: HeightField) -> HeightField:
        """Scale this height field by another height field."""
        assert self._check_match(other)
        inplace_array = np.multiply(
            other._data,
            np.reciprocal(np.invert(other._data.dtype.type(0)), dtype=np.float64),
        )
        np.multiply(self._data, inplace_array, out=inplace_array)
        if self.dtype.type in {"u", "i"}:
            inplace_array.round(out=inplace_array)
        return HeightField(inplace_array.astype(self.dtype), self.size)

    def multiply(self, other: HeightField) -> HeightField:
        """Multiply the byte values between these height fields"""
        assert self._check_match(other)
        return HeightField(np.multiply(self._data, other._data), self.size)

    def max(self, other: HeightField) -> HeightField:
        """Get the maximum value of overlapping height fields."""
        assert self._check_match(other)
        return HeightField(np.max([self._data, other._data], axis=0), self.size)

    def inverted(self) -> HeightField:
        """Invert this height field assuming 8 bit."""
        data = np.invert(self._data)
        return HeightField(data, self._size, self._metadata)

    def apply_kernel(
        self, kernel: np.ndarray, min_val=-np.inf, max_val=np.inf, pad_mode="edge"
    ):
        """Apply a kernel to the whole height field.

        The kernel can be asymmetric but still needs each dimension to be an odd value.
        """
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
        """Apply a function to each element."""
        output = np.empty_like(self.data)
        for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                output[i][j] = min(max(fn(self.data, i, j), min_val), max_val)

        return HeightField(output, self.size)

    def write_image(self, file: Union[str, Path, BinaryIO]):
        """Write this out to a greyscale image."""
        from PIL import Image

        a = self.data.astype(np.uint8)
        im = Image.fromarray(a, "L")
        im.save(file)

    @classmethod
    def load_image(cls, file: Union[str, Path]):
        """Load from any image."""
        from PIL import Image

        with Image.open(file) as im:
            data = np.asarray(im)
            assert len(data.shape) == 2
        return cls(data, data.shape[:2])

    @classmethod
    def from_rgb(cls, data: np.ndarray):
        """Load from an rgb array."""
        d = np.min(data, axis=2)
        return HeightField(d, size=(data.shape[1], data.shape[0]))
