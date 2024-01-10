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

import enum
import math
from abc import ABC
from enum import IntEnum
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Optional, Tuple, Union

import numpy as np

from smarts.core.utils.core_math import line_of_sight_test


class CoordinateSampleMode(IntEnum):
    POINT = enum.auto()
    FOUR_POINTS = enum.auto()


class HeightField(ABC):
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
        return self._data

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def size(self):
        return self._size

    @property
    def resolution(self):
        return self._resolution

    @property
    def metadata(self):
        return self._metadata

    def _check_match(self, other: HeightField):
        return np.all(self._resolution == other._resolution) and np.all(
            self._size == other._size
        )

    def add(self, other: HeightField) -> HeightField:
        assert self._check_match(other)
        return HeightField(np.add(self._data, other._data), self._size)

    def subtract(self, other: HeightField) -> HeightField:
        assert self._check_match(other)
        data = np.subtract(self._data, other._data)
        return HeightField(data, self.size)

    def scale_by(self, other: HeightField) -> HeightField:
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
        assert self._check_match(other)
        return HeightField(np.multiply(self._data, other._data), self.size)

    def max(self, other: HeightField) -> HeightField:
        assert self._check_match(other)
        return HeightField(np.max([self._data, other._data], axis=0), self.size)

    def inverted(self) -> HeightField:
        data = np.invert(self._data)
        return HeightField(data, self._size, self._metadata)

    def convert_to_data_coordinate(self, coordinate):
        return np.array(
            (
                (coordinate[0] * self._inverse_size[0] + 0.5)
                * (self._resolution[0] - 1),
                (coordinate[1] * self._inverse_size[1] + 0.5)
                * (self._resolution[1] - 1),
            ),
            dtype=np.float64,
        )

    def _direct_coordinate_sample(self, coordinate: Union[Tuple[float, float], np.ndarray]) -> float:
        # average the nearest 3 pixel coordinates
        u, v = coordinate
        return self._data[int(v)][int(u)]

    def _direct_4_point_coordinate_sample(self, coordinate) -> float:
        u1 = int(coordinate[0])
        v1 = int(coordinate[1])

        ur = coordinate[0] - u1
        vr = coordinate[1] - v1

        u2 = min(u1 + 1, int(self._resolution[0] - 1))
        v2 = min(v1 + 1, int(self._resolution[1] - 1))

        bottom_left = self._data[v1][u1]
        blw = (1 - ur) * (1 - vr)
        bottom_right = self._data[v1][u2]
        brw = (ur) * (1 - vr)
        top_left = self._data[v2][u1]
        tlw = (1 - ur) * (vr)
        top_right = self._data[v2][u2]
        trw = (ur) * (vr)

        return bottom_left * blw + bottom_right * brw + top_left * tlw + top_right * trw

    def _get_sample_averaging_function(
        self, coordinate_sample_mode: CoordinateSampleMode
    ) -> Callable[[Union[Tuple[float, float], np.ndarray]], float]:
        if coordinate_sample_mode is CoordinateSampleMode.POINT:
            return self._direct_coordinate_sample
        if coordinate_sample_mode is CoordinateSampleMode.FOUR_POINTS:
            return self._direct_4_point_coordinate_sample

        return self._direct_4_point_coordinate_sample

    def _data_sample_line(
        self,
        change_normalized: np.ndarray,
        resolution: float,
        magnitude: float,
        sample_function: Callable,
        viewer_coordinate: Union[Tuple[float, float], np.ndarray],
        factor: float,
    ):
        """Generates samples on the line between `viewer_coordinate`(excluded) and the end point `viewer_coordinate*magnitude*change_normalized`(excluded)"""
        dist = int(magnitude * np.reciprocal(resolution))
        for i in range(1, dist - 1):
            intermediary_coordinate = change_normalized * i + viewer_coordinate
            yield sample_function(intermediary_coordinate), i * factor

    def data_line_of_sight(
        self,
        data_viewer_coordinate: Union[Tuple[float, float], np.ndarray],
        data_target_coordinate: Union[Tuple[float, float], np.ndarray],
        altitude_mod: float,
        resolution: float = 1,
        coordinate_sample_mode=CoordinateSampleMode.POINT,
    ):
        # assert np.all(viewer_coordinate < self._resolution / 2) and np.all(
        #     viewer_coordinate > self._resolution / -2
        # ), f"{viewer_coordinate=} is not within bounds."
        # assert np.all(target_coordinate < self._resolution / 2) and np.all(
        #     target_coordinate > self._resolution / -2
        # ), f"{target_coordinate=} is not within bounds."

        sample_function = self._get_sample_averaging_function(coordinate_sample_mode)

        viewer_height = sample_function(data_viewer_coordinate) + altitude_mod
        target_height = sample_function(data_target_coordinate)

        change = np.subtract(data_target_coordinate, data_viewer_coordinate)
        magnitude: float = np.linalg.norm(change)
        if magnitude == 0:
            return True
        factor = resolution / magnitude

        uv_slope_normalized = np.multiply(change, factor)
        # MTA TODO: reverse iteration
        # Cull opposite facing surfaces (on target surface) to short circuit ray marching
        return line_of_sight_test(
            viewer_height,
            target_height,
            magnitude,
            self._data_sample_line(
                uv_slope_normalized,
                resolution,
                magnitude,
                sample_function,
                data_viewer_coordinate,
                factor,
            ),
        )

    def line_of_sight(
        self,
        viewer_coordinate: Tuple[float, float],
        target_coordinate: Tuple[float, float],
        altitude_mod: float,
        resolution: float = 1,
        coordinate_sample_mode=CoordinateSampleMode.POINT,
    ):
        viewer_coordinate = self.convert_to_data_coordinate(viewer_coordinate)
        target_coordinate = self.convert_to_data_coordinate(target_coordinate)

        return self.data_line_of_sight(
            viewer_coordinate,
            target_coordinate,
            altitude_mod,
            resolution,
            coordinate_sample_mode,
        )

    def to_line_of_sight(
        self,
        viewer_coordinate: Tuple[float, float],
        altitude_mod: float,
        resolution: float = 1,
        coordinate_sample_mode=CoordinateSampleMode.FOUR_POINTS,
    ):

        viewer_coordinate = self.convert_to_data_coordinate(viewer_coordinate)
        assert np.all(
            self.data.shape == tuple(reversed(self.size))
        ), f"Only currently works for images that are the same size as resolution. {self.data.shape=} and {self.size=}"

        out = np.empty(self.data.shape, self.data.dtype)
        for v in range(self.resolution[1]):
            for u in range(self.resolution[0]):
                test = self.data_line_of_sight(
                    viewer_coordinate,
                    (u, v),
                    altitude_mod,
                    resolution,
                    coordinate_sample_mode,
                )
                out[v][u] = np.uint8(test * 255)
        return HeightField(out, self.size)

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

    def write_image(self, file: Union[str, Path, BinaryIO]):
        from PIL import Image

        a = self.data.astype(np.uint8)
        im = Image.fromarray(a, "L")
        im.save(file)

    @classmethod
    def load_image(cls, file: Union[str, Path]):
        from PIL import Image

        with Image.open(file) as im:
            data = np.asarray(im)
            assert len(data.shape) == 2
        return cls(data, data.shape[:2])

    @classmethod
    def from_rgb(cls, data: np.ndarray):
        d = np.min(data, axis=2)
        return HeightField(d, size=(data.shape[1], data.shape[0]))
