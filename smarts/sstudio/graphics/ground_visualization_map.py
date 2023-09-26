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
import argparse
import os
import pickle
import warnings
from functools import cached_property, lru_cache
from typing import Any, Dict, Tuple

import cloudpickle
import numpy as np

from smarts.sstudio.graphics.heightfield import CoordinateSampleMode, HeightField


class GroundPixelViewability:
    def __init__(self, coordinate: Tuple[int, int]) -> None:
        self._coordinate = coordinate
        self._map: Dict[Tuple[int, int]] = {}

    def mark(self, coordinate: Tuple[int, int], bit: bool):
        self._map[coordinate] = bit

    def test_coordinate(self, coordinate: Tuple[int, int]):
        return self._map.get(coordinate, False)

    def to_byte_array(self, shape: np.ndarray):
        if not np.all(np.remainder(shape, 2)):
            warnings.warn(f"It is recommended to use odd value dimensions", UserWarning)
        out = np.zeros(shape, dtype=np.bool8)
        shift = np.floor_divide(shape, 2, dtype=np.int64)

        r_coordinate = np.empty(2, dtype=np.int64)
        for coord, v in self._map.items():
            if not v:
                continue
            np.subtract(coord, self._coordinate, out=r_coordinate)
            np.add(r_coordinate, shift, out=r_coordinate)
            if np.any(r_coordinate >= shape):
                continue
            if np.any(r_coordinate < 0):
                continue
            out[r_coordinate[0]][r_coordinate[1]] = v
        return out

    @property
    def coordinate(self):
        return self._coordinate


class GroundViewablityMap:
    def __init__(self, resolution: float):
        if resolution == 0:
            raise ValueError("Resolution cannot be zero")
        self._pixel_viewabilities: Dict[Tuple[int, int], GroundPixelViewability] = {}
        self._resolution = resolution
        self._inv_resolution = 1 / resolution

    @cached_property
    def _default_gpv(self):
        return GroundPixelViewability((0, 0))

    def _convert_coordinate(self, coordinate):
        return tuple(round(v * self._inv_resolution) for v in coordinate)

    def mark(
        self,
        target_coordinate: Tuple[float, float],
        spectating_coordinate: Tuple[float, float],
        bit: bool,
    ):
        spectating_coordinate_ints = self._convert_coordinate(spectating_coordinate)

        if (gpv := self._pixel_viewabilities.get(spectating_coordinate_ints)) == None:
            gpv = GroundPixelViewability(coordinate=spectating_coordinate_ints)
            self._pixel_viewabilities[spectating_coordinate_ints] = gpv

        gpv.mark(self._convert_coordinate(target_coordinate), bit)

    def test_visability(
        self,
        target_coordinate: Tuple[float, float],
        spectating_coordinate: Tuple[float, float],
    ):
        gpv = self._pixel_viewabilities.get(
            self._convert_coordinate(spectating_coordinate), self._default_gpv
        )

        return gpv.test_coordinate(self._convert_coordinate(target_coordinate))

    def coordinate_to_bytes(self, coordinate, dimensions):
        gpv = self._pixel_viewabilities.get(
            self._convert_coordinate(coordinate), self._default_gpv
        )

        return gpv.to_byte_array(dimensions)

    def generate_visibility_from_coordinate(
        self,
        road_surface: HeightField,
        topology: HeightField,
        coordinate: Tuple[int, int],
        uv_dimensions: Tuple[int, int],
    ):
        self.generate_visibility_from_coordinates(
            road_surface, topology, [coordinate], uv_dimensions
        )
        return self.coordinate_to_bytes(coordinate, dimensions=uv_dimensions)

    def generate_visibility_from_coordinates(
        self,
        road_surface: HeightField,
        topology: HeightField,
        viewer_coordinates,
        size: Tuple[int, int],
    ):
        half_u, half_v = np.floor_divide(size, 2)
        half_topology_shape = np.floor_divide(topology.size, 2)
        min_u, min_v = 0, 0
        max_u, max_v = road_surface.data.shape

        for viewer_coordinate in viewer_coordinates:
            row, column = viewer_coordinate
            if not road_surface.data[viewer_coordinate[1]][viewer_coordinate[0]]:
                continue
            min_row, max_row = max(row + -half_u, min_u), min(row + half_u, max_u)
            min_column, max_column = max(column + -half_v, min_v), min(
                column + half_v, max_v
            )

            for target_row_offset in range(min_row, max_row):
                for target_column_offset in range(min_column, max_column):
                    target_coordinate = np.array(
                        (target_row_offset, target_column_offset), dtype=np.int64
                    )
                    if not road_surface.data[target_column_offset][target_row_offset]:
                        continue
                    hit = topology.line_of_sight(
                        viewer_coordinate=np.subtract(
                            viewer_coordinate, half_topology_shape
                        ),
                        target_coordinate=np.subtract(
                            target_coordinate, half_topology_shape
                        ),
                        altitude_mod=3,
                        resolution=1,
                        coordinate_sample_mode=CoordinateSampleMode.FOUR_POINTS,
                    )
                    self.mark(
                        target_coordinate=target_coordinate,
                        spectating_coordinate=viewer_coordinate,
                        bit=hit,
                    )

    def save(self, file):
        pickle.dump(self, file)

    @classmethod
    def load(cls, file):
        return pickle.load(file)


def gen_visibility_file(road_surface_file, topology_file, output_file, width, height):
    road_surface = HeightField.load_image(road_surface_file)
    topology = HeightField.load_image(topology_file)
    gvm = GroundViewablityMap(1)
    from smarts.core.utils.logging import timeit

    def gen_viewer_coordinates(road_surface):
        max_u, max_v = road_surface.data.shape
        for row in range(max_v):
            with timeit("row took", print):
                for column in range(max_u):
                    yield row, column

    gvm.generate_visibility_from_coordinates(
        road_surface, topology, gen_viewer_coordinates(road_surface), (width, height)
    )

    with open(output_file, "wb") as f:
        gvm.save(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Utility to convert a bytemap to an edge bytemap.",
    )
    parser.add_argument("road_surface_file", help="surface file (*.bmp)", type=str)
    parser.add_argument("topology_file", help="surface file (*.bmp)", type=str)
    parser.add_argument(
        "output_file", help="where to write the edge bytemap file", type=str
    )
    parser.add_argument(
        "--width", help="the width of the output fragments", type=int, default=16
    )
    parser.add_argument(
        "--height", help="the height of the output fragments", type=int, default=16
    )
    args = parser.parse_args()

    gen_visibility_file(
        args.road_surface_file,
        args.topology_file,
        args.output_file,
        args.width,
        args.height,
    )
