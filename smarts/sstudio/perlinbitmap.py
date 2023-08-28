# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
from functools import lru_cache
import math

import numpy as np

from smarts.core.coordinates import Dimensions, Pose
from smarts.core.sensor import DrivableAreaGridMapSensor
from smarts.core.vehicle_state import VehicleState
from smarts.p3d.renderer import Renderer
from smarts.sstudio.heightfield import HeightField

@lru_cache
def table_cache(table_dim, seed):
    p = np.arange(table_dim, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    return p


class PerlinNoise:
    """This is a performant perlin noise generator heavily based on https://stackoverflow.com/a/42154921"""
    @classmethod
    def noise(cls, x, y, seed, table_dim):
        """Vectorizes the generation of noise.

        Args:
            x (np.array): The x value grid
            y (np.array): The y value grid
            seed (int): The random seed to use

        Returns:
            np.ndarray: A noise value texture with output range [-0.5:0.5].
        """
        # permutation table
        np.random.seed(seed)
        p = table_cache(table_dim, seed)

        x_mod, y_mod = x % table_dim, y % table_dim
        # coordinates of the top-left
        xi, yi = x_mod.astype(int), y_mod.astype(int)
        # internal coordinates
        xf, yf = x_mod - xi, y_mod - yi
        # fade factors
        u, v = cls.fade(xf), cls.fade(yf)
        # noise components
        n00 = cls.gradient(p[p[xi] + yi], xf, yf)
        n01 = cls.gradient(p[p[xi] + yi + 1], xf, yf - 1)
        n11 = cls.gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
        n10 = cls.gradient(p[p[xi + 1] + yi], xf - 1, yf)
        # combine noises
        x1 = cls.lerp(n00, n10, u)
        x2 = cls.lerp(n01, n11, u)  # FIX1: I was using n10 instead of n01
        return cls.lerp(x1, x2, v)  # FIX2: I also had to reverse x1 and x2 here
    
    @staticmethod
    def fade(t: float) -> float:
        """The fade function"""
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    @staticmethod
    def lerp(a, b, x):
        "linear interpolation"
        return a + x * (b - a)

    @staticmethod
    def gradient(h, x, y):
        "grad converts h to the right gradient vector and return the dot product with (x,y)"
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        return g[:, :, 0] * x + g[:, :, 1] * y



def generate_bitmap(out_bitmap_file, width, height, smooth_iterations, seed, table_dim):
    image = np.zeros((height, width))
    for i in range(0, 20):
        freq = 2**i
        lin = np.linspace(0, freq, width, endpoint=False)
        lin2 = np.linspace(0, freq, height, endpoint=False)
        x, y = np.meshgrid(lin, lin2)  # FIX3: I thought I had to invert x and y here but it was a mistake
        image = PerlinNoise.noise(x, y, seed=seed, table_dim=table_dim)/freq + image

    # print(np.min(image), np.max(image))
    image = (image + 1) * 128
    # print(np.min(image), np.max(image))
    image = image.astype(np.uint8)

    if smooth_iterations:
        hf = HeightField(image, (width, height))
        blur_arr = np.array([0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006])
        blur_arr_u = blur_arr.reshape((1, 7))
        blur_arr_v = blur_arr.reshape((7, 1))
        for i in range(smooth_iterations):
            hf = hf.apply_kernel(
                np.array(blur_arr_u)
            )
            hf = hf.apply_kernel(
                blur_arr_v
            )
            image = hf.data


    from PIL import Image

    im = Image.fromarray(image.squeeze(), "L")
    im.save(out_bitmap_file)
    im.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "perlinbitmap.py",
        description="Utility to export mesh files to bitmap.",
    )
    parser.add_argument("output_path", help="where to write the bitmap file", type=str)
    parser.add_argument("--width", help="the width pixels", type=int, default=100)
    parser.add_argument("--height", help="the height pixels", type=int, default=100)
    parser.add_argument("--smooth_iterations", help="smooth the output", type=int, default=0)
    parser.add_argument("--seed", help="the generator seed", type=int, default=87)
    parser.add_argument("--table_dim", help="the perlin permutation table", type=int, default=2048)

    args = parser.parse_args()

    generate_bitmap(args.output_path, args.width, args.height, args.smooth_iterations, args.seed, args.table_dim)
