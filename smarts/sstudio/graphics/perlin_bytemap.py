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
import importlib.resources as pkg_resources
import os
from functools import lru_cache
from typing import Tuple

import numpy as np

from smarts.sstudio.graphics.heightfield import HeightField


@lru_cache(2)
def table_cache(table_dim, seed):
    p = np.arange(table_dim, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    return p


def get_image_dimensions(image_file):
    hf = HeightField.load_image(image_file)
    return hf.resolution


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
        x2 = cls.lerp(n01, n11, u)
        return cls.lerp(x1, x2, v)

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


def generate_perlin(
    width: int,
    height: int,
    smooth_iterations: int,
    seed: int,
    table_dim: int,
    shift: Tuple[float, float],
    octaves: int = 20,
):
    image = np.zeros((height, width))
    for i in range(0, octaves):
        freq = 2**i
        lin = np.linspace(shift[0], freq + shift[0], width, endpoint=False)
        lin2 = np.linspace(shift[1], freq + shift[1], height, endpoint=False)
        x, y = np.meshgrid(lin, lin2)
        image = PerlinNoise.noise(x, y, seed=seed, table_dim=table_dim) / freq + image

    image = (image + 1) * 128
    image = image.astype(np.uint8)

    hf = HeightField(image, (width, height))
    if smooth_iterations:
        blur_arr = np.array([0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006])
        blur_arr_u = blur_arr.reshape((1, 7))
        blur_arr_v = blur_arr.reshape((7, 1))
        for i in range(smooth_iterations):
            hf = hf.apply_kernel(np.array(blur_arr_u))
            hf = hf.apply_kernel(blur_arr_v)

    return hf


def generate_simplex_p3d_gpu(
    width: int,
    height: int,
    seed,
    shift: Tuple[float, float],
    octaves: float = 2,
    granularity=0.02,
    amplitude=4,
    transformation_matrix: np.ndarray = np.identity(4),
):
    assert height % 16 == 0
    assert width % 16 == 0

    from panda3d.core import ComputeNode, Shader, ShaderAttrib, Texture

    from smarts.core import glsl
    from smarts.p3d.renderer import DEBUG_MODE, Renderer

    renderer = Renderer("noise renderer", debug_mode=DEBUG_MODE.ERROR)
    renderer._ensure_root()

    toNoiseTex = Texture("noise-texture")
    toNoiseTex.setup_2d_texture(width, height, Texture.T_unsigned_byte, Texture.F_r8i)
    toNoiseTex.set_clear_color((0, 0, 0, 0))

    node = ComputeNode("simplex")
    node.add_dispatch(width // 16, height // 16, 1)
    node_path = renderer._root_np.attach_new_node(node)

    with pkg_resources.path(glsl, "simplex.comp") as simplex_shader:
        shader = Shader.load_compute(Shader.SL_GLSL, str(simplex_shader.absolute()))
        node_path.set_shader(shader)
    node_path.set_shader_input("toNoise", toNoiseTex)

    sattr = node_path.getAttrib(ShaderAttrib)
    # renderer.render()
    gsg = renderer._showbase_instance.win.get_gsg()
    assert gsg.get_supports_compute_shaders(), f"renderer {gsg.get_class_type().name}"
    renderer._showbase_instance.graphics_engine.dispatch_compute(
        (32, 32, 1), sattr, gsg
    )


def generate_simplex(
    width: int,
    height: int,
    seed,
    shift: Tuple[float, float],
    octaves: float = 2,
    granularity=0.02,
    amplitude=4,
):
    import noise

    assert amplitude < 256
    half_amp = amplitude / 2
    noise_map = np.empty((width, height), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            nx = x * granularity
            ny = y * granularity
            noise_value = noise.snoise2(
                nx + shift[0],
                ny + shift[1],
                octaves=octaves,
                persistence=0.5,
                lacunarity=2.0,
                repeatx=800,
                repeaty=800,
            )
            noise_map[y][x] = np.floor(half_amp * (1 + noise_value))

    return HeightField(noise_map, size=(width, height))


def generate_perlin_file(
    out_bytemap_file, width, height, smooth_iterations, seed, table_dim, shift
):
    hf = generate_perlin(
        width, height, smooth_iterations, seed, table_dim, (shift, shift)
    )
    image = hf.data

    from PIL import Image

    im = Image.fromarray(image.squeeze(), "L")
    im.save(out_bytemap_file)
    im.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Utility to export mesh files to bytemap.",
    )
    parser.add_argument("output_path", help="where to write the bytemap file", type=str)
    parser.add_argument("--width", help="the width pixels", type=int, default=256)
    parser.add_argument("--height", help="the height pixels", type=int, default=256)
    parser.add_argument(
        "--smooth_iterations", help="smooth the output", type=int, default=0
    )
    parser.add_argument("--seed", help="the generator seed", type=int, default=87)
    parser.add_argument(
        "--table_dim", help="the perlin permutation table", type=int, default=2048
    )
    parser.add_argument("--shift", help="the shift on the noise", type=float, default=0)
    parser.add_argument(
        "--match_file_dimensions",
        help="uses an image file as the base for dimensions",
        type=str,
        default="",
    )

    args = parser.parse_args()

    width, height = args.width, args.height
    if args.match_file_dimensions != "":
        width, height = get_image_dimensions(args.match_file_dimensions)

    # generate_perlin_file(
    #     args.output_path,
    #     width,
    #     height,
    #     args.smooth_iterations,
    #     args.seed,
    #     args.table_dim,
    #     args.shift,
    # )

    generate_simplex_p3d_gpu(
        width,
        height,
        args.seed,
        args.shift,
    )
