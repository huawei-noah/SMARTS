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

from smarts.core.coordinates import Heading, Pose
from smarts.core.renderer_base import ShaderStep, ShaderStepVariableDependency
from smarts.sstudio.graphics.heightfield import HeightField


def get_image_dimensions(image_file):
    hf = HeightField.load_image(image_file)
    return hf.resolution


def generate_simplex_p3d_gpu(
    width: int,
    height: int,
):
    assert height % 16 == 0
    assert width % 16 == 0

    from smarts.core import glsl
    from smarts.p3d.renderer import DEBUG_MODE, Renderer

    renderer = Renderer(
        "noise renderer", debug_mode=DEBUG_MODE.ERROR, rendering_backend="p3headlessgl"
    )

    with pkg_resources.path(glsl, "simplex_shader.frag") as simplex_shader:
        camera_id = renderer.build_shader_step(
            "simplex_camera",
            simplex_shader,
            dependencies=(ShaderStepVariableDependency(1.0, "scale"),),
            priority=10,
            width=width,
            height=height,
        )
        camera = renderer.camera_for_id(camera_id)
        camera.update(Pose.from_center(np.array([0, 0, 0]), Heading(0)), 10)

    renderer.render()

    ram_image = camera.wait_for_ram_image("RGB")
    mem_view = memoryview(ram_image)
    image: np.ndarray = np.frombuffer(mem_view, np.uint8)[::3]
    image = np.reshape(image, (width, height))

    assert np.any(image > 0), image

    return HeightField(image, size=(width, height))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="Utility to export mesh files to bytemap.",
    )
    parser.add_argument("output_path", help="where to write the bytemap file", type=str)
    parser.add_argument("--width", help="the width pixels", type=int, default=256)
    parser.add_argument("--height", help="the height pixels", type=int, default=256)
    parser.add_argument(
        "--match_file_dimensions",
        help="uses an image file as the base for dimensions",
        type=str,
        default="",
    )

    args = parser.parse_args()

    a_width, a_height = args.width, args.height
    if args.match_file_dimensions != "":
        a_width, a_height = get_image_dimensions(args.match_file_dimensions)

    f_hf = generate_simplex_p3d_gpu(
        a_width,
        a_height,
    )

    image_data = f_hf.data

    from PIL import Image

    f_im = Image.fromarray(image_data.squeeze(), "L")
    f_im.save(args.output_path)
    f_im.close()
