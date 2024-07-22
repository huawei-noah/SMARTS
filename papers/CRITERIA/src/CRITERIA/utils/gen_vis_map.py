# MIT License
#
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
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

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import compress_pickle as comp_pkl
import cv2
import numpy as np
from shapely import affinity
from shapely.geometry import Polygon, box


@lru_cache(maxsize=None)
def _get_city_vis_map(city_name, map_path):
    return comp_pkl.load(Path(map_path) / f"raw_map/{city_name}_mask.pkl")


def transform_image(image, from_corner_points, to_image_size, torch_transform=None):
    to_corner_points = np.array(
        [
            [to_image_size[1], 0],
            [to_image_size[1], to_image_size[0]],
            [0, to_image_size[0]],
            [0, 0],
        ]
    )

    transform_matrix = cv2.getPerspectiveTransform(
        from_corner_points.astype(np.float32), to_corner_points.astype(np.float32)
    )

    transformed = cv2.warpPerspective(
        image, transform_matrix, (to_image_size[1], to_image_size[0])
    )

    if torch_transform is not None:
        transformed = torch_transform(transformed)

    return transformed


def get_patch(
    patch_box: Tuple[float, float, float, float], patch_angle: float = 0.0
) -> Polygon:
    """
    Convert patch_box to shapely Polygon coordinates.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :return: Box Polygon for patch_box.
    """
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w * 0.5
    y_min = patch_y - patch_h * 0.5
    x_max = patch_x + patch_w * 0.5
    y_max = patch_y + patch_h * 0.5

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(
        patch, patch_angle, origin=(patch_x, patch_y), use_radians=False
    )

    return patch


def crop_image(image, crop_patch):

    image_h, image_w = image.shape[:2]

    # Corners of the crop in the raw image's coordinate system.
    crop_coords_in_raw = np.array(crop_patch.exterior.coords)

    _, lower_right_corner, _, upper_left_corner = crop_coords_in_raw[:4]
    crop_left, crop_up = upper_left_corner
    crop_right, crop_down = lower_right_corner

    if crop_left < 0:
        pad_left = int(-crop_left) + 1
        crop_left = 0

    else:
        pad_left = 0
        crop_left = int(crop_left)

    if crop_right >= image_w + 1:
        pad_right = int(crop_right - image_w)
        crop_right = int(image_w)
    else:
        pad_right = 0
        crop_right = int(crop_right)

    if crop_up < 0:
        pad_up = int(-crop_up) + 1
        crop_up = 0
    else:
        pad_up = 0
        crop_up = int(crop_up)

    if crop_down >= image_h + 1:
        pad_down = int(crop_down - image_h)
        crop_down = int(image_h)
    else:
        pad_down = 0
        crop_down = int(crop_down)

    image_crop = image[crop_up:crop_down, crop_left:crop_right].copy()

    if pad_left or pad_right or pad_up or pad_down:
        image_crop = np.pad(
            image_crop, ((pad_up, pad_down), (pad_left, pad_right)), mode="constant"
        )

    crop_boundary = {
        "up": crop_up - pad_up,
        "down": crop_down + pad_down,
        "left": crop_left - pad_left,
        "right": crop_right + pad_right,
    }

    return image_crop, crop_boundary


def gen_vis_map(city_name, translation, map_path):
    X, Y = translation
    max_distance = 56.0
    vis_map_size = (224, 224)
    raw_vis_map = _get_city_vis_map(city_name, map_path)["map"]
    image_to_city = _get_city_vis_map(city_name, map_path)["image_to_city"]

    scale_vis_h = image_to_city[1, 1]
    translate_vis_h = image_to_city[1, 2]
    scale_vis_w = image_to_city[0, 0]
    translate_vis_w = image_to_city[0, 2]

    pixel_dims_h = scale_vis_h * max_distance * 2
    pixel_dims_w = scale_vis_w * max_distance * 2

    crop_dims_h = np.ceil(np.sqrt(2 * pixel_dims_h**2) / 10) * 10
    crop_dims_w = np.ceil(np.sqrt(2 * pixel_dims_w**2) / 10) * 10

    # Corners of the crop in the raw image's coordinate system.
    crop_box = (X + translate_vis_w, Y + translate_vis_h, crop_dims_h, crop_dims_w)
    crop_patch = get_patch(crop_box, patch_angle=0.0)

    # Do Crop
    vis_crop, crop_boundary = crop_image(raw_vis_map, crop_patch)

    # Corners of the final image in the crop image's coordinate system.
    final_box = (
        scale_vis_w * X - crop_boundary["left"] + translate_vis_w,
        scale_vis_h * Y - crop_boundary["up"] + translate_vis_h,
        pixel_dims_h,
        pixel_dims_w,
    )

    final_patch_angle = 0.0
    final_patch = get_patch(final_box, patch_angle=final_patch_angle)
    final_coords_in_crop = np.array(final_patch.exterior.coords)
    vis_corner_points = final_coords_in_crop[:4]

    vis_map = transform_image(vis_crop.copy(), vis_corner_points, vis_map_size)

    return vis_map
