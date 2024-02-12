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
from typing import Sequence

import numpy as np


def replace_rgb_image_color(
    rgb: np.ndarray,
    old_color: Sequence[np.ndarray],
    new_color: np.ndarray,
    mask: np.ndarray = np.ma.nomask,
) -> np.ndarray:
    """Convert pixels of value `old_color` to `new_color` within the masked
        region in the received RGB image.

    Args:
        rgb (np.ndarray): RGB image. Shape = (m,n,3).
        old_color (Sequence[np.ndarray]): List of old colors to be removed from the RGB image. Shape = (3,).
        new_color (np.ndarray): New color to be added to the RGB image. Shape = (3,).
        mask (np.ndarray, optional): Valid regions for color replacement. Shape = (m,n,3).
            Defaults to np.ma.nomask .

    Returns:
        np.ndarray: RGB image with `old_color` pixels changed to `new_color`
            within the masked region. Shape = (m,n,3).
    """
    # fmt: off
    assert all(color.shape == (3,) for color in old_color), (
        f"Expected old_color to be of shape (3,), but got {[color.shape for color in old_color]}.")
    assert new_color.shape == (3,), (
        f"Expected new_color to be of shape (3,), but got {new_color.shape}.")

    nc = new_color.reshape((1, 1, 3))
    nc_array = np.full_like(rgb, nc)
    rgb_masked = np.ma.MaskedArray(data=rgb, mask=mask)

    rgb_condition = rgb_masked
    result = rgb
    for color in old_color:
        result = np.ma.where((rgb_condition == color.reshape((1, 1, 3))).all(axis=-1)[..., None], nc_array, result)

    return result
    # fmt: on


def points_to_pixels(
    points: np.ndarray,
    center_position: np.ndarray,
    heading: float,
    width: int,
    height: int,
    resolution: float,
) -> np.ndarray:
    """Converts points into pixel coordinates in order to superimpose the
    points onto the RGB image.

    Args:
        points (np.ndarray): Array of points. Shape (n,3).
        center_position (np.ndarray): Center position of image. Generally, this
            is equivalent to ego position. Shape = (3,).
        heading (float): Heading of image in radians. Generally, this is
            equivalent to ego heading.
        width (int): Width of RGB image
        height (int): Height of RGB image.
        resolution (float): Resolution of RGB image in meters/pixels. Computed
            as ground_size/image_size.

    Returns:
        np.ndarray: Array of point coordinates on the RGB image. Shape = (m,3).
    """
    # fmt: off
    mask = np.array([not all(point == np.zeros(3,)) for point in points], dtype=bool)
    points_nonzero = points[mask]
    points_delta = points_nonzero - center_position
    points_rotated = rotate_axes(points_delta, theta=heading)
    points_pixels = points_rotated / np.array([resolution, resolution, resolution])
    points_overlay = np.array([width / 2, height / 2, 0]) + points_pixels * np.array([1, -1, 1])
    points_rfloat = np.rint(points_overlay)
    points_valid = points_rfloat[(points_rfloat[:,0] >= 0) & (points_rfloat[:,0] < width) & (points_rfloat[:,1] >= 0) & (points_rfloat[:,1] < height)]
    points_rint = points_valid.astype(int)
    return points_rint
    # fmt: on


def rotate_axes(points: np.ndarray, theta: float) -> np.ndarray:
    """A counterclockwise rotation of the x-y axes by an angle theta θ about
    the z-axis.

    Args:
        points (np.ndarray): x,y,z coordinates in original axes. Shape = (n,3).
        theta (np.float): Axes rotation angle in radians.

    Returns:
        np.ndarray: x,y,z coordinates in rotated axes. Shape = (n,3).
    """
    # fmt: off
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ ct, st, 0], 
                  [-st, ct, 0], 
                  [  0,  0, 1]])
    rotated_points = (R.dot(points.T)).T
    return rotated_points
    # fmt: on
