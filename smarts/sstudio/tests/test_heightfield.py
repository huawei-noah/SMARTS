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
import os

import numpy as np
import pytest

from smarts.sstudio.graphics.heightfield import CoordinateSampleMode, HeightField


@pytest.fixture
def map_data():
    with open(os.path.join(os.path.dirname(__file__), "heightfield.binary"), "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape((256, 256))


@pytest.fixture
def map_image():
    return np.array([[[0], [1], [0]], [[0], [1], [0]], [[2], [1], [2]]], dtype=np.uint8)


@pytest.fixture
def map_image2():
    return np.array(
        [
            [[0], [1], [0], [1], [0]],
            [[0], [1], [0], [0], [0]],
            [[2], [1], [2], [0], [0]],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def kernel() -> np.ndarray:
    return np.array(
        [
            [1.0, 1.0, 0.5, 1.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [0.5, 0.0, -80, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.5, 0.5, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_heightfield_from_map(map_data):
    heighfield = HeightField(map_data, (100, 100))
    assert np.all(heighfield.data == map_data)


def test_heighfield_from_image(map_image):
    heighfield = HeightField(map_image, (100, 100))
    assert np.all(heighfield.data == np.squeeze(map_image, axis=2))


def test_heightfield_kernel(map_data, kernel):
    heightfield = HeightField(map_data, (100, 100))
    field = heightfield.apply_kernel(kernel)
    assert isinstance(field, HeightField)


def test_heightfield_recoordinate(map_image):
    hf = HeightField(map_image, (100, 100))

    assert np.all(hf.resolution == (3, 3))
    assert np.all(np.equal(hf.convert_to_data_coordinate((-50, -50)), (0, 0)))
    assert np.all(np.equal(hf.convert_to_data_coordinate((0, 0)), (1, 1)))
    assert np.all(np.equal(hf.convert_to_data_coordinate((50, 50)), (2, 2)))


def test_heightfield_inverted(map_image):
    hf = HeightField(data=map_image, size=(3, 3))
    ihf: HeightField = hf.inverted()

    assert np.all(hf.data == ihf.inverted().data)


def test_heightfield_line_of_sight(map_image):
    hf = HeightField(map_image, (100, 100))

    # 2, 1, 2
    # 0, 1, 0
    # 0, 1, 0
    assert np.all(np.array(hf.convert_to_data_coordinate((-50, 49))).round() == (0, 2))
    # With point sampling the mapping on the coordinates like ledges
    #  viewing flat from the surface downward might be blocked
    # _ _ _
    #    |
    #    _ _ _
    assert hf.data_line_of_sight(
        data_viewer_coordinate=np.array((0, 2)),
        data_target_coordinate=np.array((0, 0)),
        altitude_mod=0,
        resolution=1,
        coordinate_sample_mode=CoordinateSampleMode.POINT,
    ), "Line of sight should be broken from strict coordinate mapping"
    # With 4 point sampling the height is averaged to be smooth
    # \   _
    #  \_/ \
    assert hf.line_of_sight(
        viewer_coordinate=(-50, 50),
        target_coordinate=(-50, 0),
        altitude_mod=0,
        resolution=1,
        coordinate_sample_mode=CoordinateSampleMode.FOUR_POINTS,
    ), "Line of sight should be unbroken"
    assert not hf.line_of_sight(
        viewer_coordinate=(-50, -50),
        target_coordinate=(50, -50),
        altitude_mod=0,
        resolution=0.2,
        coordinate_sample_mode=CoordinateSampleMode.FOUR_POINTS,
    ), "Line of sight should be broken"
    assert hf.line_of_sight(
        viewer_coordinate=(-50, 50),
        target_coordinate=(50, 50),
        altitude_mod=0,
        resolution=0.2,
        coordinate_sample_mode=CoordinateSampleMode.FOUR_POINTS,
    ), "Line of sight should be unbroken"
