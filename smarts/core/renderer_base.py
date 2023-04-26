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


# to allow for typing to refer to class being defined (Renderer)
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

import numpy as np

from .coordinates import Pose


class DEBUG_MODE(IntEnum):
    """The rendering debug information level."""

    SPAM = 1
    DEBUG = 2
    INFO = 3
    WARNING = 4
    ERROR = 5


@dataclass
class OffscreenCamera:
    """A camera used for rendering images to a graphics buffer."""

    renderer: RendererBase

    def wait_for_ram_image(self, img_format: str, retries=100):
        """Attempt to acquire a graphics buffer."""
        # Rarely, we see dropped frames where an image is not available
        # for our observation calculations.
        #
        # We've seen this happen fairly reliable when we are initializing
        # a multi-agent + multi-instance simulation.
        #
        # To deal with this, we can try to force a render and block until
        # we are fairly certain we have an image in ram to return to the user
        raise NotImplementedError

    def update(self, pose: Pose, height: float):
        """Update the location of the camera.
        Args:
            pose:
                The pose of the camera target.
            height:
                The height of the camera above the camera target.
        """
        raise NotImplementedError

    @property
    def image_dimensions(self) -> Tuple[int, int]:
        """The dimensions of the output camera image."""
        raise NotImplementedError

    @property
    def position(self) -> Tuple[float, float, float]:
        """The position of the camera."""
        raise NotImplementedError

    @property
    def heading(self) -> float:
        """The heading of this camera."""
        raise NotImplementedError

    def teardown(self):
        """Clean up internal resources."""
        raise NotImplementedError


class RendererBase:
    """The base class for rendering

    Returns:
        RendererBase:
    """

    @property
    def id(self):
        """The id of the simulation rendered."""
        raise NotImplementedError

    @property
    def is_setup(self) -> bool:
        """If the renderer has been fully initialized."""
        raise NotImplementedError

    @property
    def log(self) -> logging.Logger:
        """The rendering logger."""
        raise NotImplementedError

    def remove_buffer(self, buffer):
        """Remove the rendering buffer."""
        raise NotImplementedError

    def setup(self, scenario):
        """Initialize this renderer."""
        raise NotImplementedError

    def render(self):
        """Render the scene graph of the simulation."""
        raise NotImplementedError

    def reset(self):
        """Reset the render back to initialized state."""
        raise NotImplementedError

    def step(self):
        """provided for non-SMARTS uses; normally not used by SMARTS."""
        raise NotImplementedError

    def sync(self, sim_frame):
        """Update the current state of the vehicles within the renderer."""
        raise NotImplementedError

    def teardown(self):
        """Clean up internal resources."""
        raise NotImplementedError

    def destroy(self):
        """Destroy the renderer. Cleans up all remaining renderer resources."""
        raise NotImplementedError

    def create_vehicle_node(self, glb_model: str, vid: str, color, pose: Pose):
        """Create a vehicle node."""
        raise NotImplementedError

    def begin_rendering_vehicle(self, vid: str, is_agent: bool):
        """Add the vehicle node to the scene graph"""
        raise NotImplementedError

    def update_vehicle_node(self, vid: str, pose: Pose):
        """Move the specified vehicle node."""
        raise NotImplementedError

    def remove_vehicle_node(self, vid: str):
        """Remove a vehicle node"""
        raise NotImplementedError

    def camera_for_id(self, camera_id) -> OffscreenCamera:
        """Get a camera by its id."""
        raise NotImplementedError

    def build_offscreen_camera(
        self,
        name: str,
        mask: int,
        width: int,
        height: int,
        resolution: float,
    ) -> OffscreenCamera:
        """Generates a new offscreen camera."""
        raise NotImplementedError
