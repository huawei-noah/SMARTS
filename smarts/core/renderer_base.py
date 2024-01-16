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
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING, Collection, Tuple, Union

import numpy as np

from .coordinates import Pose

if TYPE_CHECKING:
    from smarts.core.shader_buffer import BufferID


class DEBUG_MODE(IntEnum):
    """The rendering debug information level."""

    SPAM = 1
    DEBUG = 2
    INFO = 3
    WARNING = 4
    ERROR = 5


class RendererNotSetUpWarning(UserWarning):
    """This occurs if a renderer is used without being set up."""


@dataclass
class OffscreenCamera(metaclass=ABCMeta):
    """A camera used for rendering images to a graphics buffer."""

    renderer: RendererBase

    @abstractmethod
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

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update the location of the camera.
        Args:
            pose:
                The pose of the camera target.
            height:
                The height of the camera above the camera target.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def image_dimensions(self) -> Tuple[int, int]:
        """The dimensions of the output camera image."""
        raise NotImplementedError

    @property
    @abstractmethod
    def position(self) -> Tuple[float, float, float]:
        """The position of the camera."""
        raise NotImplementedError

    @property
    @abstractmethod
    def heading(self) -> float:
        """The heading of this camera."""
        raise NotImplementedError

    @abstractmethod
    def teardown(self):
        """Clean up internal resources."""
        raise NotImplementedError


@dataclass(frozen=True)
class ShaderStepDependencyBase:
    """The base for shader dependencies."""


@dataclass(frozen=True)
class ShaderStepVariableDependency(ShaderStepDependencyBase):
    """The base for shader dependencies."""

    value: Union[int, float, bool, np.ndarray, list, tuple]
    script_variable_name: str

    def __post_init__(self):
        assert self.value, f"`{self.script_variable_name=}` cannot be None or empty."
        assert self.script_variable_name
        assert (
            0 < len(self.value) < 5
            if isinstance(self.value, (np.ndarray, list, tuple))
            else True
        )


@dataclass(frozen=True)
class ShaderStepBufferDependency(ShaderStepDependencyBase):
    """The base for shader dependencies."""

    buffer_id: BufferID
    script_uniform_name: str

    def __post_init__(self):
        assert self.buffer_id, f"`{self.script_uniform_name=}` cannot be None or empty."
        assert self.script_uniform_name


@dataclass(frozen=True)
class ShaderStepCameraDependency(ShaderStepDependencyBase):
    """Forwards the texture from a given camera to the"""

    camera_id: str
    script_variable_name: str

    def __post_init__(self):
        assert self.script_variable_name, "Variable name cannot be empty."
        assert (
            self.camera_id
        ), f"Camera id for {self.script_variable_name} cannot be None or empty."


@dataclass
class ShaderStep(OffscreenCamera, metaclass=ABCMeta):
    """A camera used for rendering images using a shader and a full-screen quad."""

    shader_file: str
    camera_dependencies: Collection[OffscreenCamera]
    buffer_dependencies: Collection[ShaderStepBufferDependency]


class RendererBase(metaclass=ABCMeta):
    """The base class for rendering

    Returns:
        RendererBase:
    """

    @property
    @abstractmethod
    def id(self):
        """The id of the simulation rendered."""
        raise NotImplementedError

    @property
    @abstractmethod
    def is_setup(self) -> bool:
        """If the renderer has been fully initialized."""
        raise NotImplementedError

    @property
    @abstractmethod
    def log(self) -> logging.Logger:
        """The rendering logger."""
        raise NotImplementedError

    @abstractmethod
    def remove_buffer(self, buffer):
        """Remove the rendering buffer."""
        raise NotImplementedError

    @abstractmethod
    def setup(self, scenario):
        """Initialize this renderer."""
        raise NotImplementedError

    @abstractmethod
    def render(self):
        """Render the scene graph of the simulation."""
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """Reset the render back to initialized state."""
        raise NotImplementedError

    @abstractmethod
    def step(self):
        """provided for non-SMARTS uses; normally not used by SMARTS."""
        raise NotImplementedError

    @abstractmethod
    def sync(self, sim_frame):
        """Update the current state of the vehicles within the renderer."""
        raise NotImplementedError

    @abstractmethod
    def teardown(self):
        """Clean up internal resources."""
        raise NotImplementedError

    @abstractmethod
    def destroy(self):
        """Destroy the renderer. Cleans up all remaining renderer resources."""
        raise NotImplementedError

    @abstractmethod
    def create_vehicle_node(self, glb_model: str, vid: str, color, pose: Pose):
        """Create a vehicle node."""
        raise NotImplementedError

    @abstractmethod
    def begin_rendering_vehicle(self, vid: str, is_agent: bool):
        """Add the vehicle node to the scene graph"""
        raise NotImplementedError

    @abstractmethod
    def update_vehicle_node(self, vid: str, pose: Pose):
        """Move the specified vehicle node."""
        raise NotImplementedError

    @abstractmethod
    def remove_vehicle_node(self, vid: str):
        """Remove a vehicle node"""
        raise NotImplementedError

    @abstractmethod
    def camera_for_id(self, camera_id) -> OffscreenCamera:
        """Get a camera by its id."""
        raise NotImplementedError

    @abstractmethod
    def build_offscreen_camera(
        self,
        name: str,
        mask: int,
        width: int,
        height: int,
        resolution: float,
    ) -> None:
        """Generates a new off-screen camera."""
        raise NotImplementedError

    @abstractmethod
    def build_shader_step(
        self,
        name: str,
        fshader_path: Union[str, Path],
        dependencies: Collection[
            Union[ShaderStepCameraDependency, ShaderStepVariableDependency]
        ],
        priority: int,
        height: int,
        width: int,
    ) -> None:
        """Generates a new shader camera."""
        raise NotImplementedError
