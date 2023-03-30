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
from typing import Any

from smarts.core.coordinates import BoundingBox
from smarts.core.simulation_frame import SimulationFrame
from smarts.core.vehicle_index import VehicleIndex


class PhysicsSimulation:
    """The physics engine base interface."""

    def initialize(self, fixed_timestep_sec: float):
        """Initializes the physics simulation.

        Args:
            fixed_timestep_sec (float): The step size for the simulation.
        """
        raise NotImplementedError

    def initialize_ground(self, resource_path: str, map_bb):
        """Initializes the ground for the simulation.

        Args:
            resource (str): The ground shape resource id.
        """
        raise NotImplementedError

    def reset_simulation(self):
        """Restores the physics simulation back to base state."""
        raise NotImplementedError

    def teardown(self):
        """Close down all open resources relating this physics simulation holds.
        This permanently invalidates and destroys the physics simulation.
        """
        raise NotImplementedError

    def step(
        self, dt: float, simulation_frame: SimulationFrame, vehicle_index: VehicleIndex
    ):
        """Step the current physics simulation.

        Args:
            simulation_frame (SimulationFrame): The simulation frame state.
            vehicle_index (VehicleIndex): The vehicle index current state.
        """
        raise NotImplementedError

    def collider_by_id(self, entity_id: Any):
        """Gets a collider by its given id.

        Args:
            entity_id (Any): The id of an entity to
        """
        raise NotImplementedError

    @property
    def simulation_bounding_box(self) -> BoundingBox:
        """Get the current bounding box of the simulation."""
        raise NotImplementedError

    @property
    def client(self):
        """Returns the underlying physics client."""
        raise NotImplementedError
