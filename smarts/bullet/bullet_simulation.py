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
import importlib.resources as pkg_resources
import os
from typing import Any

from smarts.core import config, models
from smarts.core.coordinates import BoundingBox
from smarts.core.physics.physics_simulation import PhysicsSimulation
from smarts.core.simulation_frame import SimulationFrame
from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc
from smarts.core.vehicle_index import VehicleIndex

MAX_PYBULLET_FREQ = 240


class BulletSimulation(PhysicsSimulation):
    """The pybullet implementation of the physics simulation.
    """
    def __init__(self) -> None:
        # For macOS GUI. See our `BulletClient` docstring for details.
        # from .utils.bullet import BulletClient
        # self._bullet_client = BulletClient(pybullet.GUI)
        # For macOS GUI. See our `BulletClient` docstring for details.
        # from .utils.bullet import BulletClient
        # self._bullet_client = BulletClient(pybullet.GUI)
        self._bullet_client = pybullet.SafeBulletClient(
            pybullet.DIRECT
        )  # pylint: disable=no-member
        self._pybullet_period = 0.1
        self._map_bb = None
        self._ground_bullet_id = None
        self._max_pybullet_freq: int = config()(
            "physics", "max_pybullet_freq", default=MAX_PYBULLET_FREQ, cast=int
        )

    def initialize(self, fixed_timestep_sec: float):
        self._bullet_client.resetSimulation()
        self._bullet_client.configureDebugVisualizer(
            pybullet.COV_ENABLE_GUI, 0  # pylint: disable=no-member
        )
        # PyBullet defaults the timestep to 240Hz. Several parameters are tuned with
        # this value in mind. For example the number of solver iterations and the error
        # reduction parameters (erp) for contact, friction and non-contact joints.
        # Attempting to get around this we set the number of substeps so that
        # timestep * substeps = 240Hz. Bullet (C++) does something to this effect as
        # well (https://git.io/Jvf0M), but PyBullet does not expose it.
        # But if our timestep is variable (due to being externally driven)
        # then we will step pybullet multiple times ourselves as necessary
        # to account for the time delta on each SMARTS step.
        self._pybullet_period = (
            fixed_timestep_sec if fixed_timestep_sec else 1 / self._max_pybullet_freq
        )
        self._bullet_client.setPhysicsEngineParameter(
            fixedTimeStep=self._pybullet_period,
            numSubSteps=int(self._pybullet_period * self._max_pybullet_freq),
            numSolverIterations=10,
            solverResidualThreshold=0.001,
            # warmStartingFactor=0.99
        )

        self._bullet_client.setGravity(0, 0, -9.8)

    def initialize_ground(self, resource_path, map_bb):
        plane_path = resource_path  # self._scenario.plane_filepath
        if not os.path.exists(plane_path):
            with pkg_resources.path(models, "plane.urdf") as path:
                plane_path = str(path.absolute())

        self._map_bb = map_bb

        if self._map_bb:
            # 1e6 is the default value for plane length and width in smarts/models/plane.urdf.
            DEFAULT_PLANE_DIM = 1e6
            ground_plane_scale = (
                2.2 * max(self._map_bb.length, self._map_bb.width) / DEFAULT_PLANE_DIM
            )
            ground_plane_center = self._map_bb.center
        else:
            # first step on undefined map, just use a big scale (1e6).
            # it should get updated as soon as vehicles are added...
            ground_plane_scale = 1.0
            ground_plane_center = (0, 0, 0)

        if self._ground_bullet_id is not None:
            self._bullet_client.removeBody(self._ground_bullet_id)
            self._ground_bullet_id = None

        self._ground_bullet_id = self._bullet_client.loadURDF(
            plane_path,
            useFixedBase=True,
            basePosition=ground_plane_center,
            globalScaling=ground_plane_scale,
        )

    def step(
        self, dt: float, simulation_frame: SimulationFrame, vehicle_index: VehicleIndex
    ):
        self._bullet_client.stepSimulation()
        pybullet_substeps = max(1, round(dt / self._pybullet_period)) - 1
        for _ in range(pybullet_substeps):
            for vehicle in vehicle_index.vehicles:
                vehicle.chassis.reapply_last_control()
            self._bullet_client.stepSimulation()

    def reset_simulation(self):
        self._bullet_client.resetSimulation()

    def teardown(self):
        self._bullet_client.disconnect()

    def collider_by_id(self, entity_id: Any):
        pass

    @property
    def simulation_bounding_box(self) -> BoundingBox:
        return self._map_bb

    @property
    def client(self):
        return self._bullet_client
