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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import random
import psutil
import itertools

import numpy as np
from panda3d.core import Quat

from .utils import pybullet
from .utils.pybullet import bullet_client as bc

from .lidar_sensor_params import SensorParams
from .utils.math import batches


class Lidar:
    def __init__(
        self, origin, sensor_params: SensorParams, bullet_client: bc.BulletClient
    ):
        self._origin = origin
        self._sensor_params = sensor_params
        self._bullet_client = bullet_client
        self._n_threads = psutil.cpu_count(logical=False)

        # As an optimization we compute a set of "base rays" once and shift translate
        # them to follow the user, and then trace for collisions.
        self._base_rays = None
        self._static_lidar_noise = self._compute_static_lidar_noise()

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = value

    def _compute_static_lidar_noise(self):
        n_rays = int(
            (self._sensor_params.end_angle - self._sensor_params.start_angle)
            / self._sensor_params.angle_resolution
        )
        n_points = n_rays * len(self._sensor_params.laser_angles)

        static_lidar_noise = []
        for _ in range(n_points):
            static_lidar_noise.append(
                random.gauss(
                    self._sensor_params.noise_mu, self._sensor_params.noise_sigma
                )
            )
        return np.array(static_lidar_noise, dtype=np.float)

    def compute_point_cloud(self):
        rays = self._compute_rays()
        point_cloud, hits = self._trace_rays(rays)
        # point_cloud = self._apply_noise(point_cloud)
        assert (
            len(point_cloud) == len(hits) == len(rays) == len(self._static_lidar_noise)
        )
        return point_cloud, hits, rays

    def _compute_rays(self):
        if self._base_rays is None:
            self._base_rays = []
            n_rays = int(
                (self._sensor_params.end_angle - self._sensor_params.start_angle)
                / self._sensor_params.angle_resolution
            )

            yaws = -self._sensor_params.laser_angles
            rolls = np.arange(n_rays) * self._sensor_params.angle_resolution
            for yaw, roll in itertools.product(yaws, rolls):
                rot = pybullet.getQuaternionFromEuler((roll, 0, yaw))
                origin = np.array([0, 0, 0])
                direction = np.array(
                    Quat(rot).xform((0, self._sensor_params.max_distance, 0))
                )
                self._base_rays.append((origin, direction))

        rays = [
            (origin + self._origin, direction + self._origin)
            for origin, direction in self._base_rays
        ]
        return rays

    def _trace_rays(self, rays):
        results = []
        for batched_rays in batches(
            rays, int(pybullet.MAX_RAY_INTERSECTION_BATCH_SIZE - 1)
        ):
            origins, directions = zip(*batched_rays)
            results.extend(
                self._bullet_client.rayTestBatch(origins, directions, self._n_threads)
            )

        hit_ids, _, _, positions, _ = zip(*results)
        positions = list(positions)
        hits = []
        for i, position in enumerate(positions):
            hit = hit_ids[i] != -1
            hits.append(hit)
            positions[i] = (
                np.array(position) if hit else np.array([np.inf, np.inf, np.inf])
            )
        return positions, hits

    def _apply_noise(self, point_cloud):
        dynamic_noise = np.random.normal(
            self._sensor_params.noise_mu,
            self._sensor_params.noise_sigma,
            size=len(point_cloud),
        )

        local_pc = point_cloud - self._origin
        noise = self._static_lidar_noise + dynamic_noise
        return point_cloud + (
            local_pc
            / np.linalg.norm(local_pc, axis=1)[:, np.newaxis]
            * noise[:, np.newaxis]
        )
