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
from typing import Set

import numpy as np

from .provider import ProviderState
from .controllers import ActionSpaceType
from .coordinates import Pose, Heading
from .bezier_motion_planner import BezierMotionPlanner
from .vehicle import VEHICLE_CONFIGS, VehicleState


class MotionPlannerProvider:
    def __init__(self):
        self._is_setup = False

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return {ActionSpaceType.TargetPose, ActionSpaceType.MultiTargetPose}

    def setup(self, scenario) -> ProviderState:
        self._motion_planner = BezierMotionPlanner()
        self._vehicle_id_to_index = {}
        self._vehicle_index_to_id = {}
        self._poses = np.empty(shape=(0, 3))  # [[x, y, heading]]; pose of vehicle
        self._is_setup = True

        return ProviderState()

    def teardown(self):
        self._is_setup = False

    def sync(self, provider_state):
        pass  # ... we ignore other sim state here

    def reset(self):
        pass

    def step(self, target_poses_at_t, dt, elapsed_sim_time) -> ProviderState:
        """Step through and update the vehicle poses

        Args:
            target_poses_at_t:
                {vehicle_id: [x, y, heading, seconds_into_future]}
                 pose we would like to have this many seconds into the future
            dt:
                sim time in seconds to advance into the future
        """
        assert self._is_setup
        self._update_membership(target_poses_at_t)

        target_poses_at_t = np.array(
            [
                self._normalize_target_pose(v_index, target_poses_at_t, dt)
                for v_index in self._vehicle_id_to_index.values()
            ],
        ).reshape(-1, 4)

        # vectorized cubic bezier computation
        indices = list(self._vehicle_id_to_index.values())
        first_point_of_traj = self._motion_planner.trajectory_batched(
            self._poses.take(indices, axis=0), target_poses_at_t, 1, dt
        ).reshape(-1, 4)
        speeds = first_point_of_traj[:, 3]
        poses = first_point_of_traj[:, :3]
        self._poses[indices] = poses
        vehicle_type = "passenger"  # TODO: allow for multiple vehicle types

        return ProviderState(
            vehicles=[
                VehicleState(
                    vehicle_id=v_id,
                    vehicle_type=vehicle_type,
                    pose=Pose.from_center(
                        [*poses[idx][:2], 0], Heading(poses[idx][2]),
                    ),
                    dimensions=VEHICLE_CONFIGS[vehicle_type].dimensions,
                    speed=speeds[idx],
                    source="BEZIER",
                )
                for idx, v_id in enumerate(self._vehicle_id_to_index.keys())
            ],
            traffic_light_systems=[],
        )

    def _normalize_target_pose(self, vehicle_index, target_poses, dt):
        # Vehicle index may or may not map to an active vehicle, it could be an
        # index of a vehicle we have not garbage collected yet

        vehicle_id = self._vehicle_index_to_id.get(vehicle_index, None)
        pose = target_poses.get(vehicle_id, None)
        if pose is None:
            # agent failed to produce a target pose, just use the previous pose
            prev_pose = self._poses[vehicle_index]
            return np.array([*prev_pose[:3], dt])
        return pose

    def _update_membership(self, active_agents):
        new_ids = {
            veh_id
            for veh_id in active_agents
            if veh_id not in self._vehicle_id_to_index
        }
        removed_ids = {
            veh_id
            for veh_id in self._vehicle_id_to_index
            if veh_id not in active_agents
        }

        assert (
            new_ids == set()
        ), f"{new_ids} should have been created ahead of time with self.create_vehicle(..)"

        for smarts_id in removed_ids:
            self._destroy_vehicle(smarts_id)

        # We should be synced up
        assert set(active_agents.keys()) == set(self._vehicle_id_to_index.keys())
        assert set(active_agents.keys()) == set(self._vehicle_index_to_id.values())

    def create_vehicle(self, provider_vehicle: VehicleState):
        assert self._is_setup

        vehicle_id = provider_vehicle.vehicle_id
        assert vehicle_id not in self._vehicle_id_to_index
        vehicle_index = self._alloc_index()

        self._vehicle_id_to_index[vehicle_id] = vehicle_index
        self._vehicle_index_to_id[vehicle_index] = vehicle_id

        position, heading = (
            provider_vehicle.pose.position,
            provider_vehicle.pose.heading,
        )
        self._poses = np.append(self._poses, [*position[:2], heading]).reshape(-1, 3)

    def _alloc_index(self) -> int:
        return len(self._poses)

    def _destroy_vehicle(self, vehicle_id):
        vehicle_index = self._vehicle_id_to_index.pop(vehicle_id)
        removed_vehicle_id = self._vehicle_index_to_id.pop(vehicle_index)
        assert removed_vehicle_id == vehicle_id

        # TODO: Currently we leak poses for the duration of the episode.
        #       This is probably fine for now since we tend to have fairly small scale sims.
        #       In the future we would like to have smarter index allocation (re-use destroyed vehicle indices)
        #       or perform periodic garbage collection to remove unused vehicle indices
