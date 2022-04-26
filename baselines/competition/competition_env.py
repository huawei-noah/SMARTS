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

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import math

import gym
import gym.spaces as spaces
import numpy as np


from smarts.core import seed as smarts_seed
from smarts.core.coordinates import Dimensions, Heading
from smarts.core.events import Events
from smarts.core.plan import EndlessGoal, Mission, Start
from smarts.core.road_map import Waypoint
from smarts.core.sensors import (
    DrivableAreaGridMap,
    EgoVehicleObservation,
    GridMapMetadata,
    Observation,
    OccupancyGridMap,
    RoadWaypoints,
    TopDownRGB,
    VehicleObservation,
    Vias,
)
import smarts.sstudio.types as t

dummy_obs: Optional[Observation] = None

MAX_MPS = 100


def _filter(obs: Observation, target_position, env):
    def _clip(formatted_obs, observation_space):
        return {
            k: np.clip(v, observation_space[k].low, observation_space[k].high)
            for k, v in formatted_obs.items()
        }

    obs = {
        "position": obs.ego_vehicle_state.position,
        "linear_velocity": obs.ego_vehicle_state.linear_velocity,
        "target_position": target_position,
        "rgb": obs.top_down_rgb.data.astype(np.uint8),
    }
    obs = _clip(obs, env.observation_space)
    assert env.observation_space.contains(
        obs
    ), "Observation mismatch with observation space. Less keys in observation space dictionary."
    return obs


class CompetitionEnv(gym.Env):
    """A generic environment for various driving tasks simulated by SMARTS."""

    metadata = {"render.modes": ["human"]}
    """Metadata for gym's use"""
    action_space = spaces.Box(low=-20.0, high=20.0, shape=(2,), dtype=np.float)
    observation_space = spaces.Dict(
        {
            # position x, y, z in meters
            "position": spaces.Box(
                low=-math.inf,
                high=math.inf,
                shape=(3,),
                dtype=np.float32,
            ),
            # Velocity
            "linear_velocity": spaces.Box(
                low=-MAX_MPS,
                high=MAX_MPS,
                shape=(3,),
                dtype=np.float32,
            ),
            # target position x, y, z in meters
            "target_position": spaces.Box(
                low=-math.inf,
                high=math.inf,
                shape=(3,),
                dtype=np.float32,
            ),
            # RGB image
            "rgb": spaces.Box(
                low=0,
                high=255,
                shape=(
                    256,
                    256,
                    3,
                ),
                dtype=np.uint8,
            ),
        }
    )

    def __init__(
        self,
        headless: bool = True,
        seed: int = 42,
        envision_endpoint: Optional[str] = None,
        envision_record_data_replay_path: Optional[str] = None,
    ):
        """
        Args:
            sim_name (Optional[str], optional): Simulation name. Defaults to
                None.
            headless (bool, optional): If True, disables visualization in
                Envision. Defaults to True.
            seed (int, optional): Random number generator seed. Defaults to 42.
            envision_endpoint (Optional[str], optional): Envision's uri.
                Defaults to None.
            envision_record_data_replay_path (Optional[str], optional):
                Envision's data replay output directory. Defaults to None.
        """
        self.seed(seed)
        self._current_time = 0.0

    def seed(self, seed: int) -> List[int]:
        """Sets random number generator seed number.

        Args:
            seed (int): Seed number.

        Returns:
            list[int]: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'.
        """
        assert isinstance(seed, int), "Seed value must be an integer."
        smarts_seed(seed)
        return [seed]

    def step(
        self, agent_action: Tuple[float, float]
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Steps the environment.

        Args:
            agent_action (Tuple[float, float]): Action taken by the agent.

        Returns:
            Tuple[Observation, float, bool, Any]:
                Observation, reward, done, and info for the environment.
        """
        global dummy_obs
        self._current_time += dummy_obs.dt
        target = [0, 0, 0]
        return (
            _filter(dummy_obs, target, self),
            0.1,
            False,
            dict(),
        )

    def reset(self) -> Observation:
        """Reset the environment and initialize to the next scenario.

        Returns:
            Observation: Agents' observation.
        """
        global dummy_obs
        self._current_time += dummy_obs.dt
        target = [0, 0, 0]
        return _filter(dummy_obs, target, self)

    def render(self, mode="human"):
        """Does nothing."""
        pass

    def close(self):
        """Closes the environment and releases all resources."""
        return


dummy_obs = Observation(
    dt=0.1,
    step_count=1,
    elapsed_sim_time=0.0,
    events=Events(
        collisions=[],
        off_road=False,
        off_route=False,
        on_shoulder=True,
        wrong_way=False,
        not_moving=False,
        reached_goal=False,
        reached_max_episode_steps=False,
        agents_alive_done=False,
    ),
    ego_vehicle_state=EgoVehicleObservation(
        id="AGENT-007-07a0ca6e",
        position=np.array([161.23485529, 3.2, 0.0]),
        bounding_box=Dimensions(length=3.68, width=1.47, height=1.0),
        heading=Heading(-1.5707963267948966),
        speed=5.0,
        steering=-0.0,
        yaw_rate=4.71238898038469,
        road_id="east",
        lane_id="east_2",
        lane_index=2,
        mission=Mission(
            start=Start(
                position=np.array([163.07485529, 3.2]),
                heading=Heading(-1.5707963267948966),
                from_front_bumper=True,
            ),
            goal=EndlessGoal(),
            route_vias=(),
            start_time=0.1,
            entry_tactic=t.TrapEntryTactic(
                wait_to_hijack_limit_s=0,
                zone=None,
                exclusion_prefixes=(),
                default_entry_speed=None,
            ),
            via=(),
            vehicle_spec=None,
        ),
        linear_velocity=np.array([5.000000e00, 3.061617e-16, 0.000000e00]),
        angular_velocity=np.array([0.0, 0.0, 0.0]),
        linear_acceleration=np.array([0.0, 0.0, 0.0]),
        angular_acceleration=np.array([0.0, 0.0, 0.0]),
        linear_jerk=np.array([0.0, 0.0, 0.0]),
        angular_jerk=np.array([0.0, 0.0, 0.0]),
    ),
    neighborhood_vehicle_states=[
        VehicleObservation(
            id="car-flow-route-west_0_0-east_0_max-7845114534199723832--7266489842092764092--0-0.0",
            position=(-1.33354215, -3.2, 0.0),
            bounding_box=Dimensions(length=3.68, width=1.47, height=1.4),
            heading=Heading(-1.5707963267948966),
            speed=5.050372796758114,
            road_id="west",
            lane_id="west_0",
            lane_index=0,
        ),
    ],
    waypoint_paths=[
        [
            Waypoint(
                pos=np.array([192.00733923, -3.2]),
                heading=Heading(-1.5707963267948966),
                lane_id="east_0",
                lane_width=3.2,
                speed_limit=5.0,
                lane_index=0,
            ),
            Waypoint(
                pos=np.array([193.0, -3.2]),
                heading=Heading(-1.5707963267948966),
                lane_id="east_0",
                lane_width=3.2,
                speed_limit=5.0,
                lane_index=0,
            ),
        ],
        [
            Waypoint(
                pos=np.array([192.00733923, 0.0]),
                heading=Heading(-1.5707963267948966),
                lane_id="east_1",
                lane_width=3.2,
                speed_limit=5.0,
                lane_index=1,
            ),
            Waypoint(
                pos=np.array([193.0, 0.0]),
                heading=Heading(-1.5707963267948966),
                lane_id="east_1",
                lane_width=3.2,
                speed_limit=5.0,
                lane_index=1,
            ),
        ],
    ],
    distance_travelled=0.0,
    lidar_point_cloud=[],
    drivable_area_grid_map=DrivableAreaGridMap(
        metadata=GridMapMetadata(
            created_at=1649853761,
            resolution=0.1953125,
            width=256,
            height=256,
            camera_pos=(161.235, 3.2, 73.6),
            camera_heading_in_degrees=-90.0,
        ),
        data=np.array(
            [
                [[0]] * 256,
            ]
            * 256,
            dtype=np.uint8,
        ),
    ),
    occupancy_grid_map=OccupancyGridMap(
        metadata=GridMapMetadata(
            created_at=1649853761,
            resolution=0.1953125,
            width=256,
            height=256,
            camera_pos=(161.235, 3.2, 73.6),
            camera_heading_in_degrees=-90.0,
        ),
        data=np.array(
            [
                [[0]] * 256,
            ]
            * 256,
            dtype=np.uint8,
        ),
    ),
    top_down_rgb=TopDownRGB(
        metadata=GridMapMetadata(
            created_at=1649853761,
            resolution=0.1953125,
            width=256,
            height=256,
            camera_pos=(161.235, 3.2, 73.6),
            camera_heading_in_degrees=-90.0,
        ),
        data=np.array(
            [
                [
                    [0, 0, 0],
                ]
                * 256,
            ]
            * 256,
            dtype=np.uint8,
        ),
    ),
    road_waypoints=RoadWaypoints(
        lanes={
            "east_0": [
                [
                    Waypoint(
                        pos=np.array([180.00587138, -3.2]),
                        heading=Heading(-1.5707963267948966),
                        lane_id="east_0",
                        lane_width=3.2,
                        speed_limit=5.0,
                        lane_index=0,
                    ),
                    Waypoint(
                        pos=np.array([181.0, -3.2]),
                        heading=Heading(-1.5707963267948966),
                        lane_id="east_0",
                        lane_width=3.2,
                        speed_limit=5.0,
                        lane_index=0,
                    ),
                ]
            ],
        }
    ),
    via_data=Vias(near_via_points=[], hit_via_points=[]),
)
