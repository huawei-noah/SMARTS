# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import os
import pathlib
from typing import Optional

from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, DoneCriteria, DrivableAreaGridMap, NeighborhoodVehicles, OGM, RGB, RoadWaypoints, Waypoints
from smarts.core.controllers import ActionSpaceType
from smarts.env.hiway_env import HiWayEnv


class IntersectionEnv(HiWayEnv):
    """An intersection environment where the agent needs to make an unprotected
    left turn in the presence of traffic.

    Observation: 
        Type: gym.spaces.Dict({
            gym.spaces.Box(low=0, high=255, shape=(256,256,3), dtype=np.uint8),
            gym.spaces.Box(low=0, high=1, shape=(256,256), dtype=np.uint8),
            gym.spaces.Box(low=0, high=1, shape=(256,256), dtype=np.uint8),
        })

        Key                      Value
        rgb	                     Top down color image (256 x 256)
        ogm	                     Top down binary occupancy grid map (256 x 256)
        driveable_area_grid_map	 Top down binary driveable are grid map (256 x 256)

    Actions:
        Type: gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        Action     Value range
        Throttle   [ 0, 1]
        Brake      [ 0, 1]
        Steering   [-1, 1]  

    Reward:
        Reward is distance travelled (in meters) in each step, including the
        termination step.

    Episode termination:
        Episode is terminated if any of the following is fulfilled.
        Steps per episode exceed 1000.
        Agent collides, drives off road, or drives off route.

    Solved requirement:
        Considered solved when the average return is greater than or equal to
        90.0 over 100 consecutive trials.
    """

    def __init__(
        self,
        headless: bool = False,
        visdom: bool = False,
        sumo_headless: bool = False,
        envision_record_data_replay_path: Optional[str] = None,
    ):
        """
        Args:
            headless (bool, optional): If True, disables visualization in
                Envision. Defaults to False.
            visdom (bool, optional): If True, enables visualization of observed
                RGB images in Visdom. Defaults to False.
            sumo_headless (bool, optional): If True, disables visualization in
                SUMO GUI. Defaults to True.
            envision_record_data_replay_path (Optional[str], optional):
                Envision's data replay output directory. Defaults to None.
        """

        scenario = str(
            pathlib.Path(__file__).absolute().parents[2]
            / "scenarios"
            / "intersections"
            / "2lane_left_turn"
        )
        build_scenario = f"scl scenario build --clean {scenario}"
        os.system(build_scenario)

        done_criteria = DoneCriteria(
            collision=False,
            off_road=True,
            off_route=True,
            on_shoulder=False,
            wrong_way=False,
            not_moving=False,
            agents_alive=None,
        )
        max_episode_steps = 3000
        img_meters = 64
        img_pixels = 256
        agent_specs = {
            "intersection": AgentSpec(
                interface=AgentInterface(
                    done_criteria=done_criteria,
                    max_episode_steps=max_episode_steps,
                    # action=ActionSpaceType.Continuous,
                    action=ActionSpaceType.LaneWithContinuousSpeed,
                    rgb=RGB(
                        width=img_pixels,
                        height=img_pixels,
                        resolution=img_meters / img_pixels,
                    ),
                    ogm=OGM(
                        width=img_pixels,
                        height=img_pixels,
                        resolution=img_meters / img_pixels,
                    ),
                    drivable_area_grid_map=DrivableAreaGridMap(
                        width=img_pixels,
                        height=img_pixels,
                        resolution=img_meters / img_pixels,
                    ),
                    neighborhood_vehicles=NeighborhoodVehicles(img_meters),
                    waypoints=Waypoints(lookahead=40),
                    road_waypoints=RoadWaypoints(horizon=40),
                    accelerometer=True,
                    lidar = True,
                ),
            )
        }

        super(IntersectionEnv, self).__init__(
            scenarios=[scenario],
            agent_specs=agent_specs,
            sim_name="LeftTurn",
            headless=headless,
            visdom=visdom,
            sumo_headless=False,
            envision_record_data_replay_path=envision_record_data_replay_path,
        )


        # fix observation space