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
from typing import Any, Dict, Optional, Tuple

import gym

from smarts import sstudio
from smarts.core.agent_interface import (
    OGM,
    RGB,
    AgentInterface,
    DoneCriteria,
    DrivableAreaGridMap,
    RoadWaypoints,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType
from smarts.env.hiway_env import HiWayEnv
from smarts.zoo.agent_spec import AgentSpec


def multi_scenario_v0_env(
    scenario: str,
    img_meters: int = 64,
    img_pixels: int = 256,
    action_space="TargetPose",
    headless: bool = True,
    visdom: bool = False,
    sumo_headless: bool = True,
    envision_record_data_replay_path: Optional[str] = None,
):
    """An environment with a mission to be completed by a single or multiple ego agents.

    Observation space for each agent:
        A `smarts.core.sensors.Observation` is returned as observation.

    Action space for each agent:
        A `smarts.core.controllers.ActionSpaceType.TargetPose`, which is a
        sequence of [x-coordinate, y-coordinate, heading, and time-delta].

        Type: gym.spaces.Box(
                low=np.array([-1e10, -1e10, -π, 0]),
                high=np.array([1e10, 1e10, π, 1e10]),
                dtype=np.float32
            )

        Action                                              Value range
        Ego's next x-coordinate on the map                  [-1e10,1e10]
        Ego's next y-coordinate on the map                  [-1e10,1e10]
        Ego's next heading with respect to the map's axes   [-π,π]
        Time delta to reach the given pose                  [0,1e10]

    Reward:
        Reward is distance travelled (in meters) in each step, including the
        termination step.

    Episode termination:
        Episode is terminated if any of the following occurs.
        + Steps per episode exceed 800.
        + Agent collides, drives off road, drives off route, or drives on wrong way.

    Solved requirement:
        If agent successfully completes the mission then `info["score"]` will
        equal 1, else it is 0. Considered solved when `info["score"] == 1` is
        achieved over 500 consecutive episodes.

    Args:
        scenario (str): Scenario name or path to scenario folder.
        img_meters (int): Ground square size covered by image observations.
            Defaults to 64 x 64 meter (height x width) square.
        img_pixels (int): Pixels representing the square image observations.
            Defaults to 256 x 256 pixels (height x width) square.
        action_space: Action space used. Defaults to "Continuous".
        headless (bool, optional): If True, disables visualization in
            Envision. Defaults to False.
        visdom (bool, optional): If True, enables visualization of observed
            RGB images in Visdom. Defaults to False.
        sumo_headless (bool, optional): If True, disables visualization in
            SUMO GUI. Defaults to True.
        envision_record_data_replay_path (Optional[str], optional):
            Envision's data replay output directory. Defaults to None.

    Returns:
        An environment described by the input argument `scenario`.
    """

    env_specs = _get_env_specs(scenario)
    sstudio.build_scenario(scenario=[env_specs["scenario"]])

    agent_specs = {
        f"Agent_{i}": AgentSpec(
            interface=resolve_agent_interface(img_meters, img_pixels, action_space)
        )
        for i in range(env_specs["num_agent"])
    }

    env = HiWayEnv(
        scenarios=[env_specs["scenario"]],
        agent_specs=agent_specs,
        sim_name="MultiScenario",
        headless=headless,
        visdom=visdom,
        sumo_headless=sumo_headless,
        envision_record_data_replay_path=envision_record_data_replay_path,
    )
    env = _InfoScore(env=env)

    return env


def _get_env_specs(scenario: str):
    """Returns the appropriate environment parameters for each scenario.

    Args:
        scenario (str): Scenario

    Returns:
        Dict[str, Any]: A parameter dictionary.
    """

    if scenario == "1_to_2lane_left_turn_c":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[1]
                / "scenarios"
                / "intersection"
                / "1_to_2lane_left_turn_c"
            ),
            "num_agent": 1,
        }
    elif scenario == "1_to_2lane_left_turn_t":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[1]
                / "scenarios"
                / "intersection"
                / "1_to_2lane_left_turn_t"
            ),
            "num_agent": 1,
        }
    elif scenario == "3lane_merge_multi_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[1]
                / "scenarios"
                / "merge"
                / "3lane_multi_agent"
            ),
            "num_agent": 2,
        }
    elif scenario == "3lane_merge_single_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[1]
                / "scenarios"
                / "merge"
                / "3lane_single_agent"
            ),
            "num_agent": 1,
        }
    elif scenario == "3lane_cruise_multi_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[1]
                / "scenarios"
                / "straight"
                / "3lane_cruise_multi_agent"
            ),
            "num_agent": 3,
        }
    elif scenario == "3lane_cruise_single_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[1]
                / "scenarios"
                / "straight"
                / "3lane_cruise_single_agent"
            ),
            "num_agent": 1,
        }
    elif scenario == "3lane_cut_in":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[1]
                / "scenarios"
                / "straight"
                / "3lane_cut_in"
            ),
            "num_agent": 1,
        }
    elif scenario == "3lane_overtake":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[1]
                / "scenarios"
                / "straight"
                / "3lane_overtake"
            ),
            "num_agent": 1,
        }
    elif os.path.isdir(scenario):
        import re

        regexp_agent = re.compile(r"agents_\d+")
        regexp_num = re.compile(r"\d+")
        matches_agent = regexp_agent.search(scenario)
        if not matches_agent:
            raise Exception(
                f"Scenario path should match regexp of 'agents_\d+', but got {scenario}"
            )
        num_agent = regexp_num.search(matches_agent.group(0))

        return {
            "scenario": str(scenario),
            "num_agent": int(num_agent.group(0)),
        }
    else:
        raise Exception(f"Unknown scenario {scenario}.")


def resolve_agent_interface(
    img_meters: int = 64, img_pixels: int = 256, action_space="TargetPose", **kwargs
):
    """Resolve an agent interface for the environments in this module."""

    done_criteria = DoneCriteria(
        collision=True,
        off_road=True,
        off_route=False,
        on_shoulder=False,
        wrong_way=False,
        not_moving=False,
        agents_alive=None,
    )
    max_episode_steps = 800
    road_waypoint_horizon = 50
    waypoints_lookahead = 50
    return AgentInterface(
        accelerometer=True,
        action=ActionSpaceType[action_space],
        done_criteria=done_criteria,
        drivable_area_grid_map=DrivableAreaGridMap(
            width=img_pixels,
            height=img_pixels,
            resolution=img_meters / img_pixels,
        ),
        lidar=True,
        max_episode_steps=max_episode_steps,
        neighborhood_vehicles=True,
        ogm=OGM(
            width=img_pixels,
            height=img_pixels,
            resolution=img_meters / img_pixels,
        ),
        rgb=RGB(
            width=img_pixels,
            height=img_pixels,
            resolution=img_meters / img_pixels,
        ),
        road_waypoints=RoadWaypoints(horizon=road_waypoint_horizon),
        waypoints=Waypoints(lookahead=waypoints_lookahead),
    )


class _InfoScore(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(_InfoScore, self).__init__(env)

    def step(
        self, action: Dict[str, Any]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Steps the environment. A modified `score` is added to the returned
        `info` of each agent.

        Args:
            action (Dict[str, Any]): Action for each agent.

        Returns:
            Tuple[ Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, Any]] ]:
                Observation, reward, done, and info, for each agent is returned.
        """
        obs, reward, done, info = self.env.step(action)

        for agent_id in obs.keys():
            reached_goal = obs[agent_id].events.reached_goal
            # Set `score=1` if ego agent successfully completes mission, else `score=0`.
            info[agent_id]["score"] = reached_goal

        return obs, reward, done, info
