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

import pathlib
from typing import Any, Dict, Optional, Tuple

import gym

from smarts.core.agent_interface import (
    OGM,
    RGB,
    AgentInterface,
    DoneCriteria,
    DrivableAreaGridMap,
    NeighborhoodVehicles,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType
from smarts.env.hiway_env import HiWayEnv
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec


def multi_scenario_v0_env(
    headless: bool = True,
    visdom: bool = False,
    sumo_headless: bool = True,
    envision_record_data_replay_path: Optional[str] = None,
    img_meters: int = 64,
    img_pixels: int = 256,
    action_space="Continuous",
):
    """A merge environment where a single agent needs to merge into a freeway
    by driving along an entrance ramp, an acceleration lane, enter into the
    freeway, and finally change lanes to the rightmost lane.

    Observation:
        A `smarts.env.wrappers.format_obs:StdObs` dict, containing enabled keys,
        is returned as observation.

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
        Episode is terminated if any of the following occurs.
        + Steps per episode exceed 3000.
        + Agent collides, drives off road, drives off route, or drives on wrong way.

    Solved requirement:
        If agent successfully completes the mission then `info["score"]` will
        equal 1, else it is 0. Considered solved when `info["score"] == 1` is
        achieved over 1000 consecutive episodes.

    Args:
        headless (bool, optional): If True, disables visualization in
            Envision. Defaults to False.
        visdom (bool, optional): If True, enables visualization of observed
            RGB images in Visdom. Defaults to False.
        sumo_headless (bool, optional): If True, disables visualization in
            SUMO GUI. Defaults to True.
        envision_record_data_replay_path (Optional[str], optional):
            Envision's data replay output directory. Defaults to None.
        img_meters (int): Ground square size covered by image observations.
            Defaults to 64 x 64 meter (height x width) square.
        img_pixels (int): Pixels representing the square image observations.
            Defaults to 256 x 256 pixels (height x width) square.

    Returns:
        A single-agent merging into a freeway.
    """

    scenario = [
        str(
            pathlib.Path(__file__).absolute().parents[1]
            / "scenarios"
            / "intersection"
            / "1_to_2lane_left_turn_c"
        ),
        str(
            pathlib.Path(__file__).absolute().parents[1]
            / "scenarios"
            / "intersection"
            / "1_to_2lane_left_turn_t"
        ),
        str(
            pathlib.Path(__file__).absolute().parents[1]
            / "scenarios"
            / "merge"
            / "3lane_multi_agent"
        ),
        str(
            pathlib.Path(__file__).absolute().parents[1]
            / "scenarios"
            / "merge"
            / "3lane_single_agent"
        ),
        str(
            pathlib.Path(__file__).absolute().parents[1]
            / "scenarios"
            / "straight"
            / "3lane_cruise_multi_agent"
        ),
        str(
            pathlib.Path(__file__).absolute().parents[1]
            / "scenarios"
            / "straight"
            / "3lane_cruise_single_agent"
        ),
        str(
            pathlib.Path(__file__).absolute().parents[1]
            / "scenarios"
            / "straight"
            / "3lane_cut_off"
        ),
        str(
            pathlib.Path(__file__).absolute().parents[1]
            / "scenarios"
            / "straight"
            / "3lane_overtake"
        ),
    ]
    build_scenario(scenario)

    done_criteria = DoneCriteria(
        collision=True,
        off_road=True,
        off_route=True,
        on_shoulder=False,
        wrong_way=True,
        not_moving=False,
        agents_alive=None,
    )
    max_episode_steps = 3000
    agent_specs = {
        "MergeAgent": AgentSpec(
            interface=AgentInterface(
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
                neighborhood_vehicles=NeighborhoodVehicles(img_meters),
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
                road_waypoints=False,
                waypoints=Waypoints(lookahead=img_meters),
            ),
        )
    }

    env = HiWayEnv(
        scenarios=scenario,
        agent_specs=agent_specs,
        sim_name="Merge",
        headless=headless,
        visdom=visdom,
        sumo_headless=sumo_headless,
        # To avoid traffic jams, exiting vehicles are not reintroduced.
        endless_traffic=False,
        envision_record_data_replay_path=envision_record_data_replay_path,
    )
    # env = FormatObs(env=env)
    # env = FormatAction(env=env, space=ActionSpaceType[action_space])
    env = _InfoScore(env=env)
    # env = SingleAgent(env=env)

    return env


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
            # Set `score=1` if ego agent successfully merges into the freeway,
            # changes lane, and reaches the end of mission route, else `score=0`.
            info[agent_id]["score"] = reached_goal

        return obs, reward, done, info
