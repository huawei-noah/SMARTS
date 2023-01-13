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

import logging
import os
import pathlib
import typing
from functools import partial

import gymnasium as gym
import numpy as np

from envision.client import Client as Envision
from envision.client import EnvisionDataFormatterArgs
from smarts import sstudio
from smarts.core.agent_interface import (
    AgentInterface,
)
from smarts.core.controllers import ActionSpaceType
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1, SumoOptions
from smarts.env.multi_scenario_env import resolve_agent_interface

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)

SUPPORTED_ACTION_TYPES = (ActionSpaceType.RelativeTargetPose,)


def driving_smarts_competition_v0_env(
    scenario: str,
    img_meters: int = 64,
    img_pixels: int = 256,
    action_space="RelativeTargetPose",
    headless: bool = True,
    seed: int = 42,
    visdom: bool = False,
    sumo_headless: bool = True,
    envision_record_data_replay_path: typing.Optional[str] = None,
):
    """An environment with a mission to be completed by a single or multiple ego agents.

    Observation space for each agent:

        A ``smarts.core.sensors.Observation`` is returned as observation.

    Action space for each agent:
    .. note::

        A ``smarts.core.controllers.ActionSpaceType.RelativeTargetPose``, which is a
        sequence of [x-coordinate, y-coordinate, heading].

        Type:

        .. code-block:: python

            gym.spaces.Box(
                    low=np.array([-28, -28, -π]),
                    high=np.array([28, 28, π]),
                    dtype=np.float32
                    )

        .. list-table:: Table
            :widths: 25 25
            :header-rows: 1

            * - Action
              - Value range
            * - Ego's next x-coordinate on the map
              - [-28,28]
            * - Ego's next y-coordinate on the map
              - [-28,28]
            * - Ego's next heading with respect to the map's axes
              - [-π,π]

    Reward:

        Reward is distance travelled (in meters) in each step, including the termination step.

    Episode termination:

    .. note::

        Episode is terminated if any of the following occurs.

            1. Steps per episode exceed 800.

            2. Agent collides, drives off road, drives off route, or drives on wrong way.

    Solved requirement:

    .. note::

        If agent successfully completes the mission then ``info["score"]`` will
        equal 1, else it is 0. Considered solved when ``info["score"] == 1`` is
        achieved over 500 consecutive episodes.

    :param scenario: Scenario name or path to scenario folder.
    :type scenario: str
    :param img_meters: Ground square size covered by image observations. Defaults to 64 x 64 meter (height x width) square.
    :type img_meters: int
    :param img_pixels: Pixels representing the square image observations. Defaults to 256 x 256 pixels (height x width) square.
    :type img_pixels: int
    :param action_space: Action space used. Defaults to ``Continuous``.
    :param headless: If True, disables visualization in Envision. Defaults to False.
    :type headless: bool, optional
    :param visdom: If True, enables visualization of observed RGB images in Visdom. Defaults to False.
    :type visdom: bool, optional
    :param sumo_headless: If True, disables visualization in SUMO GUI. Defaults to True.
    :type sumo_headless: bool, optional
    :param envision_record_data_replay_path: Envision's data replay output directory. Defaults to None.
    :type envision_record_data_replay_path: Optional[str], optional
    :return: An environment described by the input argument ``scenario``.
    """

    env_specs = _get_env_specs(scenario)
    sstudio.build_scenario(scenario=[env_specs["scenario"]])

    agent_interfaces = {
        f"Agent_{i}": resolve_agent_interface(img_meters, img_pixels, action_space)
        for i in range(env_specs["num_agent"])
    }
    action_space = resolve_env_action_space(agent_interfaces)

    visualization_client_builder = None
    if not headless:
        visualization_client_builder = partial(
            Envision,
            endpoint=None,
            output_dir=envision_record_data_replay_path,
            headless=headless,
            data_formatter_args=EnvisionDataFormatterArgs(
                "base", enable_reduction=False
            ),
        )

    env = HiWayEnvV1(
        scenarios=[env_specs["scenario"]],
        agent_interfaces=agent_interfaces,
        sim_name="Driving_SMARTS_v0",
        headless=headless,
        visdom=visdom,
        sumo_options=SumoOptions(headless=sumo_headless),
        visualization_client_builder=visualization_client_builder,
    )
    env.action_space = action_space
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
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "intersection"
                / "1_to_2lane_left_turn_c"
            ),
            "num_agent": 1,
        }
    elif scenario == "1_to_2lane_left_turn_t":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "intersection"
                / "1_to_2lane_left_turn_t"
            ),
            "num_agent": 1,
        }
    elif scenario == "3lane_merge_multi_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "merge"
                / "3lane_multi_agent"
            ),
            "num_agent": 2,
        }
    elif scenario == "3lane_merge_single_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "merge"
                / "3lane_single_agent"
            ),
            "num_agent": 1,
        }
    elif scenario == "3lane_cruise_multi_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "straight"
                / "3lane_cruise_multi_agent"
            ),
            "num_agent": 3,
        }
    elif scenario == "3lane_cruise_single_agent":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "straight"
                / "3lane_cruise_single_agent"
            ),
            "num_agent": 1,
        }
    elif scenario == "3lane_cut_in":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
                / "scenarios"
                / "straight"
                / "3lane_cut_in"
            ),
            "num_agent": 1,
        }
    elif scenario == "3lane_overtake":
        return {
            "scenario": str(
                pathlib.Path(__file__).absolute().parents[2]
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


def resolve_agent_action_space(agent_interface: AgentInterface):
    if agent_interface.action == ActionSpaceType.RelativeTargetPose:
        return gym.spaces.Box(
            low=np.array([-28, -28, -np.pi]),
            high=np.array([28, 28, np.pi]),
            dtype=np.float32,
        )
    if agent_interface.action == ActionSpaceType.TargetPose:
        return gym.spaces.Box(
            low=np.array([-1e10, -1e10, -np.pi, 0.1]),
            high=np.array([1e10, 1e10, np.pi, 0.1]),
            dtype=np.float32,
        )

    assert (
        agent_interface.action in SUPPORTED_ACTION_TYPES
    ), f"Unsupported action type `{agent_interface.action}` not in supported actions `{SUPPORTED_ACTION_TYPES}`"


def resolve_env_action_space(agent_interfaces: typing.Dict[str, AgentInterface]):
    return gym.spaces.Dict(
        {
            a_id: resolve_agent_action_space(a_inter)
            for a_id, a_inter in agent_interfaces.items()
        }
    )
