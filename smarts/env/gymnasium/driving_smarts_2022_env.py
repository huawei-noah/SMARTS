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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import copy
import logging
import math
from functools import partial
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np

from envision.client import Client as Envision
from envision.client import EnvisionDataFormatterArgs
from smarts.core.agent_interface import (
    AgentInterface,
    DoneCriteria,
    RoadWaypoints,
    Waypoints,
)
from smarts.core.controllers import ActionSpaceType
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1, SumoOptions
from smarts.env.utils.observation_conversion import ObservationOptions
from smarts.env.utils.scenario import get_scenario_specs
from smarts.sstudio.scenario_construction import build_scenario

logger = logging.getLogger(__file__)
logger.setLevel(logging.WARNING)

SUPPORTED_ACTION_TYPES = (
    ActionSpaceType.RelativeTargetPose,
    ActionSpaceType.TargetPose,
)
MAXIMUM_SPEED_MPS = 28  # 28m/s = 100.8 km/h. This is a safe maximum speed.


def driving_smarts_2022_env(
    scenario: str,
    agent_interface: AgentInterface,
    headless: bool = True,
    seed: int = 42,
    visdom: bool = False,
    sumo_headless: bool = True,
    envision_record_data_replay_path: Optional[str] = None,
):
    """An environment with a mission to be completed by a single or multiple ego agents.

    Observation space for each agent:
        An unformatted :class:`~smarts.core.observations.Observation` is returned as observation.

    Action space for each agent:
        Action space for each agent is configured through its `AgentInterface`.
        The action space could be either of the following.

        (i) :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.RelativeTargetPose`

           +------------------------------------+-------------+-------+
           | Action                             | Values      | Units |
           +====================================+=============+=======+
           | Δx-coordinate                      | [-2.8, 2.8] | m     |
           +------------------------------------+-------------+-------+
           | Δy-coordinate                      | [-2.8, 2.8] | m     |
           +------------------------------------+-------------+-------+
           | Δheading                           | [-π, π]     | rad   |
           +------------------------------------+-------------+-------+

        (ii) :attr:`~smarts.core.controllers.action_space_type.ActionSpaceType.TargetPose`

            +------------------------------------+---------------+-------+
            | Action                             | Values        | Units |
            +====================================+===============+=======+
            | Next x-coordinate                  | [-1e10, 1e10] | m     |
            +------------------------------------+---------------+-------+
            | Next y-coordinate                  | [-1e10, 1e10] | m     |
            +------------------------------------+---------------+-------+
            | Heading with respect to map's axes | [-π, π]       | rad   |
            +------------------------------------+---------------+-------+
            | ΔTime                              |  0.1          | s     |
            +------------------------------------+---------------+-------+

    Reward:
        Reward is distance travelled (in meters) in each step, including the termination step.

    Episode termination:
        Episode is terminated if any of the following occurs.

        1. Steps per episode exceed 800.
        2. Agent collides, drives off road, drives off route, or drives on wrong way.

    Args:
        scenario (str): Scenario name or path to scenario folder.
        agent_interface (AgentInterface): Agent interface specification.
        headless (bool, optional): If True, disables visualization in
            Envision. Defaults to False.
        seed (int, optional): Random number generator seed. Defaults to 42.
        visdom (bool, optional): If True, enables visualization of observed
            RGB images in `visdom`. Defaults to False.
        sumo_headless (bool, optional): If True, disables visualization in
            SUMO GUI. Defaults to True.
        envision_record_data_replay_path (Optional[str], optional):
            Envision's data replay output directory. Defaults to None.

    Returns:
        An environment described by the input argument `scenario`.
    """

    env_specs = get_scenario_specs(scenario)
    build_scenario(scenario=env_specs["scenario"])

    resolved_agent_interface = resolve_agent_interface(agent_interface)
    agent_interfaces = {
        f"Agent_{i}": resolved_agent_interface for i in range(env_specs["num_agent"])
    }
    env_action_space = resolve_env_action_space(agent_interfaces)

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
        fixed_timestep_sec=0.1,
        seed=seed,
        sumo_options=SumoOptions(headless=sumo_headless),
        visualization_client_builder=visualization_client_builder,
        observation_options=ObservationOptions.unformatted,
    )
    env.action_space = env_action_space
    if resolved_agent_interface.action == ActionSpaceType.TargetPose:
        env = _LimitTargetPose(env)
    return env


def resolve_agent_action_space(agent_interface: AgentInterface):
    """Get the competition action space for the given agent interface."""
    assert (
        agent_interface.action in SUPPORTED_ACTION_TYPES
    ), f"Unsupported action type `{agent_interface.action}` not in supported actions `{SUPPORTED_ACTION_TYPES}`"

    if agent_interface.action == ActionSpaceType.RelativeTargetPose:
        max_dist = MAXIMUM_SPEED_MPS * 0.1  # assumes 0.1 timestep
        return gym.spaces.Box(
            low=np.array([-max_dist, -max_dist, -np.pi]),
            high=np.array([max_dist, max_dist, np.pi]),
            dtype=np.float32,
        )
    if agent_interface.action == ActionSpaceType.TargetPose:
        return gym.spaces.Box(
            low=np.array([-1e10, -1e10, -np.pi, 0.1]),
            high=np.array([1e10, 1e10, np.pi, 0.1]),
            dtype=np.float32,
        )


def resolve_env_action_space(agent_interfaces: Dict[str, AgentInterface]):
    """Get the environment action space for the given set of agent interfaces."""
    return gym.spaces.Dict(
        {
            a_id: resolve_agent_action_space(a_inter)
            for a_id, a_inter in agent_interfaces.items()
        }
    )


def resolve_agent_interface(agent_interface: AgentInterface):
    """Resolve the agent interface for a given environment. Some interface
    values can be configured by the user, but others are pre-determined and
    fixed.
    """

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
        action=agent_interface.action,
        done_criteria=done_criteria,
        drivable_area_grid_map=agent_interface.drivable_area_grid_map,
        lidar_point_cloud=True,
        max_episode_steps=max_episode_steps,
        neighborhood_vehicle_states=True,
        occupancy_grid_map=agent_interface.occupancy_grid_map,
        top_down_rgb=agent_interface.top_down_rgb,
        road_waypoints=RoadWaypoints(horizon=road_waypoint_horizon),
        waypoint_paths=Waypoints(lookahead=waypoints_lookahead),
    )


class _LimitTargetPose(gym.Wrapper):
    """Uses previous observation to limit the next TargetPose action range."""

    def __init__(self, env: gym.Env):
        """
        Args:
            env (gym.Env): Environment to be wrapped.
        """
        super().__init__(env)
        self._prev_obs: Dict[str, Dict[str, Any]]

    def step(
        self, action: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, Any],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Steps the environment.

        Args:
            action (Dict[str, Any]): Action for each agent.

        Returns:
            Tuple[ Dict[str, Any], Dict[str, float], Dict[str, bool], Dict[str, Dict[str, Any]] ]:
                Observation, reward, done, and info, for each agent is returned.
        """

        limited_actions: Dict[str, np.ndarray] = {}
        for agent_name, agent_action in action.items():
            if not agent_name in self._prev_obs:
                continue
            limited_actions[agent_name] = self._limit(
                name=agent_name,
                action=agent_action,
                prev_coord=self._prev_obs[agent_name]["position"],
            )

        out = self.env.step(limited_actions)
        self._prev_obs = self._store(obs=out[0])
        return out

    def reset(self, **kwargs):
        """Resets the environment.

        Returns:
            observation (dict): Dictionary of initial-state observation for
                each agent.
            info (dict): This dictionary contains auxiliary information
                complementing ``observation``. It is analogous to the ``info``
                returned by :meth:`step`.
        """
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = self._store(obs=obs)
        return obs, info

    def _store(self, obs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        filtered_obs: Dict[str, Dict[str, Any]] = {}
        for agent_name, agent_obs in obs.items():
            filtered_obs[agent_name] = {
                "position": copy.deepcopy(agent_obs.ego_vehicle_state.position[:2])
            }
        return filtered_obs

    def _limit(
        self, name: str, action: np.ndarray, prev_coord: np.ndarray
    ) -> np.ndarray:
        """Set time delta and limit Euclidean distance travelled in TargetPose action space.

        Args:
            name (str): Agent's name.
            action (np.ndarray): Agent's action.
            prev_coord (np.ndarray): Agent's previous xy coordinate on the map.

        Returns:
            np.ndarray: Agent's TargetPose action which has fixed time-delta and constrained next xy coordinate.
        """

        time_delta = 0.1
        limited_action = np.array(
            [action[0], action[1], action[2], time_delta], dtype=np.float32
        )
        speed_max = MAXIMUM_SPEED_MPS
        dist_max = speed_max * time_delta

        # Set time-delta
        if not math.isclose(action[3], time_delta, abs_tol=1e-3):
            logger.warning(
                "%s: Expected time-delta=%s, but got time-delta=%s. "
                "Action time-delta automatically changed to %s.",
                name,
                time_delta,
                action[3],
                time_delta,
            )

        # Limit Euclidean distance travelled
        next_coord = action[:2]
        vector = next_coord - prev_coord
        dist = np.linalg.norm(vector)
        if dist > dist_max:
            unit_vector = vector / dist
            limited_action[0], limited_action[1] = prev_coord + dist_max * unit_vector
            logger.warning(
                "Action out of bounds. `%s`: Allowed max speed=%sm/s, but got speed=%sm/s. "
                "Action has be corrected from %s to %s.",
                name,
                speed_max,
                dist / time_delta,
                action,
                limited_action,
            )

        return limited_action
