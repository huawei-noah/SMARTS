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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import copy
import functools
from dataclasses import dataclass, fields
from typing import Any, Dict, NamedTuple, Set, TypeVar

import gymnasium as gym
import numpy as np

from smarts.core.agent_interface import AgentInterface
from smarts.core.coordinates import Point
from smarts.core.observations import Observation
from smarts.core.plan import PositionalGoal
from smarts.core.road_map import RoadMap
from smarts.core.scenario import Scenario
from smarts.env.gymnasium.wrappers.metric.completion import (
    Completion,
    CompletionFuncs,
    get_dist,
)
from smarts.env.gymnasium.wrappers.metric.costs import CostFuncs, Costs
from smarts.env.gymnasium.wrappers.metric.counts import Counts


@dataclass
class Record:
    """Stores an agent's scenario-completion, performance-count, and
    performance-cost values."""

    completion: Completion
    costs: Costs
    counts: Counts


@dataclass
class Data:
    """Stores an agent's performance-record, completion-functions, and
    cost-functions."""

    record: Record
    completion_funcs: CompletionFuncs
    cost_funcs: CostFuncs


class Score(NamedTuple):
    """This describes the final score given by processing observations through the metrics."""

    completion: float
    humanness: float
    rules: float
    time: float
    overall: float


class MetricsError(Exception):
    """Raised when Metrics env wrapper fails."""

    pass


class MetricsBase(gym.Wrapper):
    """Computes agents' performance metrics in a SMARTS environment."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        _check_env(env)
        self._scen: Scenario
        self._scen_name: str
        self._road_map: RoadMap
        self._cur_agents: Set[str]
        self._steps: Dict[str, int]
        self._done_agents: Set[str]
        self._records = {}

    def step(self, action: Dict[str, Any]):
        """Steps the environment by one step."""
        result = super().step(action)

        obs, _, terminated, truncated, info = result

        # Only count steps in which an ego agent is present.
        if len(obs) == 0:
            return result

        dones = {"__all__": False}
        if isinstance(terminated, dict):
            dones = {k: v or truncated[k] for k, v in terminated.items()}
        elif isinstance(terminated, bool):
            if terminated or truncated:
                dones["__all__"] = True
            dones.update({a: d["done"] for a, d in info.items()})

        obs = {agent_id: o for agent_id, o in obs.items() if o["active"]}
        # fmt: off
        for agent_name in obs:
            base_obs: Observation = info[agent_name]["env_obs"]
            self._steps[agent_name] += 1

            # Compute all cost functions.
            costs = Costs()
            for field in fields(self._records[self._scen_name][agent_name].cost_funcs):
                cost_func = getattr(self._records[self._scen_name][agent_name].cost_funcs, field.name)
                new_costs = cost_func(road_map=self._road_map, obs=base_obs)
                costs = _add_dataclass(new_costs, costs)

            # Update stored costs.
            self._records[self._scen_name][agent_name].record.costs = costs

            if dones[agent_name]:
                self._done_agents.add(agent_name)
                if not (
                    base_obs.events.reached_goal
                    or len(base_obs.events.collisions)
                    or base_obs.events.off_road
                    or base_obs.events.reached_max_episode_steps
                ):
                    raise MetricsError(
                        "Expected reached_goal, collisions, off_road, or " 
                        "max_episode_steps to be true on agent done, but got "
                        f"events: {base_obs.events}."
                    )

                # Update stored counts.
                counts = Counts(
                    episodes=1,
                    steps=self._steps[agent_name],
                    steps_adjusted=min(
                        self._steps[agent_name],
                        self.env.agent_interfaces[agent_name].max_episode_steps
                    ),
                    goals=base_obs.events.reached_goal,
                    max_steps=self.env.agent_interfaces[agent_name].max_episode_steps
                )
                self._records[self._scen_name][agent_name].record.counts = _add_dataclass(
                    counts, 
                    self._records[self._scen_name][agent_name].record.counts
                )

                # Update percentage of scenario tasks completed.
                completion = Completion(dist_tot=self._records[self._scen_name][agent_name].record.completion.dist_tot)
                for field in fields(self._records[self._scen_name][agent_name].completion_funcs):
                    completion_func = getattr(
                        self._records[self._scen_name][agent_name].completion_funcs,
                        field.name,
                    )
                    new_completion = completion_func(
                        road_map=self._road_map,
                        obs=base_obs,
                        initial_compl=completion,
                    )
                    completion = _add_dataclass(new_completion, completion)
                self._records[self._scen_name][agent_name].record.completion = completion

        # fmt: on
        if dones["__all__"] == True:
            assert (
                self._done_agents == self._cur_agents
            ), f'done["__all__"]==True but not all agents are done. Current agents = {self._cur_agents}. Agents done = {self._done_agents}.'

        return result

    def reset(self, **kwargs):
        """Resets the environment."""
        result = super().reset(**kwargs)
        self._cur_agents = set(self.env.agent_interfaces.keys())
        self._steps = dict.fromkeys(self._cur_agents, 0)
        self._done_agents = set()
        self._scen = self.env.scenario
        self._scen_name = self.env.scenario.name
        self._road_map = self.env.scenario.road_map

        # fmt: off
        if self._scen_name not in self._records:
            _check_scen(self._scen)
            self._records[self._scen_name] = {
                agent_name: Data(
                    record=Record(
                        completion=Completion(
                            dist_tot=get_dist(
                                road_map=self._road_map,
                                point_a=Point(*self._scen.missions[agent_name].start.position),
                                point_b=self._scen.missions[agent_name].goal.position,    
                            )
                        ),
                        costs=Costs(),
                        counts=Counts(),
                    ),
                    cost_funcs=CostFuncs(),
                    completion_funcs=CompletionFuncs(),
                )
                for agent_name in self._cur_agents
            }
        # fmt: on

        return result

    def records(self) -> Dict[str, Dict[str, Record]]:
        """
        Fine grained performance metric for each agent in each scenario.

        .. code-block:: bash

            $ env.records()
            $ {
                  scen1: {
                      agent1: Record(completion, costs, counts),
                      agent2: Record(completion, costs, counts),
                  },
                  scen2: {
                      agent1: Record(completion, costs, counts),
                  },
              }

        Returns:
            Dict[str, Dict[str, Record]]: Performance record in a nested
            dictionary for each agent in each scenario.
        """

        records = {}
        for scen, agents in self._records.items():
            records[scen] = {}
            for agent, data in agents.items():
                records[scen][agent] = copy.deepcopy(data.record)

        return records

    def score(self) -> Score:
        """
        Computes four sub-component scores, namely, "Completion", "Time",
        "Humanness", "Rules", and one total combined score named "Overall"
        on the wrapped environment.

        +-------------+--------+-----------------------------------------------------------------------------------------------------+
        |             | Range  | Remarks                                                                                             |
        +=============+========+=====================================================================================================+
        | Overall     | [0, 1] | Total score which combines "Completion", "Time", "Humanness", and "Rules". The higher, the better.  |
        +-------------+--------+-----------------------------------------------------------------------------------------------------+
        | Completion  | [0, 1] | Proportion of scenarios tasks completed. The higher, the better.                                    |
        +-------------+--------+-----------------------------------------------------------------------------------------------------+
        | Time        | [0, 1] | Time taken to complete scenario. The lower, the better.                                             |
        +-------------+--------+-----------------------------------------------------------------------------------------------------+
        | Humanness   | [0, 1] | Humanness indicator. The higher, the better.                                                        |
        +-------------+--------+-----------------------------------------------------------------------------------------------------+
        | Rules       | [0, 1] | Traffic rules compliance. The higher, the better.                                                   |
        +-------------+--------+-----------------------------------------------------------------------------------------------------+

        Returns:
            Dict[str, float]: Contains "Overall", "Completion", "Time",
            "Humanness", and "Rules" scores.
        """

        counts_list, costs_list, completion_list = zip(
            *[
                (data.record.counts, data.record.costs, data.record.completion)
                for agents in self._records.values()
                for data in agents.values()
            ]
        )
        agents_tot: int = len(counts_list)  # Total number of agents over all scenarios
        counts_tot: Counts = functools.reduce(
            lambda a, b: _add_dataclass(a, b), counts_list
        )
        costs_tot: Costs = functools.reduce(
            lambda a, b: _add_dataclass(a, b), costs_list
        )
        completion_tot: Completion = functools.reduce(
            lambda a, b: _add_dataclass(a, b), completion_list
        )

        completion = _completion(completion=completion_tot)
        humanness = _humanness(costs=costs_tot, agents_tot=agents_tot)
        rules = _rules(costs=costs_tot, agents_tot=agents_tot)
        time = _time(counts=counts_tot)
        overall = completion * (1 - time) * (1 - humanness) * (1 - rules)

        return Score(
            completion=completion,
            humanness=humanness,
            rules=rules,
            time=time,
            overall=overall,
        )


class Metrics(gym.Wrapper):
    """Metrics class wraps an underlying _Metrics class. The underlying
    _Metrics class computes agents' performance metrics in a SMARTS
    environment. Whereas, this Metrics class is a basic gym.Wrapper class
    which prevents external users from accessing or modifying attributes
    beginning with an underscore, to ensure security of the metrics computed.

    Args:
        env (gym.Env): A gym.Env to be wrapped.

    Raises:
        AttributeError: Upon accessing an attribute beginning with an underscore.

    Returns:
        gym.Env: A wrapped gym.Env which computes agents' performance metrics.
    """

    def __init__(self, env: gym.Env):
        env = MetricsBase(env)
        super().__init__(env)


def _check_env(env: gym.Env):
    """Checks environment suitability to compute performance metrics.
    Args:
        env (gym.Env): A gym environment
    Raises:
        AttributeError: If any required agent interface is disabled.
    """

    def check_intrfc(agent_intrfc: AgentInterface):
        intrfc = {
            "accelerometer": bool(agent_intrfc.accelerometer),
            "max_episode_steps": bool(agent_intrfc.max_episode_steps),
            "neighborhood_vehicle_states": bool(agent_intrfc.neighborhood_vehicles),
            "road_waypoints": bool(agent_intrfc.road_waypoints),
            "waypoint_paths": bool(agent_intrfc.waypoints),
            "done_criteria.collision": agent_intrfc.done_criteria.collision,
            "done_criteria.off_road": agent_intrfc.done_criteria.off_road,
        }
        return intrfc

    for agent_name, agent_interface in env.agent_interfaces.items():
        intrfc = check_intrfc(agent_interface)
        if not all(intrfc.values()):
            raise AttributeError(
                (
                    "Enable {0}'s disabled interface to "
                    "compute its metrics. Current interface is "
                    "{1}."
                ).format(agent_name, intrfc)
            )


def _check_scen(scen: Scenario):
    """Checks scenario suitability to compute performance metrics.

    Args:
        scen (Scenario): A ``smarts.core.scenario.Scenario`` class.

    Raises:
        AttributeError: If any agent's mission is not of type PositionGoal.
    """
    goal_types = {
        agent_name: type(agent_mission.goal)
        for agent_name, agent_mission in scen.missions.items()
    }
    if not all([goal_type == PositionalGoal for goal_type in goal_types.values()]):
        raise AttributeError(
            "Expected all agents to have PositionalGoal, but agents have goal type "
            "{0}".format(goal_types)
        )


T = TypeVar("T", Completion, Costs, Counts)


def _add_dataclass(first: T, second: T) -> T:
    assert type(first) is type(second)
    new = {}
    for field in fields(first):
        sum = getattr(first, field.name) + getattr(second, field.name)
        new[field.name] = sum
    output = first.__class__(**new)

    return output


def _completion(completion: Completion) -> float:
    """
    Proportion of scenarios tasks completed.

    Args:
        completion (Completion): Scenario tasks completed.

    Returns:
        float: Normalized completion value = [0, 1]. Completion value should be
            maximized. The higher the value, the better it is.
    """
    return (completion.dist_tot - completion.dist_remainder) / completion.dist_tot


def _humanness(costs: Costs, agents_tot: int) -> float:
    """
    Humanness indicator.

    Args:
        costs (Costs): Performance cost values.
        agents_tot (int): Number of agents simulated.

    Returns:
        float: Normalized humanness value = [0, 1]. Humanness value should be
            maximized. The higher the value, the better it is.
    """
    humanness_to_minimize = np.array(
        [costs.dist_to_obstacles, costs.jerk_linear, costs.lane_center_offset]
    )
    humanness_to_minimize = np.mean(humanness_to_minimize, dtype=float) / agents_tot
    return 1 - humanness_to_minimize


def _rules(costs: Costs, agents_tot: int) -> float:
    """
    Traffic rules compliance.

    Args:
        costs (Costs): Performance cost values.
        agents_tot (int): Number of agents simulated.

    Returns:
        float: Normalized rules value = [0, 1]. Rules value should be maximized.
            The higher the value, the better it is.
    """
    rules_to_minimize = np.array([costs.speed_limit, costs.wrong_way])
    rules_to_minimize = np.mean(rules_to_minimize, dtype=float) / agents_tot
    return 1 - rules_to_minimize


def _time(counts: Counts) -> float:
    """
    Time taken to complete scenario.

    Args:
        counts (Counts): Performance count values.

    Returns:
        float: Normalized time value = (0, 1]. Time value should be minimized.
            The lower the value, the better it is.
    """
    return counts.steps_adjusted / counts.max_steps
