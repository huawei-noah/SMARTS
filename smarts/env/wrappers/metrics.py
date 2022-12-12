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
from typing import Any, Dict, Set, TypeVar

import gym

from smarts.core.agent_interface import AgentInterface
from smarts.core.coordinates import Point
from smarts.core.plan import PositionalGoal
from smarts.core.road_map import RoadMap
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.env.wrappers.metric.completion import Completion, CompletionFuncs, get_dist
from smarts.env.wrappers.metric.costs import CostFuncs, Costs
from smarts.env.wrappers.metric.counts import Counts


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

class MetricsError(Exception):
    """Raised when Metrics env wrapper fails."""

    pass

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
        env = _Metrics(env)
        super().__init__(env)


class _Metrics(gym.Wrapper):
    """Computes agents' performance metrics in a SMARTS environment."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        _check_env(env)
        self._scen: Scenario
        self._cur_scen: str
        self._road_map: RoadMap 
        self._cur_agents: Set[str]
        self._steps: Dict[str, int]
        self._done_agents: Set[str]
        self._records = {}

    def step(self, action: Dict[str, Any]):
        """Steps the environment by one step."""
        obs, rewards, dones, info = super().step(action)

        # Only count steps in which an ego agent is present.
        if len(obs) == 0:
            return obs, rewards, dones, info

        for agent_name, agent_obs in obs.items():
            self._steps[agent_name] += 1

            # Compute all cost functions.
            # fmt: off
            costs = Costs()
            for field in fields(self._records[self._cur_scen][agent_name].cost_funcs):
                cost_func = getattr(self._records[self._cur_scen][agent_name].cost_funcs, field.name)
                new_costs = cost_func(agent_obs)
                costs = _add_dataclass(new_costs, costs)
            # fmt: on

            # Update stored costs.
            self._records[self._cur_scen][agent_name].record.costs = costs

            if dones[agent_name]:
                self._done_agents.add(agent_name)
                steps_adjusted, goals = 0, 0
                if agent_obs.events.reached_goal: 
                    steps_adjusted = self._steps[agent_name]
                    goals = 1
                elif (len(agent_obs.events.collisions) > 0
                    or agent_obs.events.off_road
                    or agent_obs.events.reached_max_episode_steps
                ):                
                    steps_adjusted = self.env.agent_specs[agent_name].interface.max_episode_steps
                else:
                    raise MetricsError(
                        f"Unsupported agent done reason. Events: {agent_obs.events}."
                    )

                # Update stored counts.
                # fmt: off
                counts = Counts(
                    episodes=1,
                    steps=self._steps[agent_name],
                    steps_adjusted=steps_adjusted,
                    goals=goals,
                    max_steps=self.env.agent_specs[agent_name].interface.max_episode_steps
                )
                self._records[self._cur_scen][agent_name].record.counts = _add_dataclass(
                    counts, 
                    self._records[self._cur_scen][agent_name].record.counts
                )

                # Update percentage of scenario tasks completed.
                completion = Completion(dist_tot=self._records[self._cur_scen][agent_name].record.completion.dist_tot)
                for field in fields(self._records[self._cur_scen][agent_name].completion_funcs):
                    completion_func = getattr(
                        self._records[self._cur_scen][agent_name].completion_funcs,
                        field.name,
                    )
                    new_completion = completion_func(
                        road_map=self._road_map,
                        obs=agent_obs,
                        initial_compl=completion,
                    )
                    completion = _add_dataclass(new_completion, completion)
                self._records[self._cur_scen][agent_name].record.completion = completion
                # fmt: on

                print(f"{agent_name}: {completion}")
                # s = input("Press Enter to continue...")

        if dones["__all__"] == True:
            assert (
                self._done_agents == self._cur_agents
            ), f'done["__all__"]==True but not all agents are done. Current agents = {self._cur_agents}. Agents done = {self._done_agents}.'

        return obs, rewards, dones, info

    def reset(self, **kwargs):
        """Resets the environment."""
        obs = super().reset(**kwargs)
        self._cur_agents = set(self.env.agent_specs.keys())
        self._steps = dict.fromkeys(self._cur_agents, 0)
        self._done_agents = set()
        self._scen = self.env.smarts.scenario
        self._cur_scen = self.env.smarts.scenario.name
        self._road_map = self.env.smarts.scenario.road_map


        # fmt: off
        if self._cur_scen not in self._records:
            _check_scen(self._scen)
            self._records[self._cur_scen] = {
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

        # def offset_along_lane(self, world_point: Point) -> float:
        # def from_lane_coord(self, lane_point: RefLinePoint) -> Point:
        # def to_lane_coord(self, world_point: Point) -> RefLinePoint:
        # def road_with_point(self, point: Point) -> RoadMap.Road:

        return obs

    def records(self) -> Dict[str, Dict[str, Record]]:
        """
        Fine grained performance metric for each agent in each scenario.

        Returns:
            Dict[str, Dict[str, Record]]: Performance record in a nested
                dictionary for each agent in each scenario.

        Example::

        >> records()
        >> {
                scen1: {
                    agent1: Record(completion, costs, counts),
                    agent2: Record(completion, costs, counts),
                },
                scen2: {
                    agent1: Record(completion, costs, counts),
                },
            }
        """

        records = {}
        for scen, agents in self._records.items():
            records[scen] = {}
            for agent, data in agents.items():
                records[scen][agent] = copy.deepcopy(data.record)

        return records

    def score(self) -> Dict[str, float]:
        """
        An overall performance score achieved on the wrapped environment.
        """
        # fmt: off
        counts_list, costs_list, completion_list = zip(
            *[
                (data.record.counts, data.record.costs, data.record.completion)
                for agents in self._records.values()
                for data in agents.values()
            ]
        )
        agents_tot: int = len(counts_list)  # Total number of agents over all scenarios
        counts_tot: Counts = functools.reduce(lambda a, b: _add_dataclass(a, b), counts_list)
        costs_tot: Costs = functools.reduce(lambda a, b: _add_dataclass(a, b), costs_list)
        completion_tot: Completion = functools.reduce(lambda a, b: _add_dataclass(a, b), completion_list)

        _score: Dict[str, float] = {}
        _score["completion"] = _completion(completion=completion_tot, agents_tot=agents_tot)
        _score["humanness"] = _humanness(costs=costs_tot, agents_tot=agents_tot)
        _score["rules"] = _rules(costs=costs_tot, agents_tot=agents_tot)
        _score["time"] = _time(counts=counts_tot)
        # fmt: on

        return _score


def _check_env(env):
    """Checks environment suitability to compute performance metrics.

    Args:
        env (_type_): A gym environment

    Raises:
        AttributeError: If any required agent interface is disabled.
    """

    def check_intrfc(agent_intrfc: AgentInterface):
        intrfc = {
            "accelerometer": bool(agent_intrfc.accelerometer),
            "max_episode_steps": bool(agent_intrfc.max_episode_steps),
            "neighborhood_vehicles": bool(agent_intrfc.neighborhood_vehicles),
            "road_waypoints": bool(agent_intrfc.road_waypoints),
            "waypoints": bool(agent_intrfc.waypoints),
            "done_criteria.collision": agent_intrfc.done_criteria.collision,
            "done_criteria.off_road": agent_intrfc.done_criteria.off_road,
        }
        return intrfc

    for agent_name, agent_spec in env.agent_specs.items():
        intrfc = check_intrfc(agent_spec.interface)
        if not all(intrfc.values()):
            raise AttributeError(
                "Enable {0}'s disabled interface to "
                "compute its metrics. Current interface is "
                "{1}.".format(agent_name, intrfc)
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
    new = {}
    for field in fields(first):
        sum = getattr(first, field.name) + getattr(second, field.name)
        new[field.name] = sum
    output = first.__class__(**new)

    return output


def _completion(completion: Completion, agents_tot: int) -> float:
    """
    Percentage of scenarios tasks completed, averaged over all scenarios.

    Args:
        completion (Completion): Proportion of scenario tasks completed, summed
            over all scenarios.
        scen_num (int): Number of scenarios.

    Returns:
        float: Average completion over all scenarios.
    """
    return (completion.dist_tot - completion.dist_remainder)  / agents_tot


def _humanness(costs: Costs, agents_tot: int) -> float:
    return (
        costs.dist_to_obstacles + costs.jerk_linear + costs.lane_center_offset
    ) / agents_tot


def _rules(costs: Costs, agents_tot: int) -> float:
    return (costs.speed_limit + costs.wrong_way) / agents_tot


def _time(counts: Counts) -> float:
    return counts.steps_adjusted / counts.max_steps
