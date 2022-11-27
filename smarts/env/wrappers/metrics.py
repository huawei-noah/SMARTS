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
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Set, TypeVar

import gym

from smarts.env.wrappers.metric import termination
from smarts.env.wrappers.metric.costs import COST_FUNCS, Costs
from smarts.env.wrappers.metric.counts import Counts

_MAX_STEPS = 800


@dataclass
class Record:
    """A dataclass for an agent, storing its performance counts and costs.
    """
    counts: Counts
    costs: Costs
    cost_funcs: Dict[str, Callable[[Any], Dict[str, float]]]


class Metrics(gym.Wrapper):
    """Computes agents' performance metrics in a SMARTS environment."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._cur_scen: str
        self._cur_agents: Set[str]
        self._steps: Dict[str, int]
        self._done_check: Set[str]
        self._records = {}

    def __getattribute__(self, item):
        """For security, prevents access to items beginning with an underscore.

        Args:
            item (_type_): Requested item.

        Raises:
            AttributeError: Upon accessing item beginning with an underscore.

        Returns:
            _type_: Returns requested item.
        """

        if item.startswith("_"):
            raise AttributeError(
                "Permission denied to access private attribute '{}'".format(item)
            )
        return super().__getattribute__(item)

    def step(self, action: Dict[str, Any]):
        """Steps the environment by one step."""
        obs, rewards, dones, info = super().step(action)

        # Only count steps in which an ego agent is present.
        if len(obs) == 0:
            return obs, rewards, dones, info

        for agent_name, agent_obs in obs.items():
            # Increment step count
            self._steps[agent_name] += 1

            # fmt: off
            # Compute all cost functions.
            for cost_name, cost_func in self._records[self._cur_scen][agent_name].cost_funcs.items():
                new_val = cost_func(agent_obs)
                setattr(self._records[self._cur_scen][agent_name].costs, cost_name, new_val)

            if dones[agent_name]:
                self._done_check.add(agent_name)
                self._records[self._cur_scen][agent_name].counts.episodes += 1
                self._records[self._cur_scen][agent_name].counts.steps += self._steps[agent_name]
                reason = termination.reason(obs=agent_obs)
                if reason == termination.Reason.Goal:
                    self._records[self._cur_scen][agent_name].counts.steps_adjusted += self._steps[agent_name]
                    self._records[self._cur_scen][agent_name].counts.goals += 1
                elif reason == termination.Reason.Crash:
                    self._records[self._cur_scen][agent_name].counts.steps_adjusted += _MAX_STEPS
                    self._records[self._cur_scen][agent_name].counts.crashes += 1
                else:
                    raise Exception(f"Unsupported agent done reason. Events: {agent_obs.events}.")
            # fmt: on

        if dones["__all__"] == True:
            assert (
                self._done_check == self._cur_agents
            ), f'done["__all__"]==True but not all agents are done. Current agents = {self._cur_agents}. Agents done = {self._done_check}.'

        return obs, rewards, dones, info

    def reset(self, **kwargs):
        """Resets the environment."""
        obs = super().reset(**kwargs)
        self._cur_scen = (super().scenario_log)["scenario_map"]
        self._cur_agents = set(super().agent_specs.keys())
        self._steps = dict.fromkeys(self._cur_agents, 0)
        self._done_check = set()
        if self._cur_scen not in self._records:
            for agent_name in self._cur_agents:
                cost_funcs = {
                    cost_name: cost_func()
                    for cost_name, cost_func in COST_FUNCS.items()
                }
                self._records[self._cur_scen] = {
                    agent_name: Record(
                        counts=Counts(),
                        cost_funcs=cost_funcs,
                        costs=Costs(),
                    )
                }
        return obs

    @property
    def records(self) -> Dict[str, Dict[str, Record]]:
        """
        Fine grained performance metric for each agent in each scenario.

        Example::

            self._records = {
                scen1: {
                    agent1: Record(
                        counts,
                        costs,
                        cost_funcs,
                    ),
                    agent2: Record(
                        ...
                    ),
                },
                scen2: {
                    ...
                },
            }
        """
        # Prevent modification of self._records, which is a mutable dictionary.
        return copy.deepcopy(self._records)

    @property
    def score(self) -> Dict[str, float]:
        """
        An overall performance score achieved on the wrapped environment.
        """
        # fmt: off
        agent_records = functools.reduce(lambda dict_a, dict_b: {**dict_a, **dict_b}, self.records.values())
        counts_list, costs_list = zip([record.counts, record.costs] for record in agent_records.values())
        counts_tot = functools.reduce(lambda a, b: _add_dataclass(a, b), counts_list)
        costs_tot = functools.reduce(lambda a, b: _add_dataclass(a, b), costs_list)
        # fmt: on

        _score: Dict[str, float] = {}
        _score["completion"] = _completion(counts=counts_tot)
        _score["humanness"] = _humanness(counts=counts_tot, costs=costs_tot)
        _score["rules"] = _rules(counts=counts_tot, costs=costs_tot)
        _score["time"] = _time(counts=counts_tot, costs=costs_tot)

        return _score


T = TypeVar("T", Costs, Counts)


def _add_dataclass(first: T, second: T) -> T:
    output = first.__class__()
    for key, val in asdict(first).items():
        new_val = val + getattr(second, key)
        setattr(output, key, new_val)

    return output


def _completion(counts: Counts) -> float:
    return counts.goals / counts.episodes


def _humanness(counts: Counts, costs: Costs) -> float:
    return (
        costs.dist_to_obstacles
        + costs.jerk_angular
        + costs.jerk_linear
        + costs.lane_center_offset
    ) / counts.episodes


def _rules(counts: Counts, costs: Costs) -> float:
    return (costs.speed_limit + costs.wrong_way) / counts.episodes


def _time(counts: Counts, costs: Costs) -> float:
    return (counts.steps_adjusted + costs.dist_to_goal) / counts.episodes
