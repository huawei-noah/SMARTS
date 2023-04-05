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
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Optional, Set, TypeVar

import gymnasium as gym

from smarts.core.agent_interface import AgentInterface
from smarts.core.coordinates import Point
from smarts.core.observations import Observation
from smarts.core.plan import PositionalGoal
from smarts.core.road_map import RoadMap
from smarts.core.scenario import Scenario
from smarts.core.utils.import_utils import import_module_from_file
from smarts.env.gymnasium.wrappers.metric.completion import (
    Completion,
    CompletionFuncs,
    get_dist,
)
from smarts.env.gymnasium.wrappers.metric.costs import CostFuncs, Costs
from smarts.env.gymnasium.wrappers.metric.counts import Counts
from smarts.env.gymnasium.wrappers.metric.formula import Score
from smarts.env.gymnasium.wrappers.metric.types import Data, Record
from smarts.core.road_map import RoadMap
from smarts.core.vehicle_index import VehicleIndex

class MetricsError(Exception):
    """Raised when Metrics env wrapper fails."""

    pass


class MetricsBase(gym.Wrapper):
    """Computes agents' performance metrics in a SMARTS environment."""

    def __init__(self, env: gym.Env, formula_path: Optional[Path]):
        super().__init__(env)
        # _check_env(env)
        self._scen: Scenario
        self._scen_name: str
        self._road_map: RoadMap
        self._cur_agents: Set[str]
        self._steps: Dict[str, int]
        self._done_agents: Set[str]
        self._vehicle_index: VehicleIndex 
        self._records = {}

        # Import scoring formula
        if formula_path:
            import_module_from_file("custom_formula", formula_path)
            from custom_formula import Formula
        else:
            from formula import Formula

        self._formula = Formula()
        self._params = self._formula.params()

    def step(self, action: Dict[str, Any]):
        """Steps the environment by one step."""
        result = super().step(action)

        obs, _, terminated, truncated, info = result

        # Only count steps in which an ego agent is present.
        if len(obs) == 0:
            return result

        dones = {}
        if isinstance(terminated, dict):
            # Caters to environments which use (i) ObservationOptions.multi_agent,
            # (ii) ObservationOptions.unformated, and (iii) ObservationOptions.default .
            dones = {k: v or truncated[k] for k, v in terminated.items()}
        elif isinstance(terminated, bool):
            # Caters to environments which use ObservationOptions.full .
            if terminated or truncated:
                dones["__all__"] = True
            else:
                dones["__all__"] = False
            dones.update({a: d["done"] for a, d in info.items()})

        if isinstance(next(iter(obs.values())), dict):
            # Caters to environments which use (i) ObservationOptions.multi_agent,
            # (ii) ObservationOptions.full, and (iii) ObservationOptions.default .
            active_agents = [agent_id for agent_id, agent_obs in obs.items() if agent_obs["active"]]
        else:
            # Caters to environments which uses (i) ObservationOptions.unformated .
            active_agents = list(obs.keys())

        # fmt: off
        for agent_name in active_agents:
            base_obs: Observation = info[agent_name]["env_obs"]
            self._steps[agent_name] += 1

            # Compute all cost functions.
            costs = Costs()
            for field in fields(self._records[self._scen_name][agent_name].cost_funcs):
                if not getattr(self._params, field.name).active:
                    d = field.name
                    print(d, getattr(self._params, d))
                    input("Skipping cost computation ------------")
                    continue
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
        if dones["__all__"] is True:
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
        self._scen = self.env.smarts.scenario
        self._scen_name = self.env.smarts.scenario.name
        self._road_map = self.env.smarts.scenario.road_map
        self._vehicle_index = self.env.smarts.vehicle_index

        if self._scen_name in self._records.keys():
            return result 

        # _check_scen(self._scen)
        completion = Completion()

        # Compute total scenario distance with respect to specified vehicle.
        if self._params.dist_completed.active and self._params.dist_completed.wrt is not "self":
            vehicle_name = self._params.dist_completed.wrt
            traffic_sims = self.env.smarts.traffic_sims
            routes = [traffic_sim.route_for_vehicle(vehicle_name) for traffic_sim in traffic_sims]
            route = [route for route in routes if route is not None]
            assert len(route) == 1, "Multiple traffic sims contain the vehicle of interest."
            completion = Completion(dist_tot = route[0].road_length)

        for agent_name in self._cur_agents:
            if self._params.dist_completed.active and self._params.dist_completed.wrt == "self":
                completion = Completion(
                    dist_tot=get_dist(
                        road_map=self._road_map,
                        point_a=Point(*self._scen.missions[agent_name].start.position),
                        point_b=self._scen.missions[agent_name].goal.position,    
                    )
                )
            self._records[self._scen_name] = {
                agent_name: Data(
                    record=Record(
                        completion=completion,
                        costs=Costs(),
                        counts=Counts(),
                    ),
                    cost_funcs=CostFuncs(),
                    completion_funcs=CompletionFuncs(),
                )
            }

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
        Computes score according to environment specific formula from the
        Formula class.

        Returns:
            Dict[str, float]: Contains key-value pairs denoting score
            components.
        """
        return self._formula.score(self._records)


class Metrics(gym.Wrapper):
    """Metrics class wraps an underlying MetricsBase class. The underlying
    MetricsBase class computes agents' performance metrics in a SMARTS
    environment. Whereas, this Metrics class is a basic gym.Wrapper class
    which prevents external users from accessing or modifying (i) protected
    attributes or (ii) attributes beginning with an underscore, to ensure
    security of the metrics computed.

    Args:
        env (gym.Env): A gym.Env to be wrapped.

    Raises:
        AttributeError: Upon accessing (i) a protected attribute or (ii) an
        attribute beginning with an underscore.

    Returns:
        gym.Env: A wrapped gym.Env which computes agents' performance metrics.
    """

    def __init__(self, env: gym.Env, formula_path: Path):
        env = MetricsBase(env, formula_path)
        super().__init__(env)

    def __getattr__(self, name: str):
        """Returns an attribute with ``name``, unless ``name`` is a restricted 
        attribute or starts with an underscore."""
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_") or name in [
            "smarts",
        ]:
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")

        return getattr(self.env, name)


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
