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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np

from smarts.core.agent_interface import AgentInterface, InterestDoneCriteria
from smarts.core.coordinates import Point, RefLinePoint
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.observations import Observation
from smarts.core.plan import EndlessGoal, PositionalGoal
from smarts.core.road_map import RoadMap
from smarts.core.scenario import Scenario
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.traffic_history_provider import TrafficHistoryProvider
from smarts.core.traffic_provider import TrafficProvider
from smarts.core.utils.import_utils import import_module_from_file
from smarts.core.vehicle_index import VehicleIndex
from smarts.env.gymnasium.wrappers.metric.costs import (
    CostFuncs,
    Done,
    get_dist,
    make_cost_funcs,
    on_route,
)
from smarts.env.gymnasium.wrappers.metric.formula import FormulaBase, Score
from smarts.env.gymnasium.wrappers.metric.params import Params
from smarts.env.gymnasium.wrappers.metric.types import Costs, Counts, Metadata, Record
from smarts.env.gymnasium.wrappers.metric.utils import (
    add_dataclass,
    divide,
    op_dataclass,
)


class MetricsError(Exception):
    """Raised when the Metrics environment wrapper fails."""

    pass


class MetricsBase(gym.Wrapper):
    """Computes agents' performance metrics in a SMARTS environment."""

    def __init__(self, env: gym.Env, formula_path: Optional[Path] = None):
        super().__init__(env)

        # Import scoring formula.
        if formula_path:
            import_module_from_file("custom_formula", formula_path)
            from custom_formula import Formula
        else:
            from smarts.env.gymnasium.wrappers.metric.formula import Formula

        self._formula: FormulaBase = Formula()
        self._params = self._formula.params()

        _check_env(agent_interfaces=self.env.agent_interfaces, params=self._params)

        self._scen: Scenario
        self._scen_name: str
        self._road_map: RoadMap
        self._cur_agents: Set[str]
        self._steps: Dict[str, int]
        self._done_agents: Set[str]
        self._vehicle_index: VehicleIndex
        self._cost_funcs: Dict[str, CostFuncs]
        self._records_sum: Dict[str, Dict[str, Record]] = {}

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
            # (ii) ObservationOptions.unformatted, and (iii) ObservationOptions.default .
            dones = {k: v or truncated[k] for k, v in terminated.items()}
        elif isinstance(terminated, bool):
            # Caters to environments which use (i) ObservationOptions.full .
            if terminated or truncated:
                dones["__all__"] = True
            else:
                dones["__all__"] = False
            dones.update({a: d["done"] for a, d in info.items()})

        if isinstance(next(iter(obs.values())), dict):
            # Caters to environments which use (i) ObservationOptions.multi_agent,
            # (ii) ObservationOptions.full, and (iii) ObservationOptions.default .
            active_agents = [
                agent_id for agent_id, agent_obs in obs.items() if agent_obs["active"]
            ]
        else:
            # Caters to environments which uses (i) ObservationOptions.unformatted .
            active_agents = list(obs.keys())

        for agent_name in active_agents:
            base_obs: Observation = info[agent_name]["env_obs"]
            self._steps[agent_name] += 1

            # Compute all cost functions.
            costs = Costs()
            for _, cost_func in self._cost_funcs[agent_name].items():
                new_costs = cost_func(
                    self._road_map,
                    self._vehicle_index,
                    Done(dones[agent_name]),
                    base_obs,
                )
                if dones[agent_name]:
                    costs = add_dataclass(new_costs, costs)

            if dones[agent_name] == False:
                # Skip the rest, if agent is not done yet.
                continue

            self._done_agents.add(agent_name)
            # Only these termination reasons are considered by the current metrics.
            if not (
                base_obs.events.reached_goal
                or len(base_obs.events.collisions)
                or base_obs.events.off_road
                or base_obs.events.reached_max_episode_steps
                or base_obs.events.interest_done
            ):
                raise MetricsError(
                    "Expected reached_goal, collisions, off_road, "
                    "max_episode_steps, or interest_done, to be true "
                    f"on agent done, but got events: {base_obs.events}."
                )

            # Update stored counts and costs.
            counts = Counts(
                episodes=1,
                steps=self._steps[agent_name],
                goals=base_obs.events.reached_goal,
            )
            self._records_sum[self._scen_name][agent_name].counts = add_dataclass(
                counts, self._records_sum[self._scen_name][agent_name].counts
            )
            self._records_sum[self._scen_name][agent_name].costs = add_dataclass(
                costs, self._records_sum[self._scen_name][agent_name].costs
            )

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
        self._cost_funcs = {}

        _check_scen(scenario=self._scen, agent_interfaces=self.env.agent_interfaces)

        # Get the actor of interest, if any is present in the current scenario.
        interest_actors = self.env.smarts.cached_frame.interest_actors().keys()
        if len(interest_actors) == 0:
            interest_actor = None
        elif len(interest_actors) == 1:
            interest_actor = next(iter(interest_actors))
        else:
            raise MetricsError(
                f"Expected <=1 actor of interest, but got {len(interest_actors)} "
                "actors of interest."
            )

        # fmt: off
        # Refresh the cost functions for every episode.
        for agent_name in self._cur_agents:
            cost_funcs_kwargs = {}
            if self._params.dist_to_destination.active:
                interest_criteria = self.env.agent_interfaces[agent_name].done_criteria.interest
                if interest_criteria == None:
                    end_pos = self._scen.missions[agent_name].goal.position
                    dist_tot, route = get_dist(
                        road_map=self._road_map,
                        point_a=self._scen.missions[agent_name].start.point,
                        point_b=end_pos,
                    )
                elif isinstance(interest_criteria, InterestDoneCriteria) and (interest_actor is not None):
                    end_pos, dist_tot, route = _get_end_and_dist(
                        interest_actor=interest_actor,
                        vehicle_index=self._vehicle_index,
                        traffic_sims=self.env.smarts.traffic_sims,
                        scenario=self._scen,
                        road_map=self._road_map,
                    )
                    cost_funcs_kwargs.update(
                        {
                            "vehicle_gap": {
                                "num_agents": len(self._cur_agents),
                                "actor": interest_actor,
                            }
                        }
                    )
                else:
                    raise MetricsError(
                        "Unsupported configuration for distance-to-destination cost function."
                    )
                cur_on_route, cur_route_lane, cur_route_lane_point, cur_route_displacement = on_route(
                    road_map=self._road_map, route=route, point=self._scen.missions[agent_name].start.point
                )
                assert cur_on_route, f"{agent_name} does not start nearby the desired route."
                cost_funcs_kwargs.update({
                    "dist_to_destination": {
                        "end_pos": end_pos,
                        "dist_tot": dist_tot,
                        "route": route,
                        "prev_route_lane": cur_route_lane,
                        "prev_route_lane_point": cur_route_lane_point,
                        "prev_route_displacement": cur_route_displacement,
                    }
                })

            max_episode_steps = self._scen.metadata.get("scenario_duration",0) / self.env.smarts.fixed_timestep_sec
            max_episode_steps = max_episode_steps or self.env.agent_interfaces[agent_name].max_episode_steps
            cost_funcs_kwargs.update({
                "dist_to_obstacles": {
                    "ignore": self._params.dist_to_obstacles.ignore
                },
                "steps": {
                    "max_episode_steps": max_episode_steps
                },
            })
            self._cost_funcs[agent_name] = make_cost_funcs(
                params=self._params, **cost_funcs_kwargs
            )

        # Create new entry in records_sum for new scenarios.
        if self._scen_name not in self._records_sum.keys():
            self._records_sum[self._scen_name] = {
                agent_name: Record(
                    costs=Costs(),
                    counts=Counts(),
                    metadata=Metadata(difficulty=self._scen.metadata.get("scenario_difficulty",1)),
                )
                for agent_name in self._cur_agents
            }

        return result
        # fmt: on

    def records(self) -> Dict[str, Dict[str, Record]]:
        """
        Fine grained performance metric for each agent in each scenario.

        .. code-block:: bash

            $ env.records()
            $ {
                  scen1: {
                      agent1: Record(costs, counts, metadata),
                      agent2: Record(costs, counts, metadata),
                  },
                  scen2: {
                      agent1: Record(costs, counts, metadata),
                  },
              }

        Returns:
            Dict[str, Dict[str, Record]]: Performance record in a nested
            dictionary for each agent in each scenario.
        """

        records = {}
        for scen, agents in self._records_sum.items():
            records[scen] = {}
            for agent, data in agents.items():
                data_copy = copy.deepcopy(data)
                records[scen][agent] = Record(
                    costs=op_dataclass(
                        data_copy.costs, data_copy.counts.episodes, divide
                    ),
                    counts=data_copy.counts,
                    metadata=data_copy.metadata,
                )

        return records

    def score(self) -> Score:
        """
        Computes score according to environment specific formula from the
        Formula class.

        Returns:
            Dict[str, float]: Contains key-value pairs denoting score
            components.
        """
        return self._formula.score(records=self.records())


def _get_end_and_dist(
    interest_actor: str,
    vehicle_index: VehicleIndex,
    traffic_sims: List[TrafficProvider],
    scenario: Scenario,
    road_map: RoadMap,
) -> Tuple[Point, float, RoadMap.Route]:
    """Computes the end point and route distance for a given vehicle of interest.

    Args:
        interest_actor (str): Name of vehicle of interest.
        vehicle_index (VehicleIndex): Index of all vehicles currently present.
        traffic_sims (List[TrafficProvider]): List of traffic providers.
        scenario (Scenario): Current scenario.
        road_map (RoadMap): Underlying road map.

    Returns:
        Tuple[Point, float, RoadMap.Route]: End point, route distance, and planned route.
    """
    # Check if the interest vehicle is a social agent.
    interest_social_missions = [
        mission for name, mission in scenario.missions.items() if interest_actor in name
    ]
    # Check if the actor of interest is a traffic vehicle.
    interest_traffic_sims = [
        traffic_sim
        for traffic_sim in traffic_sims
        if traffic_sim.manages_actor(interest_actor)
    ]
    if len(interest_social_missions) + len(interest_traffic_sims) != 1:
        raise MetricsError(
            "Social agents and traffic providers contain zero or "
            "more than one actor of interest."
        )

    if len(interest_social_missions) == 1:
        interest_social_mission = interest_social_missions[0]
        goal = interest_social_mission.goal
        assert isinstance(goal, PositionalGoal)
        end_pos = goal.position
        dist_tot, route = get_dist(
            road_map=road_map,
            point_a=interest_social_mission.start.point,
            point_b=end_pos,
        )
    else:
        interest_traffic_sim = interest_traffic_sims[0]
        end_pos, dist_tot, route = _get_traffic_end_and_dist(
            vehicle_name=interest_actor,
            vehicle_index=vehicle_index,
            traffic_sim=interest_traffic_sim,
            road_map=road_map,
        )

    return end_pos, dist_tot, route


def _get_traffic_end_and_dist(
    vehicle_name: str,
    vehicle_index: VehicleIndex,
    traffic_sim: TrafficProvider,
    road_map: RoadMap,
) -> Tuple[Point, float, RoadMap.Route]:
    """Computes the end point and route distance of a (i) SUMO traffic,
    (ii) SMARTS traffic, or (iii) history traffic vehicle
    specified by `vehicle_name`.

    Args:
        vehicle_name (str): Name of vehicle.
        vehicle_index (VehicleIndex): Index of all vehicles currently present.
        traffic_sim (TrafficProvider): Traffic provider.
        road_map (RoadMap): Underlying road map.

    Returns:
        Tuple[Point, float, RoadMap.Route]: End point, route distance, and planned route.
    """

    if isinstance(traffic_sim, (SumoTrafficSimulation, LocalTrafficProvider)):
        start_pos = Point(*vehicle_index.vehicle_position(vehicle_name))
        dest_road = traffic_sim.vehicle_dest_road(vehicle_name)
        end_pos = (
            road_map.road_by_id(dest_road)
            .lane_at_index(0)
            .from_lane_coord(RefLinePoint(s=np.inf))
        )
        dist_tot, route = get_dist(
            road_map=road_map, point_a=start_pos, point_b=end_pos
        )
        return end_pos, dist_tot, route
    elif isinstance(traffic_sim, TrafficHistoryProvider):
        history = traffic_sim.vehicle_history_window(vehicle_id=vehicle_name)
        start_pos = Point(x=history.start_position_x, y=history.start_position_y)
        end_pos = Point(x=history.end_position_x, y=history.end_position_y)
        # TODO : Plan.create_route() creates the shortest route which is
        # sufficient in simple maps, but it may or may not match the actual
        # roads traversed by the history vehicle in complex maps. Ideally we
        # should use the actual road ids traversed by the history vehicle to
        # compute the distance.
        dist_tot, route = get_dist(
            road_map=road_map, point_a=start_pos, point_b=end_pos
        )
        return end_pos, dist_tot, route
    else:
        raise MetricsError(f"Unsupported traffic provider {traffic_sim.source_str}.")


class Metrics(gym.Wrapper):
    """Metrics class wraps an underlying MetricsBase class. The underlying
    MetricsBase class computes agents' performance metrics in a SMARTS
    environment. Whereas, this Metrics class is a basic gym.Wrapper class
    which prevents external users from accessing or modifying (i) protected
    attributes or (ii) attributes beginning with an underscore, to ensure
    security of the metrics computed.

    Args:
        env (gym.Env): The gym environment to be wrapped.

    Raises:
        AttributeError: Upon accessing (i) a protected attribute or (ii) an \
        attribute beginning with an underscore.

    Returns:
        gym.Env: A wrapped gym environment which computes agents' performance metrics.
    """

    def __init__(self, env: gym.Env, formula_path: Optional[Path] = None):
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


def _check_env(agent_interfaces: Dict[str, AgentInterface], params: Params):
    """Checks environment suitability to compute performance metrics.
    Args:
        agent_interfaces (Dict[str,AgentInterface]): Agent interfaces.
        params (Params): Metric parameters.

    Raises:
        AttributeError: If any required agent interface is disabled or
            is ill defined.
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

    for agent_name, agent_interface in agent_interfaces.items():
        intrfc = check_intrfc(agent_interface)
        if not all(intrfc.values()):
            raise AttributeError(
                (
                    "Enable {0}'s disabled interface to "
                    "compute its metrics. Current interface is "
                    "{1}."
                ).format(agent_name, intrfc)
            )

        interest_criteria = agent_interface.done_criteria.interest
        if (
            params.dist_to_destination.active
            and isinstance(interest_criteria, InterestDoneCriteria)
        ) and not (
            len(interest_criteria.actors_filter) == 0
            and interest_criteria.include_scenario_marked == True
        ):
            raise AttributeError(
                (
                    "InterestDoneCriteria with none or multiple actors of "
                    "interest is currently not supported when "
                    "dist_to_destination cost function is enabled. Current "
                    "interface is {0}:{1}."
                ).format(agent_name, interest_criteria)
            )


def _check_scen(scenario: Scenario, agent_interfaces: Dict[str, AgentInterface]):
    """Checks scenario suitability to compute performance metrics.

    Args:
        scen (Scenario): A ``smarts.core.scenario.Scenario`` class.
        agent_interfaces (Dict[str,AgentInterface]): Agent interfaces.

    Raises:
        MetricsError: If (i) scenario difficulty is not properly normalized,
            or (ii) any agent's goal is improperly configured.
    """

    difficulty = scenario.metadata.get("scenario_difficulty", None)
    if not ((difficulty is None) or (0 < difficulty <= 1)):
        raise MetricsError(
            "Expected scenario difficulty to be normalized within (0,1], but "
            f"got difficulty={difficulty}."
        )

    goal_types = {
        agent_name: type(agent_mission.goal)
        for agent_name, agent_mission in scenario.missions.items()
    }

    aoi = scenario.metadata.get("actor_of_interest_re_filter", None)
    for agent_name, agent_interface in agent_interfaces.items():
        interest_criteria = agent_interface.done_criteria.interest
        if not (
            (goal_types[agent_name] == PositionalGoal and interest_criteria is None)
            or (
                goal_types[agent_name] == EndlessGoal
                and isinstance(interest_criteria, InterestDoneCriteria)
                and aoi != None
            )
        ):
            raise MetricsError(
                "{0} has an unsupported goal type {1} and interest done criteria {2} "
                "combination.".format(
                    agent_name, goal_types[agent_name], interest_criteria
                )
            )
