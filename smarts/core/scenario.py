# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
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
import glob
import json
import logging
import math
import os
import pickle
import random
import uuid
from functools import lru_cache
from itertools import cycle, product
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from cached_property import cached_property

from smarts.core.coordinates import Dimensions, Heading, Pose, RefLinePoint
from smarts.core.data_model import SocialAgent
from smarts.core.default_map_factory import create_road_map
from smarts.core.plan import (
    EndlessGoal,
    LapMission,
    Mission,
    PositionalGoal,
    Start,
    TraverseGoal,
    VehicleSpec,
    Via,
    default_entry_tactic,
)
from smarts.core.road_map import RoadMap
from smarts.core.traffic_history import TrafficHistory
from smarts.core.utils.file import file_md5_hash, make_dir_in_smarts_log_dir, path2hash
from smarts.core.utils.id import SocialAgentId
from smarts.core.utils.math import radians_to_vec, vec_to_radians
from smarts.sstudio import types as sstudio_types
from smarts.sstudio.types import Via as SSVia


class Scenario:
    """The purpose of the Scenario is to provide an aggregate of all
    code/configuration/assets that is specialized to a scenario using SUMO.

    Args:
        scenario_root:
            The scenario asset folder ie. './scenarios/trigger'.
        route:
            The social vehicle traffic spec.
        missions:
            agent_id to mission mapping.
    """

    def __init__(
        self,
        scenario_root: str,
        route: str = None,
        missions: Dict[str, Mission] = None,
        social_agents: Dict[str, SocialAgent] = None,
        log_dir: str = None,
        surface_patches: list = None,
        traffic_history: str = None,
    ):

        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = scenario_root
        self._route = route
        self._missions = missions or {}
        self._bubbles = Scenario._discover_bubbles(scenario_root)
        self._social_agents = social_agents or {}
        self._surface_patches = surface_patches
        self._log_dir = self._resolve_log_dir(log_dir)
        self._validate_assets_exist()

        if traffic_history:
            self._traffic_history = TrafficHistory(traffic_history)
            default_lane_width = self.traffic_history.lane_width
        else:
            self._traffic_history = None
            default_lane_width = None

        self._road_map, self._road_map_hash = create_road_map(
            self._root, 1.0, default_lane_width
        )
        self._scenario_hash = path2hash(str(Path(self.root_filepath).resolve()))

    def __repr__(self):
        return f"""Scenario(
  _root={self._root},
  _route={self._route},
  _missions={self._missions},
)"""

    @staticmethod
    def get_scenario_list(scenarios_or_scenarios_dirs: Sequence[str]):
        scenario_roots = []
        for root in scenarios_or_scenarios_dirs:
            if Scenario.is_valid_scenario(root):
                # This is the single scenario mode, only training against a single scenario
                scenario_roots.append(root)
            else:
                scenario_roots.extend(Scenario.discover_scenarios(root))
        return scenario_roots

    @staticmethod
    def scenario_variations(
        scenarios_or_scenarios_dirs: Sequence[str],
        agents_to_be_briefed: Sequence[str],
        shuffle_scenarios: bool = True,
    ):
        """Generate a cycle of the configurations of scenarios.

        Args:
            scenarios_or_scenarios_dirs:
                A sequence of either the scenario to run (see scenarios/ for some samples you
                can use) OR a directory of scenarios to sample from.
            agents_to_be_briefed:
                Agent IDs that will be assigned a mission ("briefed" on a mission).
        """
        scenario_roots = Scenario.get_scenario_list(scenarios_or_scenarios_dirs)

        if shuffle_scenarios:
            np.random.shuffle(scenario_roots)

        return Scenario.variations_for_all_scenario_roots(
            cycle(scenario_roots), agents_to_be_briefed, shuffle_scenarios
        )

    @staticmethod
    def variations_for_all_scenario_roots(
        scenario_roots, agents_to_be_briefed, shuffle_scenarios=True
    ):
        for scenario_root in scenario_roots:
            surface_patches = Scenario.discover_friction_map(scenario_root)

            agent_missions = Scenario.discover_agent_missions(
                scenario_root, agents_to_be_briefed
            )
            agent_missions = [dict(zip(agents_to_be_briefed, agent_missions))]
            social_agent_infos = Scenario._discover_social_agents_info(scenario_root)
            social_agents = [
                {
                    agent_id: (agent.to_agent_spec(), (agent, mission))
                    for agent_id, (
                        agent,
                        mission,
                    ) in per_episode_social_agent_infos.items()
                }
                for per_episode_social_agent_infos in social_agent_infos
            ]

            # `or [None]` so that product(...) will not return an empty result
            # but insted a [(..., `None`), ...].
            routes = Scenario.discover_routes(scenario_root) or [None]
            agent_missions = agent_missions or [None]
            social_agents = social_agents or [None]
            traffic_histories = Scenario.discover_traffic_histories(scenario_root) or [
                None
            ]

            roll_routes = 0
            roll_agent_missions = 0
            roll_social_agents = 0
            roll_traffic_histories = 0

            if shuffle_scenarios:
                roll_routes = random.randint(0, len(routes))
                roll_agent_missions = random.randint(0, len(agent_missions))
                roll_social_agents = random.randint(0, len(social_agents))
                roll_traffic_histories = 0  # random.randint(0, len(traffic_histories))

            for (
                concrete_route,
                concrete_agent_missions,
                concrete_social_agents,
                concrete_traffic_history,
            ) in product(
                np.roll(routes, roll_routes, 0),
                np.roll(agent_missions, roll_agent_missions, 0),
                np.roll(social_agents, roll_social_agents, 0),
                np.roll(traffic_histories, roll_traffic_histories, 0),
            ):
                concrete_social_agent_missions = {
                    agent_id: mission
                    for agent_id, (_, (_, mission)) in (
                        concrete_social_agents or {}
                    ).items()
                }

                # Filter out mission
                concrete_social_agents = {
                    agent_id: (_agent_spec, social_agent)
                    for agent_id, (_agent_spec, (social_agent, _)) in (
                        concrete_social_agents or {}
                    ).items()
                }

                yield Scenario(
                    scenario_root,
                    route=concrete_route,
                    missions={
                        **(concrete_agent_missions or {}),
                        **concrete_social_agent_missions,
                    },
                    social_agents=concrete_social_agents,
                    surface_patches=surface_patches,
                    traffic_history=concrete_traffic_history,
                )

    @staticmethod
    def discover_agent_missions_count(scenario_root):
        missions_file = os.path.join(scenario_root, "missions.pkl")
        if os.path.exists(missions_file):
            with open(missions_file, "rb") as f:
                return len(pickle.load(f))

        return 0

    @staticmethod
    def discover_agent_missions(scenario_root, agents_to_be_briefed):
        """Returns a sequence of {agent_id: mission} mappings.

        If no missions are discovered we generate random ones. If there is only one
        agent to be briefed we return a list of `{agent_id: mission}` cycling through
        each mission. If there are multiple agents to be briefed we assume that each
        one is intended to get its own mission and that `len(agents_to_be_briefed) ==
        len(missions)`. In this case a list of one dictionary is returned.
        """

        road_map, _ = create_road_map(scenario_root)

        missions = []
        missions_file = os.path.join(scenario_root, "missions.pkl")
        if os.path.exists(missions_file):
            with open(missions_file, "rb") as f:
                missions = pickle.load(f)

            missions = [
                Scenario._extract_mission(actor_and_mission.mission, road_map)
                for actor_and_mission in missions
            ]

        if not missions:
            missions = [None for _ in range(len(agents_to_be_briefed))]

        if len(agents_to_be_briefed) == 1:
            # single-agent, so we cycle through all missions individually.
            return missions
        elif len(agents_to_be_briefed) > 1:
            # multi-agent, so we assume missions "drive" the agents (i.e. one
            # mission per agent) and we will not be cycling through missions.
            assert not missions or len(missions) == len(agents_to_be_briefed), (
                "You must either provide an equal number of missions ({}) to "
                "agents ({}) or provide no missions at all so they can be "
                "randomly generated.".format(len(missions), len(agents_to_be_briefed))
            )

        return missions

    @staticmethod
    def discover_friction_map(scenario_root):
        """Returns the list of surface patches parameters defined in
        scenario file. Each element of the list contains the
        parameters of the specifiec surface patch.
        """
        surface_patches = []
        friction_map_file = os.path.join(scenario_root, "friction_map.pkl")
        if os.path.exists(friction_map_file):
            with open(friction_map_file, "rb") as f:
                map_surface_patches = pickle.load(f)
            for surface_patch in map_surface_patches:
                surface_patches.append(
                    {
                        "zone": surface_patch.zone,
                        "begin_time": surface_patch.begin_time,
                        "end_time": surface_patch.end_time,
                        "friction coefficient": surface_patch.friction_coefficient,
                    }
                )
        return surface_patches

    @staticmethod
    @lru_cache(maxsize=16)
    def _discover_social_agents_info(
        scenario,
    ) -> Sequence[Dict[str, SocialAgent]]:
        """Loops through the social agent mission pickles, instantiating corresponding
        implementations for the given types. The output is a list of
        {agent_id: (mission, locator)}, where each dictionary corresponds to the
        social agents to run for a given concrete Scenario (which translates to
        "per episode" when swapping).
        """
        scenario_root = (
            scenario.root_filepath if isinstance(scenario, Scenario) else scenario
        )
        road_map, _ = create_road_map(scenario_root)

        social_agents_path = os.path.join(scenario_root, "social_agents")
        if not os.path.exists(social_agents_path):
            return []

        # [ ( missions_file, agent_actor, Mission ) ]
        agent_bucketer = []

        # like dict.setdefault
        def setdefault(l: Sequence[Any], index: int, default):
            while len(l) < index + 1:
                l.append([])
            return l[index]

        file_match = os.path.join(social_agents_path, "*.pkl")
        for missions_file_path in glob.glob(file_match):
            with open(missions_file_path, "rb") as missions_file:
                count = 0
                missions = pickle.load(missions_file)

            for mission_and_actor in missions:
                # Each pickle file will contain a list of actor/mission pairs. The pairs
                # will most likely be generated in an M:N fashion
                # (i.e. A1: M1, A1: M2, A2: M1, A2: M2). The desired behavior is to have
                # a single pair per concrete Scenario (which would translate to
                # "per episode" when swapping)
                assert isinstance(
                    mission_and_actor.actor, sstudio_types.SocialAgentActor
                )

                actor = mission_and_actor.actor
                extracted_mission = Scenario._extract_mission(
                    mission_and_actor.mission, road_map
                )
                namespace = os.path.basename(missions_file_path)
                namespace = os.path.splitext(namespace)[0]

                setdefault(agent_bucketer, count, []).append(
                    (
                        SocialAgent(
                            id=SocialAgentId.new(actor.name, group=namespace),
                            name=actor.name,
                            is_boid=False,
                            is_boid_keep_alive=False,
                            agent_locator=actor.agent_locator,
                            policy_kwargs=actor.policy_kwargs,
                            initial_speed=actor.initial_speed,
                        ),
                        extracted_mission,
                    )
                )
                count += 1

        social_agents_info = []
        for l in agent_bucketer:
            social_agents_info.append(
                {agent.id: (agent, mission) for agent, mission in l}
            )

        return social_agents_info

    @staticmethod
    def discover_scenarios(scenario_or_scenarios_dir):
        if Scenario.is_valid_scenario(scenario_or_scenarios_dir):
            # This is the single scenario mode, only training against a single scenario
            scenario = scenario_or_scenarios_dir
            discovered_scenarios = [scenario]
        else:
            # Find all valid scenarios in the given scenarios directory
            discovered_scenarios = []
            for scenario_file in os.listdir(scenario_or_scenarios_dir):
                scenario_root = os.path.join(scenario_or_scenarios_dir, scenario_file)
                if Scenario.is_valid_scenario(scenario_root):
                    discovered_scenarios.append(scenario_root)
        assert (
            len(discovered_scenarios) > 0
        ), f"No valid scenarios found in {scenario_or_scenarios_dir}"

        return discovered_scenarios

    @staticmethod
    def discover_routes(scenario_root):
        """Discover the route files in the given scenario.

        >>> Scenario.discover_routes("scenarios/intersections/2lane")
        ['all.rou.xml', 'horizontal.rou.xml', 'turns.rou.xml', 'unprotected_left.rou.xml', 'vertical.rou.xml']
        >>> Scenario.discover_routes("scenarios/loop") # loop does not have any routes
        ['basic.rou.xml']
        """
        return sorted(
            [
                os.path.basename(r)
                for r in glob.glob(os.path.join(scenario_root, "traffic", "*.rou.xml"))
            ]
        )

    @staticmethod
    def _discover_bubbles(scenario_root):
        path = os.path.join(scenario_root, "bubbles.pkl")
        if not os.path.exists(path):
            return []

        with open(path, "rb") as f:
            bubbles = pickle.load(f)
            return bubbles

    def set_ego_missions(self, ego_missions: dict):
        self._missions = ego_missions

    def discover_missions_of_traffic_histories(self) -> Dict[str, Mission]:
        vehicle_missions = {}
        for row in self._traffic_history.first_seen_times():
            start_time = float(row[1])
            pphs = self._traffic_history.vehicle_pose_at_time(row[0], start_time)
            assert pphs
            pos_x, pos_y, heading, speed = pphs
            entry_tactic = default_entry_tactic(speed)
            v_id = str(row[0])
            veh_config_type = self._traffic_history.vehicle_config_type(v_id)
            veh_length, veh_width, veh_height = self._traffic_history.vehicle_size(v_id)
            # missions start from front bumper, but pos is center of vehicle
            hhx, hhy = radians_to_vec(heading) * (0.5 * veh_length)
            vehicle_missions[v_id] = Mission(
                start=Start(
                    (pos_x + hhx, pos_y + hhy),
                    Heading(heading),
                ),
                entry_tactic=entry_tactic,
                goal=TraverseGoal(self.road_map),
                start_time=start_time,
                vehicle_spec=VehicleSpec(
                    veh_id=v_id,
                    veh_config_type=veh_config_type,
                    dimensions=Dimensions(veh_length, veh_width, veh_height),
                ),
            )
        return vehicle_missions

    @staticmethod
    def discover_traffic_histories(scenario_root: str):
        return [
            entry
            for entry in os.scandir(scenario_root)
            if entry.is_file() and entry.path.endswith(".shf")
        ]

    @staticmethod
    def _extract_mission(mission, road_map):
        """Takes a sstudio.types.(Mission, EndlessMission, etc.) and converts it to
        the corresponding SMARTS mission types.
        """

        def resolve_offset(offset, lane_length):
            # epsilon to ensure we are within this edge and not the subsequent one
            epsilon = 1e-6
            lane_length -= epsilon
            if offset == "base":
                return epsilon
            elif offset == "max":
                return lane_length
            elif offset == "random":
                return random.uniform(epsilon, lane_length)
            else:
                return float(offset)

        def to_position_and_heading(road_id, lane_index, offset, road_map):
            road = road_map.road_by_id(road_id)
            lane = road.lane_at_index(lane_index)
            offset = resolve_offset(offset, lane.length)
            position = lane.from_lane_coord(RefLinePoint(s=offset))
            lane_vector = lane.vector_at_offset(offset)
            heading = vec_to_radians(lane_vector[:2])
            return position, Heading(heading)

        def to_scenario_via(
            vias: Tuple[SSVia, ...], road_map: RoadMap
        ) -> Tuple[Via, ...]:
            s_vias = []
            for via in vias:
                road = road_map.road_by_id(via.road_id)
                lane = road.lane_at_index(via.lane_index)
                lane_width = lane.width_at_offset(via.lane_offset)
                hit_distance = (
                    via.hit_distance if via.hit_distance > 0 else lane_width / 2
                )
                via_position = lane.from_lane_coord(RefLinePoint(via.lane_offset))

                s_vias.append(
                    Via(
                        lane_id=lane.lane_id,
                        lane_index=via.lane_index,
                        road_id=via.road_id,
                        position=tuple(via_position),
                        hit_distance=hit_distance,
                        required_speed=via.required_speed,
                    )
                )

            return tuple(s_vias)

        # For now we discard the route and just take the start and end to form our
        # missions.
        if isinstance(mission, sstudio_types.Mission):
            position, heading = to_position_and_heading(
                *mission.route.begin,
                road_map,
            )
            start = Start(position, heading)

            position, _ = to_position_and_heading(
                *mission.route.end,
                road_map,
            )
            goal = PositionalGoal(position, radius=2)

            return Mission(
                start=start,
                route_vias=mission.route.via,
                goal=goal,
                start_time=mission.start_time,
                entry_tactic=mission.entry_tactic,
                via=to_scenario_via(mission.via, road_map),
            )
        elif isinstance(mission, sstudio_types.EndlessMission):
            position, heading = to_position_and_heading(
                *mission.begin,
                road_map,
            )
            start = Start(position, heading)

            return Mission(
                start=start,
                goal=EndlessGoal(),
                start_time=mission.start_time,
                entry_tactic=mission.entry_tactic,
                via=to_scenario_via(mission.via, road_map),
            )
        elif isinstance(mission, sstudio_types.LapMission):
            start_road_id, start_lane, start_road_offset = mission.route.begin
            end_road_id, end_lane, end_road_offset = mission.route.end

            travel_road = road_map.road_by_id(start_road_id)
            if start_road_id == end_road_id:
                travel_road = travel_road.outgoing_roads[0]

            end_road = road_map.road_by_id(end_road_id)
            via_roads = [road_map.road_by_id(r) for r in mission.route.via]

            route = road_map.generate_routes(travel_road, end_road, via_roads, 1)[0]

            start_position, start_heading = to_position_and_heading(
                *mission.route.begin,
                road_map,
            )
            end_position, _ = to_position_and_heading(
                *mission.route.end,
                road_map,
            )

            return LapMission(
                start=Start(start_position, start_heading),
                goal=PositionalGoal(end_position, radius=2),
                route_vias=mission.route.via,
                num_laps=mission.num_laps,
                route_length=route.road_length,
                start_time=mission.start_time,
                entry_tactic=mission.entry_tactic,
                via_points=to_scenario_via(mission.via, road_map),
            )

        raise RuntimeError(
            f"sstudio mission={mission} is an invalid type={type(mission)}"
        )

    @staticmethod
    def is_valid_scenario(scenario_root):
        """Checks if the scenario_root directory matches our expected scenario structure

        >>> Scenario.is_valid_scenario("scenarios/loop")
        True
        >>> Scenario.is_valid_scenario("scenarios/non_existant")
        False
        """
        # just make sure we can load the map
        try:
            road_map, _ = create_road_map(scenario_root)
        except FileNotFoundError:
            return False
        return road_map is not None

    @staticmethod
    def next(scenario_iterator, log_id=""):
        """Utility to override specific attributes from a scenario iterator"""

        scenario = next(scenario_iterator)
        scenario._log_id = log_id
        return scenario

    @property
    def name(self):
        return os.path.basename(os.path.normpath(self._root))

    @property
    def root_filepath(self):
        return self._root

    @property
    def surface_patches(self):
        return self._surface_patches

    @property
    def road_map_hash(self):
        return self._road_map_hash

    @property
    def plane_filepath(self):
        return os.path.join(self._root, "plane.urdf")

    @property
    def vehicle_filepath(self):
        if not os.path.isdir(self._root):
            return None
        for fname in os.listdir(self._root):
            if fname.endswith(".urdf") and fname != "plane.urdf":
                return os.path.join(self._root, fname)
        return None

    @property
    def tire_parameters_filepath(self):
        return os.path.join(self._root, "tire_parameters.yaml")

    @property
    def controller_parameters_filepath(self):
        return os.path.join(self._root, "controller_parameters.yaml")

    @property
    def route(self):
        return self._route

    @property
    def route_files_enabled(self):
        return bool(self._route)

    @property
    def route_filepath(self):
        return os.path.join(self._root, "traffic", self._route)

    @property
    def map_glb_filepath(self):
        return os.path.join(self._root, "map.glb")

    def unique_sumo_log_file(self):
        return os.path.join(self._log_dir, f"sumo-{str(uuid.uuid4())[:8]}")

    @property
    def road_map(self):
        return self._road_map

    @property
    def missions(self):
        return self._missions

    @property
    def social_agents(self):
        return self._social_agents

    @property
    def bubbles(self):
        return self._bubbles

    def mission(self, agent_id):
        return self._missions.get(agent_id, None)

    def _resolve_log_dir(self, log_dir):
        if log_dir is None:
            log_dir = make_dir_in_smarts_log_dir("_sumo_run_logs")

        return os.path.abspath(log_dir)

    def _validate_assets_exist(self):
        assert Scenario.is_valid_scenario(self._root)
        os.makedirs(self._log_dir, exist_ok=True)

    @property
    def traffic_history(self):
        return self._traffic_history

    @property
    def scenario_hash(self):
        return self._scenario_hash
