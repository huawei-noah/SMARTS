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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import glob
import logging
import os
import pickle
import random
import uuid
import warnings
from functools import lru_cache
from itertools import cycle, product
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import cloudpickle
import numpy as np

from smarts.core.coordinates import Heading, Point, RefLinePoint
from smarts.core.data_model import SocialAgent
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
from smarts.core.utils.file import make_dir_in_smarts_log_dir, path2hash
from smarts.core.utils.id import SocialAgentId
from smarts.core.utils.math import (
    combination_pairs_with_unique_indices,
    radians_to_vec,
    vec_to_radians,
)
from smarts.sstudio import types as sstudio_types
from smarts.sstudio.types import MapSpec
from smarts.sstudio.types import Via as SSVia

VehicleWindow = TrafficHistory.TrafficHistoryVehicleWindow

# Suppress trimesh deprecation warning

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="Please use `coo_matrix` from the `scipy.sparse` namespace, the `scipy.sparse.coo` namespace is deprecated.",
        category=DeprecationWarning,
    )
    import trimesh  # only suppress the warnings caused by trimesh


class Scenario:
    """The purpose of the Scenario is to provide an aggregate of all
    code/configuration/assets that is specialized to a scenario.

    Args:
        scenario_root:
            The scenario asset folder ie. './scenarios/trigger'.
        traffic_specs:
            The social vehicle traffic specs.
        missions:
            agent_id to mission mapping.
        map_spec:
            If specified, allows specifying a MapSpec at run-time
            to override any spec that may have been pre-specified
            in the scenario folder (or the default if none were).
            Also see comments around the sstudio.types.MapSpec definition.
    """

    def __init__(
        self,
        scenario_root: str,
        traffic_specs: Sequence[str] = [],
        missions: Optional[Dict[str, Mission]] = None,
        social_agents: Optional[Dict[str, SocialAgent]] = None,
        log_dir: Optional[str] = None,
        surface_patches: Optional[Sequence[Dict[str, Any]]] = None,
        traffic_history: Optional[str] = None,
        map_spec: Optional[MapSpec] = None,
        route: Optional[str] = None,  # deprecated: use traffic_specs instead
    ):

        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = scenario_root
        self._traffic_specs = traffic_specs
        if route:
            warnings.warn(
                "Scenario route property has been deprecated in favor of traffic_specs.  Please update your code.",
                category=DeprecationWarning,
            )
            traffic_path = os.path.join(scenario_root, "build", "traffic")
            self._traffic_specs = [os.path.join(traffic_path, route)]
        self._missions = missions or {}
        self._bubbles = Scenario._discover_bubbles(scenario_root)
        self._metadata = Scenario._discover_metadata(scenario_root)
        self._social_agents = social_agents or {}
        self._surface_patches = surface_patches
        self._log_dir = self._resolve_log_dir(log_dir)

        if traffic_history:
            self._traffic_history = TrafficHistory(traffic_history)
            default_lane_width = self._traffic_history.lane_width
        else:
            self._traffic_history = None
            default_lane_width = None

        # XXX: using a map builder_fn supplied by users is a security risk
        # as SMARTS will be executing the code "as is".  We are currently
        # trusting our users to not try to sabotage their own simulations.
        # In the future, this may need to be revisited if SMARTS is ever
        # shared in a multi-user mode.
        if not map_spec:
            map_spec = Scenario.discover_map(self._root, 1.0, default_lane_width)
        self._road_map, self._road_map_hash = map_spec.builder_fn(map_spec)
        self._scenario_hash = path2hash(str(Path(self.root_filepath).resolve()))

        os.makedirs(self._log_dir, exist_ok=True)

    def __repr__(self):
        return f"""Scenario(
  _root={self._root},
  _traffic_specs={self._traffic_specs},
  _missions={self._missions},
)"""

    @staticmethod
    def get_scenario_list(scenarios_or_scenarios_dirs: Sequence[str]) -> Sequence[str]:
        """Find all specific scenario directories in the directory trees of the initial scenario
        directory.
        """
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
        circular: bool = True,
    ) -> Generator["Scenario", None, None]:
        """Generate a cycle of scenario configurations.

        Args:
            scenarios_or_scenarios_dirs:
                A sequence of either the scenario to run (see scenarios/ for some samples you
                can use) OR a directory of scenarios to sample from.
            agents_to_be_briefed:
                Agent IDs that will be assigned a mission ("briefed" on a mission).
        Returns:
            A generator that serves up Scenarios.
        """
        scenario_roots = Scenario.get_scenario_list(scenarios_or_scenarios_dirs)
        if shuffle_scenarios:
            np.random.shuffle(scenario_roots)
        if circular:
            scenario_roots = cycle(scenario_roots)
        return Scenario.variations_for_all_scenario_roots(
            scenario_roots, agents_to_be_briefed, shuffle_scenarios
        )

    @staticmethod
    def variations_for_all_scenario_roots(
        scenario_roots, agents_to_be_briefed, shuffle_scenarios=True
    ) -> Generator["Scenario", None, None]:
        """Convert scenario roots to concrete scenarios.
        Args:
            scenario_roots:
                Scenario directories containing scenario resource files.
            agents_to_be_briefed:
                Agent IDs that will be assigned a mission ("briefed" on a mission).
            shuffle_scenarios:
                Return scenarios in a pseudo-random order.
        Returns:
            A generator that serves up Scenarios.
        """
        for scenario_root in scenario_roots:
            surface_patches = Scenario.discover_friction_map(scenario_root)

            agent_missions = Scenario.discover_agent_missions(
                scenario_root, agents_to_be_briefed
            )

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
            agent_missions = agent_missions or [None]
            if len(agents_to_be_briefed) > len(agent_missions):
                warnings.warn(
                    f"Scenario `{scenario_root}` has {len(agent_missions)} missions and"
                    f" but there are {len(agents_to_be_briefed)} agents to assign"
                    " missions to. The missions will be padded with random missions."
                )
            mission_agent_groups = combination_pairs_with_unique_indices(
                agents_to_be_briefed, agent_missions
            )
            social_agents = social_agents or [None]
            traffic_histories = Scenario.discover_traffic_histories(scenario_root) or [
                None
            ]
            traffic = Scenario.discover_traffic(scenario_root) or [[]]

            roll_traffic = 0
            roll_social_agents = 0
            roll_traffic_histories = 0

            if shuffle_scenarios:
                roll_traffic = random.randint(0, len(traffic))
                roll_social_agents = random.randint(0, len(social_agents))
                roll_traffic_histories = 0  # random.randint(0, len(traffic_histories))

            for (
                concrete_traffic,
                concrete_agent_missions,
                concrete_social_agents,
                concrete_traffic_history,
            ) in product(
                np.roll(traffic, roll_traffic, 0),
                mission_agent_groups,
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
                    traffic_specs=concrete_traffic,
                    missions={
                        **{a_id: mission for a_id, mission in concrete_agent_missions},
                        **concrete_social_agent_missions,
                    },
                    social_agents=concrete_social_agents,
                    surface_patches=surface_patches,
                    traffic_history=concrete_traffic_history,
                )

    @staticmethod
    def discover_agent_missions_count(scenario_root):
        """Retrieve the agent missions from the given scenario directory."""
        missions_file = os.path.join(scenario_root, "build", "missions.pkl")
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

        road_map, _ = Scenario.build_map(scenario_root)

        missions = []
        missions_file = os.path.join(scenario_root, "build", "missions.pkl")
        if os.path.exists(missions_file):
            with open(missions_file, "rb") as f:
                missions = pickle.load(f)

            missions = [
                Scenario._extract_mission(actor_and_mission.mission, road_map)
                for actor_and_mission in missions
            ]

        if not missions:
            missions = [None for _ in range(len(agents_to_be_briefed))]

        return missions

    @staticmethod
    def discover_friction_map(scenario_root) -> List[Dict[str, Any]]:
        """Returns the list of surface patches parameters defined in
        scenario file. Each element of the list contains the
        parameters of the specified surface patch.
        """
        surface_patches = []
        friction_map_file = os.path.join(scenario_root, "build", "friction_map.pkl")
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
    ) -> Sequence[Dict[str, Tuple[SocialAgent, Mission]]]:
        """Loops through the social agent mission pickles, instantiating corresponding
        implementations for the given types. The output is a list of
        {agent_id: (mission, locator)}, where each dictionary corresponds to the
        social agents to run for a given concrete Scenario (which translates to
        "per episode" when swapping).
        """
        scenario_root = (
            scenario.root_filepath if isinstance(scenario, Scenario) else scenario
        )
        road_map, _ = Scenario.build_map(scenario_root)

        social_agents_path = os.path.join(scenario_root, "build", "social_agents")
        if not os.path.exists(social_agents_path):
            return []

        # [ ( missions_file, agent_actor, Mission ) ]
        agent_bucketer = []

        # like dict.setdefault
        def setdefault(l: list, index: int, default):
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
        """Retrieve all specific scenarios in the directory tree of the given scenario directory.
        Args:
            scenario_or_scenario_dir:
                A directory that either immediately contains a scenario or the root of a directory tree that contains multiple scenarios.
        Returns:
            All specific scenarios.
        """
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
    def build_map(scenario_root: str) -> Tuple[Optional[RoadMap], Optional[str]]:
        """Builds a road map from the given scenario's resources."""
        # XXX: using a map builder_fn supplied by users is a security risk
        # as SMARTS will be executing the code "as is".  We are currently
        # trusting our users to not try to sabotage their own simulations.
        # In the future, this may need to be revisited if SMARTS is ever
        # shared in a multi-user mode.
        map_spec = Scenario.discover_map(scenario_root)
        return map_spec.builder_fn(map_spec)

    @staticmethod
    def discover_map(
        scenario_root: str,
        lanepoint_spacing: Optional[float] = None,
        default_lane_width: Optional[float] = None,
        shift_to_origin: bool = False,
    ) -> MapSpec:
        """Generates the map specification from the given scenario's file resources.

        Args:
            scenarios_root:
                A specific scenario to run (e.g. scenarios/sumo/loop)
            lanepoint_spacing:
                The distance between lanepoints that represent a lane's geometry.
            default_lane_width:
                The default width of a lane from its centre if it does not have a specific width.
            shift_to_origin:
                Shifts the map location to near the simulation origin so that the map contains (0, 0).
        Returns:
            A new map spec.
        """
        path = os.path.join(scenario_root, "build", "map", "map_spec.pkl")
        if not os.path.exists(path):
            # Use our default map builder if none specified by scenario...
            return MapSpec(
                scenario_root,
                lanepoint_spacing,
                default_lane_width,
                shift_to_origin,
            )
        with open(path, "rb") as f:
            road_map = cloudpickle.load(f)
            return road_map

    @staticmethod
    def discover_routes(scenario_root):
        """Discover the route files in the given scenario.

        >>> Scenario.discover_routes("scenarios/sumo/intersections/2lane")
        ['all.rou.xml', 'horizontal.rou.xml', 'turns.rou.xml', 'unprotected_left.rou.xml', 'vertical.rou.xml']
        >>> Scenario.discover_routes("scenarios/sumo/loop") # loop does not have any routes
        ['basic.rou.xml']
        """
        warnings.warn(
            "Scenario.discover_routes() has been deprecated in favor of Scenario.discover_traffic().  Please update your code.",
            category=DeprecationWarning,
        )
        return sorted(
            [
                os.path.basename(r)
                for r in glob.glob(
                    os.path.join(scenario_root, "build", "traffic", "*.rou.xml")
                )
            ]
        )

    @staticmethod
    def discover_traffic(scenario_root: str) -> List[Optional[List[str]]]:
        """Discover the traffic spec files in the given scenario."""
        traffic_path = os.path.join(scenario_root, "build", "traffic")
        # combine any SMARTS and SUMO traffic together...
        sumo_traffic = glob.glob(os.path.join(traffic_path, "*.rou.xml"))
        smarts_traffic = glob.glob(os.path.join(traffic_path, "*.smarts.xml"))
        if sumo_traffic and not smarts_traffic:
            return [[ts] for ts in sumo_traffic]
        elif not sumo_traffic and smarts_traffic:
            return [[ts] for ts in smarts_traffic]
        return [list(ts) for ts in product(sumo_traffic, smarts_traffic)]

    @staticmethod
    def _discover_bubbles(scenario_root):
        path = os.path.join(scenario_root, "build", "bubbles.pkl")
        if not os.path.exists(path):
            return []

        with open(path, "rb") as f:
            bubbles = pickle.load(f)
            return bubbles

    @staticmethod
    def _discover_metadata(scenario_root):
        path = os.path.join(scenario_root, "build", "scenario_metadata.pkl")
        if not os.path.exists(path):
            return dict()

        with open(path, "rb") as f:
            metadata = pickle.load(f)
            return metadata

    def set_ego_missions(self, ego_missions: Dict[str, Mission]):
        """Replaces the ego missions within the scenario.
        Args:
            ego_missions: Ego agent ids mapped to missions.
        """
        self._missions = ego_missions

    def get_vehicle_start_at_time(
        self, vehicle_id: str, start_time: float
    ) -> Tuple[Start, float]:
        """Returns a Start object that can be used to create a Mission for
        a vehicle from a traffic history dataset starting at its location
        at start_time.  Also returns its speed at that time."""
        pphs = self._traffic_history.vehicle_pose_at_time(vehicle_id, start_time)
        assert pphs
        pos_x, pos_y, heading, speed = pphs
        # missions start from front bumper, but pos is center of vehicle
        veh_dims = self._traffic_history.vehicle_dims(vehicle_id)
        hhx, hhy = radians_to_vec(heading) * (0.5 * veh_dims.length)
        return (
            Start(
                np.array([pos_x + hhx, pos_y + hhy]),
                Heading(heading),
                from_front_bumper=True,
            ),
            speed,
        )

    def get_vehicle_goal(self, vehicle_id: str) -> Point:
        """Get the final position for a history vehicle."""
        final_exit_time = self._traffic_history.vehicle_final_exit_time(vehicle_id)
        final_pose = self._traffic_history.vehicle_pose_at_time(
            vehicle_id, final_exit_time
        )
        assert final_pose
        final_pos_x, final_pos_y, _, _ = final_pose
        return Point(final_pos_x, final_pos_y)

    def discover_missions_of_traffic_histories(self) -> Dict[str, Mission]:
        """Retrieves the missions of traffic history vehicles."""
        vehicle_missions = {}
        for row in self._traffic_history.first_seen_times():
            v_id = str(row[0])
            start_time = float(row[1])
            start, speed = self.get_vehicle_start_at_time(v_id, start_time)
            entry_tactic = default_entry_tactic(speed)
            veh_config_type = self._traffic_history.vehicle_config_type(v_id)
            veh_dims = self._traffic_history.vehicle_dims(v_id)
            vehicle_missions[v_id] = Mission(
                start=start,
                entry_tactic=entry_tactic,
                goal=TraverseGoal(self.road_map),
                start_time=start_time,
                vehicle_spec=VehicleSpec(
                    veh_id=v_id,
                    veh_config_type=veh_config_type,
                    dimensions=veh_dims,
                ),
            )
        return vehicle_missions

    def create_dynamic_traffic_history_mission(
        self, veh_id: str, trigger_time: float, positional_radius: int
    ) -> Tuple[Mission, Mission]:
        """Builds missions out of the given vehicle information.
        Args:
            veh_id:
                The id of a vehicle in the traffic history dataset.
            trigger_time:
                The time that this mission should become active.
            positional_radius:
                The goal radius for the positional goal.
        Returns:
            (positional_mission, traverse_mission): A positional mission that follows the initial
             original vehicle's travel as well as a traverse style mission which is done when the
             vehicle leaves the map.
        """
        start, speed = self.get_vehicle_start_at_time(veh_id, trigger_time)
        veh_goal = self.get_vehicle_goal(veh_id)
        entry_tactic = default_entry_tactic(speed)
        # create a positional mission and a traverse mission
        positional_mission = Mission(
            start=start,
            entry_tactic=entry_tactic,
            start_time=0,
            goal=PositionalGoal(veh_goal, radius=positional_radius),
        )
        traverse_mission = Mission(
            start=start,
            entry_tactic=entry_tactic,
            start_time=0,
            goal=TraverseGoal(self._road_map),
        )
        return positional_mission, traverse_mission

    def history_missions_for_window(
        self,
        exists_at_or_after: float,
        ends_before: float,
        minimum_vehicle_window: float,
        filter: Optional[
            Callable[
                [Iterable[VehicleWindow]],
                Iterable[VehicleWindow],
            ]
        ] = None,
    ) -> Sequence[Mission]:
        """Discovers vehicle missions for the given window of time.

        :param exists_at_or_after: The starting time of any vehicles to query for.
        :type exists_at_or_after: float
        :param ends_before: The last point in time a vehicle should be in the simulation.
            Vehicles ending after that time are not considered.
        :type ends_before: float
        :param minimum_vehicle_window: The minimum time that a vehicle must be in the simulation
            to be considered for a mission.
        :type minimum_vehicle_window: float
        :param filter: A filter in the form of ``(func(Sequence[TrafficHistoryVehicleWindow]) -> Sequence[TrafficHistoryVehicleWindow])``,
            which passes in traffic vehicle information and then should be used purely to filter the sequence down.
        :return: A set of missions derived from the traffic history.
        :rtype: List[smarts.core.plan.Mission]
        """
        vehicle_windows = self._traffic_history.vehicle_windows_in_range(
            exists_at_or_after, ends_before, minimum_vehicle_window
        )

        def _gen_mission(vw: TrafficHistory.TrafficHistoryVehicleWindow):
            assert isinstance(
                vw, TrafficHistory.TrafficHistoryVehicleWindow
            ), "`filter(..)` likely returns malformed data."
            v_id = str(vw.vehicle_id)
            start_time = float(vw.start_time)
            start = Start(
                np.array(vw.axle_start_position),
                Heading(vw.start_heading),
            )
            entry_tactic = default_entry_tactic(vw.start_speed)
            veh_config_type = vw.vehicle_type
            veh_dims = vw.dimensions
            vehicle_mission = Mission(
                start=start,
                entry_tactic=entry_tactic,
                goal=TraverseGoal(self.road_map),
                start_time=start_time,
                vehicle_spec=VehicleSpec(
                    veh_id=v_id,
                    veh_config_type=veh_config_type,
                    dimensions=veh_dims,
                ),
            )
            return vehicle_mission

        if filter is not None:
            vehicle_windows = filter(vehicle_windows)
        return [_gen_mission(vw) for vw in vehicle_windows]

    @staticmethod
    def discover_traffic_histories(scenario_root: str):
        """Finds all existing traffic history files in the specific scenario."""
        build_dir = Path(scenario_root) / "build"
        return [
            entry
            for entry in os.scandir(str(build_dir))
            if entry.is_file() and entry.path.endswith(".shf")
        ]

    @staticmethod
    def _extract_mission(mission, road_map):
        """Takes a sstudio.types.(Mission, EndlessMission, etc.) and converts it to
        the corresponding SMARTS mission types.
        """

        def resolve_offset(offset: Union[str, float], lane_length: float):
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

        def to_position_and_heading(
            road_id: str, lane_index: int, offset: Union[str, float], road_map: RoadMap
        ):
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
                lane_width, _ = lane.width_at_offset(via.lane_offset)
                hit_distance = (
                    via.hit_distance if via.hit_distance > 0 else lane_width / 2
                )
                via_position = lane.from_lane_coord(RefLinePoint(via.lane_offset))

                s_vias.append(
                    Via(
                        lane_id=lane.lane_id,
                        lane_index=via.lane_index,
                        road_id=via.road_id,
                        position=tuple(via_position[:2]),
                        hit_distance=hit_distance,
                        required_speed=via.required_speed,
                    )
                )

            return tuple(s_vias)

        # XXX: For now we discard the route and just take the start and end to form our missions.
        # In the future, we could create a Plan object here too when there's a route specified.
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
                start_time=mission.start_time,
                entry_tactic=mission.entry_tactic,
                via=to_scenario_via(mission.via, road_map),
                num_laps=mission.num_laps,
                route_length=route.road_length,
            )

        raise RuntimeError(
            f"sstudio mission={mission} is an invalid type={type(mission)}"
        )

    @staticmethod
    def is_valid_scenario(scenario_root) -> bool:
        """Checks if the scenario_root directory matches our expected scenario structure

        >>> Scenario.is_valid_scenario("scenarios/sumo/loop")
        True
        >>> Scenario.is_valid_scenario("scenarios/non_existant")
        False
        """
        # just make sure we can load the map
        try:
            road_map, _ = Scenario.build_map(scenario_root)
        except FileNotFoundError:
            return False
        return road_map is not None

    @staticmethod
    def next(scenario_iterator, log_id="") -> "Scenario":
        """Utility to override specific attributes from a scenario iterator"""

        scenario = next(scenario_iterator)
        scenario._log_id = log_id
        return scenario

    @property
    def name(self) -> str:
        """The name of the scenario."""
        return os.path.normpath(self._root)

    @property
    def root_filepath(self) -> str:
        """The root directory of the scenario."""
        return self._root

    @property
    def surface_patches(self):
        """A list of surface areas with dynamics implications (e.g. icy road.)"""
        return self._surface_patches

    @property
    def road_map_hash(self):
        """A hash value of this road map."""
        return self._road_map_hash

    @property
    def plane_filepath(self) -> str:
        """The ground plane."""
        return os.path.join(self._root, "plane.urdf")

    @property
    def vehicle_filepath(self) -> Optional[str]:
        """The filepath of the vehicle's physics model."""
        if not os.path.isdir(self._root):
            return None
        for fname in os.listdir(self._root):
            if fname.endswith("vehicle.urdf"):
                return os.path.join(self._root, fname)
        return None

    @property
    def tire_parameters_filepath(self) -> str:
        """The path of the tire model's parameters."""
        return os.path.join(self._root, "tire_parameters.yaml")

    @property
    def controller_parameters_filepath(self) -> str:
        """The path of the vehicle controller parameters."""
        return os.path.join(self._root, "controller_parameters.yaml")

    @property
    def traffic_specs(self) -> Sequence[str]:
        """The traffic spec file names to use for this scenario."""
        return self._traffic_specs

    @property
    def route(self) -> Optional[str]:
        """The traffic route file name."""
        warnings.warn(
            "Scenario route property has been deprecated in favor of traffic_specs.  Please update your code.",
            category=DeprecationWarning,
        )
        assert len(self._traffic_specs) <= 1
        return (
            os.path.basename(self._traffic_specs[0])
            if len(self._traffic_specs) == 1
            else None
        )

    @property
    def route_files_enabled(self):
        """If there is a traffic route file."""
        warnings.warn(
            "Scenario route_file_enabled property has been deprecated in favor of traffic_specs.  Please update your code.",
            category=DeprecationWarning,
        )
        return bool(self._traffic_specs)

    @property
    def route_filepath(self):
        """The filepath to the traffic route file."""
        warnings.warn(
            "Scenario route_filepath property has been deprecated in favor of traffic_specs.  Please update your code.",
            category=DeprecationWarning,
        )
        assert len(self._traffic_specs) == 1
        return self._traffic_specs[0]

    @property
    def map_glb_filepath(self):
        """The map geometry filepath."""
        return os.path.join(self._root, "build", "map", "map.glb")

    @property
    def map_glb_metadata(self):
        """The metadata for the current map glb file."""
        metadata = self.map_glb_meta_for_file(self.map_glb_filepath)
        return metadata

    @staticmethod
    @lru_cache(1)
    def map_glb_meta_for_file(filepath):
        """The map metadata given a file."""

        scene = trimesh.load(filepath)
        return scene.metadata

    def unique_sumo_log_file(self):
        """A unique logging file for SUMO logging."""
        return os.path.join(self._log_dir, f"sumo-{str(uuid.uuid4())[:8]}")

    @property
    def road_map(self) -> RoadMap:
        """The road map of the scenario."""
        return self._road_map

    @property
    def supports_sumo_traffic(self) -> bool:
        """Returns True if this scenario uses a Sumo road network."""
        from smarts.core.sumo_road_network import SumoRoadNetwork

        return isinstance(self._road_map, SumoRoadNetwork)

    @staticmethod
    def any_support_sumo_traffic(scenarios: Sequence[str]) -> bool:
        """Determines if any of the given scenarios support Sumo traffic simulation."""
        from smarts.core.sumo_road_network import SumoRoadNetwork

        for scenario_root in Scenario.get_scenario_list(scenarios):
            try:
                road_map, _ = Scenario.build_map(scenario_root)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Unable to find network file in map_source={scenario_root}."
                )
            if isinstance(road_map, SumoRoadNetwork):
                return True

        return False

    @staticmethod
    def all_support_sumo_traffic(scenarios: Sequence[str]) -> bool:
        """Determines if all given scenarios support Sumo traffic simulation."""
        from smarts.core.sumo_road_network import SumoRoadNetwork

        for scenario_root in Scenario.get_scenario_list(scenarios):
            try:
                road_map, _ = Scenario.build_map(scenario_root)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Unable to find network file in map_source={scenario_root}."
                )
            if not isinstance(road_map, SumoRoadNetwork):
                return False

        return True

    @property
    def missions(self) -> Dict[str, Mission]:
        """Agent missions contained within this scenario."""
        return self._missions

    @property
    def social_agents(self) -> Dict[str, Tuple[Any, SocialAgent]]:
        """Managed social agents within this scenario."""
        return self._social_agents

    @property
    def bubbles(self):
        """Bubbles within this scenario."""
        return self._bubbles

    def mission(self, agent_id) -> Optional[Mission]:
        """Get the mission assigned to the given agent."""
        return self._missions.get(agent_id, None)

    def _resolve_log_dir(self, log_dir):
        if log_dir is None:
            log_dir = make_dir_in_smarts_log_dir("_sumo_run_logs")

        return os.path.abspath(log_dir)

    @property
    def traffic_history(self) -> Optional[TrafficHistory]:
        """Traffic history contained within this scenario."""
        return self._traffic_history

    @property
    def scenario_hash(self) -> str:
        """A hash of the scenario."""
        return self._scenario_hash

    @property
    def metadata(self) -> Dict:
        """Scenario metadata values.

        Returns:
            Dict: The values.
        """
        return self._metadata or {}
