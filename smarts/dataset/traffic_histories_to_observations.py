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

import argparse
import logging
import math
import os
import pickle
from dataclasses import replace
from typing import Optional, Sequence

import numpy as np
from PIL import Image

from envision.client import Client as Envision
from smarts.core import seed as smarts_seed
from smarts.core.agent_interface import (
    OGM,
    RGB,
    AgentInterface,
    DoneCriteria,
    DrivableAreaGridMap,
    RoadWaypoints,
    Waypoints,
)
from smarts.core.colors import Colors
from smarts.core.controllers import ActionSpaceType, ControllerOutOfLaneException
from smarts.core.coordinates import Point
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.plan import PositionalGoal
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.vehicle import VEHICLE_CONFIGS
from smarts.sstudio.scenario_construction import build_scenario


class ObservationRecorder:
    """
    Generate SMARTS observations from the perspective of one or more
    social/history vehicles within a SMARTS scenario.

    Args:
        scenario (str):
            A path to a scenario to run.
            Note:  the scenario should already have been built using
            `scl scenario build ...`.
        output_dir (str):
            Path to the directory for the output files.
            Will be created if necessary.
        seed (int):
            Seed for random number generation.  Default:  42.
        agent_interface (AgentInterface, optional):
            Agent interface to be used for recorded vehicles. If not provided,
            will use a default interface with all sensors enabled.
        start_time (float, Optional):
            The start time (in seconds) of the window within which observations should be recorded."
        end_time (float, Optional):
            The end time (in seconds) of the window within which observations should be recorded."

    """

    def __init__(
        self,
        scenario: str,
        output_dir: Optional[str],
        seed: int = 42,
        agent_interface: Optional[AgentInterface] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ):
        assert scenario, "--scenario must be used to specify a scenario"
        scenario_iter = Scenario.variations_for_all_scenario_roots([scenario], [])
        self._scenario = next(scenario_iter)
        self._start_time = start_time if start_time is not None else 0.0
        self._end_time = end_time
        assert self._scenario
        # TAI:  also record from social vehicles?
        assert self._scenario.traffic_history is not None

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.setLevel(logging.INFO)

        smarts_seed(seed)

        self._output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not output_dir:
            self._logger.warning(
                "No output dir provided. Observations will not be saved."
            )
        self._smarts = None

        if agent_interface is not None:
            self.agent_interface = agent_interface
        else:
            self.agent_interface = self._create_default_interface()

    def _create_default_interface(
        self, img_meters: int = 64, img_pixels: int = 256, action_space="TargetPose"
    ) -> AgentInterface:
        # In future, allow for explicit mapping of vehicle_ids to agent_ids.
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
            action=ActionSpaceType[action_space],
            done_criteria=done_criteria,
            drivable_area_grid_map=DrivableAreaGridMap(
                width=img_pixels,
                height=img_pixels,
                resolution=img_meters / img_pixels,
            ),
            lidar_point_cloud=True,
            max_episode_steps=max_episode_steps,
            neighborhood_vehicle_states=True,
            occupancy_grid_map=OGM(
                width=img_pixels,
                height=img_pixels,
                resolution=img_meters / img_pixels,
            ),
            top_down_rgb=RGB(
                width=img_pixels,
                height=img_pixels,
                resolution=img_meters / img_pixels,
            ),
            road_waypoints=RoadWaypoints(horizon=road_waypoint_horizon),
            waypoint_paths=Waypoints(lookahead=waypoints_lookahead),
        )

    def collect(
        self, vehicles_with_sensors: Optional[Sequence[int]], headless: bool = True
    ):
        """Records SMARTS observations for selected vehicles.

        Args:
            vehicles_with_sensors (Sequence[int], optional):
                A list of vehicle_ids within the scenario to which to attach
                sensors and record Observations.  If not specified, this will default
                the ego vehicle of the scenario if there is one.  If not,
                this will default to all vehicles in the scenario.
            headless (bool, optional):
                Whether to run the simulation in headless mode.  Defaults to True.
        """
        assert self._scenario and self._scenario.traffic_history is not None

        # In case we have any bubbles or additional non-history traffic
        # in the scenario, we need to add some traffic providers.
        traffic_sims = []
        if self._scenario.supports_sumo_traffic:
            sumo_traffic = SumoTrafficSimulation()
            traffic_sims += [sumo_traffic]
        smarts_traffic = LocalTrafficProvider()
        traffic_sims += [smarts_traffic]

        # The actual SMARTS instance to be used for the simulation
        self._smarts = SMARTS(
            agent_interfaces=dict(),
            traffic_sims=traffic_sims,
            envision=None if headless else Envision(),
        )

        # could also include "motorcycle" or "truck" in this set if desired
        vehicle_types = frozenset({"car"})

        collected_data = {}
        off_road_vehicles = set()
        selected_vehicles = set()

        if not vehicles_with_sensors:
            ego_id = self._scenario.traffic_history.ego_vehicle_id
            if ego_id is not None:
                vehicles_with_sensors = [ego_id]
                self._logger.warning(
                    f"No vehicle IDs specifed. Defaulting to ego vehicle ({ego_id})"
                )
            else:
                vehicles_with_sensors = self._scenario.traffic_history.all_vehicle_ids()
                self._logger.warning(
                    f"No vehicle IDs specifed. Defaulting to all vehicles"
                )

        max_sim_time = 0
        all_vehicles = set(self._scenario.traffic_history.all_vehicle_ids())
        for v_id in vehicles_with_sensors:
            if v_id not in all_vehicles:
                self._logger.warning(f"Vehicle {v_id} not in scenario")
                continue
            config_type = self._scenario.traffic_history.vehicle_config_type(v_id)
            veh_type = (
                VEHICLE_CONFIGS[config_type].vehicle_type
                if config_type in VEHICLE_CONFIGS
                else None
            )
            if veh_type not in vehicle_types:
                self._logger.warning(
                    f"Vehicle type for vehicle {v_id} ({veh_type}) not in selected vehicle types ({vehicle_types})"
                )
                continue

            # TODO: get prefixed vehicle_id from TrafficHistoryProvider
            selected_vehicles.add(f"history-vehicle-{v_id}")
            exit_time = self._scenario.traffic_history.vehicle_final_exit_time(v_id)
            if exit_time > max_sim_time:
                max_sim_time = exit_time

        if not selected_vehicles:
            self._logger.error("No valid vehicles specified.  Aborting.")
            return

        _ = self._smarts.reset(self._scenario, self._start_time)
        current_vehicles = self._smarts.vehicle_index.social_vehicle_ids(
            vehicle_types=vehicle_types
        )
        self._record_data(
            collected_data,
            current_vehicles,
            off_road_vehicles,
            selected_vehicles,
            max_sim_time,
        )

        while True:
            if self._smarts.elapsed_sim_time > max_sim_time:
                self._logger.info("All observed vehicles are finished. Exiting...")
                break
            self._smarts.step({})
            current_vehicles = self._smarts.vehicle_index.social_vehicle_ids(
                vehicle_types=vehicle_types
            )
            if collected_data and not current_vehicles:
                self._logger.info("No more vehicles. Exiting...")
                break

            self._record_data(
                collected_data,
                current_vehicles,
                off_road_vehicles,
                selected_vehicles,
                max_sim_time,
            )

        if self._output_dir:
            # Get original missions for all vehicles
            missions = dict()
            orig_missions = self._scenario.discover_missions_of_traffic_histories()
            for v_id, mission in orig_missions.items():
                # TODO: get prefixed vehicle_id from TrafficHistoryProvider
                veh_id = f"history-vehicle-{v_id}"
                missions[veh_id] = mission

            # Save recorded observations as pickle files
            for car, data in collected_data.items():
                # Fill in mission with proper goal position for all observations
                last_t = max(data.keys())
                last_state = data[last_t].ego_vehicle_state
                goal_pos = Point(last_state.position[0], last_state.position[1])
                new_mission = replace(
                    missions[last_state.id], goal=PositionalGoal(goal_pos, radius=3)
                )
                for t in data.keys():
                    ego_state = data[t].ego_vehicle_state
                    new_ego_state = ego_state._replace(mission=new_mission)
                    data[t] = data[t]._replace(ego_vehicle_state=new_ego_state)

                # Create terminal state for last timestep, when the vehicle reaches the goal
                events = data[last_t].events
                new_events = events._replace(reached_goal=True)
                data[last_t] = data[last_t]._replace(events=new_events)

                outfile = os.path.join(
                    self._output_dir,
                    f"{car}.pkl",
                )
                with open(outfile, "wb") as of:
                    pickle.dump(data, of)

        self._smarts.destroy()

    def _record_data(
        self,
        collected_data,
        current_vehicles,
        off_road_vehicles,
        selected_vehicles,
        max_sim_time,
    ):
        # Record only within specified time window.
        t = self._smarts.elapsed_sim_time
        end_time = self._end_time if self._end_time is not None else max_sim_time
        if not (self._start_time <= t <= end_time):
            return

        # Attach sensors to each vehicle.
        valid_vehicles = (current_vehicles - off_road_vehicles) & selected_vehicles
        for veh_id in valid_vehicles:
            try:
                self._smarts.attach_sensors_to_vehicles(self.agent_interface, {veh_id})
            except ControllerOutOfLaneException:
                self._logger.warning(f"{veh_id} out of lane, skipped attaching sensors")
                off_road_vehicles.add(veh_id)

        # Get observations from each vehicle and record them.
        obs = dict()
        obs, _, _, _ = self._smarts.observe_from(list(valid_vehicles))
        self._logger.debug(f"t={t}, active_vehicles={len(valid_vehicles)}")
        for id_ in list(obs):
            ego_state = obs[id_].ego_vehicle_state
            if ego_state.lane_index is None:
                del obs[id_]
                continue

            top_down_rgb = obs[id_].top_down_rgb
            if top_down_rgb:
                res = top_down_rgb.metadata.resolution
                rgb = top_down_rgb.data.copy()
                h, w, _ = rgb.shape
                shape = (
                    (
                        math.floor(w / 2 - 3.68 / 2 / res),
                        math.ceil(w / 2 + 3.68 / 2 / res),
                    ),
                    (
                        math.floor(h / 2 - 1.47 / 2 / res),
                        math.ceil(h / 2 + 1.47 / 2 / res),
                    ),
                )
                color = np.array(Colors.Red.value[0:3], ndmin=3) * 255
                rgb[shape[0][0] : shape[0][1], shape[1][0] : shape[1][1], :] = color
                top_down_rgb_edited = top_down_rgb._replace(data=rgb)
                obs[id_] = obs[id_]._replace(top_down_rgb=top_down_rgb_edited)

                if self._output_dir:
                    img = Image.fromarray(rgb, "RGB")
                    img.save(os.path.join(self._output_dir, f"{t}_{id_}.png"))

        # TODO: handle case where neighboring vehicle has lane_index of None too
        for car, car_obs in obs.items():
            collected_data.setdefault(car, {})
            collected_data[car][t] = car_obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser("traffic-histories-to-observations")
    parser.add_argument(
        "scenario",
        help="The path to a SMARTS scenario to run.",
        type=str,
    )
    parser.add_argument(
        "--vehicles_with_sensors",
        "-v",
        help="A list of vehicle IDs to attach sensors to record observations from.  If none specified, defaults to the ego vehicle of the scenario if there is one; if not, defaults to all vehicles in the scenario.",
        type=int,
        nargs="*",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        help="Path into which to write collected observations.  Will be created if necessary.  If not specified, observations will not be dumped.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--start_time",
        help="The start time (in seconds) of the window within which observations should be recorded.",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--end_time",
        help="The end time (in seconds) of the window within which observations should be recorded.",
        type=float,
        default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--headless", help="Run the simulation in headless mode.", action="store_true"
    )
    parser.add_argument(
        "--no_build",
        help="Do not automatically build the scenario each time the script is run.",
        action="store_true",
    )
    args = parser.parse_args()

    if not args.no_build:
        build_scenario(clean=False, scenario=args.scenario, seed=args.seed)

    recorder = ObservationRecorder(
        scenario=args.scenario,
        output_dir=args.output_dir,
        seed=args.seed,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    recorder.collect(args.vehicles_with_sensors, headless=args.headless)
