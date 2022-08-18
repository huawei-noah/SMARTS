import argparse
import logging
import os
import pickle
from dataclasses import replace
from typing import Optional, Sequence

from PIL import Image, ImageDraw

from envision.client import Client as Envision
from smarts import sstudio
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
from smarts.core.controllers import ActionSpaceType, ControllerOutOfLaneException
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.plan import PositionalGoal
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

logging.basicConfig(level=logging.INFO)


class ObservationRecorder:
    def __init__(
        self,
        scenario: str,
        output_dir: Optional[str],
        seed: int = 42,
        agent_interface: Optional[AgentInterface] = None,
    ):
        """Generate Observations from the perspective of one or more
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
        """
        assert scenario, "--scenario must be used to specify a scenario"
        scenario_iter = Scenario.variations_for_all_scenario_roots([scenario], [])
        self._scenario = next(scenario_iter)
        assert self._scenario
        # TAI:  also record from social vehicles?
        assert self._scenario.traffic_history is not None

        self._logger = logging.getLogger(self.__class__.__name__)

        smarts_seed(seed)

        self._output_dir = output_dir
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._smarts = None
        self._create_missions()

        if agent_interface is not None:
            self.agent_interface = agent_interface
        else:
            self.agent_interface = self._create_default_interface()

    def _create_missions(self):
        self._missions = dict()
        orig_missions = self._scenario.discover_missions_of_traffic_histories()
        for v_id, mission in orig_missions.items():
            veh_goal = self._scenario._get_vehicle_goal(v_id)
            # TODO: get prefixed vehicle_id from TrafficHistoryProvider
            veh_id = f"history-vehicle-{v_id}"
            self._missions[veh_id] = replace(
                mission, goal=PositionalGoal(veh_goal, radius=3)
            )

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
            lidar=True,
            max_episode_steps=max_episode_steps,
            neighborhood_vehicles=True,
            ogm=OGM(
                width=img_pixels,
                height=img_pixels,
                resolution=img_meters / img_pixels,
            ),
            rgb=RGB(
                width=img_pixels,
                height=img_pixels,
                resolution=img_meters / img_pixels,
            ),
            road_waypoints=RoadWaypoints(horizon=road_waypoint_horizon),
            waypoints=Waypoints(lookahead=waypoints_lookahead),
        )

    def collect(
        self, vehicles_with_sensors: Optional[Sequence[int]], headless: bool = True
    ):
        """Generate Observations from the perspective of one or more
        social/history vehicles within a SMARTS scenario.

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

        self._max_sim_time = 0
        all_vehicles = set(self._scenario.traffic_history.all_vehicle_ids())
        for v_id in vehicles_with_sensors:
            if v_id not in all_vehicles:
                self._logger.warning(f"Vehicle {v_id} not in scenario")
                continue
            # TODO: get prefixed vehicle_id from TrafficHistoryProvider
            selected_vehicles.add(f"history-vehicle-{v_id}")
            exit_time = self._scenario.traffic_history.vehicle_final_exit_time(v_id)
            if exit_time > self._max_sim_time:
                self._max_sim_time = exit_time

        if not selected_vehicles:
            self._logger.error("No valid vehicles specified.  Aborting.")
            return

        _ = self._smarts.reset(self._scenario)
        current_vehicles = self._smarts.vehicle_index.social_vehicle_ids(
            vehicle_types=vehicle_types
        )
        self._record_data(
            collected_data,
            current_vehicles,
            off_road_vehicles,
            selected_vehicles,
        )

        while True:
            if self._smarts.elapsed_sim_time > self._max_sim_time:
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
            )

        if self._output_dir:
            # Save recorded observations as pickle files
            for car, data in collected_data.items():
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
    ):
        # Attach sensors to each vehicle
        valid_vehicles = (current_vehicles - off_road_vehicles) & selected_vehicles
        for veh_id in valid_vehicles:
            try:
                self._smarts.attach_sensors_to_vehicles(self.agent_interface, {veh_id})
            except ControllerOutOfLaneException:
                self._logger.warning(f"{veh_id} out of lane, skipped attaching sensors")
                off_road_vehicles.add(veh_id)

        # Get observations from each vehicle and record them
        obs = dict()
        obs, _, _, _ = self._smarts.observe_from(list(valid_vehicles))
        resolutions = {}
        for id_ in list(obs):
            if obs[id_].top_down_rgb:
                resolutions[id_] = obs[id_].top_down_rgb.metadata.resolution
            ego_state = obs[id_].ego_vehicle_state
            if ego_state.lane_index is None:
                del obs[id_]
            else:
                mission = self._missions[ego_state.id]
                if mission:
                    # doh! ego_state is immutable!
                    new_ego_state = ego_state._replace(mission=mission)
                    obs[id_] = replace(obs[id_], ego_vehicle_state=new_ego_state)
        # TODO: handle case where neighboring vehicle has lane_index of None too
        t = self._smarts.elapsed_sim_time
        for car, car_obs in obs.items():
            collected_data.setdefault(car, {}).setdefault(t, {})
            collected_data[car][t] = car_obs

        if not self._output_dir:
            return

        # Write top-down RGB image to a file for each vehicle if we have one
        for agent_id, agent_obs in obs.items():
            if agent_obs.top_down_rgb is not None:
                rgb_data = agent_obs.top_down_rgb.data
                h, w, _ = rgb_data.shape
                shape = (
                    (
                        h / 2 - 1.47 / 2 / resolutions[agent_id],
                        w / 2 - 3.68 / 2 / resolutions[agent_id],
                    ),
                    (
                        h / 2 + 1.47 / 2 / resolutions[agent_id],
                        w / 2 + 3.68 / 2 / resolutions[agent_id],
                    ),
                )
                img = Image.fromarray(rgb_data, "RGB")
                rect_image = ImageDraw.Draw(img)
                rect_image.rectangle(shape, fill="red")
                img.save(os.path.join(self._output_dir, f"{t}_{agent_id}.png"))


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--headless", help="Run the simulation in headless mode.", action="store_true"
    )
    args = parser.parse_args()

    sstudio.build_scenario([args.scenario])

    recorder = ObservationRecorder(
        scenario=args.scenario,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    recorder.collect(args.vehicles_with_sensors, headless=args.headless)
