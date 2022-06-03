import logging
import os
from pathlib import Path
import pickle
from typing import Any, Dict, Sequence

from envision.client import Client as Envision
from examples.argument_parser import default_argument_parser
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ControllerOutOfLaneException
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.math import radians_to_vec
from smarts.sstudio.types import MapSpec
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.waymo_map import WaymoMap


def main(
    script: str,
    dataset_file: Path,
    scenario_id: str,
    output_dir: Path,
    scenario_dir: Path,
):
    logger = logging.getLogger(script)
    logger.setLevel(logging.INFO)

    map_source = f"{dataset_file}#{scenario_id}"
    map_spec = MapSpec(map_source, 1.0)
    road_map = WaymoMap.from_spec(map_spec)

    # agent_spec = AgentSpec(
    #     interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
    #     agent_builder=None,
    # )

    # smarts = SMARTS(
    #     agent_interfaces={},
    #     traffic_sim=SumoTrafficSimulation(headless=headless, auto_start=True),
    #     envision=None if headless else Envision(),
    # )

    # scenario_list = Scenario.get_scenario_list(scenarios)
    # scenarios_iterator = Scenario.variations_for_all_scenario_roots(scenario_list, [])
    # for scenario in scenarios_iterator:
    #     obs = smarts.reset(scenario)

    #     collected_data = {}
    #     _record_data(smarts.elapsed_sim_time, obs, collected_data)

    #     # could also include "motorcycle" or "truck" in this set if desired
    #     vehicle_types = frozenset({"car"})

    #     # filter off-road vehicles from observations
    #     vehicles_off_road = set()

    #     while True:
    #         smarts.step({})
    #         current_vehicles = smarts.vehicle_index.social_vehicle_ids(
    #             vehicle_types=vehicle_types
    #         )

    #         if collected_data and not current_vehicles:
    #             print("no more vehicles.  exiting...")
    #             break

    #         for veh_id in current_vehicles:
    #             try:
    #                 smarts.attach_sensors_to_vehicles(agent_spec.interface, {veh_id})
    #             except ControllerOutOfLaneException:
    #                 logger.warning(f"{veh_id} out of lane, skipped attaching sensors")
    #                 vehicles_off_road.add(veh_id)

    #         valid_vehicles = {v for v in current_vehicles if v not in vehicles_off_road}
    #         obs, _, _, dones = smarts.observe_from(list(valid_vehicles))
    #         _record_data(smarts.elapsed_sim_time, obs, collected_data)

    #     # an example of how we might save the data per car
    #     observation_folder = "collected_observations"
    #     if not os.path.exists(observation_folder):
    #         os.makedirs(observation_folder)
    #     for car, data in collected_data.items():
    #         outfile = f"{observation_folder}/{scenario.name}_{scenario.traffic_history.name}_{car}.pkl"
    #         with open(outfile, "wb") as of:
    #             pickle.dump(data, of)

    # smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("waymo-data-extraction")
    parser.add_argument(
        "dataset_file",
        help="Path to the TFRecord file",
        type=str,
    )
    parser.add_argument(
        "scenario_id",
        help="ID of the scenario to extract",
        type=str,
    )
    parser.add_argument(
        "output_dir",
        help="Path to the directory to store the output files",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    scenario_dir = Path(__file__).parent.absolute() / "waymo_scenarios"
    if not os.path.exists(scenario_dir):
        os.mkdir(scenario_dir)

    main(
        script=parser.prog,
        dataset_file=args.dataset_file,
        scenario_id=args.scenario_id,
        output_dir=args.output_dir,
        scenario_dir=scenario_dir,
    )
