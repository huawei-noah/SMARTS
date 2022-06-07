import argparse
import logging
import os
import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional

from PIL import Image
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType, ControllerOutOfLaneException
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.waymo_map import WaymoMap
from smarts.env.hiway_env import HiWayEnv
from smarts.env.wrappers.format_obs import FormatObs
from smarts.sstudio import build_scenario
from smarts.sstudio.types import MapSpec
from smarts.zoo.agent_spec import AgentSpec

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

AGENT_SPEC = AgentSpec(
    interface=AgentInterface(
        accelerometer=True,
        action=ActionSpaceType.Continuous,
        neighborhood_vehicles=False,
        rgb=True,
        road_waypoints=False,
        waypoints=False,
    ),
)


def _scenario_code(dataset_path, scenario_id):
    return f"""from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

dataset_path = "{dataset_path}"
scenario_id = "{scenario_id}"

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"waymo",
        source_type="Waymo",
        input_path=dataset_path,
        scenario_id=scenario_id,
    )
]

gen_scenario(
    t.Scenario(
        map_spec=t.MapSpec(
            source=f"{{dataset_path}}#{{scenario_id}}", lanepoint_spacing=1.0
        ),
        traffic_histories=traffic_histories,
    ),
    output_dir=Path(__file__).parent,
)
"""


def _record_data(
    smarts,
    collected_data,
    output_dir,
    current_vehicles,
    off_road_vehicles,
    selected_vehicles,
):
    # Attach sensors to each vehicle
    valid_vehicles = (current_vehicles - off_road_vehicles) & selected_vehicles
    for veh_id in valid_vehicles:
        try:
            smarts.attach_sensors_to_vehicles(AGENT_SPEC.interface, {veh_id})
        except ControllerOutOfLaneException:
            logger.warning(f"{veh_id} out of lane, skipped attaching sensors")
            off_road_vehicles.add(veh_id)

    # Get observations from each vehicle and record them
    obs, _, _, _ = smarts.observe_from(list(valid_vehicles))
    t = smarts.elapsed_sim_time
    for car, car_obs in obs.items():
        collected_data.setdefault(car, {}).setdefault(t, {})
        collected_data[car][t] = car_obs
    logger.info(f"Sim time: {t}, active vehicles: {len(valid_vehicles)}")

    # Write top-down RGB image to a file for each vehicle
    for agent_id, agent_obs in obs.items():
        image_data = agent_obs.top_down_rgb.data
        img = Image.fromarray(image_data, "RGB")
        img.save(output_dir / f"{t}_{agent_id}.png")


def main(
    dataset_file: Path,
    scenario_id: str,
    output_dir: Path,
    scenarios_dir: Path,
    vehicle_ids: Optional[List[int]] = None,
    headless: bool = True,
):
    logger.info(f"Loading map from {dataset_file}, scenario {scenario_id}")
    map_source = f"{dataset_file}#{scenario_id}"
    map_spec = MapSpec(map_source, 1.0)
    road_map = WaymoMap.from_spec(map_spec)
    assert road_map is not None

    # Create & build a SMARTS scenario
    smarts_scenario_dir = scenarios_dir / f"waymo_{scenario_id}"
    if not os.path.exists(smarts_scenario_dir):
        os.mkdir(smarts_scenario_dir)

    with open(smarts_scenario_dir / "scenario.py", "w") as f:
        f.write(_scenario_code(dataset_file, scenario_id))

    scenarios = [str(smarts_scenario_dir)]
    build_scenario(scenarios)

    # Run the scenario with SMARTS
    env = HiWayEnv(
        scenarios=scenarios,
        agent_specs={"DummyAgent": AGENT_SPEC},
        sim_name="WaymoReplay",
        headless=headless,
    )
    env = FormatObs(env=env)
    smarts: SMARTS = env.env._smarts

    scenario_list = Scenario.get_scenario_list(scenarios)
    scenarios_iterator = Scenario.variations_for_all_scenario_roots(scenario_list, [])
    for scenario in scenarios_iterator:
        collected_data = {}
        vehicle_types = frozenset({"car"})
        off_road_vehicles = set()
        selected_vehicles = set()

        if vehicle_ids is None:
            ego_id = scenario.traffic_history.ego_vehicle_id
            selected_vehicles.add(f"history-vehicle-{ego_id}")
            logger.warning(
                f"No vehicle IDs specifed. Defaulting to ego vehicle ({ego_id})"
            )
        else:
            for v_id in vehicle_ids:
                selected_vehicles.add(f"history-vehicle-{v_id}")

        obs = env.reset()
        current_vehicles = smarts.vehicle_index.social_vehicle_ids(
            vehicle_types=vehicle_types
        )
        _record_data(
            smarts,
            collected_data,
            output_dir,
            current_vehicles,
            off_road_vehicles,
            selected_vehicles,
        )

        while True:
            env.step({})
            current_vehicles = smarts.vehicle_index.social_vehicle_ids(
                vehicle_types=vehicle_types
            )

            if collected_data and not current_vehicles:
                logger.info("No more vehicles. Exiting...")
                break

            _record_data(
                smarts,
                collected_data,
                output_dir,
                current_vehicles,
                off_road_vehicles,
                selected_vehicles,
            )

        # Save recorded observations as pickle files
        for car, data in collected_data.items():
            outfile = output_dir / f"{car}.pkl"
            with open(outfile, "wb") as of:
                pickle.dump(data, of)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("waymo_data_extraction.py")
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
    )
    parser.add_argument(
        "--vehicles", nargs="*", type=int, help="List of vehicle IDs to record"
    )

    # Parse arguments and ensure paths and directories are valid
    args = parser.parse_args()

    dataset_file = Path(args.dataset_file)
    if not os.path.exists(dataset_file):
        logger.error(f"Dataset file does not exist: {dataset_file}")
        exit(1)

    scenario_id = args.scenario_id
    assert type(scenario_id) == str

    output_base_dir = Path(args.output_dir)
    output_dir = output_base_dir / scenario_id
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with TemporaryDirectory() as scenarios_dir:
        main(
            dataset_file=dataset_file,
            scenario_id=scenario_id,
            output_dir=output_dir,
            scenarios_dir=Path(scenarios_dir),
            vehicle_ids=args.vehicles,
        )
