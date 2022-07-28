import logging
import os
import pickle
from typing import Any, Dict, Sequence

from envision.client import Client as Envision
from examples.utility.argument_parser import default_argument_parser
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers import ControllerOutOfLaneException
from smarts.core.local_traffic_provider import LocalTrafficProvider
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.math import radians_to_vec
from smarts.zoo.agent_spec import AgentSpec

logging.basicConfig(level=logging.INFO)


def _record_data(
    t: float,
    obs: Dict[str, Observation],
    collected_data: Dict[str, Dict[float, Dict[str, Any]]],
):
    # just a hypothetical example of how we might collect some observations to save...
    for car, car_obs in obs.items():
        car_state = car_obs.ego_vehicle_state
        collected_data.setdefault(car, {}).setdefault(t, {})
        collected_data[car][t]["ego_pos"] = car_state.position
        collected_data[car][t]["heading"] = car_state.heading
        collected_data[car][t]["speed"] = car_state.speed
        collected_data[car][t]["angular_velocity"] = car_state.yaw_rate
        # note: acceleration is a 3-vector. convert it here to a scalar
        # keeping only the acceleration in the direction of travel (the heading).
        # we will miss angular acceleration effects, but hopefully angular velocity
        # will be enough to "keep things real".  This is simpler than using
        # the angular acceleration vector because there are less degrees of
        # freedom in the resulting model.
        heading_vector = radians_to_vec(car_state.heading)
        acc_scalar = car_state.linear_acceleration[:2].dot(heading_vector)
        collected_data[car][t]["acceleration"] = acc_scalar


def main(script: str, scenarios: Sequence[str], headless: bool, seed: int):
    logger = logging.getLogger(script)
    logger.setLevel(logging.INFO)

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
        agent_builder=None,
    )

    # In case we have any bubbles or additional non-history traffic
    # in one of the scenarios, we need to add some traffic providers.
    traffic_sims = []
    if Scenario.any_support_sumo_traffic(scenarios):
        sumo_traffic = SumoTrafficSimulation(headless=headless, auto_start=True)
        traffic_sims += [sumo_traffic]
    smarts_traffic = LocalTrafficProvider()
    traffic_sims += [smarts_traffic]

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sims=traffic_sims,
        envision=None if headless else Envision(),
    )

    scenario_list = Scenario.get_scenario_list(scenarios)
    scenarios_iterator = Scenario.variations_for_all_scenario_roots(scenario_list, [])
    for scenario in scenarios_iterator:
        obs = smarts.reset(scenario)

        collected_data = {}
        _record_data(smarts.elapsed_sim_time, obs, collected_data)

        # could also include "motorcycle" or "truck" in this set if desired
        vehicle_types = frozenset({"car"})

        while True:
            smarts.step({})
            history_vehicles = smarts.vehicle_index.social_vehicle_ids(
                vehicle_types=vehicle_types
            )
            history_vehicles = {
                v
                for v in history_vehicles
                if smarts.traffic_history_provider.manages_vehicle(v)
            }

            if collected_data and not history_vehicles:
                print("no more vehicles.  exiting...")
                break

            # filter off-road vehicles from observations
            vehicles_off_road = set()

            for veh_id in history_vehicles:
                try:
                    smarts.attach_sensors_to_vehicles(agent_spec.interface, {veh_id})
                except ControllerOutOfLaneException:
                    logger.warning(f"{veh_id} out of lane, skipped attaching sensors")
                    vehicles_off_road.add(veh_id)

            valid_vehicles = {v for v in history_vehicles if v not in vehicles_off_road}
            obs, _, _, dones = smarts.observe_from(list(valid_vehicles))
            _record_data(smarts.elapsed_sim_time, obs, collected_data)

        # an example of how we might save the data per car
        observation_folder = "collected_observations"
        if not os.path.exists(observation_folder):
            os.makedirs(observation_folder)
        for car, data in collected_data.items():
            outfile = f"{observation_folder}/{scenario.name}_{scenario.traffic_history.name}_{car}.pkl"
            with open(outfile, "wb") as of:
                pickle.dump(data, of)

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("traffic-histories-to-observations-example")
    args = parser.parse_args()

    main(
        script=parser.prog,
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
    )
