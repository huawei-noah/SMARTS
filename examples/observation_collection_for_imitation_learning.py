import logging
import pickle
from typing import Any, Dict, Sequence

from envision.client import Client as Envision
from examples.argument_parser import default_argument_parser
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.utils.math import radians_to_vec

logging.basicConfig(level=logging.INFO)


def _record_data(
    t: float, obs: Observation, collected_data: Dict[str, Dict[float, Dict[str, Any]]]
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


def main(scenarios: Sequence[str], headless: bool, seed: int):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
        agent_builder=None,
        observation_adapter=None,
    )

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=SumoTrafficSimulation(headless=headless, auto_start=True),
        envision=None if headless else Envision(),
    )
    scenarios_iterator = Scenario.scenario_variations(
        scenarios,
        list([]),
    )

    scenario = next(scenarios_iterator)
    obs = smarts.reset(scenario)

    collected_data = {}
    _record_data(smarts.elapsed_sim_time, obs, collected_data)

    # could also include "motorcycle" or "truck" in this set if desired
    vehicle_types = frozenset({"car"})

    while True:
        smarts.step({})
        current_vehicles = smarts.vehicle_index.social_vehicle_ids(
            vehicle_types=vehicle_types
        )

        if collected_data and not current_vehicles:
            print("no more vehicles.  exiting...")
            break

        smarts.attach_sensors_to_vehicles(agent_spec, current_vehicles)
        obs, _, _, dones = smarts.observe_from(current_vehicles)

        _record_data(smarts.elapsed_sim_time, obs, collected_data)

    # an example of how we might save the data per car
    for car, data in collected_data.items():
        outfile = f"data_{scenario.name}_{scenario.traffic_history.name}_{car}.pkl"
        with open(outfile, "wb") as of:
            pickle.dump(data, of)

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("observation-collection-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
    )
