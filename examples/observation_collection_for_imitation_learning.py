import logging
import pickle
from typing import Sequence

from envision.client import Client as Envision
from examples.argument_parser import default_argument_parser
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

logging.basicConfig(level=logging.INFO)


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
    smarts.reset(scenario)

    collected_data = {}

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

        # just a hypothetical example of how we might collect some observations to save...
        for car, car_obs in obs.items():
            car_state = car_obs.ego_vehicle_state
            t = smarts.elapsed_sim_time
            collected_data.setdefault(car, {}).setdefault(t, {})
            collected_data[car][t]["ego_pos"] = car_state.position
            collected_data[car][t]["heading"] = car_state.heading
            collected_data[car][t]["speed"] = car_state.speed
            collected_data[car][t]["angular_velocity"] = car_state.yaw_rate
            # note: acceleration is a 3-vector...
            collected_data[car][t]["acceleration"] = car_state.linear_acceleration

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
