import logging
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

    smarts.reset(next(scenarios_iterator))

    # could also include "motorcycle" or "truck" in this set if desired
    vehicle_types = frozenset({"car"})

    for _ in range(5000):
        smarts.step({})
        current_vehicles = smarts.vehicle_index.social_vehicle_ids(
            vehicle_types=vehicle_types
        )
        smarts.attach_sensors_to_vehicles(agent_spec, current_vehicles)
        obs, _, _, dones = smarts.observe_from(current_vehicles)
        # TODO: save observations for imitation learning

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("observation-collection-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
    )
