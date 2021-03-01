import logging

from envision.client import Client as Envision
from examples import default_argument_parser
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

logging.basicConfig(level=logging.INFO)


def main(scenarios, headless, seed):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner, max_episode_steps=None),
        agent_builder=None,
        observation_adapter=None,
    )

    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=SumoTrafficSimulation(headless=True, auto_start=True),
        envision=Envision(),
    )
    scenarios_iterator = Scenario.scenario_variations(
        scenarios,
        list([]),
    )

    smarts.reset(next(scenarios_iterator))

    for _ in range(5000):
        smarts.step({})
        smarts.attach_sensors_to_vehicles(
            agent_spec, smarts.vehicle_index.social_vehicle_ids()
        )
        obs, _, _, _ = smarts.observe_from(smarts.vehicle_index.social_vehicle_ids())
        # TODO: save observations for imitation learning


if __name__ == "__main__":
    parser = default_argument_parser("observation-collection-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
    )
