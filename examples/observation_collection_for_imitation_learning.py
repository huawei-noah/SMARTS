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
        traffic_sim=SumoTrafficSimulation(headless=headless, auto_start=True),
        envision=None if headless else Envision(),
    )
    scenarios_iterator = Scenario.scenario_variations(
        scenarios,
        list([]),
    )

    smarts.reset(next(scenarios_iterator))

    prev_vehicles = set()
    done_vehicles = set()
    for _ in range(5000):
        smarts.step({})

        current_vehicles = smarts.vehicle_index.social_vehicle_ids()
        # We explicitly watch for which agent/vehicles left the simulation here
        # since we don't have a "done criteria" that detects when a vehicle's
        # traffic history has played itself out.
        done_vehicles = prev_vehicles - current_vehicles
        prev_vehicles = current_vehicles

        smarts.attach_sensors_to_vehicles(agent_spec, current_vehicles)
        obs, _, _, dones = smarts.observe_from(current_vehicles)
        # The `dones` returned above should be empty for traffic histories
        # where all vehicles are assumed to stay on the road and not collide.
        # TODO:  add the following assert once the maps are accurate enough that
        # we don't have any agents accidentally go off-road.
        # assert not done
        for v in done_vehicles:
            dones[f"Agent-{v}"] = True
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
