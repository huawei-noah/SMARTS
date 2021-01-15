import logging
import time

from smarts.core.smarts import SMARTS
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.scenario import Scenario
from envision.client import Client as Envision
from smarts.core.utils.logging import timeit

from examples import default_argument_parser


logging.basicConfig(level=logging.INFO)


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def main(scenarios, headless, seed):
    scenarios_iterator = Scenario.scenario_variations(scenarios, [])
    for _ in scenarios:
        scenario = next(scenarios_iterator)
        agent_missions = scenario.discover_missions_of_traffic_histories()
        print(f"All id length: {len(agent_missions.keys())}")
        for agent_id, mission in agent_missions.items():
            # if agent_id not in set(["1","2","3","7","10","11"]):
            #     continue
            scenario.set_ego_missions({agent_id: mission})

            agent_spec = AgentSpec(
                interface=AgentInterface.from_type(
                    AgentType.Laner, max_episode_steps=None
                ),
                agent_builder=KeepLaneAgent,
            )

            agent = agent_spec.build_agent()
            with timeit("Setup smarts"):
                smarts = SMARTS(
                    agent_interfaces={agent_id: agent_spec.interface},
                    traffic_sim=SumoTrafficSimulation(headless=True, auto_start=True),
                    envision=Envision(),
                )
            observations = smarts.reset(scenario)
            print(f"id: {agent_id}")
            dones = {agent_id: False}
            while not dones[agent_id]:
                agent_obs = observations[agent_id]
                agent_action = agent.act(agent_obs)

                observations, rewards, dones, infos = smarts.step(
                    {agent_id: agent_action}
                )
            print("\n")
            smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("history-vehicles-replacement-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios, headless=args.headless, seed=args.seed,
    )
