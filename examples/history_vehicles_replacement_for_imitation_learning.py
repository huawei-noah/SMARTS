import logging

from envision.client import Client as Envision
from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation

logging.basicConfig(level=logging.INFO)


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def main(scenarios, headless, seed):
    scenarios_iterator = Scenario.scenario_variations(scenarios, [])
    smarts = SMARTS(
        agent_interfaces={},
        traffic_sim=SumoTrafficSimulation(headless=True, auto_start=True),
        envision=Envision(),
    )

    for _ in scenarios:
        scenario = next(scenarios_iterator)
        agent_missions = scenario.discover_missions_of_traffic_histories()

        for agent_id, mission in agent_missions.items():
            scenario.set_ego_missions({agent_id: mission})

            agent_spec = AgentSpec(
                interface=AgentInterface.from_type(
                    AgentType.Laner, max_episode_steps=None
                ),
                agent_builder=KeepLaneAgent,
            )
            agent = agent_spec.build_agent()

            smarts.switch_ego_agent({agent_id: agent_spec.interface})

            observations = smarts.reset(scenario)

            dones = {agent_id: False}
            while not dones[agent_id]:
                agent_obs = observations[agent_id]
                agent_action = agent.act(agent_obs)

                observations, rewards, dones, infos = smarts.step(
                    {agent_id: agent_action}
                )

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("history-vehicles-replacement-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        seed=args.seed,
    )
