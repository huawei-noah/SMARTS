import logging
from dataclasses import replace

from envision.client import Client as Envision
from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Mission, Scenario
from smarts.core.smarts import SMARTS
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation
from smarts.core.traffic_history_provider import TrafficHistoryProvider

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
            agent_spec = AgentSpec(
                interface=AgentInterface.from_type(
                    AgentType.Laner, max_episode_steps=None
                ),
                agent_builder=KeepLaneAgent,
            )
            agent = agent_spec.build_agent()

            smarts.switch_ego_agent({agent_id: agent_spec.interface})
            # required: get traffic_history_provider and set time offset
            traffic_history_provider = smarts.get_provider_by_type(
                TrafficHistoryProvider
            )
            assert traffic_history_provider
            traffic_history_provider.set_start_time(mission.start_time)

            modified_mission = replace(mission, start_time=0.0)
            scenario.set_ego_missions({agent_id: modified_mission})
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
