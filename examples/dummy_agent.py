import logging

from envision.client import Client as Envision
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS

from examples.argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class KeepLaneAgent(Agent):
    def act(self, obs):
        return "keep_lane"


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    logger = logging.getLogger(sim_name)

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        ),
        agent_builder=KeepLaneAgent,
    )

    smarts = SMARTS(
        agent_interfaces={AGENT_ID: agent_spec.interface},
        traffic_sim=None,
        envision=None if headless else Envision(),
        fixed_timestep_sec=0.1,
    )

    scenarios_iterator = Scenario.scenario_variations(scenarios, [AGENT_ID])

    for i in range(num_episodes):
        logger.warning(f"starting episode {i}...")

        agent = agent_spec.build_agent()
        scenario = next(scenarios_iterator)
        obs = smarts.reset(scenario)

        done = False
        while not done:
            obs = agent_spec.observation_adapter(obs[AGENT_ID])
            action = agent.act(obs)
            action = agent_spec.action_adapter(action)
            obs, _, dones, _ = smarts.step({AGENT_ID: action})
            done = dones[AGENT_ID]

    smarts.destroy()


if __name__ == "__main__":
    parser = default_argument_parser("dummy-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
