import logging

import gym
import numpy

from smarts.core.utils.episodes import episodes
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, Agent
from smarts.core.sensors import Observation

from examples import default_argument_parser


logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class TrajectoryTestAgent(Agent):
    def act(self, obs: Observation):
        return numpy.array(
            [
                [0, 0.1, 0.2, 0.3, 30],
                [151.16, 152.08, 150.96, 150.96, 150.96,],
                [-83.14, -94.51, -60.71, -66.71, -53.71,],
                [0, numpy.pi / 2, numpy.pi, numpy.pi * 3 / 2, numpy.pi * 4],
                [0, 0, 0, 0, 0],
            ]
        )


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.TrajectoryInterpolator, max_episode_steps=max_episode_steps
        ),
        agent_builder=TrajectoryTestAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=False,
        sumo_auto_start=False,
        seed=seed,
        # zoo_addrs=[("10.193.241.236", 7432)], # Sample server address (ip, port), to distribute social agents in remote server.
        # envision_record_data_replay_path="./data_replay",
    )

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
