"""
This examples runs the OpEn Agent, a classical, MPC based agent using [OpEn](https://alphaville.github.io/optimization-engine/).

NOTE: You will need to install Rust to run this example. See https://rustup.rs/ for instructions.
"""

import importlib
import logging
import math

import gym

from smarts.core.utils.episodes import episodes

open_agent = importlib.import_module("zoo.policies.open-agent.open_agent")

from examples import default_argument_parser

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


def main(scenarios, sim_name, headless, num_episodes, seed):
    open_agent_spec = open_agent.entrypoint(debug=False, aggressiveness=3)
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: open_agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
        # envision_record_data_replay_path="./data_replay",
    )

    for episode in episodes(n=num_episodes):
        agent = open_agent_spec.build_agent()

        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

        del agent

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("OpEn-trajectory-optimizer-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
