import logging
import sys

import gym
from typing_extensions import final

from examples import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.zoo.registry import make as zoo_make

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


def main(scenarios, sim_name, headless, seed, speed, save_dir, write):
    from zoo import policies

    policies.replay_save_dir = save_dir
    policies.replay_read = not write
    agent_spec = zoo_make(
        "zoo.policies:replay-agent-v0",
        save_directory=save_dir,
        id="agent_007",
        wrapped_agent_locator="zoo.policies:keep-left-with-speed-agent-v0",
        wrapped_agent_params={"speed": speed},
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
        # zoo_addrs=[("10.193.241.236", 7432)], # Sample server address (ip, port), to distribute social agents in remote server.
        # envision_record_data_replay_path="./data_replay",
    )

    episode = next(episodes(n=1))
    agent = agent_spec.build_agent()
    observations = env.reset()

    dones = {"__all__": False}
    i = 0
    try:
        while not dones["__all__"] and i < 1500:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            i += 1
            print("Step: ", i)
            episode.record_step(observations, rewards, dones, infos)
    except KeyboardInterrupt:
        # discard result
        i = 2550
    finally:
        try:
            episode.record_scenario(env.scenario_log)
            env.close()
        finally:
            sys.exit(i // 10)


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    parser.add_argument(
        "--speed",
        help="The speed param for the vehicle.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save-dir",
        help="The save directory location.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--write",
        help="Replay the agent else write the agent actions out to directory.",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        seed=args.seed,
        speed=args.speed,
        save_dir=args.save_dir,
        write=args.write,
    )
