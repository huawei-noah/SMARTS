import argparse
import os
from typing import List

import gym

import agent as baseline_agent


def run(
    scenarios: List[str],
    episodes: int,
    save_rate: int,
    save_dir: str,
    headless: bool,
    seed: int = 1
):
    AGENT_ID = "AGENT-007"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    agent_spec = baseline_agent.agent_spec

    env = gym.make(
        "ultra.env:ultra-v0",
        agent_specs={AGENT_ID: agent_spec},
        scenario_info=scenarios,
        headless=headless,
        seed=seed,
        timestep_sec=0.1,
        ordered_scenarios=True,
    )
    agent = agent_spec.build_agent()

    for episode_index in range(episodes):
        print(f"Starting episode #{episode_index + 1}")
        total_reward = 0.0
        dones = {"__all__": False}
        observations = env.reset()
        print(f"Scenario: {env.scenario_log['scenario_map']}")
        while not dones["__all__"]:
            action = agent.act(observations[AGENT_ID], explore=True)
            next_observations, rewards, dones, infos = env.step({AGENT_ID: action})
            agent.step(
                observations[AGENT_ID],
                action,
                rewards[AGENT_ID],
                next_observations[AGENT_ID],
                dones[AGENT_ID],
                infos[AGENT_ID],
            )
            total_reward += rewards[AGENT_ID]
            observations = next_observations
        if (episode_index + 1) % save_rate == 0:
            agent.save(save_dir)
        print(f"Return (sum of rewards): {total_reward}")
        print(f"Reached goal? {infos[AGENT_ID]['logs']['events'].reached_goal}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("random-agent-example")
    parser.add_argument(
        "scenarios",
        help=(
            "A list of scenarios. Each element can be either the scenario to run, or a "
            "directory of scenarios to sample from."
        ),
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--episodes",
        help="The number of episodes to run the experiment.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--save-rate",
        help="The rate at which to save the agent, measured in number of episodes.",
        type=int,
        default=5
    )
    parser.add_argument(
        "--save-dir",
        help="The directory in which the neural networks will be saved.",
        type=str,
        default=".",
    )
    parser.add_argument(
        "--headless",
        help="Run the experiment without Envision.",
        action="store_true",
    )
    args = parser.parse_args()

    run(args.scenarios, args.episodes, args.save_rate, args.save_dir, args.headless)
