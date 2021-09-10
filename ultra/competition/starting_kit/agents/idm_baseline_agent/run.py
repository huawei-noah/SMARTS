import argparse
from typing import List

import gym

import agent as baseline_agent


def run(scenarios: List[str], episodes: int, headless: bool, seed: int = 1):
    AGENT_ID = "AGENT-007"

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
            action = agent.act(observations[AGENT_ID])
            observations, rewards, dones, infos = env.step({AGENT_ID: action})
            total_reward += rewards[AGENT_ID]
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
        "--headless",
        help="Run the experiment without Envision.",
        action="store_true",
    )
    args = parser.parse_args()

    run(args.scenarios, args.episodes, args.headless)
