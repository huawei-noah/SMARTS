"""This example is intended to show how a SMARTS environment would be set up without any agents."""
from pathlib import Path

import gymnasium as gym
from tools.argument_parser import default_argument_parser

from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios


def main():
    max_episode_steps = None
    num_episodes = 10
    scenarios_path = Path(__file__).absolute().parents[1] / "scenarios" / "sumo"

    # Scenarios have to be built for the scenario to be complete. This can be done
    # in two ways:
    # - From the cli like: `scl scenario build-all <scenario>`
    # - From code using `build_scenarios(scenarios=[<scenario>, ...])`
    scenarios = [str(scenarios_path / "figure_eight"), str(scenarios_path / "loop")]
    build_scenarios(scenarios=scenarios)

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces={},
        headless=True,
    )

    for episode in episodes(n=num_episodes):
        env.reset()
        episode.record_scenario(env.scenario_log)

        for _ in range(max_episode_steps or 300):
            env.step({})
            episode.record_step({}, {}, {}, {}, {})

    env.close()


if __name__ == "__main__":
    main()
