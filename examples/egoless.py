from pathlib import Path

import gym
from tools.argument_parser import default_argument_parser

from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={},
        headless=headless,
        sumo_headless=True,
    )

    if max_episode_steps is None:
        max_episode_steps = 300

    for episode in episodes(n=num_episodes):
        env.reset()
        episode.record_scenario(env.scenario_log)

        for _ in range(max_episode_steps):
            env.step({})
            episode.record_step({}, {}, {}, {})

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("egoless")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(Path(__file__).absolute().parents[2] / "scenarios" / "sumo" / "loop")
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
