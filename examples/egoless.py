import logging

import gym
from argument_parser import default_argument_parser

from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={},
        headless=headless,
        sumo_headless=True,
    )

    if max_episode_steps is None:
        max_episode_steps = 1000

    for episode in episodes(n=num_episodes):
        env.reset()
        episode.record_scenario(env.scenario_log)

        for _ in range(max_episode_steps):
            env.step({})
            episode.record_step({}, {}, {}, {})

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("egoless-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
