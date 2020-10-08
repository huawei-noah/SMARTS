import logging

import gym

from smarts.core.utils.episodes import episodes
from examples import default_argument_parser


logging.basicConfig(level=logging.INFO)


def main(scenarios, headless, num_episodes, seed):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={},
        headless=headless,
        sumo_headless=True,
        visdom=False,
        seed=seed,
        timestep_sec=0.1,
    )

    for episode in episodes(n=num_episodes):
        env.reset()
        episode.record_scenario(env.scenario_log)

        for _ in range(1000):
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
        seed=args.seed,
    )
