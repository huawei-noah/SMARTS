import pathlib

import gym
from examples.argument_parser import default_argument_parser
from smarts.zoo import registry

from smarts.core.utils.episodes import episodes


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    competition_agent = registry.make_agent("zoo.policies:competition_agent-v0")
    competition_agent_spec = registry.make("zoo.policies:competition_agent-v0")
    shared_configs = dict(
        headless=False,
        sumo_headless=True,
    )
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={"Competition_Agent": competition_agent_spec},
        **shared_configs
    )

    # Convert `env.step()` and `env.reset()` from multi-agent interface to
    # single-agent interface.
    env = competition_agent_spec.adapt_env(env)

    for episode in episodes(n=num_episodes):
        observation = env.reset()
        episode.record_scenario(env.scenario_log)

        not_done = True
        while not_done:
            agent_action = competition_agent.act(observation)
            not_done = agent_action
            observation, reward, done, info = env.step(agent_action)
            episode.record_step(observation, reward, done, info)

    env.close()
    competition_agent.close(remove_all_env=True)


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()
    if not args.scenarios:
        args.scenarios = [
            str(
                pathlib.Path(__file__).absolute().parents[0]
                / "scenarios"
                / "sumo"
                / "loop"
            )
        ]

    main(
        scenarios=args.scenarios,
        headless=True,
        num_episodes=5,
    )
