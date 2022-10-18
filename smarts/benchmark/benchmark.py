import logging

import gym

from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={},
        headless=True,
        sumo_headless=True,
        shuffle_scenarios=False,
    )
    if max_episode_steps is None:
        max_episode_steps = 1000

    first_scenario = None
    while True:
        env.reset()
        if not first_scenario:
            first_scenario = tuple([i for i in env.scenario_log.values()])
        elif first_scenario == tuple([i for i in env.scenario_log.values()]):
            break            

        for _ in range(max_episode_steps):
            env.step({})

    env.close()


    # smarts = SMARTS(
    #   ...
    # )

    # scenario_iterator = Scenarios.scenario_variations(scenarios, agent_ids=[], circular=False)
    # for scenario in scenario_iterator:
    #     smarts.reset(scenario)

    #     for _ in range(max_episode_steps):
    #         smarts.step({})
