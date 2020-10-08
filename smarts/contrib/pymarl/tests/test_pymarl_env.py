from random import randint

import numpy as np
import pytest

from smarts.contrib.pymarl import PyMARLHiWayEnv
from smarts.contrib.pymarl.adapters.action_adapter import N_ACTIONS

N_AGENTS = 3


@pytest.fixture
def env():
    configs = {
        "scenarios": ["scenarios/figure_eight"],
        "n_agents": N_AGENTS,
        "headless": True,
        "episode_limit": 1000,
        "visdom": False,
        "timestep_sec": 0.1,
    }

    env = PyMARLHiWayEnv(configs)
    yield env
    env.close()


def test_pymarl_env(env):
    def act(observation):
        # TODO: Remove randomness
        return randint(0, N_ACTIONS - 1)

    step = 0
    max_steps_across_episodes = 50

    # TODO: Switch to `for episode in episodes(env, n=MAX_EPISODES):`
    while True:  # for each episode...
        dones_registered = 0
        observations = env.reset()

        while True:  # for each step...
            actions = []
            for agent_obs in observations:
                actions.append(act(agent_obs))

            _, dones, _ = env.step(actions)
            observations = env.get_obs()

            for done in dones:
                dones_registered += 1 if done else 0
            if dones_registered == N_AGENTS:
                break

            step += 1
            if step >= max_steps_across_episodes:
                break

        # Remove this, and replace above break w/ a return?
        if step >= max_steps_across_episodes:
            break

    assert step == (
        max_steps_across_episodes
    ), "Simulation must cycle through to the final step"
