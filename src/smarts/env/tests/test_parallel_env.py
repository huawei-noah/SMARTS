# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import gym

gym.logger.set_level(40)
import pytest

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import RGB, AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.env.hiway_env import HiWayEnv
from smarts.env.wrappers.parallel_env import ParallelEnv


@pytest.fixture(scope="module")
def agent_specs():
    return {
        "Agent_"
        + agent_id: AgentSpec(
            interface=AgentInterface(
                rgb=RGB(width=256, height=256, resolution=50 / 256),
                action=ActionSpaceType.Lane,
                max_episode_steps=3,
            ),
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
            observation_adapter=lambda obs: obs.top_down_rgb.data,
            reward_adapter=lambda obs, reward: reward,
            info_adapter=lambda obs, reward, info: info["score"],
        )
        for agent_id in ["1", "2"]
    }


@pytest.fixture(scope="module")
def single_env_actions(agent_specs):
    return {agent_id: "keep_lane" for agent_id in agent_specs.keys()}


@pytest.fixture(scope="module")
def env_constructor(agent_specs):
    env_constructor = lambda: HiWayEnv(
        scenarios=["scenarios/loop"],
        agent_specs=agent_specs,
        sim_name="Test_env",
        headless=True,
    )
    return env_constructor


def test_non_callable_env_constructors(env_constructor):
    env_constructed = [
        env_constructor(),
        env_constructor(),
    ]
    with pytest.raises(TypeError):
        env = ParallelEnv(env_constructors=env_constructed, auto_reset=True)
        env.close()


def _make_parallel_env(env_constructor, num_env, auto_reset=True, seed=42):
    env_constructors = [env_constructor] * num_env
    return ParallelEnv(
        env_constructors=env_constructors,
        auto_reset=auto_reset,
        seed=seed,
    )


@pytest.mark.parametrize("num_env", [2])
def test_spaces(env_constructor, num_env):
    single_env = env_constructor()
    env = _make_parallel_env(env_constructor, num_env)

    assert env.batch_size == num_env
    assert env.observation_space == single_env.observation_space
    assert env.action_space == single_env.action_space

    single_env.close()
    env.close()


@pytest.mark.parametrize("num_env", [2])
def test_seed(env_constructor, num_env):
    env = _make_parallel_env(env_constructor, num_env)

    first_seed = 7
    seeds = env.seed(first_seed)
    assert len(seeds) == num_env
    for index, seed in enumerate(seeds):
        assert seed == first_seed + index

    env.close()


def _compare_observations(num_env, batched_observations, single_observations):
    assert len(batched_observations) == num_env
    for observations in batched_observations:
        assert observations.keys() == single_observations.keys()
        for agent_id, obs in observations.items():
            assert obs.dtype == single_observations[agent_id].dtype
            assert obs.shape == single_observations[agent_id].shape


@pytest.mark.parametrize("num_env", [2])
def test_reset(env_constructor, num_env):
    single_env = env_constructor()
    single_observations = single_env.reset()
    single_env.close()

    env = _make_parallel_env(env_constructor, num_env)
    batched_observations = env.reset()
    env.close()

    _compare_observations(num_env, batched_observations, single_observations)


@pytest.mark.parametrize("num_env", [2])
@pytest.mark.parametrize("auto_reset", [True])
def test_step(env_constructor, single_env_actions, num_env, auto_reset):
    single_env = env_constructor()
    single_env.reset()
    single_observations, _, _, _ = single_env.step(single_env_actions)
    single_env.close()

    env = _make_parallel_env(env_constructor, num_env, auto_reset=auto_reset)
    env.reset()
    batched_observations, batched_rewards, batched_dones, batched_infos = env.step(
        [single_env_actions] * num_env
    )
    env.close()

    _compare_observations(num_env, batched_observations, single_observations)

    for batched_outputs, kind in zip(
        (batched_rewards, batched_dones, batched_infos), [float, bool, float]
    ):
        assert len(batched_outputs) == num_env
        for outputs in batched_outputs:
            outputs.pop("__all__", None)
            assert outputs.keys() == single_observations.keys()
            assert all(isinstance(val, kind) for val in outputs.values())


@pytest.mark.parametrize("auto_reset", [True, False])
def test_sync_async_episodes(env_constructor, single_env_actions, auto_reset):
    num_env = 2
    env = _make_parallel_env(env_constructor, num_env, auto_reset=auto_reset)
    batched_actions = [single_env_actions] * num_env
    try:
        # Step 1
        env.reset()
        # Step 2
        _, _, batched_dones, _ = env.step(batched_actions)
        assert all(dones["__all__"] == False for dones in batched_dones)
        # Step 3
        _, _, batched_dones, _ = env.step(batched_actions)
        assert all(dones["__all__"] == True for dones in batched_dones)
        # Step 4
        _, _, batched_dones, _ = env.step(batched_actions)
        if auto_reset:
            assert all(dones["__all__"] == False for dones in batched_dones)
        else:
            assert all(dones["__all__"] == True for dones in batched_dones)
        # Step 5
        _, _, batched_dones, _ = env.step(batched_actions)
        assert all(dones["__all__"] == True for dones in batched_dones)
    finally:
        env.close()
