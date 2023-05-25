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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import gymnasium as gym

gym.logger.set_level(40)
import pytest

from smarts.core.agent import Agent
from smarts.core.agent_interface import RGB, AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.env.gymnasium.wrappers.parallel_env import ParallelEnv
from smarts.env.utils.action_conversion import ActionOptions
from smarts.zoo.agent_spec import AgentSpec

DEFAULT_SEED = 42


@pytest.fixture(scope="module")
def agent_specs():
    return {
        "Agent_"
        + agent_id: AgentSpec(
            interface=AgentInterface(
                top_down_rgb=RGB(width=256, height=256, resolution=50 / 256),
                action=ActionSpaceType.Lane,
                max_episode_steps=3,
            ),
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
        )
        for agent_id in ["1", "2"]
    }


@pytest.fixture(scope="module")
def single_env_actions(agent_specs):
    return {agent_id: "keep_lane" for agent_id in agent_specs.keys()}


@pytest.fixture(scope="module")
def env_constructor(agent_specs):
    env_constructor = lambda seed: gym.make(
        "smarts.env:hiway-v1",
        scenarios=["scenarios/sumo/figure_eight"],
        agent_interfaces={
            agent_id: agent_spec.interface
            for agent_id, agent_spec in agent_specs.items()
        },
        sim_name="Test_env",
        headless=True,
        action_options=ActionOptions.unformatted,
        seed=seed,
    )
    return env_constructor


def test_non_callable_env_constructors(env_constructor):
    env_constructed = [
        env_constructor(seed=DEFAULT_SEED),
        env_constructor(seed=DEFAULT_SEED + 1),
    ]
    with pytest.raises(TypeError):
        env = ParallelEnv(env_constructors=env_constructed, auto_reset=True)
        env.close()


def _make_parallel_env(env_constructor, num_env, auto_reset=True, seed=DEFAULT_SEED):
    env_constructors = [env_constructor] * num_env
    return ParallelEnv(
        env_constructors=env_constructors,
        auto_reset=auto_reset,
        seed=seed,
    )


@pytest.mark.parametrize("num_env", [2])
def test_spaces(env_constructor, num_env):
    single_env = env_constructor(seed=DEFAULT_SEED)
    env = _make_parallel_env(env_constructor, num_env)

    assert env.batch_size == num_env
    assert env.observation_space == single_env.observation_space
    assert env.action_space == single_env.action_space

    single_env.close()
    env.close()


@pytest.mark.parametrize("num_env", [2])
def test_seed(env_constructor, num_env):
    first_seed = DEFAULT_SEED
    env = _make_parallel_env(env_constructor, num_env, seed=first_seed)

    seeds = env.seed()
    assert len(seeds) == num_env
    for index, seed in enumerate(seeds):
        assert seed == first_seed + index

    env.close()


def _compare_outputs(num_env, batched_outputs, single_outputs):
    assert len(batched_outputs) == num_env
    for outputs in batched_outputs:
        assert outputs.keys() == single_outputs.keys()
        for agent_id, out in outputs.items():
            assert type(out) is type(single_outputs[agent_id])


@pytest.mark.parametrize("num_env", [2])
def test_reset(env_constructor, num_env):
    single_env = env_constructor(seed=DEFAULT_SEED)
    single_observations, single_infos = single_env.reset()
    single_env.close()

    env = _make_parallel_env(env_constructor, num_env)
    batched_observations, batched_infos = env.reset()
    env.close()

    _compare_outputs(num_env, batched_observations, single_observations)
    _compare_outputs(num_env, batched_infos, single_infos)


@pytest.mark.parametrize("num_env", [2])
@pytest.mark.parametrize("auto_reset", [True])
def test_step(env_constructor, single_env_actions, num_env, auto_reset):
    single_env = env_constructor(seed=DEFAULT_SEED)
    single_env.reset()
    (
        single_observations,
        single_rewards,
        single_terminateds,
        single_truncateds,
        single_infos,
    ) = single_env.step(single_env_actions)
    single_env.close()

    env = _make_parallel_env(env_constructor, num_env, auto_reset=auto_reset)
    env.reset()
    (
        batched_observations,
        batched_rewards,
        batched_terminateds,
        batched_truncateds,
        batched_infos,
    ) = env.step([single_env_actions] * num_env)
    env.close()

    for batched_outputs, single_outputs in zip(
        [
            batched_observations,
            batched_rewards,
            batched_terminateds,
            batched_truncateds,
            batched_infos,
        ],
        [
            single_observations,
            single_rewards,
            single_terminateds,
            single_truncateds,
            single_infos,
        ],
    ):
        _compare_outputs(num_env, batched_outputs, single_outputs)


@pytest.mark.parametrize("auto_reset", [True, False])
def test_sync_async_episodes(env_constructor, single_env_actions, auto_reset):
    num_env = 2
    env = _make_parallel_env(env_constructor, num_env, auto_reset=auto_reset)
    batched_actions = [single_env_actions] * num_env
    try:
        # Step 1
        env.reset()
        # Step 2
        _, _, batched_terminateds, _, _ = env.step(batched_actions)
        assert all(
            terminateds["__all__"] == False for terminateds in batched_terminateds
        )
        # Step 3
        _, _, batched_terminateds, _, _ = env.step(batched_actions)
        assert all(
            terminateds["__all__"] == True for terminateds in batched_terminateds
        )
        # Step 4
        _, _, batched_terminateds, _, _ = env.step(batched_actions)
        if auto_reset:
            assert all(
                terminateds["__all__"] == False for terminateds in batched_terminateds
            )
        else:
            assert all(
                terminateds["__all__"] == True for terminateds in batched_terminateds
            )
        # Step 5
        _, _, batched_terminateds, _, _ = env.step(batched_actions)
        assert all(
            terminateds["__all__"] == True for terminateds in batched_terminateds
        )
    finally:
        env.close()
