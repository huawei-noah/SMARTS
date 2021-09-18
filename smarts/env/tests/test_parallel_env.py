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

from smarts.env.hiway_env import HiWayEnv
from smarts.env.wrappers.frame_stack import FrameStack
from smarts.env.wrappers.parallel_env import ParallelEnv
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, RGB
from smarts.core.controllers import ActionSpaceType


@pytest.fixture(scope="module")
def agent_specs():
    return {
        "AGENT_"
        + agent_id: AgentSpec(
            interface=AgentInterface(
                rgb=RGB(),
                action=ActionSpaceType.Lane,
                max_episode_steps=3,
            ),
            agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
        )
        for agent_id in ["001", "002"]
    }


@pytest.fixture(scope="module")
def agents(agent_specs):
    return {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }


@pytest.fixture(scope="module")
def env_constructor(agent_specs):
    env_frame_stack = lambda env: FrameStack(
        env=env,
        num_stack=2,
    )
    env_constructor = lambda sim_name: env_frame_stack(
        HiWayEnv(
            scenarios=["scenarios/loop"],
            agent_specs=agent_specs,
            sim_name=sim_name,
            headless=True,
        )
    )
    return env_constructor


@pytest.mark.parametrize("num_env", [1, 2])
@pytest.mark.parametrize("auto_reset", [True, False])
def test_create_parallel_env(env_constructor, num_env, auto_reset):
    env_constructors = [env_constructor] * num_env

    # Check number of parallel environments created
    env = ParallelEnv(
        env_constructors=env_constructors,
        auto_reset=auto_reset,
        seed=1,
    )
    for process in env.processes:
        assert process.is_alive() == True
    assert len(env.processes) == len(env_constructors)

    # Check seed number of each environment
    first_seed = 7
    seeds = env.seed(first_seed)
    for index, seed in enumerate(seeds):
        assert seed == first_seed + index

    env.close()


def test_non_callable_env_constructors(env_constructor):
    env_constructed = [
        env_constructor(sim_name="Test"),
        env_constructor(sim_name="Test"),
    ]
    with pytest.raises(TypeError):
        env = ParallelEnv(env_constructos=env_constructed, auto_reset=True)
        env.close()


def _get_batched_actions(agents, batched_observations):
    batched_actions = []
    for observations in batched_observations:
        actions = {
            agent_id: agents[agent_id].act(agent_obs)
            for agent_id, agent_obs in observations.items()
        }
        batched_actions.append(actions)
    return batched_actions


@pytest.mark.parametrize("num_env", [1, 2])
@pytest.mark.parametrize("auto_reset", [True, False])
def test_step_parallel_env(env_constructor, num_env, auto_reset, agent_specs, agents):
    env_constructors = [env_constructor] * num_env
    env = ParallelEnv(env_constructors=env_constructors, auto_reset=auto_reset)
    try:
        # Verify output of reset
        batched_observations = env.reset()
        assert len(batched_observations) == num_env
        for observations in batched_observations:
            assert observations.keys() == agent_specs.keys()

        # Compute batched actions
        batched_actions = _get_batched_actions(agents, batched_observations)

        # Verify batched output of environment step
        batched_observations, batched_rewards, batched_dones, batched_infos = env.step(
            batched_actions
        )

        for batched_outputs in (
            batched_observations,
            batched_rewards,
            batched_dones,
            batched_infos,
        ):
            assert len(batched_outputs) == num_env
            for outputs in batched_outputs:
                outputs.pop("__all__", None)
                assert outputs.keys() == agents.keys()

        for batched_outputs, kind in zip(
            (batched_rewards, batched_dones), [float, bool]
        ):
            for outputs in batched_outputs:
                outputs.pop("__all__", None)
                assert all(isinstance(val, kind) for val in outputs.values())
    finally:
        env.close()


@pytest.mark.parametrize("auto_reset", [True, False])
def test_sync_async_episodes(env_constructor, agents, auto_reset):
    env_constructors = [env_constructor] * 2
    env = ParallelEnv(env_constructors=env_constructors, auto_reset=auto_reset)
    try:
        # Step 1
        batched_observations = env.reset()
        batched_actions = _get_batched_actions(agents, batched_observations)
        # Step 2
        batched_observations, _, batched_dones, _ = env.step(batched_actions)
        batched_actions = _get_batched_actions(agents, batched_observations)
        assert all(dones["__all__"] == False for dones in batched_dones)
        # Step 3
        batched_observations, _, batched_dones, _ = env.step(batched_actions)
        batched_actions = _get_batched_actions(agents, batched_observations)
        assert all(dones["__all__"] == True for dones in batched_dones)
        # Step 4
        batched_observations, _, batched_dones, _ = env.step(batched_actions)
        batched_actions = _get_batched_actions(agents, batched_observations)
        if auto_reset:
            assert all(dones["__all__"] == False for dones in batched_dones)
        else:
            assert all(dones["__all__"] == True for dones in batched_dones)
        # Step 5
        batched_observations, _, batched_dones, _ = env.step(batched_actions)
        batched_actions = _get_batched_actions(agents, batched_observations)
        assert all(dones["__all__"] == True for dones in batched_dones)
    finally:
        env.close()
