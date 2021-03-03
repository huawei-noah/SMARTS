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
from unittest import mock

import gym
import pytest

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.smarts import SMARTSNotSetupError

AGENT_ID = "AGENT-007"


def build_env(agent_spec):
    return gym.make(
        "smarts.env:hiway-v0",
        # TODO: Switch to a test scenario that has routes, and missions
        scenarios=["scenarios/loop"],
        agent_specs={AGENT_ID: agent_spec},
        headless=True,
        seed=2008,
        timestep_sec=0.01,
    )


def test_graceful_shutdown():
    """SMARTS should not throw any exceptions when shutdown."""
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner),
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
    )
    env = build_env(agent_spec)
    agent = agent_spec.build_agent()
    obs = env.reset()
    for _ in range(10):
        obs = env.step({AGENT_ID: agent.act(obs)})

    env.close()


def test_graceful_interrupt(monkeypatch):
    """SMARTS should only throw a KeyboardInterript exception."""

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(AgentType.Laner),
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
    )
    agent = agent_spec.build_agent()
    env = build_env(agent_spec)

    with pytest.raises(KeyboardInterrupt):
        obs = env.reset()

        # To simulate a user interrupting the sim (e.g. ctrl-c). We just need to
        # hook in to some function that SMARTS calls internally (like this one).
        with mock.patch(
            "smarts.core.sensors.Sensors.observe", side_effect=KeyboardInterrupt
        ):
            for episode in range(10):
                obs, _, _, _ = env.step({AGENT_ID: agent.act(obs)})

        assert episode == 0, "SMARTS should have been interrupted, ending early"

    with pytest.raises(SMARTSNotSetupError):
        env.step({AGENT_ID: agent.act(obs)})
