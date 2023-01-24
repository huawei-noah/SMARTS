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
import logging
import os
import tempfile

import gym
import importlib_resources
import pytest
import pytest_notebook.nb_regression as nb

from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.observations import Observation
from smarts.core.utils.episodes import episodes
from smarts.zoo.agent_spec import AgentSpec

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"
NOTEBOOK_NAME = "test_notebook.ipynb"


class KeepLaneAgent(Agent):
    def act(self, obs: Observation):
        return "keep_lane"


def run_scenario(
    scenarios,
    sim_name,
    headless,
    num_episodes,
    seed,
    max_episode_steps=None,
):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
        ),
        agent_builder=KeepLaneAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        fixed_timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
    )

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


@pytest.fixture(scope="module")
def notebook():
    _, tmppath = tempfile.mkstemp(suffix=".ipynb")
    with open(tmppath, "w") as f:
        import smarts.core.tests

        # pytype: disable=module-attr
        traversable = importlib_resources.files(smarts.core.tests)
        f.write(traversable.joinpath(NOTEBOOK_NAME).read_text())
        # pytype: enable=module-attr
    yield tmppath
    os.remove(tmppath)


def test_notebook1(nb_regression: nb.NBRegressionFixture, notebook):

    ## Generate from the un-run notebook
    nb_regression.force_regen = True
    try:
        nb_regression.check(notebook, False)
    except TimeoutError as te:
        assert (
            False
        ), f"pynotebook `{NOTEBOOK_NAME}` timed out after {nb_regression.exec_timeout}s during test: {te}.\nFor more details see: https://jupyterbook.org/content/execute.html#setting-execution-timeout"
    ## Run notebook against generated
    ## ignore output for now
    nb_regression.diff_ignore = (
        "/cells/*/outputs/*/text",
        "/metadata/language_info/version",
    )
    nb_regression.force_regen = False
    nb_regression.check(notebook)
