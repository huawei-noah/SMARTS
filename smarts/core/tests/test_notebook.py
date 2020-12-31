import logging
import os
import tempfile

import gym
import importlib_resources
import pytest
import pytest_notebook.nb_regression as nb

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes

logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class KeepLaneAgent(Agent):
    def act(self, obs: Observation):
        return "keep_lane"


def run_scenario(
    scenarios,
    sim_name,
    headless,
    num_episodes,
    seed,
    auth_key=None,
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
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
        auth_key=auth_key,
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
    with open(tmppath, "w") as handle:
        import smarts.core.tests

        handle.write(
            importlib_resources.read_text(smarts.core.tests, "test_notebook.ipynb")
        )
    yield tmppath
    os.remove(tmppath)


def test_notebook1(nb_regression: nb.NBRegressionFixture, notebook):

    ## Generate from the un-run notebook
    nb_regression.force_regen = True
    nb_regression.check(notebook, False)

    ## Run notebook against generated
    ## ignore output for now
    nb_regression.diff_ignore = ("/cells/*/outputs/*/text",)
    nb_regression.force_regen = False
    nb_regression.check(notebook)
