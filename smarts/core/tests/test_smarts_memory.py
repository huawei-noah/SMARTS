import gc
import time

import gym
from pympler import tracker, muppy, summary, classtracker
import pytest

from smarts.core.utils.episodes import episodes
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, AgentPolicy


@pytest.fixture
def agent_id():
    return "Agent-006"


@pytest.fixture
def primative_scenarios():
    return ["scenarios/intersections/2lane", "scenarios/cloverleaf"]


@pytest.fixture
def social_agent_scenarios():
    return ["scenarios/intersections/4lane", "scenarios/intersections/6lane_t"]


@pytest.fixture
def seed():
    return 42


@pytest.fixture(
    params=[
        # ( episodes, action, agent_type )
        (100, None, AgentType.Buddha),
        (100, (30, 1, -1), AgentType.Full),
        (10, (30, 1, -1), AgentType.Standard),  # standard is just full but less
        (100, "keep_lane", AgentType.Laner),
        (100, (30, 1, -1), AgentType.Loner),
        (100, (30, 1, -1), AgentType.Tagger),
        (100, (30, 1, -1), AgentType.StandardWithAbsoluteSteering),
        (100, (50, 0), AgentType.LanerWithSpeed),
        (100, ([1, 2, 3], [1, 2, 3], [0.5, 1, 1.5], [20, 20, 20]), AgentType.Tracker),
        # ( 5, ([0,1,2], [0,1,2], [0,1,2]), AgentType.MPCTracker),
    ]
)
def agent_params(request):
    return request.param


@pytest.fixture
def agent_type():
    return ((30, 1, -1), AgentType.Full)


def _memory_buildup(agent_id, seed, scenarios, episode_count, action, agent_type):
    class Policy(AgentPolicy):
        def act(self, obs):
            return action

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(agent_type, max_episode_steps=10),
        policy_builder=Policy,
    )
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={agent_id: agent_spec},
        headless=True,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=True,
        seed=seed,
        # envision_record_data_replay_path="./data_replay",
    )

    for episode in episodes(episode_count):
        agent = agent_spec.build_agent()
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[agent_id]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({agent_id: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


def test_smarts_basic_memory_cleanup(agent_id, seed, primative_scenarios, agent_params):
    # Run once to initialize globals and test to see if smarts is working
    _memory_buildup(
        agent_id, seed, primative_scenarios, 100, agent_params[1], agent_params[2]
    )

    gc.collect()
    initial_size = muppy.get_size(muppy.get_objects())

    gc.collect()
    try:
        _memory_buildup(agent_id, seed, primative_scenarios, *agent_params)
    except Exception:
        # Not testing for crashes
        pass

    gc.collect()
    all_objects = muppy.get_objects()
    end_size = muppy.get_size(all_objects)

    assert end_size - initial_size < 1e5


def test_smarts_social_agent_scenario(
    agent_id, seed, social_agent_scenarios, agent_type
):
    # Run once to initialize globals and test to see if smarts is working
    _memory_buildup(agent_id, seed, social_agent_scenarios, 1, *agent_type)

    gc.collect()
    initial_size = muppy.get_size(muppy.get_objects())
    gc.collect()

    try:
        _memory_buildup(agent_id, seed, social_agent_scenarios, 100, *agent_type)
    except Exception:
        # Not testing for crashes
        pass

    gc.collect()
    all_objects = muppy.get_objects()
    end_size = muppy.get_size(all_objects)

    assert initial_size / end_size < 1e3
