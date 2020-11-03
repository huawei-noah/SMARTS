import gc

import gym
import pytest

from pympler import muppy

from smarts.core.utils.episodes import episodes
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.agent import AgentSpec, AgentPolicy

SMARTS_MEMORY_GROWTH_LIMIT = 5e4
EPISODE_MEMORY_GROWTH_LIMIT = 1e4
TIMESTEP_SEC = 0.1


@pytest.fixture
def agent_id():
    return "Agent-006"


@pytest.fixture
def primative_scenarios():
    return ["scenarios/intersections/2lane"]


@pytest.fixture
def social_agent_scenarios():
    return [
        "scenarios/intersections/4lane",
        # "scenarios/intersections/6lane",
    ]


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


def env_and_spec(
    action, agent_type, max_episode_steps, scenarios, seed=42, agent_id="Agent-006"
):
    class Policy(AgentPolicy):
        def act(self, obs):
            return action

    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            agent_type, max_episode_steps=max_episode_steps
        ),
        policy_builder=Policy,
    )
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={agent_id: agent_spec},
        headless=True,
        visdom=False,
        timestep_sec=TIMESTEP_SEC,
        sumo_headless=True,
        seed=seed,
    )

    return (env, agent_spec)


def _memory_buildup(
    agent_id, seed, scenarios, episode_count, action, agent_type, max_episode_steps=10
):
    env, agent_spec = env_and_spec(
        action, agent_type, max_episode_steps, scenarios, seed, agent_id
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


def test_smarts_episode_memory_cleanup(
    agent_id, seed, primative_scenarios, agent_params
):
    MAX_EPISODE_STEPS = 100
    EPISODE_COUNT = 100

    _, action, agent_type = agent_params

    env, agent_spec = env_and_spec(
        action, agent_type, MAX_EPISODE_STEPS, primative_scenarios, seed, agent_id
    )
    last_size = 0
    episode_index = 0
    for _ in episodes(EPISODE_COUNT):
        agent = agent_spec.build_agent()
        observations = env.reset()

        gc.collect()
        all_objects = muppy.get_objects()
        next_size = muppy.get_size(all_objects)
        all_objects = None
        if episode_index > 0:
            assert (
                next_size - last_size < EPISODE_MEMORY_GROWTH_LIMIT
            ), f"End size delta {next_size - last_size}"
        last_size = next_size

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[agent_id]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({agent_id: agent_action})
            # episode.record_step(observations, rewards, dones, infos)
        agent_obs = None
        agent_action = None
        observations, rewards, dones, infos = None, None, None, None
        episode = None
        episode_index += 1
    env.close()


def test_smarts_basic_memory_cleanup(agent_id, seed, primative_scenarios, agent_params):
    # Run once to initialize globals and test to see if smarts is working
    _memory_buildup(
        agent_id, seed, primative_scenarios, 100, agent_params[1], agent_params[2]
    )

    gc.collect()
    initial_size = muppy.get_size(muppy.get_objects())

    _memory_buildup(agent_id, seed, primative_scenarios, *agent_params)

    gc.collect()
    all_objects = muppy.get_objects()
    end_size = muppy.get_size(all_objects)
    # Check for a major leak
    assert (
        end_size - initial_size < SMARTS_MEMORY_GROWTH_LIMIT
    ), f"End size delta {end_size - initial_size}"


def test_smarts_repeated_runs_memory_cleanup(
    agent_id, seed, primative_scenarios, agent_type
):
    # Run once to initialize globals and test to see if smarts is working
    _memory_buildup(agent_id, seed, primative_scenarios, 1, *agent_type)

    gc.collect()
    initial_size = muppy.get_size(muppy.get_objects())

    for i in range(100):
        _memory_buildup(agent_id, seed, primative_scenarios, 1, *agent_type)

    gc.collect()
    all_objects = muppy.get_objects()
    end_size = muppy.get_size(all_objects)

    # This "should" be roughly the same as `test_smarts_basic_memory_cleanup`
    assert (
        end_size - initial_size < SMARTS_MEMORY_GROWTH_LIMIT
    ), f"End size delta {end_size - initial_size}"


def test_smarts_fast_reset_memory_cleanup(agent_id, seed, social_agent_scenarios):
    agent_type = AgentType.Buddha
    # Run once to initialize globals and test to see if smarts is working
    _memory_buildup(agent_id, seed, social_agent_scenarios, 1, None, agent_type)

    gc.collect()
    initial_size = muppy.get_size(muppy.get_objects())

    for _ in range(100):
        _memory_buildup(
            agent_id,
            seed,
            social_agent_scenarios,
            1,
            None,
            agent_type,
            max_episode_steps=2,
        )

    gc.collect()
    all_objects = muppy.get_objects()
    end_size = muppy.get_size(all_objects)

    # Check for a major leak
    assert (
        end_size - initial_size < SMARTS_MEMORY_GROWTH_LIMIT
    ), f"End size delta {end_size - initial_size}"


def test_smarts_social_agent_scenario_memory_cleanup(
    agent_id, seed, social_agent_scenarios, agent_type
):
    # Run once to initialize globals and test to see if smarts is working
    _memory_buildup(agent_id, seed, social_agent_scenarios, 1, *agent_type)

    gc.collect()
    initial_size = muppy.get_size(muppy.get_objects())

    _memory_buildup(agent_id, seed, social_agent_scenarios, 100, *agent_type)

    gc.collect()
    all_objects = muppy.get_objects()
    end_size = muppy.get_size(all_objects)

    assert (
        end_size - initial_size < SMARTS_MEMORY_GROWTH_LIMIT
    ), f"End size delta {end_size - initial_size}"
