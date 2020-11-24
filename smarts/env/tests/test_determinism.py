import gym
import pytest
import numpy as np

# Reference: https://stackoverflow.com/a/53978543/2783780
try:
    from collections.abc import Sequence
except ImportError:
    from collections import Sequence

from smarts.core.agent_interface import AgentInterface
from smarts.core.agent import AgentSpec, Agent
from smarts.core.controllers import ActionSpaceType
from smarts.core.utils.episodes import episodes

MAX_STEPS_PER_EPISODE = 20
EPISODE_COUNT = 3
CAPTURE_STEP = 5


@pytest.fixture
def agent_spec():
    return AgentSpec(
        interface=AgentInterface(
            drivable_area_grid_map=True,
            ogm=True,
            rgb=True,
            lidar=True,
            waypoints=True,
            max_episode_steps=MAX_STEPS_PER_EPISODE,
            debug=True,
            neighborhood_vehicles=True,
            action=ActionSpaceType.Lane,
        ),
        agent_builder=lambda: Agent.from_function(lambda _: "keep_lane"),
    )


def run(agent_spec, callback):
    AGENT_ID = "Agent-007"
    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=["scenarios/intersections/2lane"],
        agent_specs={AGENT_ID: agent_spec},
        headless=True,
        timestep_sec=0.01,
        seed=42,
    )
    i = 0
    for episode in episodes(n=EPISODE_COUNT):
        agent = agent_spec.build_agent()
        observations = env.reset()

        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})

            episode.record_step(observations, rewards, dones, infos)

            if i % CAPTURE_STEP == 0:
                callback(rewards, agent_obs, dones, int(i / CAPTURE_STEP))
            i += 1

    env.close()


def vehicle_state_check(vs_now, vs_prev):
    assert all([now == prev for (now, prev) in zip(vs_now.position, vs_prev.position)])
    assert vs_now.speed == vs_prev.speed
    assert vs_now.heading == vs_prev.heading
    assert vs_now.bounding_box == vs_prev.bounding_box
    assert vs_now.lane_id == vs_prev.lane_id
    assert vs_now.lane_index == vs_prev.lane_index


def axis_dive(now_axis, prev_axis):
    for (now, prev) in zip(now_axis, prev_axis):
        if isinstance(now, np.ndarray) or isinstance(now, Sequence):
            axis_dive(now, prev)
        else:
            assert now == prev


def test_short_length_determinism(agent_spec):
    rewards_capture = []
    dones_capture = []
    observations_capture = []

    def capture_callback(rewards, agent_obs, dones, index):
        rewards_capture.append(rewards)
        dones_capture.append(dones)
        observations_capture.append(agent_obs)

    def check_callback(rewards, agent_obs, dones, index):
        assert len(rewards_capture) > index - 1
        orig_agent_obs = observations_capture[index]

        assert rewards_capture[index] == rewards

        assert len(dones) == len(dones_capture[index])
        assert all([ds == ds2 for (ds, ds2) in zip(dones, dones_capture[index])])

        # "events",
        events_now = agent_obs.events
        events_prev = orig_agent_obs.events

        assert len(events_now.collisions) == len(events_prev.collisions)
        assert all(
            [
                now == prev
                for (now, prev) in zip(events_now.collisions, events_prev.collisions)
            ]
        )
        assert events_now.off_road == events_prev.off_road
        assert events_now.reached_goal == events_prev.reached_goal
        assert (
            events_now.reached_max_episode_steps
            == events_prev.reached_max_episode_steps
        )

        # "ego_vehicle_state",
        evs_now = agent_obs.ego_vehicle_state
        evs_prev = orig_agent_obs.ego_vehicle_state
        vehicle_state_check(evs_now, evs_prev)

        # "neighborhood_vehicle_states",
        nbvs_now = agent_obs.neighborhood_vehicle_states
        nbvs_prev = orig_agent_obs.neighborhood_vehicle_states

        assert len(nbvs_now) == len(nbvs_prev)
        for now, prev in zip(nbvs_now, nbvs_prev):
            vehicle_state_check(now, prev)

        # "waypoint_paths",
        assert len(agent_obs.waypoint_paths) == len(orig_agent_obs.waypoint_paths)

        for w_paths_now, w_paths_prev in zip(
            agent_obs.waypoint_paths, orig_agent_obs.waypoint_paths
        ):
            for now, prev in zip(w_paths_now, w_paths_prev):
                assert now == prev
                # np.array is requiring this
                assert all(
                    [
                        pos_now == pos_prev
                        for (pos_now, pos_prev) in zip(now.pos, prev.pos)
                    ]
                )
                assert now.heading == prev.heading
                assert now.lane_width == prev.lane_width
                assert now.speed_limit == prev.speed_limit
                assert now.lane_id == prev.lane_id
                assert now.lane_index == prev.lane_index

        # "distance_travelled",
        assert agent_obs.distance_travelled == orig_agent_obs.distance_travelled

        # "lidar_point_cloud",
        axis_dive(agent_obs.lidar_point_cloud, orig_agent_obs.lidar_point_cloud)

        # "drivable_area_grid_map",
        axis_dive(
            agent_obs.drivable_area_grid_map.data,
            orig_agent_obs.drivable_area_grid_map.data,
        )

        # "occupancy_grid_map",
        axis_dive(
            agent_obs.occupancy_grid_map.data, orig_agent_obs.occupancy_grid_map.data
        )

        # "top_down_rgb",
        axis_dive(agent_obs.top_down_rgb.data, orig_agent_obs.top_down_rgb.data)

    run(agent_spec, capture_callback)
    run(agent_spec, check_callback)
