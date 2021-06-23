import logging
import math

import gym
import numpy as np

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.coordinates import Heading, Pose
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.core.utils.math import evaluate_bezier as bezier
from smarts.core.utils.math import (
    lerp,
    low_pass_filter,
    min_angles_difference_signed,
    radians_to_vec,
    signed_dist_to_line,
    vec_to_radians,
)
from smarts.core.planner import Waypoint


logging.basicConfig(level=logging.INFO)

AGENT_ID = "Agent-007"


class BehaviorAgentState(Enum):
    Virtual_lane_following = 0
    Approach = 1
    Interact = 2


class UTurnAgent(Agent):
    def __init__(self, aggressiveness, uturn_speed, maximum_offset, uturn_final_lane):
        # self._initial_heading = 0
        self._task_is_triggered = False
        self._uturn_agent_state = BehaviorAgentState.Virtual_lane_following
        self._prev_uturn_agent_state = None
        self._aggressiveness = aggressiveness
        self._uturn_speed = uturn_speed
        self._maximum_offset = maximum_offset
        self._uturn_final_lane = uturn_final_lane

    def act(self, obs: Observation):
        aggressiveness = 10

        vehicle = self.sim._vehicle_index.vehicles_by_actor_id("Agent-007")[0]

        miss = self.sim._vehicle_index.sensor_state_for_vehicle_id(vehicle.id).planner

        road_map = miss.road_map

        start_lane = road_map.nearest_lane(
            miss._mission.start.point,
            include_junctions=False,
        )
        neighborhood_vehicles = self.sim.neighborhood_vehicles_around_vehicle(
            vehicle=vehicle, radius=850
        )
        pose = vehicle.pose

        lane = road_map.nearest_lane(pose.point)

        def vehicle_control_commands(
            fff, des_speed, look_ahead_wp_num, look_ahead_dist
        ):
            # des_speed = 12
            # look_ahead_wp_num = 3
            # look_ahead_dist = 3
            vehicle_look_ahead_pt = [
                obs.ego_vehicle_state.position[0]
                - look_ahead_dist * math.sin(obs.ego_vehicle_state.heading),
                obs.ego_vehicle_state.position[1]
                + look_ahead_dist * math.cos(obs.ego_vehicle_state.heading),
            ]
            cum_sum = 0
            if len(fff) > 10:
                for idx in range(10):
                    cum_sum += abs(fff[idx + 1].heading - fff[idx].heading)

            reference_heading = fff[look_ahead_wp_num].heading
            heading_error = min_angles_difference_signed(
                (obs.ego_vehicle_state.heading % (2 * math.pi)), reference_heading
            )
            controller_lat_error = fff[look_ahead_wp_num].signed_lateral_error(
                vehicle_look_ahead_pt
            )

            steer = 0.34 * controller_lat_error + 0.5 * heading_error

            throttle = -0.23 * (obs.ego_vehicle_state.speed - (des_speed)) - 1.1 * abs(
                obs.ego_vehicle_state.linear_velocity[1]
            )

            if throttle >= 0:
                brake = 0
            else:
                brake = abs(throttle)
                throttle = 0
            return (throttle, brake, steer)

        if self._uturn_agent_state == BehaviorAgentState.Virtual_lane_following:
            self._prev_uturn_agent_state = BehaviorAgentState.Virtual_lane_following

            fff = obs.waypoint_paths[start_lane.index]
            fff = miss.waypoint_paths_on_lane_at_point(pose, start_lane.lane_id, 60)[0]
            ll = []
            # TODO STEVE:  for changing into left-turn lane...
            for idx in range(len(obs.waypoint_paths)):
                ll.append(len(obs.waypoint_paths[idx]))
            if len(obs.waypoint_paths[0]) == max(ll):
                fff = obs.waypoint_paths[0]
            else:
                fff = obs.waypoint_paths[ll.index(max(ll))]
            des_speed = 12
            look_ahead_wp_num = 3
            look_ahead_dist = 3
            vehicle_inputs = vehicle_control_commands(
                fff, des_speed, look_ahead_wp_num, look_ahead_dist
            )

            # if len(neighborhood_vehicles)!=0:
            #     self._uturn_agent_state=BehaviorAgentState.Approach

            return vehicle_inputs

        if self._uturn_agent_state == BehaviorAgentState.Approach:
            self._prev_uturn_agent_state = BehaviorAgentState.Approach

            start_road = start_lane.road
            oncoming_road = start_road.oncoming_roads[0]
            oncoming_lanes = oncoming_road.lanes
            target_lane = oncoming_lanes[0]

            offset = start_lane.offset_along_lane(pose.position[:2])
            oncoming_offset = max(0, target_lane.length - offset)
            nvpose = neighborhood_vehicles[0].pose
            target_l = road_map.nearest_lane(nvpose.point)
            target_p = nvpose.position[:2]
            target_offset = target_l.offset_along_lane(target_p)
            fq = target_lane.length - offset - target_offset

            paths = miss.waypoint_paths_of_lane_at_offset(
                target_lane, oncoming_offset, lookahead=30
            )

            des_speed = 12
            des_lane = 0

            if (
                fq > (aggressiveness / 10) * 65 + (1 - aggressiveness / 10) * 100
                and self._task_is_triggered is False
            ):
                fff = obs.waypoint_paths[start_lane.lane_index]
                # self._initial_heading = obs.ego_vehicle_state.heading % (2 * math.pi)

            else:
                self._task_is_triggered = True
                fff = obs.waypoint_paths[start_lane.lane_index]
                # fff = paths[des_lane]

            # fff = obs.waypoint_paths[start_lane.lane_index]
            des_speed = 12
            look_ahead_wp_num = 3
            look_ahead_dist = 3
            vehicle_inputs = vehicle_control_commands(
                fff, des_speed, look_ahead_wp_num, look_ahead_dist
            )

            if self._task_is_triggered is True:
                self._uturn_agent_state = BehaviorAgentState.Interact

            return vehicle_inputs

        if self._uturn_agent_state == BehaviorAgentState.Interact:
            self._prev_uturn_agent_state = BehaviorAgentState.Interact

            start_road = start_lane.road
            oncoming_road = start_road.oncoming_roads[0]
            oncoming_lanes = oncoming_road.lanes
            target_lane = oncoming_lanes[0]

            offset = start_lane.offset_along_lane(pose.position[:2])
            oncoming_offset = max(0, target_lane.length - offset)
            nvpose = neighborhood_vehicles[0].pose
            target_l = road_map.nearest_lane(nvpose.point)
            target_p = nvpose.position[:2]
            target_offset = target_l.offset_along_lane(target_p)
            fq = target_l.length - offset - target_offset

            paths = miss.waypoint_paths_of_lane_at_offset(
                target_lane, oncoming_offset, lookahead=30
            )

            des_speed = 12
            des_lane = 0

            if (
                fq > (aggressiveness / 10) * 65 + (1 - aggressiveness / 10) * 100
                and self._task_is_triggered is False
            ):
                fff = obs.waypoint_paths[start_lane.lane_index]
                # self._initial_heading = obs.ego_vehicle_state.heading % (2 * math.pi)
            else:
                self._task_is_triggered = True
                # fff = obs.waypoint_paths[start_lane.lane_index]
                fff = paths[des_lane]

            # fff = obs.waypoint_paths[start_lane.lane_index]
            des_speed = 12
            look_ahead_wp_num = 3
            look_ahead_dist = 3
            vehicle_inputs = vehicle_control_commands(
                fff, des_speed, look_ahead_wp_num, look_ahead_dist
            )

            if self._task_is_triggered is True:
                self._prev_uturn_agent_state = BehaviorAgentState.Interact

            return vehicle_inputs


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.StandardWithAbsoluteSteering, max_episode_steps=max_episode_steps
        ),
        agent_builder=UTurnAgent,
        agent_params=(10, 5, 100, 2),
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        timestep_sec=0.1,
        sumo_headless=False,
        sumo_auto_start=False,
        seed=seed,
    )
    global vvv
    UTurnAgent.sim = env._smarts

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        agent.sim = env._smarts
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            agent_obs = observations[AGENT_ID]
            agent_action = agent.act(agent_obs)
            observations, rewards, dones, infos = env.step({AGENT_ID: agent_action})
            episode.record_step(observations, rewards, dones, infos)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
