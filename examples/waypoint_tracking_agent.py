import logging

import gym

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent

# The following ugliness was made necessary because the `aiohttp` #
# dependency has an "examples" module too.  (See PR #1120.)
if __name__ == "__main__":
    from argument_parser import default_argument_parser
else:
    from .argument_parser import default_argument_parser

logging.basicConfig(level=logging.INFO)

AGENT_ID = "WaypointTrackingAgent"


class WaypointTrackingAgent(Agent):
    def act(self, obs: Observation):
        
        lane_index = obs.ego_vehicle_state.lane_index
        num_lane_waypoints = []
        lateral_error = []
        
        goal_position = obs.ego_vehicle_state.mission.goal.position

        for idx in range(len(obs.waypoint_paths)):
            num_waypoints = len(obs.waypoint_paths[idx])
            num_lane_waypoints.append(num_waypoints)
            lateral_error.append(abs(obs.waypoint_paths[idx][num_waypoints - 1].signed_lateral_error(goal_position)))

        # if number of waypoints are the same, choose the lane with the shortest distance to the goal
        if max(num_lane_waypoints) == min(num_lane_waypoints):
            choose_lane = obs.waypoint_paths[lateral_error.index(min(lateral_error))][len(obs.waypoint_paths[idx]) - 1].lane_index
            
            print(lane_index, choose_lane)

            if choose_lane == lane_index:
                return "keep_lane"
            elif choose_lane < lane_index:
                return "change_lane_right"
            else:
                return "change_lane_left"
        else:
            choose_lane = num_lane_waypoints.index(max(num_lane_waypoints))

            if choose_lane == lane_index:
                return "keep_lane"

            return "change_lane_left" if choose_lane > lane_index else "change_lane_right"        



def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.Laner, max_episode_steps=max_episode_steps
            # AgentType.StandardWithAbsoluteSteering, max_episode_steps=max_episode_steps
        ),
        agent_builder=WaypointTrackingAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={AGENT_ID: agent_spec},
        sim_name=sim_name,
        headless=headless,
        visdom=False,
        fixed_timestep_sec=0.1,
        sumo_headless=False,
        seed=seed,
        # zoo_addrs=[("10.193.241.236", 7432)], # Sample server address (ip, port), to distribute social agents in remote server.
        # envision_record_data_replay_path="./data_replay",
    )

    # Wrap a single-agent env with SingleAgent wrapper to make `step` and `reset`
    # output compliant with gym spaces.
    env = SingleAgent(env)

    for episode in episodes(n=num_episodes):
        agent = agent_spec.build_agent()
        observation = env.reset()
        episode.record_scenario(env.scenario_log)

        done = False
        while not done:
            agent_action = agent.act(observation)
            observation, reward, done, info = env.step(agent_action)
            episode.record_step(observation, reward, done, info)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("waypoint-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
