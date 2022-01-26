import logging

import gym
import numpy as np

from smarts.core.agent import Adapter, Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
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

AGENT_ID = "SingleAgent"

ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, -1.0]), high=np.array([100, 1.0]), dtype=np.float32
)

OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "dt": gym.spaces.Box(low=0, high=1e10, shape=(1,)),
        "step_count": gym.spaces.Box(low=0, high=1e10, shape=(1,)),
        "elapsed_sim_time": gym.spaces.Box(low=0, high=1e10, shape=(1,)),
        "events": gym.spaces.MultiBinary(9),
        "ego_vehicle_state": gym.spaces.Tuple(
            gym.spaces.Box(low=0, high=1e10, shape=(1,)),
            gym.spaces.Box(
                low=np.array([-1e10, -1e10, -1e10]),
                high=np.array([1e10, 1e10, 1e10]),
                shape=(3,),
            ),
            gym.spaces.Box(
                low=np.array([-1e10, -1e10, -1e10]),
                high=np.array([1e10, 1e10, 1e10]),
                shape=(3,),
            ),
            gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
            gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
            gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
            gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
            # road id
            # lane id
            gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
            # mission
            gym.spaces.Box(low=-1e10, high=1e10),
            gym.spaces.Box(low=-1e10, high=1e10),
            gym.spaces.Box(low=-1e10, high=1e10),
            gym.spaces.Box(low=-1e10, high=1e10),
            gym.spaces.Box(low=-1e10, high=1e10),
            gym.spaces.Box(low=-1e10, high=1e10),
        ),
    }
)


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.road_id != obs.via_data.near_via_points[0].road_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface.from_type(
            AgentType.LanerWithSpeed,
            max_episode_steps=max_episode_steps,
            # done_criteria=DoneCriteria(on_shoulder=True, off_road=False)
        ),
        action_adapter=Adapter(space=ACTION_SPACE),
        observation_adapter=Adapter(space=OBSERVATION_SPACE),
        agent_builder=ChaseViaPointsAgent,
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
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=args.episodes,
        seed=args.seed,
    )
