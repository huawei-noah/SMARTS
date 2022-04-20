import logging
import pathlib
import d3rlpy
from d3rlpy.dataset import MDPDataset
import numpy as np

import gym

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType, RGB
from smarts.core.sensors import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.wrappers.single_agent import SingleAgent
from smarts.env.wrappers.rgb_image import RGBImage
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

logging.basicConfig(level=logging.INFO)


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        # if (
        #     len(obs.via_data.near_via_points) < 1
        #     or obs.ego_vehicle_state.road_id != obs.via_data.near_via_points[0].road_id
        # ):
        #     return (obs.waypoint_paths[0][0].speed_limit, 0)

        # nearest = obs.via_data.near_via_points[0]
        # if nearest.lane_index == obs.ego_vehicle_state.lane_index:
        #     return (nearest.required_speed, 0)

        # return (
        #     nearest.required_speed,
        #     1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        # )
        return (1, 0, 0)

def main(scenarios, headless, num_episodes, max_episode_steps=None):
    agent_spec = AgentSpec(
        interface=AgentInterface(
            max_episode_steps=300,
            rgb=RGB(),
            action=getattr(ActionSpaceType, "Continuous")
        ),
        agent_builder=ChaseViaPointsAgent,
    )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=scenarios,
        agent_specs={"SingleAgent": agent_spec},
        headless=headless,
        sumo_headless=True,
    )

    # Convert `env.step()` and `env.reset()` from multi-agent interface to
    # single-agent interface.
    
    env = RGBImage(env=env, num_stack=1)
    env = SingleAgent(env=env)
    env.action_space = gym.spaces.Box(
        low=-1e6, high=1e6, shape=(3,), dtype=np.float32
    )

    # print(env.action_space)

    # dataset = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    infos = []

    # print(env.observation_space)

    for episode in episodes(n=1):
        agent = agent_spec.build_agent()
        observation = env.reset()
        episode.record_scenario(env.scenario_log)
        # print(np.array(observation).shape)
        # print(np.transpose(np.array(observation)).shape)

        done = False
        while not done:
            agent_action = agent.act(observation)
            observation, reward, done, info = env.step(agent_action)
            episode.record_step(observation, reward, done, info)
            
            # observations.append(observation)
            observations.append(np.transpose(observation))
            actions.append(np.array(agent_action))
            rewards.append(reward)
            terminals.append(done)
            timeouts.append(done)
            infos.append(info)

    dataset = MDPDataset(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))
    # dataset["observations"] = observations
    # dataset["actions"] = actions
    # dataset["rewards"] = rewards
    # dataset["terminals"] = terminals
    # dataset["timeouts"] = timeouts
    # dataset["infos"] = infos   
    # print(dataset.size())

    cql = d3rlpy.algos.DiscreteCQL(use_gpu=False)


    cql.fit(dataset, 
            eval_episodes=dataset, 
            n_epochs=100, 
            scorers={
                'environment': d3rlpy.metrics.evaluate_on_environment(env),
                'td_error': d3rlpy.metrics.td_error_scorer
            }
    )

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("single-agent-example")
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(pathlib.Path(__file__).absolute().parents[2] / "scenarios" / "loop")
        ]

    build_scenario(args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
    )
