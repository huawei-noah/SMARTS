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
from rgb_image import RGBImage
from smarts.sstudio import build_scenario
from smarts.zoo.agent_spec import AgentSpec
from smarts.core.controllers import ActionSpaceType

logging.basicConfig(level=logging.INFO)


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        
        return (1.1, 0, 0)

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

    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    infos = []

    for episode in episodes(n=2):
        agent = agent_spec.build_agent()
        observation = env.reset()
        episode.record_scenario(env.scenario_log)

        done = False
        while not done:
            agent_action = agent.act(observation)
            observation, reward, done, info = env.step(agent_action)
            episode.record_step(observation, reward, done, info)
            
            # collect data
            observations.append(observation)
            actions.append(agent_action)
            rewards.append(reward)
            terminals.append(done)
            timeouts.append(done)
            infos.append(info)

    # create dataset
    dataset = MDPDataset(np.array(observations), np.array(actions), np.array(rewards), np.array(terminals))

    cql = d3rlpy.algos.CQL(use_gpu=False)

    # train
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
