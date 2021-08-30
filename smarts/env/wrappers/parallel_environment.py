import gym
from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes
from smarts.env.hiway_env import HiWayEnv
from smarts.env.wrappers.frame_stack import FrameStack
from smarts.env.wrappers.parallel_env import ParallelEnv
from typing import Sequence, List

N_AGENTS = 4
AGENT_IDS = ["Agent %i" % i for i in range(N_AGENTS)]

class KeepLaneAgent(Agent):
    def act(self, obs):
        assert len(obs) == 3
        return "keep_lane"


def main(scenarios, sim_name, headless, num_episodes, seed, max_episode_steps=None):
    agent_specs = {
        agent_id: AgentSpec(
            interface=AgentInterface.from_type(
                AgentType.Laner, max_episode_steps=max_episode_steps
            ),
            agent_builder=KeepLaneAgent,
        )
        for agent_id in AGENT_IDS
    }

    # env = gym.make(
    #     "smarts.env:hiway-v0",
    #     scenarios=scenarios,
    #     agent_specs=agent_specs,
    #     sim_name=sim_name,
    #     headless=headless,
    #     seed=seed,
    # )

    env_frame_stack = lambda env : FrameStack(
        env = env,
        num_stack=3,
    )
    env_constructor = lambda : env_frame_stack(HiWayEnv(
        scenarios=scenarios,
        agent_specs=agent_specs,
        sim_name=sim_name,
        headless=headless,
        seed=seed
    ))


    # Parallel SMARTS environments
    env = ParallelEnv(
        env_fns = [env_constructor, env_constructor],
        seed = 31,
    )

    for episode in episodes(n=num_episodes):
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }
        batched_observations = env.reset()

        dones = {"__all__": False}
        batched_actions = []
        batch = 64
        for step in range(batch):
            for observations in batched_observations:
                actions = {
                    agent_id: agents[agent_id].act(agent_obs)
                    for agent_id, agent_obs in observations.items()
                }
                batched_actions.append(actions)

            batched_observations, rewards, dones, infos = env.step(batched_actions)

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("multi-agent-example")
    args = parser.parse_args()

    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        num_episodes=20,
        seed=args.seed,
    )
