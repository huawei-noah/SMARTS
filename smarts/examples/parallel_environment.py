# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import gym

gym.logger.set_level(40)

from typing import Dict, Sequence, Tuple

from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
from smarts.env.hiway_env import HiWayEnv
from smarts.env.wrappers.frame_stack import FrameStack
from smarts.env.wrappers.parallel_env import ParallelEnv
from smarts.examples.argument_parser import default_argument_parser


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Sequence[Observation]) -> Tuple[float, int]:
        # Here, we only utilise the newest frame from the stacked observations.
        newest_obs = obs[-1]
        speed_limit = newest_obs.waypoint_paths[0][0].speed_limit
        return (speed_limit, 0)


def main(
    scenarios: Sequence[str],
    sim_name: str,
    headless: bool,
    seed: int,
    num_agents: int,
    num_stack: int,
    num_env: int,
    auto_reset: bool,
    max_episode_steps: int = 128,
    num_steps: int = 1280,
    num_episodes: int = 10,
):

    # Agents' name
    agent_ids = [f"Agent_{i}" for i in range(num_agents)]

    # Define agent specification
    agent_specs = {
        agent_id: AgentSpec(
            interface=AgentInterface(
                rgb=True,
                waypoints=True,
                action=ActionSpaceType.LaneWithContinuousSpeed,
                max_episode_steps=max_episode_steps,
            ),
            agent_builder=ChaseViaPointsAgent,
        )
        for agent_id in agent_ids
    }

    # Create a callable env constructor. Here, for illustration purposes, each environment is
    # wrapped with a FrameStack wrapper which returns stacked observations for each environment.
    env_frame_stack = lambda env: FrameStack(
        env=env,
        num_stack=num_stack,
    )
    env_constructor = lambda sim_name: env_frame_stack(
        HiWayEnv(
            scenarios=scenarios,
            agent_specs=agent_specs,
            sim_name=sim_name,
            headless=headless,
        )
    )

    # Build multiple agents
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    # Create parallel environments
    env = ParallelEnv(
        env_constructors=[env_constructor] * num_env,
        sim_name=sim_name,
        auto_reset=auto_reset,
        seed=seed,
    )

    if auto_reset:
        parallel_env_async(agents, env, num_env, num_steps)
    else:
        parallel_env_sync(agents, env, num_env, num_episodes)


def parallel_env_async(
    agents: Dict[str, Agent], env: gym.Env, num_env: int, num_steps: int
):
    """Parallel environments with asynchronous episodes. Run multiple environments
    in parallel with `auto_reset=True`. Individual environments will automatically
    reset when their episode ends. Episodes start asynchronously in each environment.

    Args:
        agents (Dict[str, Agent]): Ego agents.
        env (gym.Env): Gym env.
        num_env (int): Number of environments.
        num_steps (int): Number of steps to step the environment.
    """

    tot_scores = {agent_id: 0 for agent_id in agents.keys()}

    batched_dones = [{"__all__": False} for _ in range(num_env)]
    batched_observations = env.reset()

    for _ in range(num_steps):
        # Compute actions for all active(i.e., not done) agents
        batched_actions = []
        for observations, dones in zip(batched_observations, batched_dones):
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
                if not dones.get(agent_id, False)
                or dones[
                    "__all__"
                ]  # `dones[__all__]==True` implies the env was auto-reset in previous iteration
            }
            batched_actions.append(actions)

        # Step all environments in parallel
        batched_observations, batched_rewards, batched_dones, batched_infos = env.step(
            batched_actions
        )

        # Sum the scores
        for dones, infos in zip(batched_dones, batched_infos):
            for agent_id, val in infos.items():
                if dones[agent_id]:
                    tot_scores[agent_id] += val["score"]

    # Add the score of not-done agents
    for dones, infos in zip(batched_dones, batched_infos):
        for agent_id, val in infos.items():
            if not dones[agent_id]:
                tot_scores[agent_id] += val["score"]

    # Print average episode score of each agent
    ave_scores = {
        agent_id: score / (num_steps * num_env)
        for agent_id, score in tot_scores.items()
    }
    print("Average step score:")
    print(f"{ave_scores}")

    env.close()


def parallel_env_sync(
    agents: Dict[str, Agent], env: gym.Env, num_env: int, num_episodes: int
):
    """Parallel environments with synchronous episodes. Run multiple environments
    in parallel with `auto_reset=False`. All environments are reset together when
    all their episodes have finished. New episodes start synchronously in all
    environments.

    Args:
        agents (Dict[str, Agent]): Ego agents.
        env (gym.Env): Gym env.
        num_env (int): Number of parallel environments.
        num_episodes (int): Number of episodes.
    """

    tot_scores = {agent_id: 0 for agent_id in agents.keys()}

    for _ in range(num_episodes):
        batched_dones = [{"__all__": False} for _ in range(num_env)]
        batched_observations = env.reset()

        # Iterate until all environments complete an episode each.
        while not all(dones["__all__"] for dones in batched_dones):
            # Compute actions for all active(i.e., not done) agents
            batched_actions = []
            for observations, dones in zip(batched_observations, batched_dones):
                actions = {
                    agent_id: agents[agent_id].act(agent_obs)
                    for agent_id, agent_obs in observations.items()
                    if not dones.get(agent_id, False)
                }
                batched_actions.append(actions)

            # Step all environments in parallel
            (
                batched_observations,
                batched_rewards,
                batched_dones,
                batched_infos,
            ) = env.step(batched_actions)

            # Sum the scores
            for dones, infos in zip(batched_dones, batched_infos):
                for agent_id, val in infos.items():
                    if dones[agent_id]:
                        tot_scores[agent_id] += val["score"]

    # Print average episode score of each agent
    ave_scores = {
        agent_id: score / (num_episodes * num_env)
        for agent_id, score in tot_scores.items()
    }
    print("Average episode score:")
    print(f"{ave_scores}")

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("parallel-environment-example")
    parser.add_argument(
        "--num-agents",
        default=2,
        type=int,
        help="Number of ego agents to simulate in each environment.",
    )
    parser.add_argument(
        "--num-stack",
        default=2,
        type=int,
        help="Number of consecutive frames to stack in each environment's observation.",
    )
    parser.add_argument(
        "--num-env",
        default=2,
        type=int,
        help="Number of parallel environments to simulate.",
    )
    parser.add_argument(
        "--max-episode-steps",
        default=128,
        type=int,
        help="Maximum number of steps per episode.",
    )
    parser.add_argument(
        "--num-steps",
        default=1280,
        type=int,
        help="Total number of steps to simulate per environment in parallel asynchronous simulation.",
    )
    args = parser.parse_args()

    print("\nParallel environments with asynchronous episodes.\n")
    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        seed=args.seed,
        num_agents=args.num_agents,
        num_stack=args.num_stack,
        num_env=args.num_env,
        auto_reset=True,
        max_episode_steps=args.max_episode_steps,
        num_steps=args.num_steps,
    )

    print("\nParallel environments with synchronous episodes.\n")
    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        seed=args.seed,
        num_agents=args.num_agents,
        num_stack=args.num_stack,
        num_env=args.num_env,
        auto_reset=False,
        max_episode_steps=args.max_episode_steps,
        num_episodes=args.episodes,
    )
