import gym
gym.logger.set_level(40)

from examples.argument_parser import default_argument_parser
from smarts.core.agent import Agent, AgentSpec
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.sensors import Observation
from smarts.env.hiway_env import HiWayEnv
from smarts.env.wrappers.async_vector_env import AsyncVectorEnv
from smarts.env.wrappers.frame_stack import FrameStack
from typing import Dict, Tuple, Sequence


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Sequence[Observation]) -> Tuple[float, int]:
        # Here, we only utilise the newest frame from the stacked observations
        obs = obs[-1]

        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.edge_id != obs.via_data.near_via_points[0].edge_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )

def info_adapter(obs, reward, info) -> float:
    return info['score']

def main(
    scenarios: Sequence[str],
    sim_name: str,
    headless: bool,
    seed: int,
    n_agents: int,
    num_env: int,
    auto_reset: bool,
    max_episode_steps: int = 150,
    num_steps: int = 128,
    num_episodes: int = 10,
):

    # Agents' name
    agent_ids = [f"Agent_{i}" for i in range(n_agents)]

    # Define agent specification
    agent_specs = {
        agent_id: AgentSpec(
            interface=AgentInterface(
                waypoints=True,
                action=ActionSpaceType.LaneWithContinuousSpeed,
                max_episode_steps=max_episode_steps,
            ),
            agent_builder=ChaseViaPointsAgent,
            info_adapter=info_adapter,
        )
        for agent_id in agent_ids
    }

    # Create a callable env constructor. Here, for illustration purposes, each environment is
    # wrapped with a FrameStack wrapper which returns stacked observations for each environment.
    env_frame_stack = lambda env: FrameStack(
        env=env,
        num_stack=2,
    )
    env_constructor = lambda: env_frame_stack(
        HiWayEnv(
            scenarios=scenarios,
            agent_specs=agent_specs,
            sim_name=sim_name,
            headless=headless,
            seed=seed,
        )
    )

    # Build multiple agents
    agents = {
        agent_id: agent_spec.build_agent()
        for agent_id, agent_spec in agent_specs.items()
    }

    # Create parallel environments
    env = AsyncVectorEnv(
        env_constructors=[env_constructor] * num_env,
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

    dones = {"__all__": False}
    batched_dones = [dones] * num_env
    batched_observations = env.reset()

    for _ in range(num_steps):
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
        batched_observations, batched_rewards, batched_dones, batched_infos = env.step(
            batched_actions
        )

        print("-------------------------------")
        print(batched_infos)
        print(batched_dones)

    # # Print score of each agent in each environment after num_steps
    # for index, infos in enumerate(batched_infos):
    #     print(f"Environment {index}:")
    #     for agent_id, val in infos.items():
    #         print(f"{agent_id}: {val['score']}")

    env.close()


def parallel_env_sync(
    agents: Dict[str, Agent], env: gym.Env, num_env: int, num_episodes: int
):
    """Parallel environments with synchronous episodes. Run multiple environments 
    in parallel with `auto_reset=False`. All environments are reset together when 
    all their episodes have finised. New episodes start synchronously in all 
    environments.

    Args:
        agents (Dict[str, Agent]): Ego agents.
        env (gym.Env): Gym env.
        num_env (int): Number of parallel environments.
        num_episodes (int): Number of episodes.
    """

    tot_scores = {agent_id: 0 for agent_id in agents.keys()}

    for _ in range(num_episodes):
        dones = {"__all__": False}
        batched_dones = [dones] * num_env
        batched_observations = env.reset()

        # Iterate until all environments complete an episode each.
        while not all(dones['__all__'] for dones in batched_dones):
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
                for agent_id, score in infos.items():
                    if dones[agent_id]:
                        tot_scores[agent_id] += score

    # Print average episode score of each agent
    ave_scores = {
        agent_id: score/(num_episodes*num_env)
        for agent_id, score in tot_scores.items()
    }
    print("Average episode score:")
    print(f"{ave_scores}")

    env.close()


if __name__ == "__main__":
    parser = default_argument_parser("parallel-environment-example")
    args = parser.parse_args()

    print("\nParallel environments with asynchronous episodes.\n")
    main(
        scenarios=args.scenarios,
        sim_name=args.sim_name,
        headless=args.headless,
        seed=args.seed,
        n_agents=2,
        num_env=2,
        auto_reset=True,
        max_episode_steps=32,
        num_steps=100,
    )

    print("\nParallel environments with synchronous episodes.\n")
    # main(
    #     scenarios=args.scenarios,
    #     sim_name=args.sim_name,
    #     headless=args.headless,
    #     seed=args.seed,
    #     n_agents=2,
    #     num_env=2,
    #     auto_reset=False,
    #     max_episode_steps=150,
    #     num_episodes=3,
    # )
