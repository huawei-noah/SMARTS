import gymnasium as gym

from smarts.env.utils.action_conversion import ActionOptions

gym.logger.set_level(40)

import sys
from functools import partial
from pathlib import Path
from typing import Dict, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parents[1].absolute()))
from examples.tools.argument_parser import default_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface
from smarts.core.controllers import ActionSpaceType
from smarts.core.observations import Observation
from smarts.env.gymnasium.wrappers.parallel_env import ParallelEnv
from smarts.env.utils.observation_conversion import ObservationOptions
from smarts.sstudio.scenario_construction import build_scenarios


class LaneAgent(Agent):
    def act(self, obs: Observation) -> Tuple[float, int]:
        speed_limit = obs.waypoint_paths[0][0].speed_limit
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
    agent_interfaces = {
        agent_id: AgentInterface(
            top_down_rgb=True,
            waypoint_paths=True,
            action=ActionSpaceType.LaneWithContinuousSpeed,
            max_episode_steps=max_episode_steps,
        )
        for agent_id in agent_ids
    }

    # Unique `sim_name` is required by each HiWayEnvV1 in order to be displayed
    # in Envision.
    env_constructor = lambda sim_name, seed: gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces=agent_interfaces,
        sim_name=sim_name,
        headless=headless,
        observation_options=ObservationOptions.unformatted,
        action_options=ActionOptions.unformatted,
        seed=seed,
    )
    # A list of env constructors of type `Callable[[int], gym.Env]`
    env_constructors = [
        partial(env_constructor, sim_name=f"{sim_name}_{ind}") for ind in range(num_env)
    ]

    # Build multiple agents
    agents = {agent_id: LaneAgent() for agent_id in agent_interfaces.keys()}

    # Create parallel environments
    env = ParallelEnv(
        env_constructors=env_constructors,
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

    batched_terminateds = [{"__all__": False} for _ in range(num_env)]
    batched_truncateds = [{"__all__": False} for _ in range(num_env)]
    batched_observations, _ = env.reset()

    for _ in range(num_steps):
        # Compute actions for all active(i.e., not done) agents
        batched_actions = []
        for observations, terminateds, truncateds in zip(
            batched_observations, batched_terminateds, batched_truncateds
        ):
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
                if (
                    not terminateds.get(agent_id, False)
                    and not truncateds.get(agent_id, False)
                )
                or terminateds[
                    "__all__"
                ]  # `terminateds[__all__]==True` implies the env was auto-reset in previous iteration
            }
            batched_actions.append(actions)

        # Step all environments in parallel
        (
            batched_observations,
            batched_rewards,
            batched_terminateds,
            batched_truncateds,
            batched_infos,
        ) = env.step(batched_actions)

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

    for _ in range(num_episodes):
        batched_terminateds = [{"__all__": False} for _ in range(num_env)]
        batched_truncateds = [{"__all__": False} for _ in range(num_env)]
        batched_observations, _ = env.reset()

        # Iterate until all environments complete an episode each.
        while not all(terminateds["__all__"] for terminateds in batched_terminateds):
            # Compute actions for all active(i.e., not done) agents
            batched_actions = []
            for observations, terminateds, truncateds in zip(
                batched_observations, batched_terminateds, batched_truncateds
            ):
                actions = {
                    agent_id: agents[agent_id].act(agent_obs)
                    for agent_id, agent_obs in observations.items()
                    if not terminateds.get(agent_id, False)
                    and not truncateds.get(agent_id, False)
                }
                batched_actions.append(actions)

            # Step all environments in parallel
            (
                batched_observations,
                batched_rewards,
                batched_terminateds,
                batched_truncateds,
                batched_infos,
            ) = env.step(batched_actions)

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

    if not args.sim_name:
        args.sim_name = "par_env"

    if not args.scenarios:
        args.scenarios = [
            str(
                Path(__file__).absolute().parents[1]
                / "scenarios"
                / "sumo"
                / "figure_eight"
            )
        ]

    build_scenarios(scenarios=args.scenarios)

    print("\nParallel environments with asynchronous episodes.")
    main(
        scenarios=args.scenarios,
        sim_name=f"{args.sim_name}_async",
        headless=True,
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
        sim_name=f"{args.sim_name}_sync",
        headless=True,
        seed=args.seed,
        num_agents=args.num_agents,
        num_stack=args.num_stack,
        num_env=args.num_env,
        auto_reset=False,
        max_episode_steps=args.max_episode_steps,
        num_episodes=args.episodes,
    )
