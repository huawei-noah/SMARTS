"""Let's play tag!

A predator-prey multi-agent example built on top of RLlib to facilitate further
developments on multi-agent support for HiWay (including design, performance,
research, and scaling).

The predator and prey use separate policies. A predator "catches" its prey when
it collides into the other vehicle. There can be multiple predators and
multiple prey in a map. Social vehicles act as obstacles where both the
predator and prey must avoid them.
"""
import argparse
import os
import random
import multiprocessing

import gym
import numpy as np
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune.schedulers import PopulationBasedTraining

from examples.rllib_agent import RLLibTFSavedModelAgent, TrainingModel
from examples.game_of_tag.custom_adapters import *

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.core.agent import AgentSpec, Agent
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria
from smarts.core.utils.episodes import episodes


class PredatorAgent(Agent):
    def act(self, obs):
        return [0.5, 0, 0]


class PreyAgent(Agent):
    def act(self, obs):
        return [0.5, 0, 0]  # throttle: 0->1, brake: 0->1, steering -1-> 1


def main(scenario, headless, resume_training, result_dir, seed):
    agent_specs = {}

    shared_interface = AgentInterface.from_type(
        AgentType.Full, max_episode_steps=300
    )  # 100s
    shared_interface.done_criteria = DoneCriteria(off_route=False, off_road=False)
    for agent_id in PREDATOR_IDS:
        agent_specs[agent_id] = AgentSpec(
            interface=shared_interface,
            agent_builder=PredatorAgent,
            observation_adapter=observation_adapter,
            reward_adapter=predator_reward_adapter,
            action_adapter=action_adapter,
        )

    for agent_id in PREY_IDS:
        agent_specs[agent_id] = AgentSpec(
            interface=shared_interface,
            agent_builder=PreyAgent,
            observation_adapter=observation_adapter,
            reward_adapter=prey_reward_adapter,
            action_adapter=action_adapter,
        )

    env = gym.make(
        "smarts.env:hiway-v0",
        scenarios=[scenario],
        agent_specs=agent_specs,
        sim_name="demo",
        headless=headless,
        seed=seed,
    )

    for episode in episodes(n=10):
        agents = {
            agent_id: agent_spec.build_agent()
            for agent_id, agent_spec in agent_specs.items()
        }
        observations = env.reset()
        episode.record_scenario(env.scenario_log)

        dones = {"__all__": False}
        while not dones["__all__"]:
            actions = {
                agent_id: agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }

            observations, rewards, dones, infos = env.step(actions)
            episode.record_step(observations, rewards, dones, infos)
        break
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("game-of-tag-example")
    parser.add_argument(
        "--scenario",
        default="scenarios/demo",
        type=str,
        help="Scenario to run (see scenarios/ for some samples you can use)",
    )
    parser.add_argument(
        "--headless", help="run simulation in headless mode", action="store_true"
    )
    parser.add_argument(
        "--resume_training",
        default=False,
        action="store_true",
        help="Resume the last trained example",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="~/ray_results",
        help="Directory containing results (and checkpointing)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        scenario=args.scenario,
        headless=args.headless,
        resume_training=args.resume_training,
        result_dir=args.result_dir,
        seed=args.seed,
    )
