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

from smarts.env.hiway_env import HiWayEnv
from smarts.core.agent import AgentSpec, Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.utils.episodes import episodes

PREDATOR_IDS = ["PRED1", "PRED2"]
PREY_IDS = ["PREY1", "PREY2"]

ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32,
)


def action_adapter(model_action):
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering])

# create different observation adaptor for prev and predator
# prev should see further than predator (All angles), predator maybe frontal vision
def observation_adapter(observations):
    nv_states = observations.neighborhood_vehicle_states

    nv_states = [
        {
            "heading": np.array([v.heading]),
            "speed": np.array([v.speed]),
            "position": np.array(v.position),
        }
        for v in nv_states
    ]

    ego = observations.ego_vehicle_state
    return {
        "steering": np.array([ego.steering]),
        "speed": np.array([ego.speed]),
        "position": np.array(ego.position),
        "neighborhood_vehicle_states": tuple(nv_states),
    }

def predator_reward_adapter(observations, env_reward_signal):
    """+ if collides with prey
    - if collides with social vehicle
    - if off road
    """
    rew = env_reward_signal
    events = observations.events
    for c in observations.events.collisions:
        if c.collidee_id in PREY_IDS:
            rew += 10
        else:
            # Collided with something other than the prey
            rew -= 10
    if events.off_road:
        rew -= 10

    predator_pos = observations.ego_vehicle_state.position

    neighborhood_vehicles = observations.neighborhood_vehicle_states
    prey_vehicles = filter(lambda v: v.id in PREY_IDS, neighborhood_vehicles)
    prey_positions = [p.position for p in prey_vehicles]

    # Decreased reward for increased distance away from prey
    rew -= 0.1 * min(
        [np.linalg.norm(predator_pos - prey_pos) for prey_pos in prey_positions],
        default=0,
    )

    return rew

def prey_reward_adapter(observations, env_reward_signal):
    """+ based off the distance away from the predator (optional)
    - if collides with prey
    - if collides with social vehicle
    - if off road
    """
    rew = env_reward_signal
    events = observations.events
    for c in events.collisions:
        rew -= 10
    if events.off_road:
        rew -= 10

    prey_pos = observations.ego_vehicle_state.position

    neighborhood_vehicles = observations.neighborhood_vehicle_states
    predator_vehicles = filter(lambda v: v.id in PREDATOR_IDS, neighborhood_vehicles)
    predator_positions = [p.position for p in predator_vehicles]

    # Increased reward for increased distance away from predators
    rew += 0.1 * min(
        [
            np.linalg.norm(prey_pos - predator_pos)
            for predator_pos in predator_positions
        ],
        default=0,
    )

    return rew

class PredatorAgent(Agent):
    def act(self, obs):
        return [0.5, 0, 1]

class PreyAgent(Agent):
    def act(self, obs):
        return [0.5, 0, -1] # throttle: 0->1, brake: 0->1, steering -1-> 1

def main(scenario, headless, resume_training, result_dir, seed):
    agent_specs = {}
    for agent_id in PREDATOR_IDS:
        agent_specs[agent_id] = AgentSpec(
            interface=AgentInterface.from_type(AgentType.Standard),
            agent_builder=PredatorAgent,
            observation_adapter=observation_adapter,
            reward_adapter=predator_reward_adapter,
            action_adapter=action_adapter,
        )

    for agent_id in PREY_IDS:
        agent_specs[agent_id] = AgentSpec(
            interface=AgentInterface.from_type(AgentType.Standard),
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