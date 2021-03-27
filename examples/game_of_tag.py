"""Let's play tag!

A predator-prey multi-agent example built on top of RLlib to facilitate further
developments on multi-agent support for HiWay (including design, performance,
research, and scaling).

The predator and prey use separate policies. A predator "catches" its prey when
it collides into the other vehicle. There can be multiple predators and
multiple prey in a map. Social vehicles act as obstacles where both the
predator and prey must avoid them.
"""

### Find a paper for tag: Not found yet
# continuous controller, vs. lane controller
# 2 preys and 2 predators. preys learn to run away from predators.
# should still use ray and rllib? (Or pytorch)

# 9:30 standups M,W,Th
# Tech: Tuesday, Friday
# BM: Monday, Thursday


import argparse
import os
import random
import multiprocessing

import gym
import numpy as np
from typing import List
from ray import tune
from ray.rllib.utils import try_import_tf
from ray.tune.schedulers import PopulationBasedTraining

from smarts.env.rllib_hiway_env import RLlibHiWayEnv
from smarts.core.agent import AgentSpec
from smarts.core.agent_interface import AgentInterface, AgentType, DoneCriteria

from rllib_agent import RLLibTFSavedModelAgent, TrainingModel

tf = try_import_tf()


PREDATOR_IDS = ["PRED1", "PRED2"]
PREY_IDS = ["PREY1", "PREY2"]

# NUM_SOCIAL_VEHICLES = 10 ######### Why we have to define how many social vehicles?

ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]),
    high=np.array([1.0, 1.0, 1.0]),
    dtype=np.float32,
)

NEIGHBORHOOD_VEHICLE_STATES = gym.spaces.Dict(
    {
        "heading": gym.spaces.Box(low=-1 * np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "position": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
    }
)

# Input layer: input layer can be dictionary
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "position": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        # "drivable_area_grid_map": gym.spaces.Box(low=0, high=256, shape=(256, 256, 1)),
        "predator_vehicles": gym.spaces.Tuple(tuple([NEIGHBORHOOD_VEHICLE_STATES]*len(PREDATOR_IDS))),
        "prey_vehicles": gym.spaces.Tuple(tuple([NEIGHBORHOOD_VEHICLE_STATES]*len(PREY_IDS))),
    }
)


def action_adapter(model_action):
    print("Entered action_adapter")
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering])


def get_specfic_vehicle_states(nv_states, wanted_ids: List[str]):
    """ return vehicle states of vehicle that has id in wanted_ids
        append 0 if not enough
    """
    states = [
        {
            "heading": np.array([v.heading]),
            "speed": np.array([v.speed]),
            "position": np.array(v.position),
        }
        for v in nv_states
        if v.id in wanted_ids
    ] 
    states += [
        {
            "heading": np.array([0]),
            "speed": np.array([0]),
            "position": np.array([0, 0, 0]),
        }
    ] * (len(wanted_ids) - len(states))
    return states
    

def observation_adapter(observations):
    nv_states = observations.neighborhood_vehicle_states
    # drivable_area_grid_map = (
    #     np.zeros((256, 256, 1))
    #     if drivable_area_grid_map is None
    #     else observations.drivable_area_grid_map.data
    # )

    predator_states = get_specfic_vehicle_states(nv_states, PREDATOR_IDS)
    prey_states = get_specfic_vehicle_states(nv_states, PREY_IDS)

    ego = observations.ego_vehicle_state
    print("Entered observation_adapter")
    return {
        "steering": np.array([ego.steering]),
        "speed": np.array([ego.speed]),
        "position": np.array(ego.position),
        "predator_vehicles": tuple(predator_states),
        "prey_vehicles": tuple(prey_states),
        # "drivable_area_grid_map": drivable_area_grid_map,
    }

# add a bit of reward for staying alive
def predator_reward_adapter(observations, env_reward_signal):
    """+ if collides with prey
    - if collides with social vehicle
    - if off road
    """
    print("Entered predator_reward_adapter")
    rew = env_reward_signal
    events = observations.events
    for c in observations.events.collisions:
        if c.collidee_id in PREY_IDS:
            rew += 10
        else:
            # Collided with something other than the prey
            rew -= 10
    if events.off_road:
        # have a time limit for 
        rew -= 10 # if 10 then after 100 steps, then it tries to suicide

    predator_pos = observations.ego_vehicle_state.position

    neighborhood_vehicles = observations.neighborhood_vehicle_states
    prey_vehicles = filter(lambda v: v.id in PREY_IDS, neighborhood_vehicles)
    prey_positions = [p.position for p in prey_vehicles]

    # Decreased reward for increased distance away from prey
    # ! check if the penalty is reasonable, staying alive should be sizable enough to keep agent on road or reduce this penalty
    # use the absolute of environment reward to encourage predator to drive around.
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
    print("Entered prey_reward_adapter")
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
    # not neccessary? just reward for staying alive. Remove this reward?

    # rew += 0.1 * min(
    #     [
    #         np.linalg.norm(prey_pos - predator_pos)
    #         for predator_pos in predator_positions
    #     ],
    #     default=0,
    # )

    return rew


rllib_agents = {}
# add custom done criteria
# 1: add on_shoulder as event in observation
# map offset difference between sumo-gui and envision
# agent_interface full

shared_interface = AgentInterface.from_type(AgentType.Standard)
shared_interface.done_criteria = DoneCriteria(off_route=False, off_road=True)
# shared_interface.neighborhood_vehicles = NeighborhoodVehicles(radius=50) # To-do have different radius for prey vs predator

# predator_neighborhood_vehicles=NeighborhoodVehicles(radius=30)
for agent_id in PREDATOR_IDS:
    rllib_agents[agent_id] = {
        "agent_spec": AgentSpec(
            interface=shared_interface,
            agent_builder=lambda: RLLibTFSavedModelAgent( ## maybe fine since it might understand which mode it is in. Try 2 models at first
                os.path.join(os.path.dirname(os.path.realpath(__file__)), "model"),
                OBSERVATION_SPACE,
            ),
            observation_adapter=observation_adapter,
            reward_adapter=predator_reward_adapter,
            action_adapter=action_adapter,
        ),
        "observation_space": OBSERVATION_SPACE,
        "action_space": ACTION_SPACE,
    }

for agent_id in PREY_IDS:
    rllib_agents[agent_id] = {
        "agent_spec": AgentSpec(
            interface=shared_interface,
            agent_builder=lambda: RLLibTFSavedModelAgent(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "model"
                ),  # assume model exists
                OBSERVATION_SPACE,
            ),
            observation_adapter=observation_adapter,
            reward_adapter=prey_reward_adapter,
            action_adapter=action_adapter,
        ),
        "observation_space": OBSERVATION_SPACE,
        "action_space": ACTION_SPACE,
    }


# Add custom metrics to your tensorboard using these callbacks
# see: https://ray.readthedocs.io/en/latest/rllib-training.html#callbacks-and-custom-metrics
def on_episode_start(info):
    episode = info["episode"]
    print("episode {} started".format(episode.episode_id))
    episode.user_data["ego_speed"] = []


def on_episode_step(info):
    episode = info["episode"]
    single_agent_id = list(episode._agent_to_last_obs)[0]
    obs = episode.last_raw_obs_for(single_agent_id)
    episode.user_data["ego_speed"].append(obs["speed"])


def on_episode_end(info):
    episode = info["episode"]
    mean_ego_speed = np.mean(episode.user_data["ego_speed"])
    print(
        "episode {} ended with length {} and mean ego speed {:.2f}".format(
            episode.episode_id, episode.length, mean_ego_speed
        )
    )
    episode.custom_metrics["mean_ego_speed"] = mean_ego_speed


def explore(config):
    # ensure we collect enough timesteps to do sgd
    if config["train_batch_size"] < config["sgd_minibatch_size"] * 2:
        config["train_batch_size"] = config["sgd_minibatch_size"] * 2
    # ensure we run at least one sgd iter
    if config["num_sgd_iter"] < 1:
        config["num_sgd_iter"] = 1
    return config


def main(args):
    pbt = PopulationBasedTraining(
        time_attr="time_total_s",
        metric="episode_reward_mean",
        mode="max",
        perturbation_interval=300,
        resample_probability=0.25,
        # Specifies the mutations of these hyperparams
        hyperparam_mutations={
            "lambda": lambda: random.uniform(0.9, 1.0),
            "clip_param": lambda: random.uniform(0.01, 0.5),
            "lr": [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "num_sgd_iter": lambda: random.randint(1, 30),
            # "sgd_minibatch_size": lambda: random.randint(128, 16384),
            # "train_batch_size": lambda: random.randint(2000, 160000),
            "train_batch_size": lambda: 2000,
        },
        custom_explore_fn=explore,
    )

    rllib_policies = {
        f"{agent_id}_policy": (
            None,
            rllib_agent["observation_space"],
            rllib_agent["action_space"],
            {"model": {"custom_model": TrainingModel.NAME}}, # choose a default model?
        )
        for agent_id, rllib_agent in rllib_agents.items()
    }

    tune_config = {
        "env": RLlibHiWayEnv,
        "log_level": "WARN",
        "num_workers": 1,
        # 'sample_batch_size': 1,  # XXX: 200
        # 'train_batch_size': 1,
        # 'sgd_minibatch_size': 1,
        # 'num_sgd_iter': 1,
        "horizon": 10000,
        "env_config": {
            "seed": 42,
            "sim_name": "game_of_tag",
            "scenarios": [os.path.abspath(args.scenario)],
            "headless": False,
            "sumo_headless": False,
            "agent_specs": {
                agent_id: rllib_agent["agent_spec"]
                for agent_id, rllib_agent in rllib_agents.items()
            },
        },
        "multiagent": {
            "policies": rllib_policies,
            "policy_mapping_fn": lambda agent_id: f"{agent_id}_policy",
        },
        "callbacks": {
            "on_episode_start": on_episode_start,
            "on_episode_step": on_episode_step,
            "on_episode_end": on_episode_end,
        },
    }

    local_dir = os.path.expanduser(args.result_dir)

    analysis = tune.run(
        "PPO",
        name="lets_play_tag",
        # stop={'time_total_s': 60 * 60 * 24},  # 24 hours
        # XXX: Every X iterations perform a _ray actor_ checkpoint (this is
        #      different than _exporting_ a TF/PT checkpoint).
        checkpoint_freq=1,
        checkpoint_at_end=True,
        # XXX: Beware, resuming after changing tune params will not pick up
        #      the new arguments as they are stored alongside the checkpoint.
        resume=args.resume_training,
        local_dir=local_dir,
        reuse_actors=True,
        max_failures=2,
        config=tune_config,
        scheduler=pbt,
    )

    print(analysis.dataframe().head())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("rllib-example")
    parser.add_argument(
        "scenario",
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
    args = parser.parse_args()
    main(args)


# Failure # 1 (occurred at 2021-03-25_18-32-07)
# Traceback (most recent call last):
#   File "/home/kyber/work/SMARTS/.venv/lib/python3.7/site-packages/ray/tune/trial_runner.py", line 726, in _process_trial
#     result = self.trial_executor.fetch_result(trial)
#   File "/home/kyber/work/SMARTS/.venv/lib/python3.7/site-packages/ray/tune/ray_trial_executor.py", line 489, in fetch_result
#     result = ray.get(trial_future[0], timeout=DEFAULT_GET_TIMEOUT)
#   File "/home/kyber/work/SMARTS/.venv/lib/python3.7/site-packages/ray/worker.py", line 1454, in get
#     raise value
# ray.exceptions.RayActorError: The actor died unexpectedly before finishing this task.

