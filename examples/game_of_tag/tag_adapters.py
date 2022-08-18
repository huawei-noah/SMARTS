import gym
import numpy as np
import random
import math
from typing import List
import time
from dataclasses import dataclass

PREDATOR_IDS = ["PRED1"]
PREY_IDS = ["PREY1"]


@dataclass
class Rewards:
    collesion_with_target: float = 10
    offroad: float = 10
    collesion_with_other_deduction: float = -1.5


global_rewards = Rewards()

# vehicles collide at around 3.8 if from behind
# colldie at 2.11 if from side
COLLIDE_DISTANCE = 3.8

ACTION_SPACE = gym.spaces.Tuple(
    (
        gym.spaces.Discrete(4),  # 4 types of speed
        gym.spaces.Discrete(3),  # -1 0 or 1 for lane change
    )
)

NEIGHBORHOOD_VEHICLE_STATES = gym.spaces.Dict(
    {
        "heading": gym.spaces.Box(low=-2 * np.pi, high=2 * np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=-2e2, high=2e2, shape=(1,)),
        "position": gym.spaces.Box(low=-1e4, high=1e4, shape=(2,)),
        "distance": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        "lane_index": gym.spaces.Discrete(5),
    }
)

OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "heading": gym.spaces.Box(low=-1 * np.pi, high=np.pi, shape=(1,)),
        "speed": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        "position": gym.spaces.Box(low=-1e3, high=1e3, shape=(2,)),
        "lane_index": gym.spaces.Discrete(5),
        "target_vehicles": gym.spaces.Tuple(
            tuple([NEIGHBORHOOD_VEHICLE_STATES] * len(PREDATOR_IDS))
        ),
    }
)


def action_adapter(model_action):
    speed, laneChange = model_action
    speeds = [0, 3, 6, 9]
    adapted_action = [speeds[speed], laneChange - 1]
    return adapted_action


def _is_vehicle_wanted(id, wanted_ids: List[str]):
    """This function is needed since agent-id during training would be
    'PREY1-xxxxxxxx' instead of 'PREY1'
    """
    for wanted_id in wanted_ids:
        if wanted_id in id:
            return True
    return False


def get_specific_vehicle_states(nv_states, wanted_ids: List[str], ego_state):
    """return vehicle states of vehicle that has id in wanted_ids"""
    states = [
        {
            "heading": np.array([v.heading]),
            "speed": np.array([v.speed]),
            "position": np.array(v.position[:2]),
            "lane_index": v.lane_index,
            "distance": np.array(
                [np.linalg.norm(v.position[:2] - ego_state.position[:2])]
            ),
        }
        for v in nv_states
        if _is_vehicle_wanted(v.id, wanted_ids)
    ]
    # ego is predator, prey went off road
    if wanted_ids == PREY_IDS:
        # make the last observation bad for prey to discourage off road
        states += [
            {
                "heading": np.array([0]),
                "speed": np.array([0]),
                "position": ego_state.position[:2],
                "lane_index": ego_state.lane_index,
                "distance": np.array([COLLIDE_DISTANCE]),  # give max reward to predator
            }
        ] * (len(wanted_ids) - len(states))
    elif wanted_ids == PREDATOR_IDS:
        # ego is prey, predator went off road
        # make the last observation bad for predator
        states += [
            {
                "heading": np.array([0]),
                "speed": np.array([0]),
                "position": np.array([1000, 1000]),
                "lane_index": ego_state.lane_index,
                "distance": np.array([1e3 - 1]),  # makes  position far from predator
            }
        ] * (len(wanted_ids) - len(states))

    return states


def min_distance_to_rival(ego_position, rival_ids, neighbour_states):
    rival_vehicles = filter(
        lambda v: _is_vehicle_wanted(v.id, rival_ids), neighbour_states
    )
    rival_positions = [p.position for p in rival_vehicles]

    return min(
        [np.linalg.norm(ego_position - prey_pos) for prey_pos in rival_positions],
        default=0,
    )


def observation_adapter(observations):
    nv_states = observations.neighborhood_vehicle_states
    ego = observations.ego_vehicle_state

    target_vehicles = None
    if _is_vehicle_wanted(ego.id, PREY_IDS):
        target_vehicles = get_specific_vehicle_states(nv_states, PREDATOR_IDS, ego)
    elif _is_vehicle_wanted(ego.id, PREDATOR_IDS):
        target_vehicles = get_specific_vehicle_states(nv_states, PREY_IDS, ego)

    return {
        "heading": np.array([ego.heading]),
        "speed": np.array([ego.speed]),
        "position": np.array(ego.position[:2]),
        "lane_index": ego.lane_index,
        "target_vehicles": tuple(target_vehicles),
    }


def dominant_reward(distance):
    if distance == COLLIDE_DISTANCE:
        return 10
    return min(0.5 / ((distance - COLLIDE_DISTANCE) ** 2), 10)


def predator_reward_adapter(observations, env_reward_signal):
    rew = 0
    ego = observations.ego_vehicle_state

    # Primary reward
    distance_to_target = min_distance_to_rival(
        ego.position,
        PREY_IDS,
        observations.neighborhood_vehicle_states,
    )

    rew += dominant_reward(distance_to_target)

    events = observations.events
    for c in observations.events.collisions:
        if _is_vehicle_wanted(c.collidee_id, PREY_IDS):
            rew += global_rewards.collesion_with_target
            print(
                f"predator {ego.id} collided with prey {c.collidee_id} distance {distance_to_target}"
            )
        # # keeping this commented code for expanding to mutiple preys and predators in the future
        # else:
        #     # Collided with something other than the prey
        #     rew += global_rewards.collesion_with_other_deduction
        #     print(f"predator {ego.id} collided with others {c.collidee_id}")

    if events.off_road:
        rew -= global_rewards.offroad

    # if no prey vehicle avaliable, have 0 reward instead
    # TODO: Test to see if this is neccessary
    prey_vehicles = list(
        filter(
            lambda v: _is_vehicle_wanted(v.id, PREY_IDS),
            observations.neighborhood_vehicle_states,
        )
    )
    return rew if len(prey_vehicles) > 0 else 0


def prey_reward_adapter(observations, env_reward_signal):

    rew = 0
    ego = observations.ego_vehicle_state

    # Primary reward
    distance_to_target = min_distance_to_rival(
        ego.position,
        PREDATOR_IDS,
        observations.neighborhood_vehicle_states,
    )
    rew -= dominant_reward(distance_to_target)

    events = observations.events
    for c in events.collisions:
        if _is_vehicle_wanted(c.collidee_id, PREDATOR_IDS):
            rew -= global_rewards.collesion_with_target
            print(
                f"prey {ego.id} collided with Predator {c.collidee_id} distance {distance_to_target}"
            )
        # # keeping this commented code for expanding to mutiple preys and predators in the future
        # else:
        #     # Collided with something other than the prey
        #     rew += global_rewards.collesion_with_other_deduction
        #     print(f"prey {ego.id} collided with other vehicle {c.collidee_id}")

    if events.off_road:
        rew -= global_rewards.offroad

    # if no predator vehicle avaliable, have 0 reward instead
    # TODO: Test to see if this is neccessary
    predator_vehicles = list(
        filter(
            lambda v: _is_vehicle_wanted(v.id, PREDATOR_IDS),
            observations.neighborhood_vehicle_states,
        )
    )
    return rew if len(predator_vehicles) > 0 else 0
