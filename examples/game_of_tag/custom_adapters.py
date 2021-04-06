import gym
import numpy as np
import random
from typing import List
import time
from dataclasses import dataclass

@dataclass
class Rewards:
    prey_base_reward: float = 1
    pred_base_reward: float = 1
    collesion_with_target: float = 10.0
    game_ended: float = 10
    collesion_with_other_deduction: float = -12.0
    off_road_deduction: float = -12
    on_shoulder_deduction: float = -2

global_rewards = Rewards()

PREDATOR_IDS = ["PRED1", "PRED2"]
PREY_IDS = ["PREY1", "PREY2"]

ACTION_SPACE = gym.spaces.Box(
    low=np.array([0.0, 0.0, -1.0]), # throttle can be negative?
    high=np.array([1.0, 1.0, 1.0]),
    dtype=np.float32,
)

## To-do
# Try with a less complicated map first
# Try to see if predator distance with prey award can be positive


NEIGHBORHOOD_VEHICLE_STATES = gym.spaces.Dict(
    {
        "heading": gym.spaces.Box(low=-1 * np.pi, high=np.pi, shape=(1,)),
        # "speed": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
        "position": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,)),
    }
)

# Input layer: input layer can be dictionary
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        "steering": gym.spaces.Box(low=-1e3, high=1e3, shape=(1,)),
        "speed": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        "position": gym.spaces.Box(low=-1e3, high=1e3, shape=(2,)),
        # add acceleration?
        # add distance between prey and predator
        "min_distance_to_prey": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        "min_distance_to_predator": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        "drivable_area_grid_map": gym.spaces.Box(
            low=0, high=1, shape=(6, 4)
        ),  # bitmap, change to a python integer/bitmap(binary number with 0 or 1)
        "predator_vehicles": gym.spaces.Tuple(
            tuple([NEIGHBORHOOD_VEHICLE_STATES] * len(PREDATOR_IDS))
        ),
        "prey_vehicles": gym.spaces.Tuple(
            tuple([NEIGHBORHOOD_VEHICLE_STATES] * len(PREY_IDS))
        ),
    }
)

# break 256x256 to a sequence of integers
# by grids of 64 and encode it,
# 00001000
# 00000000
# 00000000
# 00000000
# 00001000
# 00000000
# 00000000
# 00000000


def action_adapter(model_action):
    throttle, brake, steering = model_action
    return np.array([throttle, brake, steering])


def _is_vehicle_wanted(id, wanted_ids: List[str]):
    for wanted_id in wanted_ids:
        if wanted_id in id:
            return True
    return False


def get_specfic_vehicle_states(nv_states, wanted_ids: List[str]):
    """return vehicle states of vehicle that has id in wanted_ids
    append 0 if not enough
    """

    states = [
        {
            "heading": np.array([v.heading]),
            # "speed": np.array([v.speed]),
            "position": np.array(v.position[:2]),
        }
        for v in nv_states
        if _is_vehicle_wanted(v.id, wanted_ids)
    ]
    states += [
        {
            "heading": np.array([0]),
            # "speed": np.array([0]),
            "position": np.array([0, 0]),
        }
    ] * (len(wanted_ids) - len(states))
    return states


def congregate_map(grid_map):
    block_size = 5
    drivable_area = 1
    non_drivable_area = 0
    scaled_width = int(grid_map.shape[0] / block_size)
    scaled_height = int(grid_map.shape[1] / block_size)
    congregated_map = np.zeros((scaled_width, scaled_height))
    for vertical in range(0, grid_map.shape[0], block_size):
        for horizontal in range(0, grid_map.shape[1], block_size):
            portion = grid_map[
                vertical : vertical + block_size, horizontal : horizontal + block_size
            ]
            drivable = np.count_nonzero(portion > non_drivable_area) > int(
                block_size * block_size / 2
            )  # 12
            congregated_map[
                int(vertical / block_size), int(horizontal / block_size)
            ] = (drivable_area if drivable else non_drivable_area)
    return congregated_map


def resize_grid_map(grid_map):
    grid_map = grid_map[
        100:130, 120:140
    ]  # vertical take 100->130, horizontal take 120->140,
    # grid_map is 30x20, 6x4 2d array
    map_6x4 = congregate_map(grid_map)
    return map_6x4


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
    # if observations.drivable_area_grid_map:
    #     np.save("grid_map", observations.drivable_area_grid_map.data)
    drivable_area_grid_map = (
        np.zeros((6, 4))
        if observations.drivable_area_grid_map is None
        else resize_grid_map(observations.drivable_area_grid_map.data)
    )

    predator_states = get_specfic_vehicle_states(nv_states, PREDATOR_IDS)
    prey_states = get_specfic_vehicle_states(nv_states, PREY_IDS)

    ego = observations.ego_vehicle_state
    return {
        "steering": np.array([ego.steering]),
        "speed": np.array([ego.speed]),
        "position": np.array(ego.position[:2]),
        "min_distance_to_prey": np.array([min_distance_to_rival(
            observations.ego_vehicle_state.position,
            PREY_IDS,
            observations.neighborhood_vehicle_states,
        )]),
        "min_distance_to_predator": np.array([min_distance_to_rival(
            observations.ego_vehicle_state.position,
            PREDATOR_IDS,
            observations.neighborhood_vehicle_states,
        )]),
        "predator_vehicles": tuple(predator_states),
        "prey_vehicles": tuple(prey_states),
        "drivable_area_grid_map": drivable_area_grid_map,
    }


# add a bit of reward for staying alive
def predator_reward_adapter(observations, env_reward_signal):
    """+ if collides with prey
    - if collides with social vehicle
    - if off road
    """
    ## small negative reward for drivable_grid_map have on_drivable blocks

    collided_with_prey = False
    rew = global_rewards.pred_base_reward
    # rew = 0.2 * np.sum(
    #     np.absolute(observations.ego_vehicle_state.linear_velocity)
    # )  # encourage predator to drive
    events = observations.events
    for c in observations.events.collisions:
        if _is_vehicle_wanted(c.collidee_id, PREY_IDS):
            rew += global_rewards.collesion_with_target
            collided_with_prey = True
            print(f"predator collided with prey {c.collidee_id}")
        else:
            # Collided with something other than the prey
            rew += global_rewards.collesion_with_other_deduction
            print(f"predator collided with others {c.collidee_id}")

    if events.off_road:
        # if both prey or both predator went off_road, the other agent will receive 0 rewards onwards.
        print("predator offroad")
        # have a time limit for
        rew += global_rewards.off_road_deduction 
    elif events.on_shoulder:
        rew -= global_rewards.on_shoulder_deduction

    # Decreased reward for increased distance away from prey
    # ! check if the penalty is reasonable, staying alive should be sizable enough to keep agent on road or reduce this penalty
    # rew -= 0.1 * min_distance_to_rival(
    #     observations.ego_vehicle_state.position,
    #     PREY_IDS,
    #     observations.neighborhood_vehicle_states,
    # )
    if not collided_with_prey and events.reached_max_episode_steps:
        # predator failed to catch the prey
        rew -= global_rewards.game_ended

    # if no prey vehicle avaliable, have 0 reward instead
    prey_vehicles = list(filter(
        lambda v: _is_vehicle_wanted(v.id, PREY_IDS), observations.neighborhood_vehicle_states,
    ))
    return rew if len(prey_vehicles) > 0 else 0


def prey_reward_adapter(observations, env_reward_signal):
    """+ based off the distance away from the predator (optional)
    - if collides with prey
    - if collides with social vehicle
    - if off road
    """
    collided_with_pred = False
    rew = -1*global_rewards.prey_base_reward
    # rew = 0.2 * np.sum(
    #     np.absolute(observations.ego_vehicle_state.linear_velocity)
    # )  # encourages driving
    events = observations.events
    for c in events.collisions:
        if _is_vehicle_wanted(c.collidee_id, PREDATOR_IDS):
            rew -= global_rewards.collesion_with_target
            collided_with_pred = True
            print(f"prey collided with Predator {c.collidee_id}")
        else:
            # Collided with something other than the prey
            rew += global_rewards.collesion_with_other_deduction 
            print(f"prey collided with other vehicle {c.collidee_id}")
    if events.off_road:
        print("prey offroad")
        rew += global_rewards.off_road_deduction
    elif events.on_shoulder:
        rew += global_rewards.on_shoulder_deduction

    # Increased reward for increased distance away from predators
    # rew += 0.1 * min_distance_to_rival(
    #     observations.ego_vehicle_state.position,
    #     PREDATOR_IDS,
    #     observations.neighborhood_vehicle_states,
    # )

    if not collided_with_pred and events.reached_max_episode_steps:
        # prey survived
        rew += global_rewards.game_ended

    # if no predator vehicle avaliable, have 0 reward instead
    predator_vehicles = list(filter(
        lambda v: _is_vehicle_wanted(v.id, PREDATOR_IDS), observations.neighborhood_vehicle_states,
    ))
    return rew if len(predator_vehicles) > 0 else 0
