import gym
import numpy as np
import random
from typing import List
import time


PREDATOR_IDS = ["PRED1", "PRED2"]
PREY_IDS = ["PREY1", "PREY2"]

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
        # add acceleration?
        # add distance between prey and predator
        "position": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
        "drivable_area_grid_map": gym.spaces.Box(low=0, high=1, shape=(6, 4)), # bitmap, change to a python integer/bitmap(binary number with 0 or 1)
        "predator_vehicles": gym.spaces.Tuple(tuple([NEIGHBORHOOD_VEHICLE_STATES]*len(PREDATOR_IDS))),
        "prey_vehicles": gym.spaces.Tuple(tuple([NEIGHBORHOOD_VEHICLE_STATES]*len(PREY_IDS))),
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
        if _is_vehicle_wanted(v.id, wanted_ids)
    ] 
    states += [
        {
            "heading": np.array([0]),
            "speed": np.array([0]),
            "position": np.array([0, 0, 0]),
        }
    ] * (len(wanted_ids) - len(states))
    return states

def congregate_map(grid_map):
    block_size = 5
    drivable_area = 1
    non_drivable_area = 0
    scaled_width = int(grid_map.shape[0]/block_size)
    scaled_height = int(grid_map.shape[1]/block_size)
    congregated_map = np.zeros((scaled_width, scaled_height))
    for vertical in range(0,grid_map.shape[0],block_size):
        for horizontal in range(0, grid_map.shape[1], block_size):
            portion = grid_map[vertical:vertical+block_size,horizontal:horizontal+block_size]
            drivable = np.count_nonzero(portion > non_drivable_area) > int(block_size*block_size/2) # 12
            congregated_map[int(vertical/block_size),int(horizontal/block_size)] = drivable_area if drivable else non_drivable_area
    return congregated_map

def resize_grid_map(grid_map):
    grid_map = grid_map[100:130,120:140] # vertical take 100->130, horizontal take 120->140, 
    # grid_map is 30x20, 6x4 2d array
    map_6x4 = congregate_map(grid_map)
    return map_6x4

def observation_adapter(observations):
    nv_states = observations.neighborhood_vehicle_states
    drivable_area_grid_map = (
        np.zeros((6,4))
        if observations.drivable_area_grid_map is None
        else resize_grid_map(observations.drivable_area_grid_map.data)
    )

    predator_states = get_specfic_vehicle_states(nv_states, PREDATOR_IDS)
    prey_states = get_specfic_vehicle_states(nv_states, PREY_IDS)

    ego = observations.ego_vehicle_state
    return {
        "steering": np.array([ego.steering]),
        "speed": np.array([ego.speed]),
        "position": np.array(ego.position),
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
    # maybe remove this
    rew = 0.2*np.sum(np.absolute(observations.ego_vehicle_state.linear_velocity)) # encourage predator to drive
    events = observations.events
    for c in observations.events.collisions:
        if _is_vehicle_wanted(c.collidee_id, PREY_IDS):
            rew += 20
            print(f"predator collided with prey {c.collidee_id}")
        else:
            # Collided with something other than the prey
            rew -= 25
            print(f"predator collided with others {c.collidee_id}")

    ###  Check why off_road event is not generated when off_road done creteria is False
    if events.off_road: 
        # if 1 agent goes off_road, both agent receive 0 reward. 
        # if both prey or both predator went off_road, the other agent will receive 0 rewards onwards.
        print("predator offroad")
        # have a time limit for 
        rew -= 30 # if 10 then after 100 steps, then it tries to suicide, too high
    elif events.on_shoulder:
        rew -= 10

    predator_pos = observations.ego_vehicle_state.position
    neighborhood_vehicles = observations.neighborhood_vehicle_states
    prey_vehicles = filter(lambda v: _is_vehicle_wanted(v.id, PREY_IDS), neighborhood_vehicles)
    prey_positions = [p.position for p in prey_vehicles]

    # Decreased reward for increased distance away from prey
    # ! check if the penalty is reasonable, staying alive should be sizable enough to keep agent on road or reduce this penalty
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
    rew = 0.2*np.sum(np.absolute(observations.ego_vehicle_state.linear_velocity)) # encourages driving
    events = observations.events
    for c in events.collisions:
        if _is_vehicle_wanted(c.collidee_id, PREDATOR_IDS):
            rew -= 20
            print(f"prey collided with Predator {c.collidee_id}")
        else:
            # Collided with something other than the prey
            rew -= 25
            print(f"prey collided with other vehicle {c.collidee_id}")
    if events.off_road:
        print("prey offroad")
        rew -= 30
    elif events.on_shoulder:
        rew -= 10

    prey_pos = observations.ego_vehicle_state.position

    neighborhood_vehicles = observations.neighborhood_vehicle_states
    predator_vehicles = filter(lambda v: _is_vehicle_wanted(v.id, PREDATOR_IDS), neighborhood_vehicles)
    predator_positions = [p.position for p in predator_vehicles]

    # Increased reward for increased distance away from predators
    # not neccessary? just reward for staying alive. Remove this reward?
    
    # Penalize "on_shoulder" event after on_shoulder is added. 

    rew += 0.1 * min(
        [
            np.linalg.norm(prey_pos - predator_pos)
            for predator_pos in predator_positions
        ],
        default=0,
    )

    return rew