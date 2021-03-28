import gym
import numpy as np
from typing import List


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
    
    # Penalize "on_shoulder" event after on_shoulder is added. 

    # rew += 0.1 * min(
    #     [
    #         np.linalg.norm(prey_pos - predator_pos)
    #         for predator_pos in predator_positions
    #     ],
    #     default=0,
    # )

    return rew