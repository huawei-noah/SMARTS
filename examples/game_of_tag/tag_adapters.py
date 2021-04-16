import gym
import numpy as np
import random
import math
from typing import List
import time
from dataclasses import dataclass

# questions:
# use state in reward adaptor? have negative reward based on previous states
# use the same model on a different map?
# training procedures

#  predator agent interface 

# start with mininum - just collision reward
## Prey with caught: -1*(0.1)^(timesteps), more deduction 

## continuous agentinterface on figure 8 map?

## messaging channel between agents? sending and receiving actions


PREDATOR_IDS = ["PRED1", "PRED2", "PRED3", "PRED4"]
PREY_IDS = ["PREY1", "PREY2"]

# PREDATOR_IDS = ["PRED1"]
# PREY_IDS = ["PREY1"]

class TrainingState:
    """a class to keep track of training states"""
    timestamp = 0
    """ agent_id: [action_1, action_2]"""
    last_two_actions_of_agent = {}

    @classmethod
    def step(cls):
        cls.timestamp += 1
    
    @classmethod
    def reset(cls):
        cls.timestamp = 0
        cls.last_two_actions_of_agent = {}
        for agent_id in PREY_IDS+PREDATOR_IDS:
            cls.last_two_actions_of_agent[agent_id] = []

    @classmethod
    def update_agent_actions(cls, agent_id, action) -> float:
        """ stores each vehicle's last action, give deduction if action changes
        """
        assert len(action) == 2
        actions_queue = cls.last_two_actions_of_agent[agent_id]
        if len(actions_queue) == 2:
            actions_queue.pop(0)
            actions_queue.append(action)
        else:
            actions_queue.append(action)

    @classmethod
    def punish_if_action_changed(cls, agent_id):
        actions_queue = cls.last_two_actions_of_agent[agent_id]
        if len(actions_queue) < 2:
            return 0
        
        if actions_queue[0][0] != actions_queue[1][0] or actions_queue[0][1] != actions_queue[1][1]:
            return global_rewards.making_change_deduction

        return 0


@dataclass
class Rewards:
    prey_base_reward: float = 0
    pred_base_reward: float = 0
    collesion_with_target: float = 1
    game_ended: float = 1
    collesion_with_other_deduction: float = -1.5
    off_road_deduction: float = -1.5
    on_shoulder_deduction: float = -0.2
    following_prey_reward: float = 0.05
    blocking_prey_reward: float  = 0.03
    making_change_deduction: float = -0.002
    discount_factor: float = 0.999

global_rewards = Rewards()

# ACTION_SPACE = gym.spaces.Box(
#     low=np.array([0.0, 0.0, -1.0]), # throttle can be negative?
#     high=np.array([1.0, 1.0, 1.0]),
#     dtype=np.float32,
# )

ACTION_SPACE = gym.spaces.Tuple((
    gym.spaces.Discrete(5), # 5 types of speed
    gym.spaces.Discrete(3)) # -1 0 or 1 for lane change
    # no message: 0, come_to_me: 1 received by predator vehicles, from observation
)


NEIGHBORHOOD_VEHICLE_STATES = gym.spaces.Dict(
    {
        #"rel_heading": gym.spaces.Box(low=-2 * np.pi, high=2*np.pi, shape=(1,)),
        "rel_speed": gym.spaces.Box(low=-2e2, high=2e2, shape=(1,)),
        "position": gym.spaces.Box(low=-1e4, high=1e4, shape=(2,)),
        "distance": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        # rel_lane_index can be -4, -3, -2, -1, 0, 1, 2, 3, 4
        # -4, -3, -2, -1: larger than current lane
        # 0: same as current lane
        # 1, 2, 3, 4: smaller than current lane
        "rel_lane_index": gym.spaces.Discrete(9), 
    }
)

# Input layer: input layer can be dictionary
OBSERVATION_SPACE = gym.spaces.Dict(
    {
        #"heading": gym.spaces.Box(low=-1 * np.pi, high=np.pi, shape=(1,)),
        #"speed": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        "position": gym.spaces.Box(low=-1e3, high=1e3, shape=(2,)),
        # add distance between prey and predator
        #"min_distance_to_prey": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        #"min_distance_to_predator": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        # "drivable_area_grid_map": gym.spaces.Box(
        #     low=0, high=1, shape=(16, 16)
        # ),  # bitmap, change to a python integer/bitmap(binary number with 0 or 1)
        "predator_vehicles": gym.spaces.Tuple(
            tuple([NEIGHBORHOOD_VEHICLE_STATES] * len(PREDATOR_IDS))
        ),
        "prey_vehicles": gym.spaces.Tuple(
            tuple([NEIGHBORHOOD_VEHICLE_STATES] * len(PREY_IDS))
        ),
    }
)


def action_adapter(model_action, agent_id):
    speed, laneChange = model_action
    speeds = [0, 3, 6, 9, 12]
    adapted_action = [speeds[speed], laneChange-1]
    TrainingState.update_agent_actions(agent_id, adapted_action)
    return adapted_action

def _is_vehicle_wanted(id, wanted_ids: List[str]):
    for wanted_id in wanted_ids:
        if wanted_id in id:
            return True
    return False

def get_specfic_vehicle_states(nv_states, wanted_ids: List[str], ego_state):
    """return vehicle states of vehicle that has id in wanted_ids
    append 0 if not enough
    """
    rel_lane_index_mapping = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    states = [
        {
            #"rel_heading": np.array([v.heading-ego_state.heading]),
            "rel_speed": np.array([v.speed-ego_state.speed]),
            "position": np.array(v.position[:2]), 
            "rel_lane_index": rel_lane_index_mapping.index(v.lane_index - ego_state.lane_index),
            "distance": np.array([np.linalg.norm(v.position[:2] - ego_state.position[:2])]),
        }
        for v in nv_states
        if _is_vehicle_wanted(v.id, wanted_ids)
    ]
    states += [
        {
            #"rel_heading": np.array([0]),
            "rel_speed": np.array([2e2]),
            "position": np.array([10000, 10000]),
            "rel_lane_index": 0,
            "distance":np.array([1e3])
        }
    ] * (len(wanted_ids) - len(states))
    return states

# only used for drivable_grid_map
# def congregate_map(grid_map):
#     block_size = 16
#     drivable_area = 1
#     non_drivable_area = 0
#     scaled_width = int(grid_map.shape[0] / block_size)
#     scaled_height = int(grid_map.shape[1] / block_size)
#     congregated_map = np.zeros((scaled_width, scaled_height))
#     for vertical in range(0, grid_map.shape[0], block_size):
#         for horizontal in range(0, grid_map.shape[1], block_size):
#             portion = grid_map[
#                 vertical : vertical + block_size, horizontal : horizontal + block_size
#             ]
#             drivable = np.count_nonzero(portion > non_drivable_area) > int(
#                 block_size * block_size / 2
#             )  # 12
#             congregated_map[
#                 int(vertical / block_size), int(horizontal / block_size)
#             ] = (drivable_area if drivable else non_drivable_area)
#     return congregated_map


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
    # drivable_area_grid_map = (
    #     np.zeros((6, 4))
    #     if observations.drivable_area_grid_map is None
    #     else congregate_map(observations.drivable_area_grid_map.data)
    # )

    predator_states = get_specfic_vehicle_states(nv_states, PREDATOR_IDS, observations.ego_vehicle_state)
    prey_states = get_specfic_vehicle_states(nv_states, PREY_IDS, observations.ego_vehicle_state)

    ego = observations.ego_vehicle_state
    # min_distance_to_prey = min_distance_to_rival(
    #     observations.ego_vehicle_state.position,
    #     PREY_IDS,
    #     observations.neighborhood_vehicle_states,
    # )
    # min_distance_to_predator = min_distance_to_rival(
    #     observations.ego_vehicle_state.position,
    #     PREDATOR_IDS,
    #     observations.neighborhood_vehicle_states,
    # )
    return {
        #"heading": np.array([ego.heading]),
        #"speed": np.array([ego.speed]),
        "position": np.array(ego.position[:2]),
        #"min_distance_to_prey": np.array([min_distance_to_prey]),
        #"min_distance_to_predator": np.array([min_distance_to_predator]),
        "predator_vehicles": tuple(predator_states),
        "prey_vehicles": tuple(prey_states),
        # "drivable_area_grid_map": drivable_area_grid_map,
    }

def apply_discount(reward):
    return math.pow(global_rewards.discount_factor, TrainingState.timestamp) * reward

def range_within(val, target, range):
    return val - range <= target and target <= val + range

def predator_reward_adapter(observations, env_reward_signal):
    """+ if collides with prey
    - if collides with social vehicle
    - if off road
    """
    collided_with_prey = False
    rew = global_rewards.pred_base_reward
    # rew = 0.2 * np.sum(
    #     np.absolute(observations.ego_vehicle_state.linear_velocity)
    # )  # encourage predator to drive
    events = observations.events
    for c in observations.events.collisions:
        if _is_vehicle_wanted(c.collidee_id, PREY_IDS):
            rew += apply_discount(global_rewards.collesion_with_target)
            collided_with_prey = True
            print(f"predator {observations.ego_vehicle_state.id} collided with prey {c.collidee_id}")
        else:
            # Collided with something other than the prey
            rew += apply_discount(global_rewards.collesion_with_other_deduction)
            print(f"predator {observations.ego_vehicle_state.id} collided with others {c.collidee_id}")

    if events.off_road:
        # if both prey or both predator went off_road, the other agent will receive 0 rewards onwards.
        print("predator offroad")
        # have a time limit for
        rew += apply_discount(global_rewards.off_road_deduction)
    elif events.on_shoulder:
        rew += apply_discount(global_rewards.on_shoulder_deduction)

    # give 0.05 reward for following prey, give 0.1 reward for following and having higher speed than prey
    ego_pos = observations.ego_vehicle_state.position[:2]
    ego_heading = observations.ego_vehicle_state.heading
    ego_speed = observations.ego_vehicle_state.speed
    for v in observations.neighborhood_vehicle_states:
        if not _is_vehicle_wanted(v.id, PREY_IDS):
            continue
        prey_heading = v.heading
        prey_pos = v.position[:2]
        x = abs(ego_pos[0]-prey_pos[0])
        y = abs(ego_pos[1]-prey_pos[1])
        # prey at top left
        cal_heading = math.atan(x/y)
        if prey_pos[0] > ego_pos[0] and prey_pos[1] < ego_pos[1]:
            # prey at bottom right
            cal_heading = -1*math.pi+cal_heading
        elif prey_pos[0] > ego_pos[0] and prey_pos[1] >= ego_pos[1]:
            # prey at top right
            cal_heading = -1 * cal_heading 
        elif prey_pos[0] < ego_pos[0] and prey_pos[1] < ego_pos[1]:
            cal_heading = math.pi - cal_heading
        # checks if predator is chasing after the prey within 20 meters
        distance_to_target = math.sqrt(x*x+y*y)
        if range_within(cal_heading, ego_heading, 0.05) and distance_to_target <= 20:
            # if two vehicle distance <= 20 meters, add 
            rew += apply_discount(global_rewards.following_prey_reward)
            if ego_speed > v.speed:
                rew += apply_discount(global_rewards.following_prey_reward)
                #print(f"predator {observations.ego_vehicle_state.id} chasing prey: {v.id} cal heading {cal_heading} ego_heading {ego_heading}")
        elif range_within(abs(cal_heading) + abs(ego_heading), np.pi, 0.05) and distance_to_target < 10 and ego_speed < v.speed:
            rew += apply_discount(global_rewards.blocking_prey_reward)
            #print(f"predator {observations.ego_vehicle_state.id} blocking prey: {v.id}")
            

    # Decreased reward for increased distance away from prey
    # rew -= (0.005) * min_distance_to_rival(
    #     observations.ego_vehicle_state.position,
    #     PREY_IDS,
    #     observations.neighborhood_vehicle_states,
    # )
    if not collided_with_prey and events.reached_max_episode_steps:
        # predator failed to catch the prey
        rew -= global_rewards.game_ended
    
    rew += apply_discount(TrainingState.punish_if_action_changed(observations.ego_vehicle_state.id.split('-')[0]))

    # if no prey vehicle avaliable, have 0 reward instead
    prey_vehicles = list(filter(
        lambda v: _is_vehicle_wanted(v.id, PREY_IDS), observations.neighborhood_vehicle_states,
    ))
    rew = rew if len(prey_vehicles) > 0 else 0
    #print(f"predator {observations.ego_vehicle_state.id.split('-')[0]} reward: {rew}")
    return rew


def prey_reward_adapter(observations, env_reward_signal):
    """+ based off the distance away from the predator (optional)
    - if collides with prey
    - if collides with social vehicle
    - if off road
    """

    collided_with_pred = False
    rew = global_rewards.prey_base_reward
    # rew = 0.2 * np.sum(
    #     np.absolute(observations.ego_vehicle_state.linear_velocity)
    # )  # encourages driving
    events = observations.events
    for c in events.collisions:
        if _is_vehicle_wanted(c.collidee_id, PREDATOR_IDS):
            rew -= apply_discount(global_rewards.collesion_with_target)
            collided_with_pred = True
            print(f"prey {observations.ego_vehicle_state.id} collided with Predator {c.collidee_id}")
        else:
            # Collided with something other than the prey
            rew += apply_discount(global_rewards.collesion_with_other_deduction)
            print(f"prey {observations.ego_vehicle_state.id} collided with other vehicle {c.collidee_id}")
    if events.off_road:
        print("prey offroad")
        rew += apply_discount(global_rewards.off_road_deduction)
    elif events.on_shoulder:
        rew += apply_discount(global_rewards.on_shoulder_deduction)

    # # Increased reward for increased distance away from predators
    rew += (0.005) * min_distance_to_rival(
        observations.ego_vehicle_state.position,
        PREDATOR_IDS,
        observations.neighborhood_vehicle_states,
    )
    rew += apply_discount(TrainingState.punish_if_action_changed(observations.ego_vehicle_state.id.split('-')[0]))

    if not collided_with_pred and events.reached_max_episode_steps:
        # prey survived
        rew += global_rewards.game_ended

    # if no predator vehicle avaliable, have 0 reward instead
    predator_vehicles = list(filter(
        lambda v: _is_vehicle_wanted(v.id, PREDATOR_IDS), observations.neighborhood_vehicle_states,
    ))
    rew = rew if len(predator_vehicles) > 0 else 0
    #print(f"prey {observations.ego_vehicle_state.id.split('-')[0]} reward: {rew}")
    return rew
