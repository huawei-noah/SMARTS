import gym
import numpy as np
import random
import math
from typing import List
import time
from dataclasses import dataclass

# PREDATOR_IDS = ["PRED1", "PRED2", "PRED3", "PRED4"]
# PREY_IDS = ["PREY1", "PREY2"]

PREDATOR_IDS = ["PRED1"]
PREY_IDS = ["PREY1"]

@dataclass
class Rewards:
    collesion_with_target: float = 10
    game_ended: float = 1
    collesion_with_other_deduction: float = -1.5
    off_road_deduction: float = -1.5
    on_shoulder_deduction: float = -0.2
    following_prey_reward: float = 0.05
    blocking_prey_reward: float  = 0.03

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
        "speed": gym.spaces.Box(low=-2e2, high=2e2, shape=(1,)),
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
        "speed": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        "position": gym.spaces.Box(low=-1e3, high=1e3, shape=(2,)),
        #"distance_to_curb": gym.spaces.Box(low=-1e2, high=1e2, shape=(1,)),
        # Prey or predator

        # add distance between prey and predator
        #"min_distance_to_prey": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        #"min_distance_to_predator": gym.spaces.Box(low=0, high=1e3, shape=(1,)),
        # "drivable_area_grid_map": gym.spaces.Box(
        #     low=0, high=1, shape=(16, 16)
        # ),  # bitmap, change to a python integer/bitmap(binary number with 0 or 1)
        "target_vehicles": gym.spaces.Tuple(
            tuple([NEIGHBORHOOD_VEHICLE_STATES] * len(PREDATOR_IDS))
        ),
        # "prey_vehicles": gym.spaces.Tuple(
        #     tuple([NEIGHBORHOOD_VEHICLE_STATES] * len(PREY_IDS))
        # ),
    }
)


def action_adapter(model_action, agent_id):
    speed, laneChange = model_action
    speeds = [0, 3, 6, 9, 12]
    adapted_action = [speeds[speed], laneChange-1]
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
            "speed": np.array([v.speed]),
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
            "speed": np.array([2e2]),
            "position": np.array([10000, 10000]),
            "rel_lane_index": 0,
            "distance":np.array([1e3])
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

def calculate_x_y_of_angle(tangent,angle):
    return tangent * math.cos(angle), tangent * math.sin(angle)

def new_axis_position(pos1, pos2, axis, value):
    # since the lane width portion should always make the distance between 2 points longer
    if abs(pos1[axis] + value - pos2[axis]) > abs(pos1[axis] - pos2[axis]):
        return pos1[axis] + value
    
    return pos1[axis] - value

def calculate_distance_to_road_curb(observations):
    ego = observations.ego_vehicle_state
    pos_data = [[points[0].pos, points[0].lane_width] for points in observations.waypoint_paths]
    if len(pos_data) > 1 and np.any(pos_data[0][0] != pos_data[-1][0]):
        first_pos = pos_data[0][0]
        last_pos = pos_data[-1][0]
        first_width = pos_data[0][1]/2 
        last_width = pos_data[-1][1]/2

        # find slope
        slope = (first_pos[1]-last_pos[1]) / (first_pos[0]-last_pos[0])

        # find angle
        angle = math.atan(slope)

        # find 2 edge points
        x, y  = calculate_x_y_of_angle(first_width, angle)
        y = new_axis_position(first_pos, last_pos, 1, y)
        x = new_axis_position(first_pos, last_pos, 0, x)
        first_pos_edge_ptr = [x, y]

        x, y = calculate_x_y_of_angle(last_width, angle)
        y = new_axis_position(last_pos, first_pos, 1, y)
        x = new_axis_position(last_pos, first_pos, 0, x)
        last_pos_edge_ptr = [x, y]

        # find distance to the edge points
        distance =  min(
            np.linalg.norm(ego.position[:2] - np.array(first_pos_edge_ptr)),
            np.linalg.norm(ego.position[:2] - np.array(last_pos_edge_ptr)),
        )
        print(f"ego id: {ego.id} distance: {distance}")
        return distance
    else:
        pos_data = [[[p.pos, p.lane_width] for p in points] for points in observations.waypoint_paths]
        first_pos = pos_data[0][0][0]
        last_pos = pos_data[0][2][0] # assume at least 3 waypoints
        first_width = pos_data[0][0][1]/2 

        slope = (first_pos[1]-last_pos[1]) / (first_pos[0]-last_pos[0])
        # perpendicular slope
        slope = -1 * 1/slope
        angle = math.atan(slope)
        # find 2 edge points
        x, y  = calculate_x_y_of_angle(first_width, angle)
        p1 = [first_pos[0] + x, first_pos[1] + y]
        p2 = [first_pos[0] - x, first_pos[1] - y]
        distance =  min(
            np.linalg.norm(ego.position[:2] - np.array(p1)),
            np.linalg.norm(ego.position[:2] - np.array(p2)),
        )
        print(f"ego id: {ego.id} distance: {distance}")
        return distance


def observation_adapter(observations):
    distance = calculate_distance_to_road_curb(observations)

    nv_states = observations.neighborhood_vehicle_states
    # if observations.drivable_area_grid_map:
    #     np.save("grid_map", observations.drivable_area_grid_map.data)
    # drivable_area_grid_map = (
    #     np.zeros((6, 4))
    #     if observations.drivable_area_grid_map is None
    #     else congregate_map(observations.drivable_area_grid_map.data)
    # )
    target_vehicles = None
    if _is_vehicle_wanted(observations.ego_vehicle_state.id, PREY_IDS):
        target_vehicles = get_specfic_vehicle_states(nv_states, PREDATOR_IDS, observations.ego_vehicle_state)
    elif _is_vehicle_wanted(observations.ego_vehicle_state.id, PREDATOR_IDS):
        target_vehicles = get_specfic_vehicle_states(nv_states, PREY_IDS, observations.ego_vehicle_state)

    # predator_states = get_specfic_vehicle_states(nv_states, PREDATOR_IDS, observations.ego_vehicle_state)
    # prey_states = get_specfic_vehicle_states(nv_states, PREY_IDS, observations.ego_vehicle_state)

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
        "speed": np.array([observations.ego_vehicle_state.speed]),
        "position": np.array(observations.ego_vehicle_state.position[:2]),
        "target_vehicles": tuple(target_vehicles)
        #"distance_to_curb":np.array([distance]),
        #"min_distance_to_prey": np.array([min_distance_to_prey]),
        #"min_distance_to_predator": np.array([min_distance_to_predator]),
        # "predator_vehicles": tuple(predator_states),
        # "prey_vehicles": tuple(prey_states),
        # "drivable_area_grid_map": drivable_area_grid_map,
    }

def distance_to_curb_reward(observations):
    # 3 log(distance) - 0.5
    distance = calculate_distance_to_road_curb(observations)
    return 3 * math.log10(calculate_distance_to_road_curb(observations)) - 1.5


def range_within(val, target, range):
    return val - range <= target and target <= val + range

def dominant_reward(distance):
    return 5*(1/(distance*distance))

def predator_reward_adapter(observations, env_reward_signal):
    """+ if collides with prey
    - if collides with social vehicle
    - if off road
    """
    rew = 0

    distance_to_target = min_distance_to_rival(
        observations.ego_vehicle_state.position,
        PREY_IDS,
        observations.neighborhood_vehicle_states,
    )
    rew += max(dominant_reward(distance_to_target), 10)

    # rew += distance_to_curb_reward(observations)
    # rew = 0.2 * np.sum(
    #     np.absolute(observations.ego_vehicle_state.linear_velocity)
    # )  # encourage predator to drive
    events = observations.events
    for c in observations.events.collisions:
        if _is_vehicle_wanted(c.collidee_id, PREY_IDS):
            rew += global_rewards.collesion_with_target
            print(f"predator {observations.ego_vehicle_state.id} collided with prey {c.collidee_id}")
        # else:
        #     # Collided with something other than the prey
        #     rew += global_rewards.collesion_with_other_deduction
        #     print(f"predator {observations.ego_vehicle_state.id} collided with others {c.collidee_id}")

    # if events.off_road:
    #     # if both prey or both predator went off_road, the other agent will receive 0 rewards onwards.
    #     print("predator offroad")
    #     # have a time limit for
    #     rew += global_rewards.off_road_deduction

    # give 0.05 reward for following prey, give 0.1 reward for following and having higher speed than prey
    # ego_pos = observations.ego_vehicle_state.position[:2]
    # ego_heading = observations.ego_vehicle_state.heading
    # ego_speed = observations.ego_vehicle_state.speed
    # for v in observations.neighborhood_vehicle_states:
    #     if not _is_vehicle_wanted(v.id, PREY_IDS):
    #         continue
    #     prey_heading = v.heading
    #     prey_pos = v.position[:2]
    #     x = abs(ego_pos[0]-prey_pos[0])
    #     y = abs(ego_pos[1]-prey_pos[1])
    #     # prey at top left
    #     cal_heading = math.atan(x/y)
    #     if prey_pos[0] > ego_pos[0] and prey_pos[1] < ego_pos[1]:
    #         # prey at bottom right
    #         cal_heading = -1*math.pi+cal_heading
    #     elif prey_pos[0] > ego_pos[0] and prey_pos[1] >= ego_pos[1]:
    #         # prey at top right
    #         cal_heading = -1 * cal_heading 
    #     elif prey_pos[0] < ego_pos[0] and prey_pos[1] < ego_pos[1]:
    #         cal_heading = math.pi - cal_heading
    #     # checks if predator is chasing after the prey within 20 meters
    #     distance_to_target = math.sqrt(x*x+y*y)
    #     if range_within(cal_heading, ego_heading, 0.05) and distance_to_target <= 20:
    #         # if two vehicle distance <= 20 meters, add 
    #         rew += global_rewards.following_prey_reward
    #         if ego_speed > v.speed:
    #             rew += global_rewards.following_prey_reward
    #             #print(f"predator {observations.ego_vehicle_state.id} chasing prey: {v.id} cal heading {cal_heading} ego_heading {ego_heading}")
    #     elif range_within(abs(cal_heading) + abs(ego_heading), np.pi, 0.05) and distance_to_target < 10 and ego_speed < v.speed:
    #         rew += global_rewards.blocking_prey_reward
    #         #print(f"predator {observations.ego_vehicle_state.id} blocking prey: {v.id}")
            

    # Decreased reward for increased distance away from prey
    # rew -= (0.005) * min_distance_to_rival(
    #     observations.ego_vehicle_state.position,
    #     PREY_IDS,
    #     observations.neighborhood_vehicle_states,
    # )
    # if not collided_with_prey and events.reached_max_episode_steps:
    #     # predator failed to catch the prey
    #     rew -= global_rewards.game_ended
    
    # if no prey vehicle avaliable, have 0 reward instead
    prey_vehicles = list(filter(
        lambda v: _is_vehicle_wanted(v.id, PREY_IDS), observations.neighborhood_vehicle_states,
    ))
    rew = rew if len(prey_vehicles) > 0 else 0
    print(f"predator {observations.ego_vehicle_state.id.split('-')[0]} reward: {rew}")
    return rew


def prey_reward_adapter(observations, env_reward_signal):
    """+ based off the distance away from the predator (optional)
    - if collides with prey
    - if collides with social vehicle
    - if off road
    """

    rew = 0
    #rew += distance_to_curb_reward(observations)
    # rew = 0.2 * np.sum(
    #     np.absolute(observations.ego_vehicle_state.linear_velocity)
    # )  # encourages driving
    distance_to_target = min_distance_to_rival(
        observations.ego_vehicle_state.position,
        PREDATOR_IDS,
        observations.neighborhood_vehicle_states,
    )
    # set min on the reward
    rew -= min(dominant_reward(distance_to_target, -10),

    events = observations.events
    for c in events.collisions:
        if _is_vehicle_wanted(c.collidee_id, PREDATOR_IDS):
            rew -= global_rewards.collesion_with_target
            print(f"prey {observations.ego_vehicle_state.id} collided with Predator {c.collidee_id}")
        # else:
        #     # Collided with something other than the prey
        #     rew += global_rewards.collesion_with_other_deduction
        #     print(f"prey {observations.ego_vehicle_state.id} collided with other vehicle {c.collidee_id}")
    # if events.off_road:
    #     print("prey offroad")
    #     rew += global_rewards.off_road_deduction


    # # Increased reward for increased distance away from predators
    # rew += (0.005) * min_distance_to_rival(
    #     observations.ego_vehicle_state.position,
    #     PREDATOR_IDS,
    #     observations.neighborhood_vehicle_states,
    # )

    # if no predator vehicle avaliable, have 0 reward instead
    predator_vehicles = list(filter(
        lambda v: _is_vehicle_wanted(v.id, PREDATOR_IDS), observations.neighborhood_vehicle_states,
    ))
    rew = rew if len(predator_vehicles) > 0 else 0
    print(f"prey {observations.ego_vehicle_state.id.split('-')[0]} reward: {rew}")
    return rew
