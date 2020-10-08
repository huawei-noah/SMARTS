import gym
import numpy as np

ACTIONS = [
    "keep_lane",
    "slow_down",
    "change_lane_left",
    "change_lane_right",
]
N_ACTIONS = len(ACTIONS)

DEFAULT_ACTION_SPACE = gym.spaces.Discrete(N_ACTIONS)


def default_action_adapter(model_action):
    if type(model_action) in (int, np.int, np.int8, np.int16, np.int32, np.int64):
        action = model_action
    elif type(model_action) in (list, tuple, np.ndarray):
        action = model_action[0]
    else:
        action = int(model_action.numpy())
    return ACTIONS[action]
