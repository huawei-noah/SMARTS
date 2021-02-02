from ultra.baselines.agent_spec import BaselineAgentSpec
import gym
import numpy as np
import string
import random
from ultra.baselines.adapter import Dummy
from smarts.core.sensors import VehicleObservation

class String(gym.Space):
    def __init__(
                self,
                shape=None,
                min_length=1,
                max_length=180,
            ):
        self.shape = shape
        self.min_length = min_length
        self.max_length = max_length
        self.letters = string.ascii_letters + " .,!-"

    def sample(self):
        length = random.randint(self.min_length, self.max_length)
        string = ""
        for i in range(length):
            letter = random.choice(self.letters)
            string += letter
        return string

    def contains(self, x):
        return type(x) is "str" and len(x) > self.min and len(x) < self.max
#
# class Socials(gym.Space):
#     def __init__(
#                 self,
#                 shape,
#                 dummies=[]
#             ):
#         # self.dummies = dummies
#         self.shape = shape
#         self.dummies = dummies
#         super().__init__((1,),Dummy);
#
#     def sample(self):
#         return np.random.choice(self.dummies)
#
#     def contains(self, x):
#         return  x in self.dummies

class RLlibAgent:
    def __init__(self, action_type, policy_class):
        self._spec = BaselineAgentSpec(
            action_type=action_type, policy_class=policy_class,is_rllib=True
        )

    @property
    def spec(self):
        return self._spec

    # ('Observation ({}) outside given space ({})!', {'speed': 8.68038462032637,
    #'relative_goal_position': array([-133.79996542,   17.0920858 ]),
    #'distance_from_center': 0.0,
    # 'steering': -0.0, 'angle_error': Heading(0.0), 'road_speed': 13.89,
    # 'start': array([101.6       ,  84.40493594]),
    # 'goal': (-32.199965420171935, 101.49702174544765),
    # 'heading': Heading(0.0),
    # 'ego_position': array([101.6       ,  84.40493594,   0.        ])},
    # Dict(angle_error:Box(-3.1415927410125732, 3.1415927410125732, (1,), float32),
    # distance_from_center:Box(-10000000000.0, 10000000000.0, (1,), float32),
    # ego_position:Box(0.0, 10000000000.0, (2,), float32),
    # goal:Box(0.0, 10000000000.0, (2,), float32),
    # relative_goal_position:Box(-10000000000.0, 10000000000.0, (1,), float32),
    # road_speed:Box(0.0, 10000000000.0, (1,), float32),
    # speed:Box(0.0, 10000000000.0, (1,), float32),
    # start:Box(0.0, 10000000000.0, (2,), float32),
    # steering:Box(-10000000000.0, 10000000000.0, (1,), float32)))

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                # "speed": gym.spaces.Box(low=0, high=1e10, shape=(1,)),
                # "relative_goal_position": gym.spaces.Box(
                #     low=np.array([-1e10, -1e10]), high=np.array([1e10, 1e10])
                # ),
                # "distance_from_center": gym.spaces.Box(
                #     low=-1e10, high=1e10, shape=(1,)
                # ),
                # "steering": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                # "angle_error": gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
                "social_vehicles": gym.spaces.Tuple([gym.spaces.Dict({
                    'position':gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
                    "heading":gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                    "speed":gym.spaces.Box(low=0, high=1e10, shape=(1,)),
                })]),
                # "road_speed": gym.spaces.Box(low=0, high=1e10, shape=(1,)),
                # # # ----------
                # # # don't normalize the following:
                # "start": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,)),
                # "goal": gym.spaces.Box(low=-1e10, high=1e10, shape=(2,)),
                # "heading": gym.spaces.Box(low=-1e10, high=1e10, shape=(1,)),
                # # # "goal_path": gym.spaces.Box(low=0, high=1e10, shape=(300,)),
                # "ego_position": gym.spaces.Box(low=-1e10, high=1e10, shape=(3,)),
                # "waypoint_paths": gym.spaces.Box(low=0, high=1e10, shape=(300,)),
                # "events": gym.spaces.Box(low=0, high=100, shape=(15,)),
            }
        )

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=np.array([0.0, 0.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
