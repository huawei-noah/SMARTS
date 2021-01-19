# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
from ultra.scenarios.common.begin_time_init_funcs import *

# -----------------------------------------
#             cheat sheet
# -----------------------------------------
# 'north-south': ["north-NS", "south-NS"]
# 'south-north': ["south-SN", "north-SN"]
# 'south-east':  ["south-SN", "east-WE"]
# 'south-west':  ["south-SN", "west-EW"]
# 'north-east':  ["north-NS", "east-WE"]
# 'north-west':  ["north-NS", "west-EW"]
# 'east-north':  ["east-EW", "north-SN"]
# 'east-south': ["east-EW", "south-NS"]
# 'east-west':  ["east-EW", "west-EW"]
# 'west-north': ["west-WE", "north-SN"]
# 'west-south': ["west-WE", "south-NS"]
# 'west-east':  ["west-WE", "east-WE"]
# -----------------------------------------


prob_easy = 0.02
prob_medium = 0.04
prob_heavy = 0.06
behavior_distribution = {
    "default": 0.85,
    "aggressive": 0.05,
    "cautious": 0.05,
    "blocker": 0.05,
}
t_patterns = {
    "no-traffic": {  # t-intersection
        "routes": {
            "south-west": {
                "vehicles": 2,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "south-east": None,
            "south-north": None,  # blocking
            "west-east": None,
            "west-north": None,  # blocking
            "west-south": None,
            "east-west": None,
            "east-north": None,  # blocking
            "east-south": {
                "vehicles": 2,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
        },
        # t-intersection has no north route
    },
    "low-density": {  # t-intersection
        "routes": {
            "south-west": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "south-east": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "south-north": None,  # blocking
            "west-east": {
                "vehicles": 1000,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "west-north": None,  # blocking
            "west-south": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "east-west": {
                "vehicles": 1000,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "east-north": None,  # blocking
            "east-south": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 1000,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
        # t-intersection has no north route
    },
    "mid-density": {  # t-intersection
        "routes": {
            "south-west": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "south-east": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "south-north": None,  # blocking
            "west-east": {
                "vehicles": 1000,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "west-north": None,  # blocking
            "east-west": {
                "vehicles": 1000,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "east-north": None,  # blocking
            "east-south": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "west-south": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 1000,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    "high-density": {  # t-intersection
        "routes": {
            "south-west": {
                "vehicles": 100,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },  # blocking
            "south-east": {
                "vehicles": 100,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },  # blocking
            "south-north": None,  # blocking
            "west-east": {
                "vehicles": 1000,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "west-north": None,  # blocking
            "west-south": {
                "vehicles": 100,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },  # blocking
            "east-west": {
                "vehicles": 1000,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "east-north": None,  # blocking
            "east-south": {
                "vehicles": 100,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": True,
                "deadlock_optimization": False,
            },
            # t-intersection has no north route
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 1000,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    "p-stopwatchers": {  # t-intersection
        "routes": {
            "south-west": None,
            "south-east": None,
            "south-north": None,  # blocking
            "west-east": None,
            "west-north": None,  # blocking
            "west-south": None,
            "east-south": None,
            "east-north": None,  # blocking
            "east-west": {
                "vehicles": 10,
                "distribution": {"default": 0.4, "aggressive": 0.4, "cautious": 0.1},
                "start_end_on_different_lanes_probability": 1,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": 0.0},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
        }
    },
    "p-test": {  # t-intersection
        "routes": {
            "south-west": {
                "vehicles": 100,
                "distribution": {"default": 0.4, "aggressive": 0.4, "cautious": 0.1},
                "start_end_on_different_lanes_probability": 1,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": 0.0},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "south-east": {
                "vehicles": 100,
                "distribution": {"default": 0.4, "aggressive": 0.4, "cautious": 0.1},
                "start_end_on_different_lanes_probability": 1,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": 0.0},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "south-north": None,  # blocking
            "west-east": None,
            "west-north": None,  # blocking
            "west-south": None,
            "east-south": None,
            "east-north": None,  # blocking
            "east-west": {
                "vehicles": 100,
                "distribution": {"default": 1},
                "start_end_on_different_lanes_probability": 1,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": 0.0},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 100,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
}

cross_patterns = {
    "no-traffic": {  # t-intersection
        "routes": {
            "south-west": {
                "vehicles": 2,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "south-east": None,
            "south-north": None,  # blocking
            "west-east": None,
            "west-north": None,  # blocking
            "west-south": None,
            "east-west": None,
            "east-north": None,  # blocking
            "east-south": {
                "vehicles": 2,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "north-south": None,
            "north-east": None,
            "north-west": None,
        },
        # t-intersection has no north route
    },
    "low-density": {  # t-intersection
        "routes": {
            "south-west": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "south-east": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "south-north": {
                "vehicles": 1000,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "west-east": None,
            "west-north": None,  # blocking
            "west-south": None,
            "east-west": None,
            "east-north": None,  # blocking
            "east-south": None,
            "north-south": {
                "vehicles": 1000,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "north-west": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "north-east": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 1000,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    "mid-density": {  # t-intersection
        "routes": {
            "south-west": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "south-east": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "south-north": {
                "vehicles": 1000,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "west-east": None,
            "west-north": None,  # blocking
            "east-west": None,
            "east-north": None,  # blocking
            "east-south": None,
            "west-south": None,
            "north-south": {
                "vehicles": 1000,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "north-east": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
            "north-west": {
                "vehicles": 100,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_medium},
                },
                "has_turn": False,
                "deadlock_optimization": False,
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 1000,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    "high-density": {  # t-intersection
        "routes": {
            "south-west": {
                "vehicles": 100,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },  # blocking
            "south-east": {
                "vehicles": 100,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },  # blocking
            "south-north": {
                "vehicles": 1000,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "west-east": None,
            "west-north": None,  # blocking
            "west-south": None,  # blocking
            "east-west": None,
            "east-north": None,  # blocking
            "east-south": None,
            "north-south": {
                "vehicles": 1000,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "north-east": {
                "vehicles": 100,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "north-west": {
                "vehicles": 100,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_heavy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 1000,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
}


def get_pattern(idx, intersection_type):
    if intersection_type[-2:] == "_t":
        patterns = t_patterns
    elif intersection_type[-2:] == "_c":
        patterns = cross_patterns
    else:
        raise ("Intersection type not detected.")
    if idx in patterns:
        return patterns[idx]
