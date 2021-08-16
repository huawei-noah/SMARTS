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
prob_heavy = 0.08
behavior_distribution = {
    "default": 0.70,
    "aggressive": 0.20,
    "cautious": 0.08,
    "blocker": 0.02,
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
    "blocks": {  # t-intersection
        "routes": {
            "south-north": None,
            "south-east": None,
            "south-west": {
                "vehicles": 10,
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
            "west-south": None,
            "east-west": {
                "vehicles": 10,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "east-south": None,
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 10,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
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
                "vehicles": 200,
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
                "vehicles": 200,
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
            "wait_to_hijack_limit_s": 10,
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
                "vehicles": 200,
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
                "vehicles": 200,
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
            "wait_to_hijack_limit_s": 10,
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
                "distribution": {
                    "default": 0.80,
                    "aggressive": 0.20,
                    "cautious": 0.00,
                    "blocker": 0.0,
                },
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (60, 80),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": True,
                "deadlock_optimization": False,
            },  # blocking
            "south-north": None,  # blocking
            "west-east": {
                "vehicles": 200,
                "distribution": {
                    "default": 0.80,
                    "aggressive": 0.20,
                    "cautious": 0.00,
                    "blocker": 0.0,
                },
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (2, 4),
                        "time_between_cluster": (15, 35),
                        "time_for_each_cluster": 10,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
            },
            "west-north": None,  # blocking
            "west-south": {
                "vehicles": 100,
                "distribution": {
                    "default": 0.7,
                    "aggressive": 0.29,
                    "cautious": 0.01,
                    "blocker": 0.00,
                },
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 2),
                        "time_between_cluster": (30, 60),
                        "time_for_each_cluster": 10,
                    },
                },
                "has_turn": True,
                "deadlock_optimization": False,
            },  # blocking
            "east-west": {
                "vehicles": 200,
                "distribution": {
                    "default": 0.80,
                    "aggressive": 0.20,
                    "cautious": 0.00,
                    "blocker": 0.0,
                },
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (2, 4),
                        "time_between_cluster": (15, 35),
                        "time_for_each_cluster": 10,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
            },
            "east-north": None,  # blocking
            "east-south": {
                "vehicles": 100,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 2),
                        "time_between_cluster": (60, 70),
                        "time_for_each_cluster": 10,
                    },
                },
                "has_turn": True,
                "deadlock_optimization": False,
            },
            # t-intersection has no north route
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 10,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    # -------------------------------------------------- Intersection specific traffic distribution ------------------------------------------------
    # The following traffic distributions [low-interaction, mid-interaction, high-interaction] are created to emphasize interaction between ego and
    # social vehicles at the intersections (T or Cross). The key differences between these distributions and the {low, mid, high}-density are that they
    # do not focus on interactions beyond the intersection, use a very limited number of social vehicles, and there are no social vehicles in the ego
    # mission route. In terms of implementation, these distribution will be used inside the simple level, where the goal is to make the ego agent
    # familiar with the intersection
    "low-interaction": {
        "routes": {
            "south-west": None,
            "south-east": None,
            "south-north": None,  # blocking
            "west-east": {
                "vehicles": 1,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 15),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (160, 165),
                    "end": (20, 20),
                },
            },
            "west-north": None,  # blocking
            "west-south": None,
            "east-south": None,
            "east-north": None,  # blocking
            "east-west": None,
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 2,
            "start_time": 3,  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    "mid-interaction": {
        "routes": {
            "south-west": None,
            "south-east": None,
            "south-north": None,  # blocking
            "west-east": {
                "vehicles": 1,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 10),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (160, 165),
                    "end": (50, 50),
                },
            },
            "west-north": None,  # blocking
            "west-south": None,
            "east-south": {
                "vehicles": 1,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 20),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (160, 165),
                    "end": (50, 50),
                },
            },
            "east-north": None,  # blocking
            "east-west": {
                "vehicles": 1,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 10),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (160, 165),
                    "end": (50, 50),
                },
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 2,
            "start_time": 3,  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    "high-interaction": {  # t-intersection
        "routes": {
            "south-west": None,
            "south-east": None,
            "south-north": None,  # blocking
            "west-east": {
                "vehicles": 2,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 10),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (130, 140),
                    "end": (150, 160),
                },
            },
            "east-west": {
                "vehicles": 2,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 10),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (130, 140),
                    "end": (150, 160),
                },
            },
            "west-north": None,  # blocking
            "west-south": {
                "vehicles": 1,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (10, 20),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (130, 140),
                    "end": (150, 160),
                },
            },
            "east-south": {
                "vehicles": 1,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (10, 20),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (130, 140),
                    "end": (150, 160),
                },
            },
            "east-north": None,  # blocking
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 2,
            "start_time": 3,  # any value or default for LANE_LENGTH / speed_m_per_s
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
    "no-traffic": {  # c-intersection
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
    "blocks": {  # c-intersection
        "routes": {
            "south-west": None,
            "south-east": None,
            "south-north": {
                "vehicles": 10,
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
            "east-west": {
                "vehicles": 10,
                "start_end_on_different_lanes_probability": 0.0,
                "distribution": behavior_distribution,
                "begin_time_init": {
                    "func": basic_begin_time_init_func,
                    "params": {"probability": prob_easy},
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "east-north": None,
            "east-south": None,
            "north-east": None,
            "north-west": None,
            "north-south": None,
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 10,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
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
                "vehicles": 200,
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
                "vehicles": 200,
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
            "wait_to_hijack_limit_s": 10,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    "mid-density": {  # c-intersection
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
                "vehicles": 200,
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
                "vehicles": 200,
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
            "wait_to_hijack_limit_s": 10,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    "high-density": {  # c-intersection
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
                "distribution": {
                    "default": 0.70,
                    "aggressive": 0.30,
                    "cautious": 0.00,
                    "blocker": 0.0,
                },
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 2),
                        "time_between_cluster": (50, 60),
                        "time_for_each_cluster": 5,
                    },
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "south-north": {
                "vehicles": 200,
                "distribution": {
                    "default": 0.7,
                    "aggressive": 0.3,
                    "cautious": 0.0,
                    "blocker": 0.0,
                },
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 2),
                        "time_between_cluster": (60, 70),
                        "time_for_each_cluster": 5,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
            },
            "west-north": None,  # blocking
            "west-south": None,  # blocking
            "east-west": None,
            "west-east": None,
            "east-north": None,  # blocking
            "east-south": None,
            "north-south": {
                "vehicles": 200,
                "distribution": {
                    "default": 0.80,
                    "aggressive": 0.20,
                    "cautious": 0.00,
                    "blocker": 0.0,
                },
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (2, 4),
                        "time_between_cluster": (15, 35),
                        "time_for_each_cluster": 5,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
            },
            "north-east": {
                "vehicles": 100,
                "distribution": {
                    "default": 0.70,
                    "aggressive": 0.29,
                    "cautious": 0.01,
                    "blocker": 0.0,
                },
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 2),
                        "time_between_cluster": (30, 60),
                        "time_for_each_cluster": 10,
                    },
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
            "north-west": {
                "vehicles": 100,
                "distribution": {
                    "default": 0.30,
                    "aggressive": 0.70,
                    "cautious": 0.00,
                    "blocker": 0.0,
                },
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 2),
                        "time_between_cluster": (40, 50),
                        "time_for_each_cluster": 5,
                    },
                },
                "has_turn": True,
                "deadlock_optimization": True,
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 10,
            "start_time": "default",  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    # -------------------------------------------------- Intersection specific traffic distribution ------------------------------------------------
    # The following traffic distributions [low-interaction, mid-interaction, high-interaction] are created to emphasize interaction between ego and
    # social vehicles at the intersections (T or Cross). The key differences between these distributions and the {low, mid, high}-density are that they
    # do not focus on interactions beyond the intersection, use a very limited number of social vehicles, and there are no social vehicles in the ego
    # mission route. In terms of implementation, these distribution will be used inside the simple level, where the goal is to make the ego agent
    # familiar with the intersection
    "low-interaction": {
        "routes": {
            "south-west": None,
            "south-east": None,
            "south-north": None,  # blocking
            "west-east": None,
            "west-north": None,  # blocking
            "west-south": None,
            "east-south": None,
            "east-north": None,  # blocking
            "north-south": {
                "vehicles": 1,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 2),
                        "time_between_cluster": (5, 15),
                        "time_for_each_cluster": 5,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (120, 120),
                    "end": (150, 160),
                },
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 2,
            "start_time": 3,  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    "mid-interaction": {
        "routes": {
            "south-west": None,
            "south-east": None,
            "south-north": None,  # blocking
            "west-east": None,
            "west-north": None,  # blocking
            "west-south": None,
            "east-south": None,
            "east-north": None,  # blocking
            "north-east": {
                "vehicles": 2,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 10),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (120, 120),
                    "end": (50, 50),
                },
            },
            "north-south": {
                "vehicles": 2,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 15),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (120, 120),
                    "end": (50, 50),
                },
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 2,
            "start_time": 3,  # any value or default for LANE_LENGTH / speed_m_per_s
        },
    },
    "high-interaction": {  # c-intersection
        "routes": {
            "south-west": None,
            "south-east": None,
            "south-north": None,  # blocking
            "west-east": None,
            "west-north": None,
            "west-south": {
                "vehicles": 1,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 25),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (120, 140),
                    "end": (150, 160),
                },
            },
            "east-south": {
                "vehicles": 1,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 25),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (120, 140),
                    "end": (150, 160),
                },
            },
            "east-north": {
                "vehicles": 2,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 15),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (120, 140),
                    "end": (150, 160),
                },
            },
            "north-south": {
                "vehicles": 2,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 10),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (120, 140),
                    "end": (150, 160),
                },
            },
            "north-west": {
                "vehicles": 1,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 25),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (120, 140),
                    "end": (150, 160),
                },
            },
            "north-east": {
                "vehicles": 2,
                "distribution": behavior_distribution,
                "start_end_on_different_lanes_probability": 0.0,
                "begin_time_init": {
                    "func": burst_begin_time_init_func,
                    "params": {
                        "vehicle_cluster_size": (1, 1),
                        "time_between_cluster": (5, 25),
                        "time_for_each_cluster": 1,
                    },
                },
                "has_turn": False,
                "deadlock_optimization": True,
                "pos_offsets": {
                    "start": (120, 140),
                    "end": (150, 160),
                },
            },
        },
        "ego_hijacking_params": {
            "zone_range": [5, 10],
            "wait_to_hijack_limit_s": 2,
            "start_time": 3,  # any value or default for LANE_LENGTH / speed_m_per_s
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
