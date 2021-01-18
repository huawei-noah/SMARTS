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
from smarts.sstudio.types import (
    TrafficActor,
    Distribution,
    LaneChangingModel,
    JunctionModel,
)
import re

social_vehicle_colors = {
    "default": (255, 255, 255),
    "aggressive": (255, 128, 0),
    "cautious": (100, 178, 255),
    "blocker": (102, 0, 255),
    # "bus": (255, 255, 0),
    # "crusher": (255, 0, 0),
    "stopwatcher": (255, 158, 251),
}

depart_speed = "max"  # sets to max
social_vehicles = {
    "default": TrafficActor(
        name="default",
        accel=3.0,
        decel=7.5,
        tau=1.0,
        depart_speed=depart_speed,
        min_gap=Distribution(mean=2.5, sigma=0.05),
        speed=Distribution(mean=1.0, sigma=0.05),
        imperfection=Distribution(mean=0.5, sigma=0.05),
        vehicle_type="passenger",
        lane_changing_model=LaneChangingModel(),
        junction_model=JunctionModel(),
    ),
    "aggressive": TrafficActor(
        name="aggressive",
        accel=10.0,
        decel=10.0,
        tau=0.5,
        depart_speed=depart_speed,
        min_gap=Distribution(mean=1.0, sigma=0.1),
        speed=Distribution(mean=1.5, sigma=0.1),
        imperfection=Distribution(mean=0.1, sigma=0.1),
        vehicle_type="passenger",
        lane_changing_model=LaneChangingModel(
            strategic=10,
            pushy=1.0,
            overtake_right=1,
            speed_gain=200,
            time_to_impatience=0.01,
            impatience=1.0,
            lookahead_left=0.0,
        ),
        junction_model=JunctionModel(
            impatience=0.98, timegap_minor=0.1, ignore_foe_prob=0.98
        ),
    ),
    "cautious": TrafficActor(
        name="cautious",
        accel=3.0,
        decel=10.0,
        tau=1.0,
        depart_speed=depart_speed,
        min_gap=Distribution(mean=5.0, sigma=0.05),
        speed=Distribution(mean=0.4, sigma=0.1),
        imperfection=Distribution(mean=0.5, sigma=0.05),
        vehicle_type="passenger",
        lane_changing_model=LaneChangingModel(
            strategic=1,
            pushy=0.2,
            overtake_right=0,
            speed_gain=1,
            keep_right=1,
            impatience=0,
            lookahead_left=2.9,
        ),
        junction_model=JunctionModel(
            impatience=0.0, timegap_minor=1.0, ignore_foe_prob=0.0
        ),
    ),
    "blocker": TrafficActor(
        name="blocker",
        accel=3.0,
        decel=7.5,
        tau=1.0,
        depart_speed=depart_speed,
        min_gap=Distribution(mean=1.0, sigma=0.1),
        speed=Distribution(mean=0.2, sigma=0.1),
        imperfection=Distribution(mean=0.0, sigma=0.1),
        vehicle_type="passenger",
        lane_changing_model=LaneChangingModel(
            strategic=0,
            pushy=0,
            overtake_right=0,
            speed_gain=1,
            time_to_impatience=1000,
            cooperative=0.0,
            opposite=0,  # less lane changing behavior
            impatience=0.0,
            keep_right=0.0,
        ),
        junction_model=JunctionModel(
            impatience=0.0, timegap_minor=1.0, ignore_foe_prob=0.0
        ),
    ),
    "bus": TrafficActor(
        name="bus",
        accel=1.2,
        decel=4.0,
        tau=1.0,
        depart_speed=depart_speed,
        min_gap=Distribution(mean=2.5, sigma=0.05),
        speed=Distribution(mean=1.0, sigma=0.05),
        imperfection=Distribution(mean=0.5, sigma=0.05),
        vehicle_type="bus",
        lane_changing_model=LaneChangingModel(),
        junction_model=JunctionModel(),
    ),
    "truck": TrafficActor(
        name="truck",
        accel=1.3,
        decel=4.0,
        tau=1.0,
        depart_speed=depart_speed,
        min_gap=Distribution(mean=2.5, sigma=0.05),
        speed=Distribution(mean=1.0, sigma=0.05),
        imperfection=Distribution(mean=0.5, sigma=0.05),
        vehicle_type="truck",
        lane_changing_model=LaneChangingModel(),
        junction_model=JunctionModel(),
    ),
    "trailer": TrafficActor(
        name="trailer",
        accel=1.0,
        decel=4.0,
        tau=1.0,
        depart_speed=depart_speed,
        min_gap=Distribution(mean=2.5, sigma=0.05),
        speed=Distribution(mean=1.0, sigma=0.05),
        imperfection=Distribution(mean=0.5, sigma=0.05),
        vehicle_type="trailer",
        lane_changing_model=LaneChangingModel(),
        junction_model=JunctionModel(),
    ),
    "coach": TrafficActor(
        name="coach",
        accel=2.0,
        decel=4.0,
        tau=1.0,
        depart_speed=depart_speed,
        min_gap=Distribution(mean=2.5, sigma=0.05),
        speed=Distribution(mean=1.0, sigma=0.05),
        imperfection=Distribution(mean=0.5, sigma=0.05),
        vehicle_type="coach",
        lane_changing_model=LaneChangingModel(),
        junction_model=JunctionModel(),
    ),
    # "crusher": TrafficActor(
    #     name="crusher",
    #     accel=10.0,
    #     decel=1.0,
    #     tau=0.0,
    #     depart_speed=depart_speed,
    #     emergency_decel=0.0,
    #     sigma=1.0,
    #     min_gap=Distribution(mean=0.0, sigma=0),
    #     speed=Distribution(mean=2.0, sigma=0.1),
    #     imperfection=Distribution(mean=1.0, sigma=0),
    #     vehicle_type="passenger",
    #     lane_changing_model=LaneChangingModel(
    #         strategic=100,
    #         pushy=1.0,
    #         overtake_right=1,
    #         speed_gain=200,
    #         time_to_impatience=0.1,
    #         cooperative=0.0,
    #         opposite=0, # less lane changing behavior
    #         impatience=1.0
    #     ),
    #     junction_model=JunctionModel(
    #         impatience=1, timegap_minor=0.0, ignore_foe_prob=1
    #     ),
    # ),
}


def get_social_vehicle_behavior(behavior_id):
    if behavior_id in social_vehicles:
        return social_vehicles[behavior_id]
    return None


def get_all_behaviors():
    all_behaviors = sorted(list(social_vehicle_colors.keys()))
    all_behaviors.remove("stopwatcher")
    return all_behaviors


def get_social_vehicle_color(vehicle_id):
    behavior_id = vehicle_id.split("-")[0]
    if behavior_id in social_vehicle_colors.keys():
        return (behavior_id, social_vehicle_colors[behavior_id])
    else:
        # some processing:  actor-default-2244061760535854552, actor-stopwatcher_aggressive--2078631594661947501
        behavior_match = behavior_id
        stopwatcher_behavior = None

        behavior_key = behavior_match
        if "_" in behavior_match:
            behavior_key, stopwatcher_behavior = behavior_match.split("_")

        if stopwatcher_behavior:
            return (behavior_match, social_vehicle_colors[stopwatcher_behavior])
        if behavior_key in social_vehicle_colors.keys():
            return (behavior_match, social_vehicle_colors[behavior_key])

    return (f" Unknown ***** {behavior_id} *******", (255, 50, 255))
