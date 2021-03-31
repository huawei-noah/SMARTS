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
import unittest
from ultra.scenarios.common.social_vehicle_definitions import *


class SocialVehiclesTest(unittest.TestCase):
    def test_get_social_vehicles(self):
        all_types = ["default", "aggressive", "cautious", "blocker"]
        for behavior in all_types:
            vehicle = get_social_vehicle_behavior(behavior)
            self.assertTrue(vehicle is not None)
        vehicle = get_social_vehicle_behavior("Unknown")
        self.assertTrue(vehicle is None)

    def test_get_all_behaviors(self):
        all_types = ["default", "aggressive", "cautious", "blocker"]
        vehicles = get_all_behaviors()
        self.assertTrue(sorted(vehicles) == sorted(all_types))

    def test_get_social_vehicle_color(self):
        vehicles = [
            "default-2244061760535854552",
            "aggressive-224406176053585455",
            "cautious-224406176053585455",
            "blocker-224406176053585455",
            "stopwatcher_aggressive--2078631594661947501",
        ]
        for vid in vehicles:
            behavior_id = vid.split("-")[0]
            b_id, color = get_social_vehicle_color(vid)
            self.assertTrue(b_id == behavior_id)

        unknown_type = "dummy-None"
        b_id, color = get_social_vehicle_color(unknown_type)
        self.assertTrue(b_id, f" Unknown ***** {unknown_type} *******")
