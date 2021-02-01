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
