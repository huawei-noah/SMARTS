from smarts.core.sensors import Observation
from typing import Set


class Overtake:
    """This is a rudimentary overtake detector specifically designed for the
    single agent `SMARTS/smarts/scenarios/straight/3lane_overtake/scenario.py`.
    Not meant for other scenarios.
    """

    def __init__(self):
        self._infront = set() # Set of vehicles that were in front the ego agent in the past
        self._behind = set() # Set of vehicles that were in front the ego agent in the past

    def __call__(self, obs: Observation):
        ego = obs.ego_vehicle_state
        # ego.lane_index
        nghbs = obs.neighborhood_vehicle_states
        nghbs = [(nghb.id, nghb.position[0], nghb.lane_index) for nghb in nghbs]

        return True

    def check(agent_name: str):
        overtake = 0

        return 0
