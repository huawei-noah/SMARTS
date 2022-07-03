from smarts.core.sensors import Observation


class Overtake:
    """A rudimentary overtake detector specifically designed for the scenario
    `SMARTS/smarts/scenarios/straight/3lane_overtake/scenario.py`. Not meant
    for other scenarios.
    """

    def __init__(self):
        self._prev_front = set()  # Vehicles that were last seen in front the ego agent.
        self._prev_rear = set()  # Vehicles that were last seen behind the ego agent.
        self._once = True
        self._desired_lane_index: int
        self._overtake = False

    def __call__(self, obs: Observation):
        ego = obs.ego_vehicle_state

        if self._once:
            self._desired_lane_index = ego.lane_index
            self._once = False

        # Overtake is only complete when ego returns to its original lane.
        # Thus, only check for `overtake` event when ego is in its original lane.
        if ego.lane_index == self._desired_lane_index:
            ego_x = ego.position[0]

            # Get current neighbours at the front and at the rear.
            cur_front = set()
            cur_rear = set()
            nghbs = obs.neighborhood_vehicle_states
            nghbs = filter(
                lambda nghb: nghb.lane_index == self._desired_lane_index, nghbs
            )
            for nghb in nghbs:
                if nghb.position[0] > ego_x:
                    cur_front.add(nghb.id)
                else:
                    cur_rear.add(nghb.id)

            # Check whether vehicle order has changed.
            for v_id in cur_rear:
                if v_id in self._prev_front:
                    self._overtake = True

            # Update last known vehicles at the front and rear.
            self._prev_front.difference_update(cur_rear)
            self._prev_rear.difference_update(cur_front)
            self._prev_front.update(cur_front)
            self._prev_rear.update(cur_rear)

    def check(self):
        return self._overtake
