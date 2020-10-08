import numpy as np

from smarts.core.agent import AgentPolicy


class NonInteractivePolicy(AgentPolicy):
    def __init__(self, speed=5, target_lane_index=None):
        self.speed = speed
        if target_lane_index is None:
            target_lane_index = {}
        self.target_lanes = [
            f"{edge}_{lane_index}" for edge, lane_index in target_lane_index.items()
        ]

    def act(self, obs):
        # Waypoint searching approach:
        # 1. Use the first waypoint path as default
        # 2. Look for current waypoint path
        # 3. Look for a waypoint path in the target lanes
        wp = obs.waypoint_paths[0][:5][-1]
        current_lane_id = obs.ego_vehicle_state.lane_id
        for waypoints in obs.waypoint_paths:
            if waypoints[0].lane_id == current_lane_id:
                wp = waypoints[:5][-1]
                break
        for waypoints in obs.waypoint_paths:
            if waypoints[0].lane_id in self.target_lanes:
                wp = waypoints[:5][-1]
                break
        dist_to_wp = wp.dist_to(obs.ego_vehicle_state.position)
        return np.array([*wp.pos, wp.heading, dist_to_wp / self.speed])
