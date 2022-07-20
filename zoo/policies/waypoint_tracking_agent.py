from smarts.core.agent import Agent

MIN_WAYPOINTS = 10


class WaypointTrackingAgent(Agent):
    def __init__(self):
        self.waypoint_path = []

    def act(self, obs):

        current_lane = obs.ego_vehicle_state.lane_index
        # Desired speed is in m/s
        desired_speed = 10
        wp_index = 0
        num_lanes = 0

        for path in obs.waypoint_paths:
            num_lanes = max(num_lanes, path[0].lane_index + 1)

        # This is when there are waypoint paths that change lanes
        if num_lanes < len(obs.waypoint_paths) <= num_lanes**2:

            goal_position = obs.ego_vehicle_state.mission.goal.position
            min_lateral_error = 100
            for idx in range(len(obs.waypoint_paths)):
                num_waypoints = len(obs.waypoint_paths[idx])

                # choose waypoint paths that start on the same lane
                if obs.waypoint_paths[idx][0].lane_index == current_lane:

                    lateral_error = obs.waypoint_paths[idx][
                        num_waypoints - 1
                    ].signed_lateral_error(goal_position)

                    # choose waypoint path with end closest to the goal
                    if abs(lateral_error) < min_lateral_error:
                        min_lateral_error = abs(lateral_error)
                        wp_index = idx

            self.waypoint_path = obs.waypoint_paths[wp_index]

        else:
            for i in range(len(obs.waypoint_paths)):
                if obs.waypoint_paths[i][0].lane_index == current_lane:
                    wp_index = i
                    break

        if self.waypoint_path:
            chosen_waypoint_path = self.waypoint_path
            self.waypoint_path.pop(0)
        else:
            chosen_waypoint_path = obs.waypoint_paths[wp_index]

        num_trajectory_points = min([MIN_WAYPOINTS, len(chosen_waypoint_path)])
        trajectory = [
            [chosen_waypoint_path[i].pos[0] for i in range(num_trajectory_points)],
            [chosen_waypoint_path[i].pos[1] for i in range(num_trajectory_points)],
            [chosen_waypoint_path[i].heading for i in range(num_trajectory_points)],
            [desired_speed for i in range(num_trajectory_points)],
        ]

        return trajectory
