import numpy as np
from typing import Tuple
from smarts.core.agent import Agent
from smarts.core.observations import Observation
from smarts.core.sensors import LANE_ID_CONSTANT


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        LANE_CHANGE_DIST = 40

        assert obs.waypoint_paths, f"Waypoint paths = {obs.waypoint_paths}; " \
            "cannot be empty or None. Enable waypoint paths in agent interface."

        for ind, wp in enumerate(obs.waypoint_paths):
            print("+ Waypoint:", ind)
            print("    Waypoints= ", wp[0].pos, wp[0].lane_id)
            print("    Waypoints= ", wp[-1].pos, wp[-1].lane_id)
        print(
            "+ Leader: ", obs.ego_vehicle_state.lane_id, obs.ego_vehicle_state.position
        )
        print("+ NVP= ", obs.via_data.near_via_points)
        print("+ Hit= ", obs.via_data.hit_via_points)

        # Truncate all paths to be of the same length
        min_len = min(map(len, obs.waypoint_paths))
        trunc_waypoints = list(map(lambda x: x[:min_len], obs.waypoint_paths))
        waypoints = [list(map(lambda x: x.pos, path)) for path in trunc_waypoints]
        waypoints = np.array(waypoints, dtype=np.float64)

        # Ego status
        ego_lane_id = obs.ego_vehicle_state.lane_id
        assert ego_lane_id is not LANE_ID_CONSTANT, f"Ego lane cannot be {ego_lane_id}."
        ego_pos = obs.ego_vehicle_state.position[:2]    
        ego_wp_ind = [path[0].lane_id for path in trunc_waypoints].index(ego_lane_id)

        # Filter via points within LANE_CHANGE_DIST from ego.
        candidate_via_points = [via_point.position for via_point in obs.via_data.near_via_points if np.linalg.norm(via_point.position-ego_pos) <= LANE_CHANGE_DIST]
        # No nearby via points
        if len(candidate_via_points) == 0:
            return (obs.waypoint_paths[ego_wp_ind][0].speed_limit, 0)

        # 
        index = get_nearest_index(waypoints, np.array(candidate_via_points[0]))
        print("Nearest waypoint", waypoints[index])
        print("via point", candidate_via_points[0])





        # trajectories = [(path[0],path[-1])  for path in obs.waypoint_paths]
        # wp_dist = [np.linalg.norm(path[0] - ego_pos) for path in obs.waypoint_paths]
        # ego_waypoint_path_index = np.argmin(wp_dist)
        # # Sanity check: Ensure ego's lane id matches that of the nearest waypoint.
        # assert obs.waypoint_paths[ego_waypoint_path_index][0].lane_id == ego_lane_id

        # nearby_via_points = obs.via_data.near_via_points
        # target_via_point = [np.linalg.norm(path[0] - ego_pos) for path in nearby_via_points]
        # for via_point in nearby_via_points:


        # if len(nearby_via_points) == 0:
        #     # No nearby via points
        #     return (obs.waypoint_paths[0][0].speed_limit, 0)
        #     input()

        # put threshold to change lane

        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.road_id != obs.via_data.near_via_points[0].road_id
        ):
            print("No via points or road id is different \n")
            input()
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            print("Nearest lane index matched ego road id \n")
            input()
            return (nearest.required_speed, 0)

        print("Changing lane \n")
        input()
        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )


def get_nearest_index(matrix:np.ndarray, points:np.ndarray)->Tuple[int,int]:
    assert len(matrix.shape) == 3
    assert matrix.shape[2] == 2
    assert len(points.shape) == 2
    assert points.shape[1] == 2 

    points_expanded = np.expand_dims(points,(1,2))
    diff = matrix - points_expanded
    dist = np.linalg.norm(diff, axis=-1)
    index = np.argmin(dist)
    ncol = dist.shape[1]
    return index//ncol, index%ncol


import numpy as np
f = np.array([[[1,2],[2,3],[3,4]],[[4,5],[5,6],[6,7]]])
f.shape
g = np.array([[1,1],[2,2]])
g.shape
t = np.expand_dims(g,(1,2))
q = f-t
q[0]