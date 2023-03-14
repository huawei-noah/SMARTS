import numpy as np
from typing import List, Tuple
from smarts.core.agent import Agent
from smarts.core.observations import Observation
from smarts.core.sensors import LANE_ID_CONSTANT


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        LANE_CHANGE_DIST = 40

        assert obs.waypoint_paths, f"Waypoint paths = {obs.waypoint_paths}; " \
            "cannot be empty or None. Enable waypoint paths in agent interface."

        # for ind, wp in enumerate(obs.waypoint_paths):
        #     print("+ Waypoint:", ind)
        #     print("    Waypoints= ", wp[0].pos, wp[0].lane_id)
        #     print("    Waypoints= ", wp[-1].pos, wp[-1].lane_id)
        # print(
        #     "+ Leader: ", obs.ego_vehicle_state.lane_id, obs.ego_vehicle_state.position
        # )
        # print("+ NVP= ", obs.via_data.near_via_points)
        # print("+ Hit= ", obs.via_data.hit_via_points)

        # Truncate all paths to be of the same length
        min_len = min(map(len, obs.waypoint_paths))
        assert min_len >= LANE_CHANGE_DIST
        trunc_waypoints = list(map(lambda x: x[:min_len], obs.waypoint_paths))
        waypoints = [list(map(lambda x: x.pos, path)) for path in trunc_waypoints]
        waypoints = np.array(waypoints, dtype=np.float64)

        # Ego status
        ego_lane_id = obs.ego_vehicle_state.lane_id
        assert ego_lane_id is not LANE_ID_CONSTANT, f"Ego lane cannot be {ego_lane_id}."
        ego_pos = obs.ego_vehicle_state.position[:2]  
        _, ego_wp_ind = _nearest_point_to_waypoints(waypoints, ego_pos[np.newaxis,:])
        assert ego_wp_ind[1] == 0

        # Filter via points within LANE_CHANGE_DIST from ego.
        candidate_via_points = [via_point.position for via_point in obs.via_data.near_via_points if np.linalg.norm(via_point.position-ego_pos) <= LANE_CHANGE_DIST]

        # No nearby via points. Hence, remain in same lane.
        if len(candidate_via_points) == 0:
            print("No via points wihtin lane change distance. \n")
            return (obs.waypoint_paths[ego_wp_ind[0]][0].speed_limit, 0)

        # Get traget via point.
        via_point_wp_ind, via_point_ind = _nearest_point_to_waypoints(waypoints, np.array(candidate_via_points))
        if via_point_target is None:
            print("No via points within waypoint radius. \n")
            return (obs.waypoint_paths[ego_wp_ind[0]][0].speed_limit, 0)

        # Target via point is in the same path. Hence, remain in same lane.
        if obs.waypoint_paths[ego_wp_ind[0]][0].lane_id == obs.waypoint_paths[via_point_wp_ind[0]][0].lane_id:
            return (via_point_target.required_speed, 0)

        # Change to left lane since target via point is on the left lane. 
        if ego_wp_ind[0] < via_point_wp_ind[0]:
            return (via_point_target.required_speed, 1)

        # Change to right lane since target via point is on the right lane. 
        if ego_wp_ind[0] > via_point_wp_ind[0]:
            return (via_point_target.required_speed, -1)

        raise Exception("ChaseViaPointsAgent did not catch any preprogrammed actions.")


        # if (
        #     len(obs.via_data.near_via_points) < 1
        #     or obs.ego_vehicle_state.road_id != obs.via_data.near_via_points[0].road_id
        # ):
        #     print("No via points or road id is different \n")
        #     input()
        #     return (obs.waypoint_paths[0][0].speed_limit, 0)

        # nearest = obs.via_data.near_via_points[0]
        # if nearest.lane_index == obs.ego_vehicle_state.lane_index:
        #     print("Nearest lane index matched ego road id \n")
        #     input()
        #     return (nearest.required_speed, 0)

        # print("Changing lane \n")
        # input()
        # return (
        #     nearest.required_speed,
        #     1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        # )


def _nearest_point_to_waypoints(matrix:np.ndarray, points:np.ndarray, radius:float=2):
    assert len(matrix.shape) == 3
    assert matrix.shape[2] == 2
    assert len(points.shape) == 2
    assert points.shape[1] == 2 

    points_expanded = np.expand_dims(points,(1,2))
    diff = matrix - points_expanded
    dist = np.linalg.norm(diff, axis=-1)
    cur_point_index = (None,(np.intp(1e10),np.intp(1e10)))
    for ii in range(points.shape[0]):
        index = np.argmin(dist[ii])
        index_unravel = np.unravel_index(index, dist[ii].shape)
        min_dist = dist[ii][index_unravel]
        if min_dist <= radius and index_unravel[1] < cur_point_index[1][1]:
            cur_point_index = (points[ii],index_unravel) 
        
        # print("----------------------------------------------")
        # print("Min Dist", min_dist)
        # print("Index_unravel", index_unravel)
        # print("Point", points[ii])

    # print("Final point index", cur_point_index)
    return cur_point_index

# import numpy as np
# f = np.array([[[1,2],[2,3],[3,4]],[[4,5],[5,6],[6,7]]])
# f.shape
# g = np.array([[1,1],[2,2]])
# g.shape
# t = np.expand_dims(g,(1,2))
# q = f-t
# q[0]
# w = np.linalg.norm(q, axis=-1)
# c = np.array([[[3,1,1],[4,0,0]],[[-5,9,1],[6,3,2]]])

# Some thoughts on presenting SMARTS support for external datasets. 
# We could address this in a separate PR too.

# Consider putting up an example, in the example folder, on using Waymo map and traffic. 
# Add the Waymo example to the CI tests. 

# Consider updating how to use SMARTS for Waymo under Ecosystem section in ReadTheDocs. 
# In the Example section we could provide intructions to run a simple 
# ego agent (e.g., `AgentType.Laner`, etc) on a Waymo map with traffic. 
# We could also post a gif of the replayed Waymo scenario. 
# To reproduce the result, we could instruct the users to download the dataset and modify the dataset path in the provided script.

# We could move readmes such as (i)`SMARTS/scenarios/waymo/README.md`, and (ii) `SMARTS/waymo_open_dataset/README.md`,
# to ReadTheDocs.

# An example Waymo gif which we could post:
