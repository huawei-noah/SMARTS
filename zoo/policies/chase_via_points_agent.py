import numpy as np

from smarts.core.agent import Agent
from smarts.core.observations import Observation
from smarts.core.sensors import LANE_ID_CONSTANT


class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        assert obs.waypoint_paths, (
            f"Waypoint paths = {obs.waypoint_paths}; "
            "cannot be empty or None. Enable waypoint paths in agent interface."
        )

        lane_change_dist = 80

        # Truncate all paths to be of the same length
        min_len = min(lane_change_dist, min(map(len, obs.waypoint_paths)))
        trunc_waypoints = list(map(lambda x: x[:min_len], obs.waypoint_paths))
        waypoints = [list(map(lambda x: x.pos, path)) for path in trunc_waypoints]
        waypoints = np.array(waypoints, dtype=np.float64)

        # Ego status
        ego_lane_id = obs.ego_vehicle_state.lane_id
        assert ego_lane_id is not LANE_ID_CONSTANT, f"Ego lane cannot be {ego_lane_id}."
        ego_pos = obs.ego_vehicle_state.position[:2]
        dist = np.linalg.norm(waypoints[:, 0, :] - ego_pos, axis=-1)
        ego_wp_inds = np.where(dist == dist.min())[0]

        # Get target via point.
        via_points = np.array(
            [via_point.position for via_point in obs.via_data.near_via_points]
        )
        via_point_wp_ind, via_point_ind = _nearest_waypoint(waypoints, via_points)

        # No nearby via points. Hence, remain in same lane.
        if via_point_ind is None:
            return (obs.waypoint_paths[ego_wp_inds[0]][0].speed_limit, 0)

        # Target via point is in the same path. Hence, remain in same lane.
        if via_point_wp_ind[0] in ego_wp_inds:
            return (obs.via_data.near_via_points[via_point_ind].required_speed, 0)

        # Turn leftwards if (via_point_wp_ind[0] - ego_wp_inds[0]) > 0 , as target via point is on the left.
        # Turn rightwards if (via_point_wp_ind[0] - ego_wp_inds[0]) < 0 , as target via point is on the right.
        return (
            obs.via_data.near_via_points[via_point_ind].required_speed,
            via_point_wp_ind[0] - ego_wp_inds[0],
        )


def _nearest_waypoint(matrix: np.ndarray, points: np.ndarray, radius: float = 2):
    cur_point_index = ((np.intp(1e10), np.intp(1e10)), None)

    if points.shape == (0,):
        return cur_point_index

    assert len(matrix.shape) == 3
    assert matrix.shape[2] == 2
    assert len(points.shape) == 2
    assert points.shape[1] == 2

    points_expanded = np.expand_dims(points, (1, 2))
    diff = matrix - points_expanded
    dist = np.linalg.norm(diff, axis=-1)
    for ii in range(points.shape[0]):
        index = np.argmin(dist[ii])
        index_unravel = np.unravel_index(index, dist[ii].shape)
        min_dist = dist[ii][index_unravel]
        if min_dist <= radius and index_unravel[1] < cur_point_index[0][1]:
            cur_point_index = (index_unravel, ii)

    return cur_point_index
