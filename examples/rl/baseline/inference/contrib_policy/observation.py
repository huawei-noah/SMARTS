import copy
import math
from typing import Any, Dict, Sequence

import gymnasium as gym
import numpy as np

from smarts.core.agent_interface import RGB
from smarts.core.colors import Colors, SceneColors

# class SaveObs(gym.ObservationWrapper):
#     """Saves several selected observation parameters."""

#     def __init__(self, env: gym.Env):
#         """
#         Args:
#             env (gym.Env): Environment to be wrapped.
#         """
#         super().__init__(env)
#         self.saved_obs: Dict[str, Dict[str, Any]]

#     def observation(self, obs: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
#         """Saves the wrapped environment's observation.

#         Note: Users should not directly call this method.
#         """

#         obs_data = {}
#         for agent_id, agent_obs in obs.items():
#             obs_data.update(
#                 {
#                     agent_id: {
#                         "pos": copy.deepcopy(agent_obs["ego"]["pos"]),
#                         "heading": copy.deepcopy(agent_obs["ego"]["heading"]),
#                     }
#                 }
#             )
#         self.saved_obs = obs_data

#         return obs


class FilterObs:
    """Filter only the selected observation parameters."""

    def __init__(self, top_down_rgb: RGB):
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "rgb": gym.spaces.Box(
        #             low=0,
        #             high=255,
        #             shape=(agent_interface.top_down_rgb.height,
        #                 agent_interface.top_down_rgb.width,
        #                 3),
        #             dtype=np.uint8,
        #         ),
        #         "goal_distance": gym.spaces.Box(
        #             low=-1e10,
        #             high=+1e10,
        #             shape=(1, 1),
        #             dtype=np.float32,
        #         ),
        #         "goal_heading": gym.spaces.Box(
        #             low=-np.pi,
        #             high=np.pi,
        #             shape=(1, 1),
        #             dtype=np.float32,
        #         ),
        #     }
        # )
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(top_down_rgb.height, top_down_rgb.width, 3),
            dtype=np.uint8,
        )

        self._wps_color = np.array(Colors.GreenTransparent.value[0:3]) * 255
        self._traffic_color = np.array(SceneColors.SocialVehicle.value[0:3]) * 255
        self._road_color = np.array(SceneColors.Road.value[0:3]) * 255
        self._lane_divider_color = np.array(SceneColors.LaneDivider.value[0:3]) * 255
        self._edge_divider_color = np.array(SceneColors.EdgeDivider.value[0:3]) * 255
        self._ego_color = np.array(SceneColors.Agent.value[0:3]) * 255

        self._res = top_down_rgb.resolution
        h = top_down_rgb.height
        w = top_down_rgb.width
        shape = (
            (
                math.floor(w / 2 - 3.68 / 2 / self._res),
                math.ceil(w / 2 + 3.68 / 2 / self._res),
            ),
            (
                math.floor(h / 2 - 1.47 / 2 / self._res),
                math.ceil(h / 2 + 1.47 / 2 / self._res),
            ),
        )
        self._rgb_mask = np.zeros(shape=(h, w, 3), dtype=np.uint8)
        self._rgb_mask[shape[0][0] : shape[0][1], shape[1][0] : shape[1][1], :] = 1

    def filter(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapts the environment's observation."""
        # fmt: off
        # Distance between ego and goal.
        # goal_distance = np.array(
        #     [
        #         [
        #             np.linalg.norm(
        #                 obs["mission"]["goal_position"] - obs["ego"]["position"]
        #             )
        #         ]
        #     ],
        #     dtype=np.float32,
        # )

        # Ego's heading with respect to the map's coordinate system.
        # Note: All angles returned by smarts is with respect to the map's coordinate system.
        #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
        ego_heading = (obs["ego_vehicle_state"]["heading"] + np.pi) % (2 * np.pi) - np.pi
        ego_pos = obs["ego_vehicle_state"]["position"]

        # Goal's angle with respect to the ego's position.
        # Note: In np.angle(), angle is zero at positive x axis, and increases anti-clockwise.
        #       Hence, map_angle = np.angle() - π/2
        # goal_pos = obs["mission"]["goal_position"]
        # rel_pos = goal_pos - ego_pos
        # goal_angle = np.angle(rel_pos[0] + 1j * rel_pos[1]) - np.pi / 2
        # goal_angle = (goal_angle + np.pi) % (2 * np.pi) - np.pi

        # Goal heading is the angle correction required by ego agent to face the goal.
        # goal_heading = goal_angle - ego_heading
        # goal_heading = (goal_heading + np.pi) % (2 * np.pi) - np.pi
        # goal_heading = np.array([[goal_heading]], dtype=np.float32)

        # Get rgb image, remove road, and replace other egos (if any) as background vehicles
        rgb = obs["top_down_rgb"]
        h, w, _ = rgb.shape
        rgb_noroad = replace_color(rgb=rgb, old_color=[self._road_color, self._lane_divider_color, self._edge_divider_color], new_color=np.zeros((3,)))
        rgb_ego = replace_color(rgb=rgb_noroad, old_color=[self._ego_color], new_color=self._traffic_color, mask=self._rgb_mask)

        # Superimpose waypoints onto rgb image
        wps = obs["waypoint_paths"]["position"][0:3, 3:, 0:3]
        for path in wps[:]:
            wps_valid = wps_to_pixels(
                wps=path,
                ego_pos=ego_pos,
                ego_heading=ego_heading,
                w=w,
                h=h,
                res=self._res,
            )
            for point in wps_valid:
                img_x, img_y = point[0], point[1]
                if all(rgb_ego[img_y, img_x, :] == self._traffic_color):
                    # Ignore waypoints in the current path which lie ahead of an obstacle.
                    # break
                    # Ignore waypoints overlapping obstacles and continue with other waypoints ahead.
                    continue
                rgb_ego[img_y, img_x, :] = self._wps_color

        # Channel first rgb
        # rgb_ego = rgb_ego.transpose(2, 0, 1)

        # filtered_obs = {
        #     "rgb": np.uint8(rgb_ego),
        #     "goal_distance": goal_distance,
        #     "goal_heading": goal_heading,
        # }
        filtered_obs = np.uint8(rgb_ego)

        return filtered_obs
        # fmt: on


def replace_color(
    rgb: np.ndarray,
    old_color: Sequence[np.ndarray],
    new_color: np.ndarray,
    mask: np.ndarray = np.ma.nomask,
) -> np.ndarray:
    """Convert pixels of value `old_color` to `new_color` within the masked
        region in the received RGB image.

    Args:
        rgb (np.ndarray): RGB image. Shape = (m,n,3).
        old_color (Sequence[np.ndarray]): List of old colors to be removed from the RGB image. Shape = (3,).
        new_color (np.ndarray): New color to be added to the RGB image. Shape = (3,).
        mask (np.ndarray, optional): Valid regions for color replacement. Shape = (m,n,3).
            Defaults to np.ma.nomask .

    Returns:
        np.ndarray: RGB image with `old_color` pixels changed to `new_color`
            within the masked region. Shape = (m,n,3).
    """
    # fmt: off
    assert all(color.shape == (3,) for color in old_color), (
        f"Expected old_color to be of shape (3,), but got {[color.shape for color in old_color]}.")
    assert new_color.shape == (3,), (
        f"Expected new_color to be of shape (3,), but got {new_color.shape}.")

    nc = new_color.reshape((1, 1, 3))
    nc_array = np.full_like(rgb, nc)
    rgb_masked = np.ma.MaskedArray(data=rgb, mask=mask)

    rgb_condition = rgb_masked
    result = rgb
    for color in old_color:
        result = np.ma.where((rgb_condition == color.reshape((1, 1, 3))).all(axis=-1)[..., None], nc_array, result)

    return result
    # fmt: on


def wps_to_pixels(
    wps: np.ndarray, ego_pos: np.ndarray, ego_heading: float, w: int, h: int, res: float
) -> np.ndarray:
    """Converts waypoints into pixel coordinates in order to superimpose the
    waypoints onto the RGB image.

    Args:
        wps (np.ndarray): Waypoints for a single route. Shape (n,3).
        ego_pos (np.ndarray): Ego position. Shape = (3,).
        ego_heading (float): Ego heading in radians.
        w (int): Width of RGB image
        h (int): Height of RGB image.
        res (float): Resolution of RGB image in meters/pixels. Computed as
            ground_size/image_size.

    Returns:
        np.ndarray: Array of waypoint coordinates on the RGB image. Shape = (m,3).
    """
    # fmt: off
    mask = [False if all(point == np.zeros(3,)) else True for point in wps]
    wps_nonzero = wps[mask]
    wps_delta = wps_nonzero - ego_pos
    wps_rotated = rotate_axes(wps_delta, theta=ego_heading)
    wps_pixels = wps_rotated / np.array([res, res, res])
    wps_overlay = np.array([w / 2, h / 2, 0]) + wps_pixels * np.array([1, -1, 1])
    wps_rint = np.rint(wps_overlay).astype(np.uint8)
    wps_valid = wps_rint[(wps_rint[:,0] >= 0) & (wps_rint[:,0] < w) & (wps_rint[:,1] >= 0) & (wps_rint[:,1] < h)] 
    return wps_valid
    # fmt: on

def rotate_axes(points: np.ndarray, theta: float) -> np.ndarray:
    """A counterclockwise rotation of the x-y axes by an angle theta θ about
    the z-axis.

    Args:
        points (np.ndarray): x,y,z coordinates in original axes. Shape = (n,3).
        theta (np.float): Axes rotation angle in radians.

    Returns:
        np.ndarray: x,y,z coordinates in rotated axes. Shape = (n,3).
    """
    # fmt: off
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    ct, st = np.cos(theta), np.sin(theta)
    R = np.array([[ ct, st, 0], 
                  [-st, ct, 0], 
                  [  0,  0, 1]])
    rotated_points = (R.dot(points.T)).T
    return rotated_points
    # fmt: on


# class Concatenate(gym.ObservationWrapper):
#     """Concatenates data from stacked dictionaries. Only works with nested gym.spaces.Box .
#     Dimension to stack over is determined by `channels_order`.
#     """

#     def __init__(self, env: gym.Env, channels_order: str = "first"):
#         """
#         Args:
#             env (gym.Env): Environment to be wrapped.
#             channels_order (str): A string, either "first" or "last", specifying
#                 the dimension over which to stack each observation.
#         """
#         super().__init__(env)

#         self._repeat_axis = {
#             "first": 0,
#             "last": -1,
#         }.get(channels_order)

#         for agent_name, agent_space in env.observation_space.spaces.items():
#             for subspaces in agent_space:
#                 for key, space in subspaces.spaces.items():
#                     assert isinstance(space, gym.spaces.Box), (
#                         f"Concatenate only works with nested gym.spaces.Box. "
#                         f"Got agent {agent_name} with key {key} and space {space}."
#                     )

#         _, agent_space = next(iter(env.observation_space.spaces.items()))
#         self._num_stack = len(agent_space)
#         self._keys = agent_space[0].spaces.keys()

#         obs_space = {}
#         for agent_name, agent_space in env.observation_space.spaces.items():
#             subspaces = {}
#             for key, space in agent_space[0].spaces.items():
#                 low = np.repeat(space.low, self._num_stack, axis=self._repeat_axis)
#                 high = np.repeat(space.high, self._num_stack, axis=self._repeat_axis)
#                 subspaces[key] = gym.spaces.Box(low=low, high=high, dtype=space.dtype)
#             obs_space.update({agent_name: gym.spaces.Dict(subspaces)})
#         self.observation_space = gym.spaces.Dict(obs_space)

#     def observation(self, obs):
#         """Adapts the wrapped environment's observation.

#         Note: Users should not directly call this method.
#         """

#         wrapped_obs = {}
#         for agent_id, agent_obs in obs.items():
#             stacked_obs = {}
#             for key in self._keys:
#                 val = [obs[key] for obs in agent_obs]
#                 stacked_obs[key] = np.concatenate(val, axis=self._repeat_axis)
#             wrapped_obs.update({agent_id: stacked_obs})

#         return wrapped_obs
