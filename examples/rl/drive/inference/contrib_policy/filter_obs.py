import math
from typing import Any, Dict, Sequence, Tuple

import gym
import numpy as np

from smarts.core.agent_interface import RGB
from smarts.core.colors import Colors, SceneColors


class FilterObs:
    """Filter only the selected observation parameters."""

    def __init__(
        self, top_down_rgb: RGB, crop: Tuple[int, int, int, int] = (0, 0, 0, 0)
    ):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                3,
                top_down_rgb.height - crop[2] - crop[3],
                top_down_rgb.width - crop[0] - crop[1],
            ),
            dtype=np.uint8,
        )

        self._no_color = np.zeros((3,))
        self._wps_color = np.array(Colors.GreenTransparent.value[0:3]) * 255
        self._traffic_color = np.array(SceneColors.SocialVehicle.value[0:3]) * 255
        self._road_color = np.array(SceneColors.Road.value[0:3]) * 255
        self._lane_divider_color = np.array(SceneColors.LaneDivider.value[0:3]) * 255
        self._edge_divider_color = np.array(SceneColors.EdgeDivider.value[0:3]) * 255
        self._ego_color = np.array(SceneColors.Agent.value[0:3]) * 255
        self._goal_color = np.array(Colors.Purple.value[0:3]) * 255

        self._blur_radius = 2
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

        self._crop = crop
        assert (
            self._crop[0] < np.floor(w / 2)
            and self._crop[1] < np.floor(w / 2)
            and self._crop[2] < np.floor(h / 2)
            and self._crop[3] < np.floor(h / 2)
        ), f"Expected crop to be less than half the image size, got crop={self._crop} for an image of size[h,w]=[{h},{w}]."

    def filter(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """Adapts the environment's observation."""
        # fmt: off

        # Ego's heading with respect to the map's coordinate system.
        # Note: All angles returned by smarts is with respect to the map's coordinate system.
        #       On the map, angle is zero at positive y axis, and increases anti-clockwise.
        ego_heading = (obs["ego_vehicle_state"]["heading"] + np.pi) % (2 * np.pi) - np.pi
        ego_pos = obs["ego_vehicle_state"]["position"]

        # Get rgb image, remove road, and replace other egos (if any) as background vehicles
        rgb = obs["top_down_rgb"]
        h, w, _ = rgb.shape
        rgb_noroad = replace_color(rgb=rgb, old_color=[self._road_color, self._lane_divider_color, self._edge_divider_color], new_color=self._no_color)
        rgb_ego = replace_color(rgb=rgb_noroad, old_color=[self._ego_color], new_color=self._traffic_color, mask=self._rgb_mask)

        # Superimpose waypoints onto rgb image
        wps = obs["waypoint_paths"]["position"][0:11, 3:, 0:3]
        for path in wps[:]:
            wps_valid = points_to_pixels(
                points=path,
                ego_pos=ego_pos,
                ego_heading=ego_heading,
                w=w,
                h=h,
                res=self._res,
            )
            for point in wps_valid:
                img_x, img_y = point[0], point[1]
                if all(rgb_ego[img_y, img_x, :] == self._no_color):
                    rgb_ego[img_y, img_x, :] = self._wps_color

        # Superimpose goal position onto rgb image       
        if not all((goal:=obs["ego_vehicle_state"]["mission"]["goal_position"]) == np.zeros((3,))):       
            goal_pixel = points_to_pixels(
                points=np.expand_dims(goal,axis=0),
                ego_pos=ego_pos,
                ego_heading=ego_heading,
                w=w,
                h=h,
                res=self._res,
            )
            if len(goal_pixel) != 0:
                img_x, img_y = goal_pixel[0][0], goal_pixel[0][1]
                if all(rgb_ego[img_y, img_x, :] == self._no_color) or all(rgb_ego[img_y, img_x, :] == self._wps_color):
                    rgb_ego[
                        max(img_y-self._blur_radius,0):min(img_y+self._blur_radius,h), 
                        max(img_x-self._blur_radius,0):min(img_x+self._blur_radius,w), 
                        :,
                    ] = self._goal_color

        # Crop image
        rgb_ego = rgb_ego[self._crop[2]:h-self._crop[3],self._crop[0]:w-self._crop[1],:]

        # Channel first rgb
        rgb_ego = rgb_ego.transpose(2, 0, 1)

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


def points_to_pixels(
    points: np.ndarray,
    ego_pos: np.ndarray,
    ego_heading: float,
    w: int,
    h: int,
    res: float,
) -> np.ndarray:
    """Converts points into pixel coordinates in order to superimpose the
    points onto the RGB image.

    Args:
        points (np.ndarray): Array of points. Shape (n,3).
        ego_pos (np.ndarray): Ego position. Shape = (3,).
        ego_heading (float): Ego heading in radians.
        w (int): Width of RGB image
        h (int): Height of RGB image.
        res (float): Resolution of RGB image in meters/pixels. Computed as
            ground_size/image_size.

    Returns:
        np.ndarray: Array of point coordinates on the RGB image. Shape = (m,3).
    """
    # fmt: off
    mask = [False if all(point == np.zeros(3,)) else True for point in points]
    points_nonzero = points[mask]
    points_delta = points_nonzero - ego_pos
    points_rotated = rotate_axes(points_delta, theta=ego_heading)
    points_pixels = points_rotated / np.array([res, res, res])
    points_overlay = np.array([w / 2, h / 2, 0]) + points_pixels * np.array([1, -1, 1])
    points_rfloat = np.rint(points_overlay)
    points_valid = points_rfloat[(points_rfloat[:,0] >= 0) & (points_rfloat[:,0] < w) & (points_rfloat[:,1] >= 0) & (points_rfloat[:,1] < h)]
    points_rint = points_valid.astype(int)
    return points_rint
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
