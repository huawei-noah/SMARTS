# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import math
import numpy as np


def batches(list_, n):
    """Split an indexable container into `n` batches.

    Args:
      list_:
        The iterable to split into parts
      n:
        The number of batches
    """
    for i in range(0, len(list_), n):
        yield list_[i : i + n]


def yaw_from_quaternion(quaternion) -> float:
    """Converts a quaternion to the yaw value.

    Args:
      np.narray: np.array([x, y, z, w])

    Returns:
      A float angle in radians.
    """
    assert len(quaternion) == 4, f"len({quaternion}) != 4"
    siny_cosp = 2 * (quaternion[0] * quaternion[1] + quaternion[3] * quaternion[2])
    cosy_cosp = (
        quaternion[3] ** 2
        + quaternion[0] ** 2
        - quaternion[1] ** 2
        - quaternion[2] ** 2
    )
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return yaw


def fast_quaternion_from_angle(angle: float) -> np.ndarray:
    """Converts a float to a quaternion.

    Args:
      angle: An angle in radians.

    Returns:
      np.ndarray: np.array([x, y, z, w])
    """

    half_angle = angle * 0.5
    return np.array([0, 0, math.sin(half_angle), math.cos(half_angle)])


def clip(val, min_val, max_val):
    assert (
        min_val <= max_val
    ), f"min_val({min_val}) must be less than max_val({max_val})"
    return min_val if val < min_val else max_val if val > max_val else val


def squared_dist(a, b) -> float:
    """Computes the squared distance between a and b.

    Args:
      a, b: same dimension numpy.array([..])
    Returns:
      float: dist**2
    """
    delta = b - a
    return np.dot(delta, delta)


def signed_dist_to_line(point, line_point, line_dir_vec) -> float:
    """Computes the signed distance to a directed line

    The signed of the distance is:
      - negative if point is on the right of the line
      - positive if point is on the left of the line

    >>> import numpy as np
    >>> signed_dist_to_line(np.array([2, 0]), np.array([0, 0]), np.array([0, 1.]))
    -2.0
    >>> signed_dist_to_line(np.array([-1.5, 0]), np.array([0, 0]), np.array([0, 1.]))
    1.5
    """
    p = vec_2d(point)
    p1 = line_point
    p2 = line_point + line_dir_vec

    u = abs(
        line_dir_vec[1] * p[0] - line_dir_vec[0] * p[1] + p2[0] * p1[1] - p2[1] * p1[0]
    )
    d = u / np.linalg.norm(line_dir_vec)

    line_normal = np.array([-line_dir_vec[1], line_dir_vec[0]])
    _sign = np.sign(np.dot(p - p1, line_normal))
    return d * _sign


def vec_2d(v) -> np.ndarray:
    """Converts a higher order vector to a 2D vector."""

    assert len(v) >= 2
    return np.array(v[:2])


def sign(x) -> int:
    """Finds the sign of a numeric type.

    Args:
        x: A signed numeric type
    Returns:
        The sign [-1|1] of the input number
    """

    return 1 - (x < 0) * 2


def lerp(a, b, p):
    """Linear interpolation between a and b with p

    .. math:: a * (1.0 - p) + b * p

    Args:
        a, b: interpolated values
        p: [0..1] float describing the weight of a to b
    """

    assert 0 <= p and p <= 1

    return a * (1.0 - p) + b * p


def low_pass_filter(
    input_value,
    previous_filter_state,
    filter_constant,
    time_step,
    lower_bound=-1,
    raw_value=0,
):
    previous_filter_state += (
        time_step * filter_constant * (input_value - previous_filter_state)
    )
    previous_filter_state = np.clip(previous_filter_state + raw_value, lower_bound, 1)
    return previous_filter_state


def radians_to_vec(radians) -> np.ndarray:
    # +y = 0 rad.
    angle = (radians + math.pi * 0.5) % (2 * math.pi)
    return np.array([math.cos(angle), math.sin(angle)])


def vec_to_radians(v) -> float:
    # See: https://stackoverflow.com/a/15130471
    assert len(v) == 2, f"Vector must be 2D: {repr(v)}"

    x, y = v
    r = math.atan2(abs(y), abs(x))

    quad = 0
    if x < 0:
        if y < 0:
            quad = 3
        else:
            quad = 2
    else:
        if y < 0:
            quad = 4

    # Adjust angle based on quadrant
    if 2 == quad:
        r = math.pi - r
    elif 3 == quad:
        r = math.pi + r
    elif 4 == quad:
        r = 2 * math.pi - r

    # +y = 0 rad.
    return (r - (math.pi) * 0.5) % (2 * math.pi)


def rotate_around_point(point, radians, origin=(0, 0)) -> np.ndarray:
    """Rotate a point around a given origin."""
    x, y = point
    ox, oy = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return np.array([qx, qy])


def min_angles_difference_signed(first, second) -> float:
    return ((first - second) + math.pi) % (2 * math.pi) - math.pi


def position_to_ego_frame(position, ego_position, ego_heading):
    """
    Get the position in ego vehicle frame given the pose (of either a vehicle or some point) in global frame.
    Egocentric frame: The ego position becomes origin, and ego heading direction is positive x-axis.
    Args:
        position: [x,y,z]
        ego_position: Ego vehicle [x,y,z]
        ego_heading: Ego vehicle heading in radians

    Returns:
        new_pose: The pose [x,y,z] in egocentric view
    """
    transform_matrix = np.eye(3)
    ego_rel_position = np.asarray(position) - np.asarray(ego_position)
    transform_matrix[0, 0] = np.cos(-ego_heading)
    transform_matrix[0, 1] = -np.sin(-ego_heading)
    transform_matrix[1, 0] = np.sin(-ego_heading)
    transform_matrix[1, 1] = np.cos(-ego_heading)

    new_position = np.matmul(transform_matrix, ego_rel_position.T).T
    return new_position.tolist()


def trajectory_batched(current_poses, target_poses_at_t, n, dt) -> np.array:
    """Generate a batch of trajectories

    Args:
        current_poses: np.array([[x, y, heading]])
        target_poses_at_t: np.array([[x, y, heading, seconds_into_future]]
                pose we would like to have this many seconds into the future
        n: number of trajectory points to return
        dt: time delta between trajectory points
    Returns:
        Stacked np.array of the form:
            np.array([[x], [y], [heading], [desired_speed]])
    """
    extend_bias = 0.5
    extend = 0.9
    assert len(current_poses) == len(target_poses_at_t)
    # vectorized cubic bezier computation
    target_headings = target_poses_at_t[:, 2] + np.pi * 0.5
    target_dir_vecs = np.array(
        [np.cos(target_headings), np.sin(target_headings)]
    ).T.reshape(-1, 2)

    current_headings = current_poses[:, 2] + np.pi * 0.5
    current_dir_vecs = np.array(
        [np.cos(current_headings), np.sin(current_headings)]
    ).T.reshape(-1, 2)

    extension = (
        np.linalg.norm(target_poses_at_t[:, :2] - current_poses[:, :2], axis=1).reshape(
            -1, 1
        )
        * extend
    )

    p0s = current_poses[:, :2].repeat(n, axis=0)
    p1s = (current_poses[:, :2] + current_dir_vecs * extension * extend_bias).repeat(
        n, axis=0
    )
    p2s = (
        target_poses_at_t[:, :2] - target_dir_vecs * extension * (1 - extend_bias)
    ).repeat(n, axis=0)
    p3s = target_poses_at_t[:, :2].repeat(n, axis=0)
    dts = (np.array(range(1, n + 1)) * dt).reshape(-1, 1).repeat(
        len(current_poses), axis=1
    ).T.reshape(-1, 1) / target_poses_at_t[:, 3:4].repeat(n, axis=0).clip(dt, None)

    def linear_bezier(t, p0, p1):
        return (1 - t) * p0 + t * p1

    def quadratic_bezier(t, p0, p1, p2):
        return linear_bezier(t, linear_bezier(t, p0, p1), linear_bezier(t, p1, p2))

    def cubic_bezier(t, p0, p1, p2, p3):
        return linear_bezier(
            t, quadratic_bezier(t, p0, p1, p2), quadratic_bezier(t, p1, p2, p3)
        )

    def cubic_bezier_derivative(t, p0, p1, p2, p3):
        return (
            3 * (1 - t) ** 2 * (p1 - p0)
            + 6 * (1 - t) * t * (p2 - p1)
            + 3 * t ** 2 * (p3 - p2)
        )

    positions = cubic_bezier(dts, p0s, p1s, p2s, p3s)
    tangents = cubic_bezier_derivative(dts, p0s, p1s, p2s, p3s)
    speeds = np.linalg.norm(tangents, axis=1)

    # angle interp. equations come from:
    # https://stackoverflow.com/questions/2708476/rotation-interpolation#14498790
    heading_correction = ((target_headings - current_headings) + np.pi) % (
        2 * np.pi
    ) - np.pi
    headings = (
        current_headings
        + (
            (dts.reshape(-1) * heading_correction + np.pi) % (2 * np.pi) - np.pi
        ).reshape(-1)
        - np.pi * 0.5
    )

    trajectories = np.array(
        [positions[:, 0], positions[:, 1], headings, speeds]
    ).T.reshape(-1, 4, n)
    return trajectories


import numpy as np
from math import factorial


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def get_bezier_curve(points):
    n = len(points) - 1
    return lambda t: sum(
        comb(n, i) * t ** i * (1 - t) ** (n - i) * points[i] for i in range(n + 1)
    )


def evaluate_bezier(points, total):
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:, 0], new_points[:, 1]
