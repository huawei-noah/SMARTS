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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import math
from dataclasses import dataclass
from math import factorial
from typing import Callable, List, Sequence, Tuple, Union


@dataclass(frozen=True)
class CubicPolynomial:
    """A cubic polynomial."""

    a: float
    b: float
    c: float
    d: float

    @classmethod
    def from_list(cls, coefficients: List[float]):
        """Generates CubicPolynomial.
        Args:
            coefficients: The list of coefficients [a, b, c, d]
        Returns:
            A new CubicPolynomial.
        """
        return cls(
            a=coefficients[0],
            b=coefficients[1],
            c=coefficients[2],
            d=coefficients[3],
        )

    def eval(self, ds: float) -> float:
        """Evaluate a value along the polynomial."""
        return self.a + self.b * ds + self.c * ds * ds + self.d * ds * ds * ds


def constrain_angle(angle: float) -> float:
    """Constrain an angle within the inclusive range [-pi, pi]"""
    angle %= 2 * math.pi
    if angle > math.pi:
        angle -= 2 * math.pi
    return angle


import numpy as np


def batches(list_, n):
    """Split an indexable container into `n` batches.

    :param list_: The iterable to split into parts
    :param n: The number of batches
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


def mult_quat(q1, q2):
    """Specialized quaternion multiplication as required by the unique attributes of quaternions.
    Returns:
        The product of the quaternions.
    """
    q3 = np.copy(q1)
    q3[0] = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    q3[1] = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    q3[2] = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    q3[3] = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    return q3


def rotate_quat(quat, vect):
    """Rotate a vector with the rotation defined by a quaternion."""
    # Transform a vector into an quaternion
    vect = np.append([0], vect)
    # Normalize it
    norm_vect = np.linalg.norm(vect)
    vect /= norm_vect
    # Computes the conjugate of quat
    quat_ = np.append(quat[0], -quat[1:])
    # The result is given by: quat * vect * quat_
    res = mult_quat(quat, mult_quat(vect, quat_)) * norm_vect
    return res[1:]


def clip(val, min_val, max_val):
    """Constrain a value between a min and max by clamping exterior values to the extremes."""
    assert (
        min_val <= max_val
    ), f"min_val({min_val}) must be less than max_val({max_val})"
    return min_val if val < min_val else max_val if val > max_val else val


def get_linear_segments_for_range(
    s_start: float, s_end: float, segment_size: float
) -> List[float]:
    """Given a range from s_start to s_end, give a linear segment of size segment_size."""
    num_segments = int((s_end - s_start) / segment_size) + 1
    return [s_start + seg * segment_size for seg in range(num_segments)]


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

    ..code-block:: python

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
    """Filters out large value jumps by taking a filter state and returning a filter state.
    This is generally intended for filtering out high frequencies from raw signal values.
    Args:
        input_value: The raw signal value.
        previous_filter_state: The last generated value from the filter.
        filter_constant: The scale of the filter
        time_step: The length of time between the previously processed signal and the current signal.
        lower_bound: The lowest possible value allowed.
        raw_value: A scalar addition to the signal value.
    Returns:
        The processed raw signal value.

    """
    previous_filter_state += (
        time_step * filter_constant * (input_value - previous_filter_state)
    )
    previous_filter_state = np.clip(previous_filter_state + raw_value, lower_bound, 1)
    return previous_filter_state


def radians_to_vec(radians) -> np.ndarray:
    """Convert a radian value to a unit directional vector. 0 rad relates to [0x, 1y] with
    counter-clockwise rotation.
    """
    # +y = 0 rad.
    angle = (radians + math.pi * 0.5) % (2 * math.pi)
    return np.array((math.cos(angle), math.sin(angle)))


def vec_to_radians(v) -> float:
    """Converts a vector to a radian value. [0x,+y] is 0 rad with counter-clockwise rotation."""
    # See: https://stackoverflow.com/a/15130471
    assert len(v) == 2, f"Vector must be 2D: {repr(v)}"

    x, y = v
    r = math.atan2(abs(y), abs(x))

    # Adjust angle based on quadrant where +y = 0 rad.
    # Standard quadrants
    #    +y
    #   2 | 1
    # -x - - - +x
    #   3 | 4
    #    -y
    if x < 0:
        if y < 0:
            return (r + 0.5 * math.pi) % (2 * math.pi)  # quad 3
        return (0.5 * math.pi - r) % (2 * math.pi)  # quad 2
    elif y < 0:
        return (1.5 * math.pi - r) % (2 * math.pi)  # quad 4
    return (r - 0.5 * math.pi) % (2 * math.pi)  # quad 1


def circular_mean(vectors: Sequence[np.ndarray]) -> float:
    """Given a sequence of equal-length 2D vectors (e.g., unit vectors),
    returns their circular mean in radians, but with +y = 0 rad.
    See: https://en.wikipedia.org/wiki/Circular_mean"""
    return (
        math.atan2(sum(v[1] for v in vectors), sum(v[0] for v in vectors))
        - 0.5 * math.pi
    )


def is_close(a: float, b: float, rel_tol: float = 1e-09, abs_tol: float = 0.0) -> bool:
    """Determines if two values are close as defined by the inputs.
    Args:
        a: The first value.
        b: The other value.
        rel_tol: Difference required to be close relative to the magnitude
        abs_tol: Absolute different allowed to be close.
    Returns:
        If the two values are "close".
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def rotate_cw_around_point(point, radians, origin=(0, 0)) -> np.ndarray:
    """Rotate a point clockwise around a given origin."""
    x, y = point
    ox, oy = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return np.array([qx, qy])


def line_intersect(a, b, c, d) -> Union[np.ndarray, None]:
    """Check if the lines ``[a, b]`` and ``[c, d]`` intersect, and return the
    intersection point if so. Otherwise, return None.
    """

    r = b - a
    s = d - c
    d = r[0] * s[1] - r[1] * s[0]

    if d == 0:
        return None

    u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / d
    t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / d

    if 0 <= u <= 1 and 0 <= t <= 1:
        return a + t * r

    return None


def line_intersect_vectorized(
    a: np.ndarray,
    b: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    ignore_start_pt: bool = False,
) -> bool:
    """Vectorized version of `line_intersect(...)`, where C and D represent
    the segment points for an entire line, and a and b are points of a single
    line segment to be tested against.
    If ignore_start_pt is True, then two diverging lines that *only* intersect at
    their starting points will cause this to return False.
    """
    r = b - a
    S = D - C
    rs1 = np.multiply(r[0], S[:, 1])
    rs2 = np.multiply(r[1], S[:, 0])
    d = rs1 - rs2

    if not np.any(d):
        return False

    u_numerator = np.multiply(C[:, 0] - a[0], r[1]) - np.multiply(C[:, 1] - a[1], r[0])
    t_numerator = np.multiply(C[:, 0] - a[0], S[:, 1]) - np.multiply(
        C[:, 1] - a[1], S[:, 0]
    )

    # Use where=d!=0 to avoid divisions by zero. The out parameter is an
    # array of [-1, ..., -1], to make sure we're defaulting to something
    # outside of our expected range for our return result.
    u = np.divide(u_numerator, d, out=np.zeros_like(u_numerator) - 1, where=d != 0)
    t = np.divide(t_numerator, d, out=np.zeros_like(t_numerator) - 1, where=d != 0)

    u_in_range = (0 <= u) & (u <= 1)
    t_in_range = (0 <= t) & (t <= 1)
    combined = u_in_range & t_in_range

    return np.any(combined) and (not ignore_start_pt or any(combined[1:]) or t[0] > 0.0)


def ray_boundary_intersect(
    ray_start, ray_end, boundary_pts, early_return=True
) -> Union[np.ndarray, None]:
    """Iterate over the boundary segments, returning the nearest intersection point if a ray intersection is found.
    If early_return is True, this will return the first intersection point that is found."""
    vl = len(ray_start)
    assert vl == len(ray_end)
    nearest_pt = None
    min_dist = math.inf
    for j in range(len(boundary_pts) - 1):
        b0 = boundary_pts[j][:vl]
        b1 = boundary_pts[j + 1][:vl]
        pt = line_intersect(b0, b1, ray_start, ray_end)
        if pt is not None:
            if early_return:
                return pt
            dist = np.linalg.norm(pt - ray_start)
            if dist < min_dist:
                min_dist = dist
                nearest_pt = pt
    return nearest_pt


def min_angles_difference_signed(first, second) -> float:
    """The minimum signed difference between angles(radians)."""
    return ((first - second) + math.pi) % (2 * math.pi) - math.pi


def wrap_value(value: Union[int, float], _min: float, _max: float) -> float:
    """Wraps the value around if it goes over max or under min."""
    v = value
    assert isinstance(value, (int, float))
    diff = _max - _min
    if value <= _min:
        v = _max - (_min - value) % diff
    if value > _max:
        v = _min + (value - _max) % diff
    return v


def _gen_ego_frame_matrix(ego_heading):
    transform_matrix = np.eye(3)
    transform_matrix[0, 0] = np.cos(-ego_heading)
    transform_matrix[0, 1] = -np.sin(-ego_heading)
    transform_matrix[1, 0] = np.sin(-ego_heading)
    transform_matrix[1, 1] = np.cos(-ego_heading)
    return transform_matrix


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
    transform_matrix = _gen_ego_frame_matrix(ego_heading)
    ego_rel_position = np.asarray(position) - np.asarray(ego_position)
    new_position = np.matmul(transform_matrix, ego_rel_position.T).T
    return new_position.tolist()


def world_position_from_ego_frame(position, ego_world_position, ego_world_heading):
    """
    Restore the position from ego given the pose (of either a vehicle or some point) in world frame.
    world frame: The world (0, 0, 0) becomes origin.
    Args:
        position: [x,y,z]
        ego_world_position: Ego vehicle [x,y,z]
        ego_world_heading: Ego vehicle heading in radians
    Returns:
        new_pose: The pose [x,y,z] in world frame
    """
    transform_matrix = _gen_ego_frame_matrix(ego_world_heading)
    transform_matrix = np.linalg.inv(transform_matrix)
    rot_position = np.matmul(transform_matrix, np.asarray(position).T).T
    new_position = np.asarray(rot_position) + np.asarray(ego_world_position)
    return new_position.tolist()


def comb(n, k):
    """Binomial coefficient"""
    return factorial(n) // (factorial(k) * factorial(n - k))


def get_bezier_curve(points):
    """Get the curve function given a series of points.
    Returns:
        A curve function that takes a normalized offset [0:1] into the curve.
    """
    n = len(points) - 1
    return lambda t: sum(
        comb(n, i) * t**i * (1 - t) ** (n - i) * points[i] for i in range(n + 1)
    )


def evaluate_bezier(points, total):
    """Generate the approximated points of a bezier curve given a series of control points.
    Args:
        points: The bezier control points.
        total: The number of points generated from approximating the curve.
    Returns:
        An approximation of the bezier curve.
    """
    bezier = get_bezier_curve(points)
    new_points = np.array([bezier(t) for t in np.linspace(0, 1, total)])
    return new_points[:, 0], new_points[:, 1]


def inplace_unwrap(wp_array):
    """Unwraps an array in place."""
    ## minor optimization hack adapted from
    ##  https://github.com/numpy/numpy/blob/v1.20.0/numpy/lib/function_base.py#L1492-L1546
    ## to avoid unnecessary (slow) np array copy
    ## (as seen in profiling).
    p = np.asarray(wp_array)
    dd = np.subtract(p[1:], p[:-1])
    ddmod = np.mod(dd + math.pi, 2 * math.pi) - math.pi
    np.copyto(ddmod, math.pi, where=(ddmod == -math.pi) & (dd > 0))
    ph_correct = ddmod - dd
    np.copyto(ph_correct, 0, where=abs(dd) < math.pi)
    p[1:] += ph_correct.cumsum(axis=-1)
    return p


def round_param_for_dt(dt: float) -> int:
    """For a given dt, returns what to pass as the second parameter
    to the `round()` function in order to not lose precision.
    Note that for whole numbers, like 100, the result will be negative.
    For example, `round_param_for_dt(100) == -2`,
    such that `round(190, -2) = 200`."""
    strep = np.format_float_positional(dt)
    decimal = strep.find(".")
    if decimal >= len(strep) - 1:
        return 1 - decimal
    return len(strep) - decimal - 1


def rounder_for_dt(dt: float) -> Callable[[float], float]:
    """Return a rounding function appropriate for timestepping."""
    rp = round_param_for_dt(dt)
    return lambda f: round(f, rp)


def welford() -> Tuple[
    Callable[[float], None], Callable[[], float], Callable[[], float], Callable[[], int]
]:
    """Welford's online mean and std computation.

    Reference
        + https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
        + https://www.adamsmith.haus/python/answers/how-to-find-a-running-standard-deviation-in-python

    Returns:
        Tuple[ Callable[[float], None], Callable[[], float], Callable[[], float], Callable[[], int] ]: Callable functions to update, get mean, get std, and get steps.
    """

    import math

    n = 0  # steps
    M = 0
    S = 0

    def update(val: float):
        nonlocal n, M, S
        n = n + 1
        newM = M + (val - M) / n
        newS = S + (val - M) * (val - newM)
        M = newM
        S = newS

    def mean() -> float:
        nonlocal M
        return M

    def std() -> float:
        nonlocal n, M, S
        if n == 1:
            return 0

        std = math.sqrt(S / (n - 1))
        return std

    def steps() -> int:
        nonlocal n
        return n

    return update, mean, std, steps


def running_mean(prev_mean: float, prev_step: int, new_val: float) -> Tuple[float, int]:
    """
    Returns a new running mean value, when given previous mean, previous step
    count, and new value,

    Args:
        prev_mean (float): Previous mean value.
        prev_step (int): Previous step count.
        new_val (float): New value to be averaged.

    Returns:
        Tuple[float, int]: Updated mean and step count.
    """
    new_step = prev_step + 1
    new_mean = prev_mean + (new_val - prev_mean) / new_step
    return new_mean, new_step
