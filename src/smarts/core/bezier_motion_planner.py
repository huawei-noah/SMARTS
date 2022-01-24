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
import numpy as np

from .utils.math import vec_to_radians


class BezierMotionPlanner:
    def __init__(self, extend=0.9, extend_bias=0.5):
        self._extend = extend
        self._extend_bias = extend_bias

    def trajectory(self, current_pose, target_pose_at_t, n, dt):
        return self.trajectory_batched(
            np.array([current_pose]), np.array([target_pose_at_t]), n, dt
        )[0]

    def trajectory_batched(self, current_poses, target_poses_at_t, n, dt) -> np.array:
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
            np.linalg.norm(
                target_poses_at_t[:, :2] - current_poses[:, :2], axis=1
            ).reshape(-1, 1)
            * self._extend
        )

        p0s = current_poses[:, :2].repeat(n, axis=0)
        p1s = (
            current_poses[:, :2] + current_dir_vecs * extension * self._extend_bias
        ).repeat(n, axis=0)
        p2s = (
            target_poses_at_t[:, :2]
            - target_dir_vecs * extension * (1 - self._extend_bias)
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
