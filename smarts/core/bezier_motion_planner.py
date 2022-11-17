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
import numpy as np


class BezierMotionPlanner:
    """A bezier trajectory builder."""

    def __init__(self, extend=0.9, extend_bias=0.5, speed_calculation_resolution=5):
        self._extend = extend
        self._extend_bias = extend_bias
        self._speed_calculation_resolution = speed_calculation_resolution

    def trajectory(self, current_pose, target_pose_at_t, n, dt):
        """Generate a bezier trajectory to a target pose."""
        return self.trajectory_batched(
            np.array([current_pose]), np.array([target_pose_at_t]), n, dt
        )[0]

    def trajectory_batched(self, current_poses, target_poses_at_t, n, dt) -> np.ndarray:
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

        # FIXME: speed control still needed for values of t != dt !!!
        #  Reason being that the tangents (therefore speed) along a bezier curve will vary.
        real_times = target_poses_at_t[:, 3:4].repeat(n, axis=0).clip(dt, None)
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
        ).T.reshape(-1, 1) / real_times

        def linear_bezier(t, p0, p1):
            return (1 - t) * p0 + t * p1

        def quadratic_bezier(t, p0, p1, p2):
            return linear_bezier(t, linear_bezier(t, p0, p1), linear_bezier(t, p1, p2))

        def cubic_bezier(t, p0, p1, p2, p3):
            return linear_bezier(
                t, quadratic_bezier(t, p0, p1, p2), quadratic_bezier(t, p1, p2, p3)
            )

        def curve_lengths(subsections, t, p0, p1, p2, p3):
            # TAI: subsections could be scaled by time, magnitude between p0 and p3,
            # and directional difference between p1 and p2
            lengths = []
            inverse_subsection = 1 / subsections
            for (ti, p0i, p1i, p2i, p3i) in zip(t, p0, p1, p2, p3):
                # get s subsection points in [p(0):p(t)]
                tss = [ts * inverse_subsection * ti for ts in range(subsections + 1)]
                points = [cubic_bezier(ts, p0i, p1i, p2i, p3i) for ts in tss]
                subsection_length_total = 0
                # accumulate position deltas [s[t+1] - s[t]]
                for (ps, ps1) in zip(points[:-1], points[1:]):
                    # convert deltas to magnitudes
                    delta_dist = ps1 - ps
                    # add magnitudes
                    subsection_length = np.linalg.norm(delta_dist)
                    # add to lengths
                    subsection_length_total += subsection_length
                lengths.append(subsection_length_total)
            return np.array(lengths)

        def length_to_speed(t, length):
            speeds = [l / t if t > 0 else -1 for (t, l) in zip(t, length)]
            return np.array(speeds)

        positions = cubic_bezier(dts, p0s, p1s, p2s, p3s)
        # TODO: this could be optimized to use the positions already generated
        lengths = curve_lengths(
            self._speed_calculation_resolution, dts, p0s, p1s, p2s, p3s
        )
        speeds = length_to_speed(real_times.reshape(n), lengths)

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
