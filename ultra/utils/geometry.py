# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import math


class Box(object):
    def __init__(
        self,
        width=1,
        height=1,
        length=1,
        centerX=0,
        centerY=0,
        centerZ=0,
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        translationX=0,
        translationY=0,
        translationZ=0,
    ):
        # In webots length is in z-axis, width is in x-axis and height is in y-axis
        # Center is the rotation center for the box
        #  -> in webots, this should be the rear axle location relative to the center of the box
        #  -> center is the vector from the true center of the box to the rotation center of the box
        # In webots yaw is CC around the y-axis!
        # In webots pitch is CC around the z-axis!
        # In webots roll is CC around the x-axis!
        # NOTE: this geometry class applies a translation to get the center of rotation,
        #  rotates the box and then applies a global translation to move the rectangle in a global coordinate system
        self.dimensions = np.array([width, height, length])
        self.center = np.array([centerX, centerY, centerZ])
        self.translation = np.array([translationX, translationY, translationZ])
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.unrotatedegocorners = self._getunrotatedegocorners()
        self.rotation = self.getyawrollpitchrotation(self.yaw, self.pitch, self.roll)
        # The transpose is the inverse rotation matrix
        self.reverserotation = np.transpose(self.rotation)
        self.corners = self.getcorners()

    def __str__(self):
        return "[({},{},{}), center=({},{},{}), rotation=({},{},{}), translation=({},{},{})]".format(
            self.dimensions[0],
            self.dimensions[1],
            self.dimensions[2],
            self.center[0],
            self.center[1],
            self.center[2],
            self.yaw,
            self.pitch,
            self.roll,
            self.translation[0],
            self.translation[1],
            self.translation[2],
        )

    def getyawrollpitchrotation(self, yaw, pitch, roll):
        sin_p = math.sin(pitch)
        cos_p = math.cos(pitch)
        sin_y = math.sin(yaw)
        cos_y = math.cos(yaw)
        sin_r = math.sin(roll)
        cos_r = math.cos(roll)
        return np.array(
            [
                [
                    cos_p * cos_y,
                    cos_p * sin_y * sin_r - sin_p * cos_r,
                    cos_p * sin_y * cos_r + sin_p * sin_r,
                ],
                [
                    sin_p * cos_y,
                    sin_p * sin_y * sin_r + cos_p * cos_r,
                    sin_p * sin_y * cos_r - cos_p * sin_r,
                ],
                [-sin_y, cos_y * sin_r, cos_y * cos_r],
            ]
        )

    def _getunrotatedegocorners(self):
        x_diff1, y_diff1, z_diff1 = -self.dimensions / 2.0 - self.center
        x_diff2, y_diff2, z_diff2 = self.dimensions / 2.0 - self.center
        x1, y1, z1 = [
            min(x_diff1, x_diff2),
            min(y_diff1, y_diff2),
            min(z_diff1, z_diff2),
        ]
        x2, y2, z2 = [
            max(x_diff1, x_diff2),
            max(y_diff1, y_diff2),
            max(z_diff1, z_diff2),
        ]
        corners = np.array(
            [
                [x1, y1, z1],
                [x1, y1, z2],
                [x1, y2, z1],
                [x1, y2, z2],
                [x2, y1, z1],
                [x2, y1, z2],
                [x2, y2, z1],
                [x2, y2, z2],
            ]
        )
        return corners

    def getcorners(self):
        corners = self._getunrotatedegocorners()

        if abs(self.yaw) > 1e-30 or abs(self.pitch) > 1e-30 or abs(self.roll) > 1e-30:
            corners = np.inner(corners, self.rotation)

        corners += self.translation

        return corners

    def getvolume(self):
        return np.prod(self.dimensions)

    def containspoint(self, point):
        return self.containspoints(np.array([point]))

    def containspoints(self, points):
        # 1.) Rotate the point around the center
        # 2.) Check to see if the points lie inside the co-linear rectangle
        N, d = points.shape
        ego_points = points - self.translation
        if abs(self.yaw) > 1e-30 or abs(self.pitch) > 1e-30 or abs(self.roll) > 1e-30:
            rotated_points = np.inner(ego_points, self.reverserotation)
        else:
            rotated_points = ego_points
        low_corner = self.unrotatedegocorners[0]
        high_corner = self.unrotatedegocorners[7]

        # This is why we rotate the points rather than the box -> simpler to check if the box is
        #  co-linear with the axis of the local coordinate system
        return np.all(
            np.logical_and(
                (high_corner >= rotated_points), (rotated_points >= low_corner)
            ),
            axis=1,
        )

    # Note to be used externly
    def _unrotated_containspoints(self, unrotated_points):
        low_corner = self.unrotatedegocorners[0]
        high_corner = self.unrotatedegocorners[7]

        # This is why we rotate the points rather than the box -> simpler to check if the box is
        #  co-linear with the axis of the local coordinate system
        return np.all(
            np.logical_and(
                (high_corner >= unrotated_points), (unrotated_points >= low_corner)
            ),
            axis=1,
        )

    def _getnormals(self):
        # Just need three normals of the unrotated box
        p1, p2, p3, p4, p5, p6, p7, p8 = self.unrotatedegocorners
        xn = np.cross(p3 - p1, p2 - p1)
        yn = np.cross(p2 - p1, p5 - p1)
        zn = np.cross(p5 - p1, p3 - p1)

        return xn, yn, zn

    def getlines(self):
        p1, p2, p3, p4, p5, p6, p7, p8 = self.corners
        start_points = np.array([p1, p1, p1, p2, p2, p3, p3, p4, p5, p5, p6, p7])
        end_points = np.array([p2, p3, p5, p4, p6, p7, p4, p8, p6, p7, p8, p8])
        return start_points, end_points

    def intersects(self, box):
        # NOTE: the order of the points in self.corners and self.unrotatedegocorners must not change!

        # Calculates whether any corners of rect fall within self
        start1, end1 = box.getlines()
        intersect1 = self.intersectswithlines(points=start1, end_points=end1)

        # Also need to see if any of the corners of self fall in rect
        start2, end2 = self.getlines()
        intersect2 = box.intersectswithlines(points=start2, end_points=end2)

        return np.any(np.concatenate((intersect1, intersect2)))

    # Calculates intersection point between two parallel planes with norm and defined by points 1 and 2 respectively
    # norm must be the outer norm for plane1 defined by point pts_on_plane1
    def _get_line_intersect_with_planes_3d(
        self, points, directions, norm, pts_on_plane1, pts_on_plane2
    ):
        r = directions
        n1 = norm
        n2 = -norm
        d1 = -np.inner(n1, pts_on_plane1[0])
        d2 = -np.inner(n2, pts_on_plane2[0])

        r_n1 = np.inner(r, n1)
        r_n2 = np.inner(r, n2)
        n1_px = np.inner(n1, points) + d1
        n2_px = np.inner(n2, points) + d2
        n1_p = np.inner(n1, points)
        n2_p = np.inner(n2, points)

        t1 = np.zeros(len(points))
        t2 = np.zeros(len(points))

        # Check for parallel
        z1 = np.abs(r_n1) < 1e-20
        z2 = np.abs(r_n2) < 1e-20
        nz1 = np.logical_not(z1)
        nz2 = np.logical_not(z2)
        # Check for points on plane
        on1 = np.abs(n1_px) < 1e-20
        on2 = np.abs(n2_px) < 1e-20
        non1 = np.logical_not(on1)
        non2 = np.logical_not(on2)

        # All points that are not on the plane but are perpendicular -> inf
        t1[np.logical_and(z1, non1)] = -np.inf
        t2[np.logical_and(z2, non2)] = np.inf

        # All points not perpendicular and not on the plane
        nz_non1 = np.logical_and(nz1, non1)
        nz_non2 = np.logical_and(nz2, non2)
        t1[nz_non1] = -(d1 + n1_p[nz_non1]) / r_n1[nz_non1]
        t2[nz_non2] = -(d2 + n2_p[nz_non2]) / r_n2[nz_non2]

        # Re-order points if necessary
        t = np.stack((t1, t2), axis=1)
        tpos = np.min(t, axis=1)
        tneg = np.max(t, axis=1)

        # print("POS {}\nNEG {}".format(tpos, tneg))

        return tpos, tneg

    # NOTE: directions must be vectors from points (start) to end_points
    def intersectswithlines(self, points, end_points):
        # Method is described here: https://math.stackexchange.com/questions/1477930/does-a-line-intersect-a-box-in-a-3d-space
        rot_points = points - self.translation
        rot_end_points = end_points - self.translation
        if abs(self.yaw) > 1e-30 or abs(self.pitch) > 1e-30 or abs(self.roll) > 1e-30:
            rot_points = np.inner(rot_points, self.reverserotation)
            rot_end_points = np.inner(rot_end_points, self.reverserotation)

        rot_directions = rot_end_points - rot_points

        xn, yn, zn = self._getnormals()

        with np.errstate(divide="ignore"):
            low_xpoints = [self.unrotatedegocorners[0]]
            low_ypoints = [self.unrotatedegocorners[0]]
            low_zpoints = [self.unrotatedegocorners[0]]
            high_xpoints = [self.unrotatedegocorners[7]]
            high_ypoints = [self.unrotatedegocorners[7]]
            high_zpoints = [self.unrotatedegocorners[7]]

            t_xpos, t_xneg = self._get_line_intersect_with_planes_3d(
                rot_points, rot_directions, xn, high_xpoints, low_xpoints
            )
            t_ypos, t_yneg = self._get_line_intersect_with_planes_3d(
                rot_points, rot_directions, yn, high_ypoints, low_ypoints
            )
            t_zpos, t_zneg = self._get_line_intersect_with_planes_3d(
                rot_points, rot_directions, zn, high_zpoints, low_zpoints
            )

        pos_ts = np.stack((t_xpos, t_ypos, t_zpos), axis=1)
        neg_ts = np.stack((t_xneg, t_yneg, t_zneg), axis=1)

        # print("{} {}".format(pos_ts, neg_ts))

        maxpos = np.max(pos_ts, axis=1)
        minneg = np.min(neg_ts, axis=1)

        condition = np.array([False] * len(points))

        start_contains = self._unrotated_containspoints(rot_points)
        end_contains = self._unrotated_containspoints(rot_end_points)
        both = np.logical_and(start_contains, end_contains)
        one = np.logical_xor(start_contains, end_contains)
        none = np.logical_not(np.logical_or(both, one))

        # print("MAX {}; MIN {}".format(maxpos[none], minneg[none]))
        # print("POS {}; NEG {}".format(pos_ts[none], neg_ts[none]))

        # Handle the case where both points are in the box
        condition[both] = True
        # Handle the case where one point is in the box
        condition[one] = np.logical_and(
            maxpos[one] <= minneg[one],
            np.logical_and(maxpos[one] <= 1, minneg[one] >= 0),
        )
        # Handle the case where both points are outside the box
        if np.any(none):
            possibles = np.array([False] * len(points))
            possibles[none] = np.logical_and(
                maxpos[none] <= minneg[none],
                np.logical_and(
                    maxpos[none] >= 0,
                    np.logical_and(
                        minneg[none] <= 1,
                        np.logical_and(maxpos[none] <= 1, minneg[none] >= 0),
                    ),
                ),
            )
            if np.any(possibles):
                none_start_points = rot_points[possibles]
                none_directions = rot_directions[possibles]
                none_surface1 = none_start_points + np.transpose(
                    np.transpose(none_directions) * maxpos[possibles]
                )
                none_surface2 = none_start_points + np.transpose(
                    np.transpose(none_directions) * minneg[possibles]
                )
                # Update any possibles that were potentially true
                possibles[possibles] = np.logical_and(
                    self._unrotated_containspoints(none_surface1),
                    self._unrotated_containspoints(none_surface2),
                )
            condition[none] = possibles[none]

        return condition


# Support library for converting a LIDAR point (where the points are in scan order) to an image
class SphericalCartesianConverter(object):
    def __init__(self, hfov, vfov, width, height):
        # Assume tilt angle is 0!
        # Scan from left to right
        az = np.linspace(
            math.pi + hfov / 2.0, math.pi - hfov / 2.0, width, dtype=np.float32
        )
        # Scan from top layer to bottom layer
        el = np.linspace(vfov / 2.0, -vfov / 2.0, height, dtype=np.float32)
        az_grid, el_grid = np.meshgrid(az, el, sparse=True)
        self.z_factor = np.cos(az_grid) * np.cos(el_grid)
        self.x_factor = np.sin(az_grid) * np.cos(el_grid)
        self.y_factor = np.sin(el_grid)

    def calculate_point_cloud(self, depth):
        # These x, y, z are webots axes!
        z = depth * self.z_factor  # Points behind vehicle
        x = depth * self.x_factor  # Points to right of vehicle
        y = depth * self.y_factor  # Points up

        points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
        return points


def filter_points(points, ground_height=0.0, max_distance=99.9):
    distances = np.linalg.norm(points, axis=1)
    filtered_points = points[distances < max_distance]
    return filtered_points[filtered_points[:, 1] > ground_height]


def calculate_point_cloud(depth, hfov, vfov, width, height):
    # Assume tilt angle is 0!
    # Scan from left to right
    az = np.linspace(
        math.pi + hfov / 2.0, math.pi - hfov / 2.0, width, dtype=np.float32
    )
    # Scan from top layer to bottom layer
    el = np.linspace(vfov / 2.0, -vfov / 2.0, height, dtype=np.float32)
    az_grid, el_grid = np.meshgrid(az, el, sparse=True)
    z_factor = np.cos(az_grid) * np.cos(el_grid)
    x_factor = np.sin(az_grid) * np.cos(el_grid)
    y_factor = np.sin(el_grid)

    # These x, y, z are webots axes!
    z = depth * z_factor  # Points behind vehicle
    x = depth * x_factor  # Points to right of vehicle
    y = depth * y_factor  # Points up

    points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=-1)
    return points


def yaw_rotation(yaw):
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    return np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])


def yaw_rotate(yaw, vectors):
    rotation = yaw_rotation(yaw)
    return np.inner(vectors, rotation)


# In webots yaw is CC around the y-axis!
# In webots pitch is CC around the z-axis!
# In webots roll is CC around the x-axis!
def yawrollpitch_rotation(yaw, pitch, roll):
    sin_p = math.sin(pitch)
    cos_p = math.cos(pitch)
    sin_y = math.sin(yaw)
    cos_y = math.cos(yaw)
    sin_r = math.sin(roll)
    cos_r = math.cos(roll)
    return np.array(
        [
            [
                cos_p * cos_y,
                cos_p * sin_y * sin_r - sin_p * cos_r,
                cos_p * sin_y * cos_r + sin_p * sin_r,
            ],
            [
                sin_p * cos_y,
                sin_p * sin_y * sin_r + cos_p * cos_r,
                sin_p * sin_y * cos_r - cos_p * sin_r,
            ],
            [-sin_y, cos_y * sin_r, cos_y * cos_r],
        ]
    )


def rotate(yaw, pitch, roll, vectors):
    rotation = yawrollpitch_rotation(yaw, pitch, roll)
    return np.inner(vectors, rotation)


def visualize_boxes(ax, boxes, markers=None, colors=None, faces_color=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if markers is None:
        markers = ["o"] * len(boxes)
    if colors is None:
        colors = ["red"]
    if faces_color is None:
        faces_color = ["cyan"]

    for box, marker, color, face_color in zip(boxes, markers, colors, faces_color):
        Z = box.corners
        ax.scatter(
            box.corners[:, 0],
            box.corners[:, 1],
            box.corners[:, 2],
            marker=marker,
            color=color,
        )

        # generate list of sides' polygons of our pyramid
        verts = [
            [Z[0], Z[1], Z[3], Z[2]],
            [Z[4], Z[5], Z[7], Z[6]],
            [Z[0], Z[1], Z[5], Z[4]],
            [Z[2], Z[3], Z[7], Z[6]],
            [Z[1], Z[5], Z[7], Z[3]],
            [Z[4], Z[0], Z[2], Z[6]],
        ]

        # plot sides
        ax.add_collection3d(
            Poly3DCollection(
                verts, facecolors=face_color, linewidths=0.5, edgecolors="k", alpha=0.5
            )
        )
    ax.set_zlim(0, 50)
    ax.set_xlim(50, 150)
    ax.set_ylim(50, 150)

    return ax
