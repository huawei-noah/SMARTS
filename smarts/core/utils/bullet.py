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
from typing import NamedTuple, Tuple
from multiprocessing import Process, Pipe

import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc

from smarts.core.coordinates import Pose


class BulletClient:
    """This wrapper class is a hack for macOS where running PyBullet in GUI mode,
    alongside Panda3D segfaults. It seems due to running two OpenGL applications
    in the same process. Here we spawn a process to run PyBullet and forward all
    calls to it over unix pipes.

    N.B. This class can be directly subbed-in for pybullet_utils's BulletClient
    but your application must start from a,

        if __name__ == "__main__:":
            # https://turtlemonvh.github.io/python-multiprocessing-and-corefoundation-libraries.html
            import multiprocessing as mp
            mp.set_start_method('spawn', force=True)
    """

    def __init__(self, bullet_connect_mode=pybullet.GUI):
        self._parent_conn, self._child_conn = Pipe()
        self.process = Process(
            target=BulletClient.consume, args=(bullet_connect_mode, self._child_conn,),
        )
        self.process.start()

    def __getattr__(self, name):
        def wrapper(*args, **kwargs):
            self._parent_conn.send((name, args, kwargs))
            return self._parent_conn.recv()

        return wrapper

    @staticmethod
    def consume(bullet_connect_mode, connection):
        # runs in sep. process
        client = bc.BulletClient(bullet_connect_mode)

        while True:
            method, args, kwargs = connection.recv()
            result = getattr(client, method)(*args, **kwargs)
            connection.send(result)


class ContactPoint(NamedTuple):
    bullet_id: str
    contact_point: Tuple[float, float, float]
    contact_point_other: Tuple[float, float, float]


class JointInfo(NamedTuple):
    index: int
    type_: int
    lower_limit: float
    upper_limit: float
    max_force: float
    max_velocity: float


class JointState(NamedTuple):
    position: Tuple[float, ...]
    velocity: Tuple[float, ...]


class BulletBoxShape:
    def __init__(self, bbox, bullet_client):
        self._client = bullet_client

        width, length, height = bbox
        collision_box = bullet_client.createCollisionShape(
            shapeType=pybullet.GEOM_BOX,
            halfExtents=(length * 0.5, width * 0.5, height * 0.5),
        )

        self._height = height
        self._bullet_id = self._client.createMultiBody(1, collision_box)

    def reset_pose(self, pose: Pose):
        """Only call this before it needs to do anything physics-wise"""
        position, orientation = pose.as_bullet()
        self._client.resetBasePositionAndOrientation(
            self._bullet_id,
            np.sum([position, [0, 0, self._height * 0.5]], axis=0),
            orientation,
        )

    def teardown(self):
        self._client.removeBody(self._bullet_id)
        self._bullet_id = None


class BulletPositionConstraint:
    def __init__(self, bullet_shape, bullet_client):
        self._client = bullet_client

        self._bullet_shape = bullet_shape
        self._bullet_cid = None

    def _make_constraint(self, pose: Pose):
        self._bullet_shape.reset_pose(pose)
        relative_pos = [0, 0, 0]
        relative_ori = [0, 0, 0, 1]
        self._bullet_cid = self._client.createConstraint(
            self._bullet_shape._bullet_id,
            -1,
            -1,
            -1,
            self._client.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, -self._bullet_shape._height * 0.5],
            relative_pos,
            relative_ori,
        )

    def move_to(self, pose: Pose):
        if not self._bullet_cid:
            self._make_constraint(pose)
        position, orientation = pose.as_bullet()
        # TODO: Consider to remove offset when collision is improved
        # Move contraints slightly up to avoid ground collision
        ground_position = position + [0, 0, 0.2]
        self._client.changeConstraint(self._bullet_cid, ground_position, orientation)

    def teardown(self):
        if self._bullet_cid is not None:
            self._client.removeConstraint(self._bullet_cid)
        self._bullet_cid = None
