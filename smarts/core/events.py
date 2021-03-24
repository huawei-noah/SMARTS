# MIT License

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from typing import List, NamedTuple, Tuple

from smarts.zoo import worker_pb2


class Events(NamedTuple):
    collisions: List[Tuple[str, str]]
    off_route: bool
    reached_goal: bool
    reached_max_episode_steps: bool
    off_road: bool
    wrong_way: bool
    not_moving: bool


def events_to_proto(events: Events) -> worker_pb2.Events:
    return worker_pb2.Events(
        collisions=[
            collision_to_proto(collision) for collision in events.collisions
        ],
        off_route=events.off_route,
        reached_goal=events.reached_goal,
        reached_max_episode_steps=events.reached_max_episode_steps,
        off_road=events.off_road,
        wrong_way=events.wrong_way,
        not_moving=events.not_moving,
    )


class Collision(NamedTuple):
    collidee_id: str = None


def collision_to_proto(collision: Collision) -> worker_pb2.Collision:
    return worker_pb2.Collision(collidee_id=collision.collidee_id)
