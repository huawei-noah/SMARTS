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

from smarts.proto import observation_pb2


class Events(NamedTuple):
    collisions: List[Tuple[str, str]]
    not_moving: bool
    off_road: bool
    off_route: bool
    on_shoulder: bool
    reached_goal: bool
    reached_max_episode_steps: bool
    wrong_way: bool
    agents_alive_done: bool


def events_to_proto(events: Events) -> observation_pb2.Events:
    return observation_pb2.Events(
        collisions=[collision_to_proto(collision) for collision in events.collisions],
        off_route=events.off_route,
        on_shoulder=events.on_shoulder,
        reached_goal=events.reached_goal,
        reached_max_episode_steps=events.reached_max_episode_steps,
        off_road=events.off_road,
        wrong_way=events.wrong_way,
        not_moving=events.not_moving,
    )


def proto_to_events(proto: observation_pb2.Events) -> Events:
    return Events(
        collisions=[proto_to_collision(collision) for collision in proto.collisions],
        off_route=proto.off_route,
        on_shoulder=proto.on_shoulder,
        reached_goal=proto.reached_goal,
        reached_max_episode_steps=proto.reached_max_episode_steps,
        off_road=proto.off_road,
        wrong_way=proto.wrong_way,
        not_moving=proto.not_moving,
    )


class Collision(NamedTuple):
    collidee_id: str = None


def collision_to_proto(collision: Collision) -> observation_pb2.Collision:
    return observation_pb2.Collision(collidee_id=collision.collidee_id)


def proto_to_collision(proto: observation_pb2.Collision) -> Collision:
    return Collision(collidee_id=proto.collidee_id)
