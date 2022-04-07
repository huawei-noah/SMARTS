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
from enum import Enum
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

from smarts.core.events import Events


class TrafficActorType(str, Enum):
    """Traffic actor type information to help distinguish actors from each other."""

    SocialVehicle = "social_vehicle"
    SocialAgent = "social_agent"
    Agent = "agent"

    # TODO: make as override
    def serialize_to_context(self, context):
        mapping = {
            TrafficActorType.SocialVehicle: 0,
            TrafficActorType.SocialAgent: 1,
            TrafficActorType.Agent: 2,
        }
        # using context.primative(type(self)) as cc:
        #   cc.add("traffic_actor", this, select=(ta)=>mapping[ta])


class VehicleType(str, Enum):
    """Vehicle classification type information."""

    Bus = "bus"
    Coach = "coach"
    Truck = "truck"
    Trailer = "trailer"
    Car = "car"

    # TODO: make as override
    def serialize_to_context(self, context):
        mapping = {
            VehicleType.Bus: 0,
            VehicleType.Coach: 1,
            VehicleType.Truck: 2,
            VehicleType.Trailer: 3,
            VehicleType.Car: 4,
        }
        # using context.primative(type(self)) as cc:
        #   cc.add("vehicle_type", this, select=(vt)=>mapping[vt])


class TrafficActorState(NamedTuple):
    """Individual traffic actor state and meta information."""

    actor_type: TrafficActorType
    vehicle_type: Union[VehicleType, str]  # TODO: Restrict to VehicleType only
    position: Tuple[float, float, float]
    heading: float
    speed: float
    name: str = ""
    actor_id: Optional[str] = None
    events: Events = None
    waypoint_paths: Sequence = []
    driven_path: Sequence = []
    point_cloud: Sequence = []
    mission_route_geometry: Sequence[Sequence[Tuple[float, float]]] = None

    # TODO: make as override
    def serialize_to_context(self, context):
        # TODO implement
        # using context.layer(type(self), self.actor_id) as cc:
        #   cc.add("actor_id", self.actor_id, op=Context.REDUCE)
        #   cc.add("lane_id", self.lane_id, op=Context.DELTA | Context.REDUCE)
        #   cc.add("position", self.position, op=Context.FLATTEN)
        #   cc.add("heading", self.heading)
        #   cc.add("speed", self.speed)
        #   cc.add("events", self.events, op=Context.LOCAL_DELTA)
        #   cc.add("score", self.score, op=Context.OPTIONAL)
        #   cc.add("waypoint_paths", self.waypoint_paths, op=Context.OPTIONAL)
        #   cc.add("driven_path", self.driven_path, op=Context.OPTIONAL)
        #   cc.add("point_cloud", self.point_cloud, op=Context.OPTIONAL)
        #   cc.add("mission_route_geometry", self.mission_route_geometry, op=Context.OPTIONAL)
        #   cc.add("actor_type", self.actor_type, op=Context.ONCE | Context.REDUCE)
        #   cc.add("vehicle_type", self.agent_type, op=Context.ONCE | Context.REDUCE)
        pass


class State(NamedTuple):
    """A full representation of a single frame of an envision simulation step."""

    traffic: Dict[str, TrafficActorState]
    scenario_id: str
    scenario_name: str
    # sequence of x, y coordinates
    bubbles: Sequence[Sequence[Tuple[float, float]]]
    scene_colors: Dict[str, Tuple[float, float, float, float]]
    scores: Dict[str, float]
    ego_agent_ids: list
    position: Dict[str, Tuple[float, float]]
    speed: Dict[str, float]
    heading: Dict[str, float]
    lane_ids: Dict[str, str]
    frame_time: float

    # TODO: make as override
    def serialize_to_context(self, context):
        # TODO implement
        # using context.layer(type(self), self.actor_id) as cc:
        #   cc.add("frame_time", self.frame_time)
        #   cc.add("scenario_id", self.scenario_id)
        #   cc.add("scenario_name", self.scenario_name, op=Context.ONCE)
        #   cc.add("bubbles", self.bubbles, select=(bbl)=>(bbl.geometry, bbl.pose), alternate=(bbl)=>bbl.pose, op=Context.USE_ALTERNATE) # On delta use alternative
        #   cc.add("ego_agent_ids", self.ego_agent_ids, op=Context.DELTA)
        pass


# TODO: Output the following:
# _x = float
# _y = float
# _z = float
# _pos=Tuple[_x,_y,_z]
# _heading = float
# _id = int
# _pose = Tuple[_x, _y, _z, _heading]
# from gym.spaces import Space
# _Layout=Space
# _Waypoint=[
#     *_pose,
#     _id, # lane_id
#     float, # lane_width_at_waypoint
#     float, # speed_limit
#     int, # lane_index
# ]
# _Traffic = [
#     int, # actor_id # mapped to id in scenario level state
#     int, # lane_id # mapped to name in scenario level state
#     *_pose, # [x,y,z,heading]
#     float, #speed
#     Optional[Sequence[_Waypoint]], # waypoint_paths
#     Optional[Sequence[_pos]], # point cloud positions [x,y,x2,y2,...,xn,yn]
#     Optional[Sequence[_pos]], # mission route geometry positions [x,y,x2,y2,...,xn,yn]
#     int, #actor_type # Send once
#     int, #vehicle_type # Send once
#     # bitfield, events
#     # remove name
# ]
# _Geometry = Tuple[Sequence[Tuple[_x, _y]], _pose]
# _State = [
#     int, # frame time
#     str, # scenario id
#     Optional[str], # scenario name # send once
#     *Sequence[Tuple[_id, Union[_Geometry, _pose]]], # should be sent updates only Optional[id,geometry] optional[pos,heading]
#     *Sequence[_Traffic],
# ]
# _Payload = [
#     [
#         *_State,
#         Sequence[Tuple[_id, str]], # delta lane ids # new ids sent once
#         Sequence[Tuple[_id, str]], # added vehicle ids
#         Sequence[_id], # removed vehicle ids
#         Sequence[Tuple[_id, str]], # added agent ids
#         Sequence[_id], # removed agent ids
#     ],
#     Optional[_Layout]
# ]


def format_actor_id(actor_id: str, vehicle_id: str, is_multi: bool):
    """A conversion utility to ensure that an actor id conforms to envision's actor id standard.
    Args:
        actor_id: The base id of the actor.
        vehicle_id: The vehicle id of the vehicle the actor associates with.
        is_multi: If an actor associates with multiple vehicles.

    Returns:
        An envision compliant actor id.
    """
    if is_multi:
        return f"{actor_id}-{{{vehicle_id[:4]}}}"
    return actor_id
