# MIT License
#
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


class ActionSpaceType(Enum):
    """Available vehicle action spaces."""

    Continuous = 0
    """
    Action=(throttle, brake, steering)
    + throttle: Range=[0, 1]. Type=float.
    + brake: Range=[0, 1]. Type=float.
    + steering: Range=[-1, 1]. Type=float.
    Steering maps to full turn in one direction to a full turn in the other.
    Direction of turn for the steering depends on the vehicle.
    """
    Lane = 1
    """
    Action= ``str``. Discrete lane action from one of
    
    + "keep_lane",
    + "slow_down", 
    + "change_lane_left", and 
    + "change_lane_right".
    """
    ActuatorDynamic = 2
    """
    Action=(throttle, brake, steering_rate)
        
    + throttle: Range=[0, 1]. Type=float. 
    + brake: Range=[0, 1]. Type=float.
    + steering_rate: Range[-1, 1]. Type=float.
    
    Steering rate means the amount of steering change *per second*.
    (either positive or negative) to be applied to the current steering.
    This gets clipped to the available steering of the vehicle (which may vary.)
    """
    LaneWithContinuousSpeed = 3
    """
    Action=(target_speed, lane_change).

    + target_speed: Baseline target speed (controller may give more or less regardless). Type=float.
    + lane_change: Discrete lane change value. Can be one of 
        + -1 : change to right lane
        + 0 : keep to current lane
        + 1 : change to left lane
    """
    TargetPose = 4
    """
    Action=`(x_coord, y_coord, heading, time_delta)`. Type= ``Sequence[float,
    float, float, float]``. Continuous action space of vehicle's next x 
    coordinate, y coordinate, heading, and time delta to reach the given pose. 
    """
    Trajectory = 5
    """
    Action=`(x_coords, y_coords, headings, speeds)`. Type= ``(Sequence[float],
    Sequence[float], Sequence[float], Sequence[float])``. Continuous action 
    space using trajectory comprising of x coordinates, y coordinates, headings,
    and speeds, to directly move a vehicle.
    """
    MultiTargetPose = 6  # For boid control.
    """
    Action= ``Dict[str, (float, float, float, float)]``. Continuous action space
    that provides ``TargetPose`` actions for multiple vehicles.
    """
    MPC = 7
    """
    Action=`(x_coords, y_coords, headings, speeds)`. Type= ``(Sequence[float], 
    Sequence[float], Sequence[float], Sequence[float])``. Adaptive control 
    performed on the vehicle model to match the given trajectory comprising 
    of vehicle's x coordinates, y coordinates, headings, and speeds.
    """
    TrajectoryWithTime = 8  # For pure interpolation provider.
    """
    Action=([time],[x_coord],[y_coord],[heading],[speed]). 
    Type= ``(Sequence[float], Sequence[float], Sequence[float], Sequence[float],
    Sequence[float])``. Interpolates vehicle along the given trajectory 
    comprising of times, x coordinates, y coordinates, headings, and speeds.
    """
    Direct = 9
    """
    Action=(speed) OR (acceleration, angular_velocity). Type= ``Union[float, 
    (float,float)]``. Continuous action space where we can pass either (a) an 
    initial speed upon reset, or (b) linear acceleration and angular velocity 
    for other steps.
    """
    Empty = 10
    """
    Action=(). Type= ``Tuple[]. This action is empty.``
    """
    RelativeTargetPose = 11
    """
    Action=(delta_x, delta_y, delta_heading). Type= ``Sequence[float,
    float, float]``. Continuous action space of vehicle's next pose in terms of delta x 
    coordinate, delta y coordinate, and delta heading, to be reached in 0.1 seconds.
    """
