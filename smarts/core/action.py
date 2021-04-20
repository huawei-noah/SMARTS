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

from smarts.core.controllers import ActionSpaceType
from smarts.proto import action_pb2
from typing import Dict


def actions_to_proto(action_space_type: Dict, actions) -> action_pb2.ActionsBoid:
    keys = list(actions.keys())

    # if actions is boid agent, i.e., actions={<*-boid-*>: {<vehicle_id>: <ActionSpace>} }
    if "boid" in keys[0]:
        assert len(keys) == 1, "Incorrect boid dictionary structure in action."
        boid_key = keys[0]
        actions = actions[boid_key]
        proto = {
            boid_key: action_pb2.Actions(
                vehicles={
                    vehicle_id: action_to_proto(
                        action_space_type[boid_key], vehicle_action
                    )
                    for vehicle_id, vehicle_action in actions.items()
                }
            )
        }

    # if actions is empty, i.e., actions=={}, or
    # if actions is non boid agent, i.e., actions={<vehicle_id>: <ActionSpace>}
    else:
        proto = {
            "unused": action_pb2.Actions(
                vehicles={
                    vehicle_id: action_to_proto(
                        action_space_type[vehicle_id], vehicle_action
                    )
                    for vehicle_id, vehicle_action in actions.items()
                }
            )
        }

    return proto


def action_to_proto(action_space_type, action) -> action_pb2.Action:

    proto = action_pb2.Action()

    if action_space_type == ActionSpaceType.Continuous:
        proto.continous.action.extend(action)

    elif action_space_type == ActionSpaceType.Lane:
        proto.lane.action = action

    elif action_space_type == ActionSpaceType.ActuatorDynamic:
        proto.actuator_dynamic.action.extend(action)

    elif action_space_type == ActionSpaceType.LaneWithContinuousSpeed:
        proto.lane_with_continuous_speed.action.extend(action)

    elif action_space_type == ActionSpaceType.TargetPose:
        proto.target_pose.action.extend(action)

    elif action_space_type == ActionSpaceType.Trajectory:
        proto.trajectory.action_1.extend(action[0])
        proto.trajectory.action_2.extend(action[1])
        proto.trajectory.action_3.extend(action[2])
        proto.trajectory.action_4.extend(action[3])

    elif action_space_type == ActionSpaceType.MultiTargetPose:
        raise NotImplementedError(
            f"Conversion of MultiTargetPose action space to proto not implemented yet."
        )

    elif action_space_type == ActionSpaceType.MPC:
        proto.mpc.action_1.extend(action[0])
        proto.mpc.action_2.extend(action[1])
        proto.mpc.action_3.extend(action[2])
        proto.mpc.action_4.extend(action[3])

    else:
        raise ValueError(
            f"ActionSpaceType {action_space_type} not found in conversion of action to proto."
        )

    return proto


def proto_to_actions(proto: action_pb2.ActionsBoid):
    boids = proto.boids
    keys = list(boids.keys())
    assert len(keys) == 1, "Incorrect action proto structure."
    boid_key = keys[0]
    vehicles = boids[boid_key].vehicles

    if "boid" in boid_key:
        action = {
            boid_key: {
                vehicle_id: proto_to_action(vehicle_proto)
                for vehicle_id, vehicle_proto in vehicles.items()
            }
        }
    elif "unused" in boid_key:
        action = {
            vehicle_id: proto_to_action(vehicle_proto)
            for vehicle_id, vehicle_proto in vehicles.items()
        }
    else:
        raise ValueError(f"Incorrect action proto structure: {proto}.")

    return action


def proto_to_action(proto: action_pb2.Action):

    if proto.HasField("continuous"):
        return list(proto.continous.action)

    elif proto.HasField("lane"):
        return str(proto.lane.action)

    elif proto.HasField("actuator_dynamic"):
        return list(proto.actuator_dynamic.action)

    elif proto.HasField("lane_with_continuous_speed"):
        return list(proto.lane_with_continuous_speed.action)

    elif proto.HasField("target_pose"):
        return list(proto.target_pose.action)

    elif proto.HasField("trajectory"):
        action = []
        action.append(list(proto.trajectory.action_1))
        action.append(list(proto.trajectory.action_2))
        action.append(list(proto.trajectory.action_3))
        action.append(list(proto.trajectory.action_4))
        return action

    elif proto.HasField("multi_target_pose"):
        raise NotImplementedError(
            f"Conversion of proto to MultiTargetPose action space not implemented yet."
        )

    elif proto.HasField("mpc"):
        action = []
        action.append(list(proto.mpc.action_1))
        action.append(list(proto.mpc.action_2))
        action.append(list(proto.mpc.action_3))
        action.append(list(proto.mpc.action_4))
        return action

    else:
        raise ValueError(
            f"ActionSpaceType {action_space_type} not found in conversion of proto to action."
        )
