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
from smarts.zoo import worker_pb2


def actions_to_proto(action_space_type, action) -> worker_pb2.Actions:
    # if action is non_boid_agent
    if not isinstance(action, dict):
        vehicle_action = action_to_proto(action_space_type, action)
        proto = {"NON_BOID": vehicle_action}

    # if action is empty, i.e., action=={}, or
    # if action is boid_agent, i.e., action={<vehicle_id>: <ActionSpace>}
    else:
        proto = {
            vehicle_id: action_to_proto(action_space_type, vehicle_action)
            for vehicle_id, vehicle_action in action.items()
        }

    return proto


def action_to_proto(action_space_type, action) -> worker_pb2.Action:

    proto = worker_pb2.Action()

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


def proto_to_actions(proto: worker_pb2.Actions):
    vehicles = proto.vehicles

    if "NON_BOID" in vehicles.keys():
        action = proto_to_action(vehicles["NON_BOID"])
    else:
        action = {
            vehicle_id: proto_to_action(vehicle_proto)
            for vehicle_id, vehicle_proto in vehicles.items()
        }

    return action


def proto_to_action(proto: worker_pb2.Action):

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
