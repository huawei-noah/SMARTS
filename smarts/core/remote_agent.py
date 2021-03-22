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

import logging
import time
from concurrent import futures

import cloudpickle
import grpc

from smarts.core.agent import AgentSpec
from smarts.zoo import manager_pb2, manager_pb2_grpc, worker_pb2, worker_pb2_grpc


class RemoteAgentException(Exception):
    pass


class RemoteAgent:
    def __init__(self, manager_address, worker_address):
        self._log = logging.getLogger(self.__class__.__name__)

        # Track the last action future.
        self._act_future = None

        self._manager_channel = grpc.insecure_channel(
            f"{manager_address[0]}:{manager_address[1]}"
        )
        self._worker_address = worker_address
        self._worker_channel = grpc.insecure_channel(
            f"{worker_address[0]}:{worker_address[1]}"
        )
        try:
            # Wait until the grpc server is ready or timeout after 30 seconds.
            grpc.channel_ready_future(self._manager_channel).result(timeout=30)
            grpc.channel_ready_future(self._worker_channel).result(timeout=30)
        except grpc.FutureTimeoutError as e:
            raise RemoteAgentException(
                "Timeout while connecting to remote worker process."
            ) from e
        self._manager_stub = manager_pb2_grpc.ManagerStub(self._manager_channel)
        self._worker_stub = worker_pb2_grpc.WorkerStub(self._worker_channel)

    def act(self, obs):
        # Run task asynchronously and return a Future.
        self._act_future = self._worker_stub.act.future(
            worker_pb2.Observation(
                payload=cloudpickle.dumps(obs),
                observe=obs_tuple_to_proto(obs))
        )

        return self._act_future

    def start(self, agent_spec: AgentSpec):
        # Send the AgentSpec to the agent runner.
        # Cloudpickle used only for the agent_spec to allow for serialization of lambdas.
        self._worker_stub.build(
            worker_pb2.Specification(payload=cloudpickle.dumps(agent_spec))
        )

    def terminate(self):
        # If the last action future returned is incomplete, cancel it first.
        if (self._act_future is not None) and (not self._act_future.done()):
            self._act_future.cancel()

        # Close worker channel
        self._worker_channel.close()

        # Stop the remote worker process
        response = self._manager_stub.stop_worker(
            manager_pb2.Port(num=self._worker_address[1])
        )

        # Close manager channel
        self._manager_channel.close()


def obs_tuple_to_proto(obs):

    proto = worker_pb2.Observe(
        vehicles={
                agent_id: agent_obs_tuple_to_proto(agent_obs)
                for agent_id, agent_obs in obs.items()
            }
    )

    print("printing below -------------------------------")
    agent_id = 0
    for agent, agent_obs in obs.items():
        agent_id = agent
        break

    print("proto === ", proto)
    print("^^^^^^^^^^^ -------------------------------\n")

    return proto

def agent_obs_tuple_to_proto(obs):
    # events.collisions
    collisions = [ 
        worker_pb2.Collisions(
                collidee_id=collision.collidee_id
            )
        for collision in obs.events.collisions
    ]

    # events
    events = worker_pb2.Events(
        collisions=collisions,
        off_route=obs.events.off_route,
        reached_goal=obs.events.reached_goal,
        reached_max_episode_steps=obs.events.reached_max_episode_steps,
        off_road=obs.events.off_road,
        wrong_way=obs.events.wrong_way,
        not_moving=obs.events.not_moving,
    )

    # ego_vehicle_state.mission.via
    via = [ 
        worker_pb2.Via(
            lane_id=elem.lane_id,
            edge_id=elem.edge_id,
            lane_index=elem.lane_index,
            position=elem.position,
            hit_distance=elem.hit_distance,
            required_speed=elem.required_speed,
        )
        for elem in obs.ego_vehicle_state.mission.via
    ]

    # ego_vehicle_state.mission
    mission = worker_pb2.Mission(
        start=worker_pb2.Start(
            position=obs.ego_vehicle_state.mission.start.position,
            heading=obs.ego_vehicle_state.mission.start.heading,
        ),
        goal=worker_pb2.Goal(
            position=obs.ego_vehicle_state.mission.goal.position,
            radius=obs.ego_vehicle_state.mission.goal.radius,
        ),
        route_vias=obs.ego_vehicle_state.mission.route_vias,
        start_time=obs.ego_vehicle_state.mission.start_time,
        via=via,
        route_length=obs.ego_vehicle_state.mission.route_length,
        num_laps=obs.ego_vehicle_state.mission.num_laps,
    )

    # ego_vehicle_state 
    ego_vehicle_state = worker_pb2.EgoVehicleObservation(
        id=obs.ego_vehicle_state.id,
        position=obs.ego_vehicle_state.position,
        bounding_box=worker_pb2.BoundingBox(
            length=obs.ego_vehicle_state.bounding_box.length,
            width=obs.ego_vehicle_state.bounding_box.width,
            height=obs.ego_vehicle_state.bounding_box.height,
        ),
        heading=obs.ego_vehicle_state.heading,
        speed=obs.ego_vehicle_state.speed,
        steering=obs.ego_vehicle_state.steering,
        yaw_rate=obs.ego_vehicle_state.yaw_rate,
        edge_id=obs.ego_vehicle_state.edge_id,
        lane_id=obs.ego_vehicle_state.lane_id,
        lane_index=obs.ego_vehicle_state.lane_index,
        mission=mission,
        linear_velocity=obs.ego_vehicle_state.linear_velocity,
        angular_velocity=obs.ego_vehicle_state.angular_velocity,
        linear_acceleration=obs.ego_vehicle_state.linear_acceleration,
        angular_acceleration=obs.ego_vehicle_state.angular_acceleration,
        linear_jerk=obs.ego_vehicle_state.linear_jerk,
        angular_jerk=obs.ego_vehicle_state.angular_jerk,
    )  

    # neighborhood_vehicle_states
    neighborhood_vehicle_states = [ 
        worker_pb2.VehicleObservation(
            id=elem.id,
            position=elem.position,
            bounding_box=worker_pb2.BoundingBox(
                length=elem.bounding_box.length,
                width=elem.bounding_box.width,
                height=elem.bounding_box.height,
            ),
            heading=elem.heading,
            speed=elem.speed,
            edge_id=elem.edge_id,
            lane_id=elem.lane_id,
            lane_index=elem.lane_index,
        )
        for elem in obs.neighborhood_vehicle_states
    ]

    # vehicle_state
    vehicle_state = worker_pb2.VehicleState(
        events=events,
        ego_vehicle_state=ego_vehicle_state, 
        neighborhood_vehicle_states=neighborhood_vehicle_states,
        # repeated ListWaypoint Waypoint_paths=4,
        distance_travelled=obs.distance_travelled,
    )

    # print("vehicle_state.ego_vehicle_state ====>>> ", vehicle_state.ego_vehicle_state)
    # print("\n")
    # print("vehicle_state.ego_vehicle_state.position ====>>> ", vehicle_state.ego_vehicle_state.position)
    # print("\n")
    # print("type of vehicle_state.ego_vehicle_state.position ====>>> ", type(vehicle_state.ego_vehicle_state.position))
    # if vehicle_state.ego_vehicle_state.position:
    #     print("positive")
    # else:
    #     print("negative")

    # print("\n")
    # print("obs.ego_vehicle_state ====>>> ", obs.ego_vehicle_state)

    return vehicle_state


    # import numpy as np
    # gf = [np.array([1,0,0]),np.array([2,0,0])]
    # print(gf)

    # fe = np.ravel(gf)
    # print(fe)
    # dw = list(fe.reshape((2,3)))
    # print(dw)
    # exit()