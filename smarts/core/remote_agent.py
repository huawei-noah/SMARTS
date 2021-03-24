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
import numpy as np
import grpc

from smarts.core import events, scenario, sensors, waypoints
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
                payload=cloudpickle.dumps(obs), observe=obs_to_proto(obs),
            )
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


def obs_to_proto(obs):

    proto = worker_pb2.Observe(
        vehicles={
            agent_id: agent_obs_to_proto(agent_obs)
            for agent_id, agent_obs in obs.items()
        }
    )

    # if obs == {}:
    #     print("printing first proto below -------------------------------")
    #     print("proto ================ \n ", proto)
    #     print("^^^^^^^^^^^ -------------------------------\n\n")
    # else:
    #     k = 0
    #     for agent_id, agent_obs in obs.items():
    #         print(agent_id)
    #         k = agent_id
    #         break

    #     print("printing second proto below -------------------------------")
    #     print("proto ================ \n ", proto)
    #     print(
    #         "proto.vehicles[].drivable_area_grid_map ======================== \n ",
    #         proto.vehicles[k].drivable_area_grid_map,
    #     )
    #     print("^^^^^^^^^^^ -------------------------------\n\n")

    return proto

def agent_obs_to_proto(obs):
    # obs.waypoint_paths
    waypoint_paths = [
        worker_pb2.ListWaypoint(
            waypoints=[waypoints.waypoint_to_proto(elem) for elem in list_elem]
        )
        for list_elem in obs.waypoint_paths
    ]

    # obs.lidar_point_cloud
    lidar_point_cloud = worker_pb2.Lidar(
        points=worker_pb2.Matrix(
            data=np.ravel(obs.lidar_point_cloud[0]),
            rows=len(obs.lidar_point_cloud[0]),
            cols=3,
        ),
        hits=worker_pb2.Matrix(
            data=np.ravel(obs.lidar_point_cloud[1]),
            rows=len(obs.lidar_point_cloud[1]),
            cols=3,
        ),
        ray=[
            worker_pb2.Matrix(
                data=np.ravel(elem),
                rows=2,
                cols=3,
            )
            for elem in obs.lidar_point_cloud[2]
        ],
    )

    # vehicle_state
    vehicle_state = worker_pb2.VehicleState(
        events=events.events_to_proto(obs.events),
        ego_vehicle_state=sensors.ego_vehicle_observation_to_proto(
            obs.ego_vehicle_state
        ),
        neighborhood_vehicle_states=[
            sensors.vehicle_observation_to_proto(elem)
            for elem in obs.neighborhood_vehicle_states
        ],
        waypoint_paths=waypoint_paths,
        distance_travelled=obs.distance_travelled,
        lidar_point_cloud=lidar_point_cloud,
        drivable_area_grid_map=sensors.grid_map_to_proto(obs.drivable_area_grid_map),
        occupancy_grid_map=sensors.grid_map_to_proto(obs.occupancy_grid_map),
        top_down_rgb=sensors.grid_map_to_proto(obs.top_down_rgb),
        road_waypoints=sensors.road_waypoints_to_proto(obs.road_waypoints),
        via_data=sensors.vias_to_proto(obs.via_data),
    )

    # print("agent_obs_tuple_to_proto VVVVVVVVV\n", obs)
    # print("-----------------------------------------")

    return vehicle_state

    # import numpy as np
    # gf = [np.array([1,0,0]),np.array([2,0,0])]
    # print(gf)

    # fe = np.ravel(gf)
    # print(fe)
    # dw = list(fe.reshape((2,3)))
    # print(dw)
    # exit()

def proto_to_obs(proto):

    print(proto)
    print("first printing ********************")
    
    obs = {}
    # proto = proto(
    #     vehicles={
    #         agent_id: agent_obs_tuple_to_proto(agent_obs)
    #         for agent_id, agent_obs in obs.items()
    #     }
    # )

    # if obs == {}:
    #     print("printing first proto below -------------------------------")
    #     print("proto ================ \n ", proto)
    #     print("^^^^^^^^^^^ -------------------------------\n\n")
    # else:
    #     k = 0
    #     for agent_id, agent_obs in obs.items():
    #         print(agent_id)
    #         k = agent_id
    #         break

    #     print("printing second proto below -------------------------------")
    #     print("proto ================ \n ", proto)
    #     print(
    #         "proto.vehicles[].drivable_area_grid_map ======================== \n ",
    #         proto.vehicles[k].drivable_area_grid_map,
    #     )
    #     print("^^^^^^^^^^^ -------------------------------\n\n")

    return obs