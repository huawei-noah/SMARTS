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

import numpy as np

from smarts.core import events, sensors, scenario, waypoints
from smarts.core.utils import sequence
from smarts.proto import observation_pb2
from typing import Dict, NamedTuple


def fix_observation_size(obs_config: Dict, obs: Dict) -> Dict:
    fixed_obs = {
        agent_id: fix_agent_observation_size(obs_config, agent_obs)
        for agent_id, agent_obs in obs.items()
    }
    return fixed_obs


def fix_agent_observation_size(
    obs_config: Dict, obs: NamedTuple
) -> sensors.Observation:

    # Truncate/pad observation.events
    collisions = sequence.truncate_pad_li(
        obs.events.collisions,
        obs_config["observation"]["events"]["collisions"],
        events.Collision(),
    )
    obs_events = obs.events._replace(collisions=collisions)

    # Truncate/pad observation.ego_vehicle_state
    yaw_rate = (sequence.truncate_pad_arr(obs.ego_vehicle_state.yaw_rate, 3, 0),)
    route_vias = sequence.truncate_pad_li(
        obs.ego_vehicle_state.mission.route_vias,
        obs_config["observation"]["ego_vehicle_state"]["mission"]["route_vias"],
        None,
    )
    via = (
        sequence.truncate_pad_li(
            obs.ego_vehicle_state.mission.via,
            obs_config["observation"]["ego_vehicle_state"]["mission"]["via"],
            scenario.Via(),
        ),
    )
    mission = obs.ego_vehicle_state.mission._replace(
        entry_tactic=None,  # EntryTactic removed from observation output
        task=None,  # Task removed from observation output
        via=via,
        route_vias=route_vias,
    )
    linear_velocity = (
        sequence.truncate_pad_arr(obs.ego_vehicle_state.linear_velocity, 3, 0),
    )
    angular_velocity = (
        sequence.truncate_pad_arr(obs.ego_vehicle_state.angular_velocity, 3, 0),
    )
    linear_acceleration = (
        sequence.truncate_pad_arr(obs.ego_vehicle_state.linear_acceleration, 3, 0),
    )
    angular_acceleration = (
        sequence.truncate_pad_arr(obs.ego_vehicle_state.angular_acceleration, 3, 0),
    )
    linear_jerk = (sequence.truncate_pad_arr(obs.ego_vehicle_state.linear_jerk, 3, 0),)
    angular_jerk = (
        sequence.truncate_pad_arr(obs.ego_vehicle_state.angular_jerk, 3, 0),
    )
    ego_vehicle_state = obs.ego_vehicle_state._replace(
        mission=mission,
        linear_velocity=linear_velocity,
        angular_velocity=angular_velocity,
        linear_acceleration=linear_acceleration,
        angular_acceleration=angular_acceleration,
        linear_jerk=linear_jerk,
        angular_jerk=angular_jerk,
    )

    # Truncate/pad observation.neighborhood_vehicle_states
    neighborhood_vehicle_states = obs.neighborhood_vehicle_states or []
    neighborhood_vehicle_states = (
        sequence.truncate_pad_li(
            neighborhood_vehicle_states,
            obs_config["observation"]["neighborhood_vehicle_states"],
            sensors.VehicleObservation(),
        ),
    )

    # Truncate/pad observation.lidar_point_cloud
    lidar_points = sequence.truncate_pad_li(
        obs.lidar_point_cloud[0],
        obs_config["observation"]["lidar_point_cloud"][0],
        np.array([0, 0, 0], dtype=np.float32),
    )
    lidar_hits = sequence.truncate_pad_li(
        obs.lidar_point_cloud[1],
        obs_config["observation"]["lidar_point_cloud"][1],
        np.array([0, 0, 0], dtype=np.float32),
    )
    lidar_ray = sequence.truncate_pad_li(
        obs.lidar_point_cloud[2],
        obs_config["observation"]["lidar_point_cloud"][2],
        (np.array([0, 0, 0], dtype=np.float32), np.array([0, 0, 0], dtype=np.float32)),
    )
    lidar_point_cloud = (lidar_points, lidar_hits, lidar_ray)

    # Truncate/pad observation.road_waypoints
    if obs.road_waypoints:
        lanes = {
            k: sequence.truncate_pad_li(
                v,
                obs_config["observation"]["road_waypoints"]["lanes"],
                waypoints.Waypoint(),
            )
            for k, v in obs.road_waypoints.lanes.items()
        }
        route_waypoints = sequence.truncate_pad_li(
            obs.road_waypoints.route_waypoints,
            obs_config["observation"]["road_waypoints"]["route_waypoints"],
            waypoints.Waypoint(),
        )
        road_waypoints = sensors.RoadWaypoints(
            lanes=lanes, route_waypoints=route_waypoints
        )
    else:
        road_waypoints = obs.road_waypoints

    # Truncate/pad observation.via_data
    near_via_points = sequence.truncate_pad_li(
        obs.via_data.near_via_points,
        obs_config["observation"]["via_data"]["near_via_points"],
        sensors.ViaPoint(),
    )
    hit_via_points = sequence.truncate_pad_li(
        obs.via_data.hit_via_points,
        obs_config["observation"]["via_data"]["hit_via_points"],
        sensors.ViaPoint(),
    )
    via_data = sensors.Vias(
        near_via_points=near_via_points, hit_via_points=hit_via_points
    )

    # Fixed-size observation
    fixed_obs = sensors.Observation(
        events=obs_events,
        ego_vehicle_state=ego_vehicle_state,
        neighborhood_vehicle_states=neighborhood_vehicle_states,
        waypoint_paths=sequence.truncate_pad_li_2d(
            obs.waypoint_paths,
            obs_config["observation"]["waypoint_paths"],
            ([], waypoints.Waypoint()),
        ),
        distance_travelled=obs.distance_travelled,
        lidar_point_cloud=lidar_point_cloud,
        drivable_area_grid_map=obs.drivable_area_grid_map,
        occupancy_grid_map=obs.occupancy_grid_map,
        top_down_rgb=obs.top_down_rgb,
        road_waypoints=road_waypoints,
        via_data=via_data,
    )

    return fixed_obs


def observations_to_proto(obs) -> observation_pb2.ObservationsBoid:
    keys = list(obs.keys())

    # if obs is boid agent, i.e., obs={<*-boid-*>: {<vehicle_id>: Sensors.Observation()} }
    if "boid" in keys[0]:
        assert len(keys) == 1, "Incorrect boid dictionary structure in observation."
        boid_key = keys[0]
        obs = obs[boid_key]
        proto = {
            boid_key: observation_pb2.Observations(
                vehicles={
                    vehicle_id: observation_to_proto(vehicle_obs)
                    for vehicle_id, vehicle_obs in obs.items()
                }
            )
        }

    # if obs is empty, i.e., obs=={}, or
    # if obs is non boid agent, i.e., obs={<vehicle_id>: Sensors.Observation()}
    else:
        proto = {
            "unused": observation_pb2.Observations(
                vehicles={
                    vehicle_id: observation_to_proto(vehicle_obs)
                    for vehicle_id, vehicle_obs in obs.items()
                }
            )
        }

    return proto


def observation_to_proto(obs):
    # obs.waypoint_paths
    waypoint_paths = [
        observation_pb2.ListWaypoint(
            waypoints=[waypoints.waypoint_to_proto(elem) for elem in list_elem]
        )
        for list_elem in obs.waypoint_paths
    ]

    # obs.lidar_point_cloud
    lidar_point_cloud = observation_pb2.Lidar(
        points=observation_pb2.Matrix(
            data=np.ravel(obs.lidar_point_cloud[0]),
            rows=len(obs.lidar_point_cloud[0]),
            cols=3,
        ),
        hits=observation_pb2.Matrix(
            data=np.ravel(obs.lidar_point_cloud[1]),
            rows=len(obs.lidar_point_cloud[1]),
            cols=3,
        ),
        ray=[
            observation_pb2.Matrix(
                data=np.ravel(elem),
                rows=2,
                cols=3,
            )
            for elem in obs.lidar_point_cloud[2]
        ],
    )

    return observation_pb2.Observation(
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


def proto_to_observations(proto: observation_pb2.ObservationsBoid):
    boids = proto.boids
    keys = list(boids.keys())
    assert len(keys) == 1, "Incorrect observation proto structure."
    boid_key = keys[0]
    vehicles = boids[boid_key].vehicles

    if "boid" in boid_key:
        obs = {
            boid_key: {
                vehicle_id: proto_to_observation(vehicle_proto)
                for vehicle_id, vehicle_proto in vehicles.items()
            }
        }
    elif "unused" in boid_key:
        obs = {
            vehicle_id: proto_to_observation(vehicle_proto)
            for vehicle_id, vehicle_proto in vehicles.items()
        }
    else:
        raise ValueError(f"Incorrect observation proto structure: {proto}.")

    return obs


def proto_to_observation(proto: observation_pb2.Observation) -> sensors.Observation:
    # proto.waypoint_paths
    waypoint_paths = [
        [waypoints.proto_to_waypoint(elem) for elem in list_elem.waypoints]
        for list_elem in proto.waypoint_paths
    ]

    # proto.lidar_point_cloud
    lidar_point_cloud = (
        list(sensors.proto_matrix_to_obs(proto.lidar_point_cloud.points)),
        list(sensors.proto_matrix_to_obs(proto.lidar_point_cloud.hits)),
        [
            tuple(sensors.proto_matrix_to_obs(elem))
            for elem in proto.lidar_point_cloud.ray
        ],
    )

    obs = sensors.Observation(
        events=events.proto_to_events(proto.events),
        ego_vehicle_state=sensors.proto_to_ego_vehicle_observation(
            proto.ego_vehicle_state
        ),
        neighborhood_vehicle_states=[
            sensors.proto_to_vehicle_observation(elem)
            for elem in proto.neighborhood_vehicle_states
        ],
        waypoint_paths=waypoint_paths,
        distance_travelled=proto.distance_travelled,
        lidar_point_cloud=lidar_point_cloud,
        drivable_area_grid_map=sensors.proto_to_grid_map(
            proto.drivable_area_grid_map, sensors.DrivableAreaGridMap
        ),
        occupancy_grid_map=sensors.proto_to_grid_map(
            proto.occupancy_grid_map, sensors.OccupancyGridMap
        ),
        top_down_rgb=sensors.proto_to_grid_map(proto.top_down_rgb, sensors.TopDownRGB),
        road_waypoints=sensors.proto_to_road_waypoints(proto.road_waypoints),
        via_data=sensors.proto_to_vias(proto.via_data),
    )

    return obs
