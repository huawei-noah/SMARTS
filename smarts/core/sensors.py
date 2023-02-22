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
import dataclasses
import logging
import re
import multiprocessing as mp
import sys
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import psutil

import smarts.core.serialization.default as serializer
from smarts.core import config
from smarts.core.agent_interface import ActorsAliveDoneCriteria, AgentsAliveDoneCriteria
from smarts.core.coordinates import Heading, Point
from smarts.core.events import Events
from smarts.core.observations import (
    DrivableAreaGridMap,
    EgoVehicleObservation,
    GridMapMetadata,
    Observation,
    OccupancyGridMap,
    RoadWaypoints,
    SignalObservation,
    TopDownRGB,
    VehicleObservation,
    ViaPoint,
    Vias,
)
from smarts.core.plan import Plan, PlanFrame
from smarts.core.road_map import RoadMap
from smarts.core.sensor import (
    AccelerometerSensor,
    CameraSensor,
    DrivableAreaGridMapSensor,
    DrivenPathSensor,
    LanePositionSensor,
    LidarSensor,
    NeighborhoodVehiclesSensor,
    OGMSensor,
    RGBSensor,
    RoadWaypointsSensor,
    Sensor,
    SignalsSensor,
    TripMeterSensor,
    ViaSensor,
    WaypointsSensor,
)
from smarts.core.simulation_frame import SimulationFrame
from smarts.core.simulation_local_constants import SimulationLocalConstants
from smarts.core.utils.file import replace
from smarts.core.utils.logging import timeit
from smarts.core.utils.math import squared_dist
from smarts.core.vehicle_state import VehicleState

logger = logging.getLogger(__name__)

LANE_ID_CONSTANT = "off_lane"
ROAD_ID_CONSTANT = "off_road"
LANE_INDEX_CONSTANT = -1


def _make_vehicle_observation(road_map, neighborhood_vehicle):
    nv_lane = road_map.nearest_lane(neighborhood_vehicle.pose.point, radius=3)
    if nv_lane:
        nv_road_id = nv_lane.road.road_id
        nv_lane_id = nv_lane.lane_id
        nv_lane_index = nv_lane.index
    else:
        nv_road_id = ROAD_ID_CONSTANT
        nv_lane_id = LANE_ID_CONSTANT
        nv_lane_index = LANE_INDEX_CONSTANT

    return VehicleObservation(
        id=neighborhood_vehicle.actor_id,
        position=neighborhood_vehicle.pose.position,
        bounding_box=neighborhood_vehicle.dimensions,
        heading=neighborhood_vehicle.pose.heading,
        speed=neighborhood_vehicle.speed,
        road_id=nv_road_id,
        lane_id=nv_lane_id,
        lane_index=nv_lane_index,
        lane_position=None,
    )


class SensorState:
    """Sensor state information"""

    def __init__(self, max_episode_steps: int, plan_frame: PlanFrame):
        self._max_episode_steps = max_episode_steps
        self._plan_frame = plan_frame
        self._step = 0
        self._seen_interest_actors = False

    def step(self):
        """Update internal state."""
        self._step += 1

    @property
    def seen_interest_actors(self) -> bool:
        """If a relevant actor has been spotted before."""
        return self._seen_interest_actors

    @seen_interest_actors.setter
    def seen_interest_actors(self, value: bool):
        self._seen_interest_actors = value

    @property
    def reached_max_episode_steps(self) -> bool:
        """Inbuilt sensor information that describes if episode step limit has been reached."""
        if self._max_episode_steps is None:
            return False

        return self._step >= self._max_episode_steps

    def get_plan(self, road_map: RoadMap):
        """Get the current plan for the actor."""
        return Plan.from_frame(self._plan_frame, road_map)

    @property
    def steps_completed(self) -> int:
        """Get the number of steps where this sensor has been updated."""
        return self._step


class SensorResolver:
    """An interface describing sensor observation and update systems."""

    # TODO: Remove renderer and bullet client from the arguments
    def observe(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        renderer,
        bullet_client,
    ):
        """Generate observations

        Args:
            sim_frame (SimulationFrame): The simulation frame.
            sim_local_constants (SimulationLocalConstants): Constraints defined by the local simulator.
            agent_ids (Set[str]): The agents to run.
            renderer (Renderer): The renderer to use.
            bullet_client (Any): The bullet client. This parameter is likely to be removed.
        """
        raise NotImplementedError()

    def step(self, sim_frame, sensor_states):
        """Step the sensor state."""
        raise NotImplementedError()


class LocalSensorResolver(SensorResolver):
    """This implementation of the sensor resolver completes observations serially."""

    def observe(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        renderer,
        bullet_client,
    ):
        with timeit("serial run", logger.info):
            (
                observations,
                dones,
                updated_sensors,
            ) = Sensors.observe_serializable_sensor_batch(
                sim_frame,
                sim_local_constants,
                agent_ids,
            )

        # While observation processes are operating do rendering
        with timeit("rendering", logger.info):
            rendering = {}
            for agent_id in agent_ids:
                for vehicle_id in sim_frame.vehicles_for_agents[agent_id]:
                    rendering[agent_id] = Sensors.process_serialization_unsafe_sensors(
                        sim_frame,
                        sim_local_constants,
                        agent_id,
                        sim_frame.sensor_states[vehicle_id],
                        vehicle_id,
                        renderer,
                        bullet_client,
                    )

        with timeit(f"merging observations", logger.info):
            # Merge sensor information
            for agent_id, r_obs in rendering.items():
                observations[agent_id] = replace(observations[agent_id], **r_obs)

        return observations, dones, updated_sensors

    def step(self, sim_frame, sensor_states):
        """Step the sensor state."""
        for sensor_state in sensor_states:
            sensor_state.step()


class ParallelSensorResolver(SensorResolver):
    """This implementation of the sensor resolver completes observations in parallel."""

    def __init__(self, process_count_override: Optional[int] = None) -> None:
        super().__init__()
        self._logger = logging.getLogger("Sensors")
        self._sim_local_constants: SimulationLocalConstants = None
        self._workers: List[SensorsWorker] = []
        self._process_count_override = process_count_override

    def observe(
        self,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids: Set[str],
        renderer,
        bullet_client,
    ):
        """Runs observations in parallel where possible.
        Args:
            sim_frame (SimulationFrame):
                The current state from the simulation.
            sim_local_constants (SimulationLocalConstants):
                The values that should stay the same for a simulation over a reset.
            agent_ids ({str, ...}):
                The agent ids to process.
            renderer (Optional[Renderer]):
                The renderer (if any) that should be used.
            bullet_client (bc.BulletClient):
                The physics client.
        """
        observations, dones, updated_sensors = {}, {}, {}

        num_spare_cpus = max(0, psutil.cpu_count(logical=False) - 1)
        used_processes = (
            min(
                config()("core", "observation_workers", default=128, cast=int),
                num_spare_cpus,
            )
            if self._process_count_override == None
            else max(1, self._process_count_override)
        )

        workers: List[SensorsWorker] = self.get_workers(
            used_processes, sim_local_constants=sim_local_constants
        )
        used_workers: List[SensorsWorker] = []
        with timeit(
            f"parallizable observations with {len(agent_ids)} and {len(workers)}",
            logger.info,
        ):
            agent_ids_for_grouping = list(agent_ids)
            agent_groups = [
                agent_ids_for_grouping[i::used_processes] for i in range(used_processes)
            ]
            worker_args = WorkerKwargs(sim_frame=sim_frame)
            for i, agent_group in enumerate(agent_groups):
                if not agent_group:
                    break
                with timeit(f"submitting {len(agent_group)} agents", logger.info):
                    workers[i].send(
                        SensorsWorker.Request(
                            SensorsWorkerRequestId.SIMULATION_FRAME,
                            worker_args.merged(WorkerKwargs(agent_ids=agent_group)),
                        )
                    )
                    used_workers.append(workers[i])

            # While observation processes are operating do rendering
            with timeit("rendering", logger.info):
                rendering = {}
                for agent_id in agent_ids:
                    for vehicle_id in sim_frame.vehicles_for_agents[agent_id]:
                        rendering[
                            agent_id
                        ] = Sensors.process_serialization_unsafe_sensors(
                            sim_frame,
                            sim_local_constants,
                            agent_id,
                            sim_frame.sensor_states[vehicle_id],
                            vehicle_id,
                            renderer,
                            bullet_client,
                        )

            # Collect futures
            with timeit("waiting for observations", logger.info):
                if used_workers:
                    while agent_ids != set(observations):
                        assert all(
                            w.running for w in used_workers
                        ), "A process worker crashed."
                        for result in mp.connection.wait(
                            [worker.connection for worker in used_workers], timeout=5
                        ):
                            # pytype: disable=attribute-error
                            obs, ds, u_sens = result.recv()
                            # pytype: enable=attribute-error
                            observations.update(obs)
                            dones.update(ds)
                            updated_sensors.update(u_sens)

            with timeit(f"merging observations", logger.info):
                # Merge sensor information
                for agent_id, r_obs in rendering.items():
                    observations[agent_id] = replace(observations[agent_id], **r_obs)

        return observations, dones, updated_sensors

    def __del__(self):
        try:
            self.stop_all_workers()
        except AttributeError:
            pass

    def stop_all_workers(self):
        """Stop all current workers and clear reference to them."""
        for worker in self._workers:
            worker.stop()
        self._workers = []

    def _validate_configuration(self, local_constants: SimulationLocalConstants):
        """Check that constants have not changed which might indicate that the workers need to be updated."""
        return local_constants == self._sim_local_constants

    def generate_workers(
        self, count, workers_list: List[Any], worker_kwargs: "WorkerKwargs"
    ):
        """Generate the given number of workers requested."""
        while len(workers_list) < count:
            new_worker = SensorsWorker()
            workers_list.append(new_worker)
            new_worker.run()
            new_worker.send(
                request=SensorsWorker.Request(
                    SensorsWorkerRequestId.SIMULATION_LOCAL_CONSTANTS, worker_kwargs
                )
            )

    def get_workers(
        self, count, sim_local_constants: SimulationLocalConstants, **kwargs
    ) -> List["SensorsWorker"]:
        """Get the give number of workers."""
        if not self._validate_configuration(sim_local_constants):
            self.stop_all_workers()
            self._sim_local_constants = sim_local_constants
        if len(self._workers) < count:
            worker_kwargs = WorkerKwargs(
                **kwargs, sim_local_constants=sim_local_constants
            )
            self.generate_workers(count, self._workers, worker_kwargs)
        return self._workers[:count]

    def step(self, sim_frame, sensor_states):
        """Step the sensor state."""
        for sensor_state in sensor_states:
            sensor_state.step()

    @property
    def process_count_override(self) -> Optional[int]:
        """The number of processes this implementation should run.

        Returns:
            int: Number of processes.
        """
        return self._process_count_override

    @process_count_override.setter
    def process_count_override(self, count: Optional[int]):
        self._process_count_override = count


class Sensors:
    """Sensor related utilities"""

    _log = logging.getLogger("Sensors")
    _instance = None
    _sim_local_constants: SimulationLocalConstants = None

    @classmethod
    def observe_serializable_sensor_batch(
        cls,
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_ids_for_group,
    ):
        """Run the serializable sensors in a batch."""
        observations, dones, updated_sensors = {}, {}, {}
        for agent_id in agent_ids_for_group:
            vehicle_ids = sim_frame.vehicles_for_agents.get(agent_id)
            if not vehicle_ids:
                continue
            for vehicle_id in vehicle_ids:
                (
                    observations[agent_id],
                    dones[agent_id],
                    updated_sensors[vehicle_id],
                ) = cls.process_serialization_safe_sensors(
                    sim_frame,
                    sim_local_constants,
                    agent_id,
                    sim_frame.sensor_states[vehicle_id],
                    vehicle_id,
                )
        return observations, dones, updated_sensors

    @staticmethod
    def process_serialization_unsafe_sensors(
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_id,
        sensor_state,
        vehicle_id,
        renderer,
        bullet_client,
    ):
        """Run observations that can only be done on the main thread."""
        vehicle_sensors: Dict[str, Any] = sim_frame.vehicle_sensors[vehicle_id]

        vehicle_state = sim_frame.vehicle_states[vehicle_id]
        lidar = None
        lidar_sensor = vehicle_sensors.get("lidar_sensor")
        if lidar_sensor:
            lidar_sensor.follow_vehicle(vehicle_state)
            lidar = lidar_sensor(bullet_client)

        def get_camera_sensor_result(sensors, sensor_name, renderer):
            return (
                sensors[sensor_name](renderer=renderer)
                if renderer and sensors.get(sensor_name)
                else None
            )

        return dict(
            drivable_area_grid_map=get_camera_sensor_result(
                vehicle_sensors, "drivable_area_grid_map_sensor", renderer
            ),
            occupancy_grid_map=get_camera_sensor_result(
                vehicle_sensors, "ogm_sensor", renderer
            ),
            top_down_rgb=get_camera_sensor_result(
                vehicle_sensors, "rgb_sensor", renderer
            ),
            lidar_point_cloud=lidar,
        )

    @staticmethod
    def process_serialization_safe_sensors(
        sim_frame: SimulationFrame,
        sim_local_constants: SimulationLocalConstants,
        agent_id,
        sensor_state,
        vehicle_id,
    ):
        """Observations that can be done on any thread."""
        vehicle_sensors = sim_frame.vehicle_sensors[vehicle_id]
        vehicle_state = sim_frame.vehicle_states[vehicle_id]
        plan = sensor_state.get_plan(sim_local_constants.road_map)
        neighborhood_vehicle_states = None
        neighborhood_vehicle_states_sensor = vehicle_sensors.get(
            "neighborhood_vehicle_states_sensor"
        )
        if neighborhood_vehicle_states_sensor:
            neighborhood_vehicle_states = []
            for nv in neighborhood_vehicle_states_sensor(
                vehicle_state, sim_frame.vehicle_states.values()
            ):
                veh_obs = _make_vehicle_observation(sim_local_constants.road_map, nv)
                lane_position_sensor = vehicle_sensors.get("lane_position_sensor")
                nv_lane_pos = None
                if veh_obs.lane_id is not LANE_ID_CONSTANT and lane_position_sensor:
                    nv_lane_pos = lane_position_sensor(
                        sim_local_constants.road_map.lane_by_id(veh_obs.lane_id), nv
                    )
                neighborhood_vehicle_states.append(
                    veh_obs._replace(lane_position=nv_lane_pos)
                )

        waypoints_sensor = vehicle_sensors.get("waypoints_sensor")
        if waypoints_sensor:
            waypoint_paths = waypoints_sensor(
                vehicle_state, plan, sim_local_constants.road_map
            )
        else:
            waypoint_paths = sim_local_constants.road_map.waypoint_paths(
                vehicle_state.pose,
                lookahead=1,
                within_radius=vehicle_state.dimensions.length,
            )

        closest_lane = sim_local_constants.road_map.nearest_lane(
            vehicle_state.pose.point
        )
        ego_lane_pos = None
        if closest_lane:
            ego_lane_id = closest_lane.lane_id
            ego_lane_index = closest_lane.index
            ego_road_id = closest_lane.road.road_id
            lane_position_sensor = vehicle_sensors.get("lane_position_sensor")
            if lane_position_sensor:
                ego_lane_pos = lane_position_sensor(closest_lane, vehicle_state)
        else:
            ego_lane_id = LANE_ID_CONSTANT
            ego_lane_index = LANE_INDEX_CONSTANT
            ego_road_id = ROAD_ID_CONSTANT

        acceleration_params = {
            "linear_acceleration": None,
            "angular_acceleration": None,
            "linear_jerk": None,
            "angular_jerk": None,
        }

        accelerometer_sensor = vehicle_sensors.get("accelerometer_sensor")
        if accelerometer_sensor:
            acceleration_values = accelerometer_sensor(
                vehicle_state.linear_velocity,
                vehicle_state.angular_velocity,
                sim_frame.last_dt,
            )
            acceleration_params.update(
                dict(
                    zip(
                        [
                            "linear_acceleration",
                            "angular_acceleration",
                            "linear_jerk",
                            "angular_jerk",
                        ],
                        acceleration_values,
                    )
                )
            )

        ego_vehicle = EgoVehicleObservation(
            id=vehicle_state.actor_id,
            position=vehicle_state.pose.point.as_np_array,
            bounding_box=vehicle_state.dimensions,
            heading=Heading(vehicle_state.pose.heading),
            speed=vehicle_state.speed,
            steering=vehicle_state.steering,
            yaw_rate=vehicle_state.yaw_rate,
            road_id=ego_road_id,
            lane_id=ego_lane_id,
            lane_index=ego_lane_index,
            mission=plan.mission,
            linear_velocity=vehicle_state.linear_velocity,
            angular_velocity=vehicle_state.angular_velocity,
            lane_position=ego_lane_pos,
            **acceleration_params,
        )

        road_waypoints_sensor = vehicle_sensors.get("road_waypoints_sensor")
        road_waypoints = (
            road_waypoints_sensor(vehicle_state, plan, sim_local_constants.road_map)
            if road_waypoints_sensor
            else None
        )

        near_via_points = []
        hit_via_points = []

        via_sensor = vehicle_sensors.get("via_sensor")
        if via_sensor:
            (
                near_via_points,
                hit_via_points,
            ) = via_sensor(vehicle_state, plan, sim_local_constants.road_map)
        via_data = Vias(
            near_via_points=near_via_points,
            hit_via_points=hit_via_points,
        )

        distance_travelled = 0
        trip_meter_sensor: Optional[TripMeterSensor] = vehicle_sensors.get(
            "trip_meter_sensor"
        )
        if trip_meter_sensor:
            if waypoint_paths:
                trip_meter_sensor.update_distance_wps_record(
                    waypoint_paths, vehicle_state, plan, sim_local_constants.road_map
                )
            distance_travelled = trip_meter_sensor(increment=True)

        driven_path_sensor = vehicle_sensors.get("driven_path_sensor")
        if driven_path_sensor:
            driven_path_sensor.track_latest_driven_path(
                sim_frame.elapsed_sim_time, vehicle_state
            )

        if not waypoints_sensor:
            waypoint_paths = None

        done, events = Sensors._is_done_with_events(
            sim_frame,
            sim_local_constants,
            agent_id,
            vehicle_state,
            sensor_state,
            plan,
            vehicle_sensors,
        )

        if done and sensor_state.steps_completed == 1:
            agent_type = "Social agent"
            if agent_id in sim_frame.ego_ids:
                agent_type = "Ego agent"
            logger.warning(
                "%s with Agent ID: %s is done on the first step", agent_type, agent_id
            )

        signals = None
        signals_sensor = vehicle_sensors.get("signals_sensor")
        if signals_sensor:
            provider_state = sim_frame.last_provider_state
            signals = signals_sensor(
                closest_lane,
                ego_lane_pos,
                vehicle_state,
                plan,
                provider_state,
            )

        agent_controls = agent_id == sim_frame.agent_vehicle_controls.get(
            vehicle_state.actor_id
        )
        updated_sensors = {
            sensor_name: sensor
            for sensor_name, sensor in vehicle_sensors.items()
            if sensor.mutable
        }

        # TODO MTA: Return updated sensors or make sensors stateless
        return (
            Observation(
                dt=sim_frame.last_dt,
                step_count=sim_frame.step_count,
                steps_completed=sensor_state.steps_completed,
                elapsed_sim_time=sim_frame.elapsed_sim_time,
                events=events,
                ego_vehicle_state=ego_vehicle,
                under_this_agent_control=agent_controls,
                neighborhood_vehicle_states=neighborhood_vehicle_states,
                waypoint_paths=waypoint_paths,
                distance_travelled=distance_travelled,
                road_waypoints=road_waypoints,
                via_data=via_data,
                signals=signals,
            ),
            done,
            updated_sensors,
        )

    @classmethod
    def observe_vehicle(
        cls,
        sim_frame: SimulationFrame,
        sim_local_constants,
        agent_id,
        sensor_state,
        vehicle,
        renderer,
        bullet_client,
    ) -> Tuple[Observation, bool, Dict[str, "Sensor"]]:
        """Generate observations for the given agent around the given vehicle."""
        args = [sim_frame, sim_local_constants, agent_id, sensor_state, vehicle.id]
        base_obs, dones, updated_sensors = cls.process_serialization_safe_sensors(*args)
        complete_obs = dataclasses.replace(
            base_obs,
            **cls.process_serialization_unsafe_sensors(*args, renderer, bullet_client),
        )
        return (complete_obs, dones, updated_sensors)

    @classmethod
    def _agents_alive_done_check(
        cls, ego_agent_ids, agent_ids, agents_alive: Optional[AgentsAliveDoneCriteria]
    ):
        if agents_alive is None:
            return False

        if (
            agents_alive.minimum_ego_agents_alive
            and len(ego_agent_ids) < agents_alive.minimum_ego_agents_alive
        ):
            return True
        if (
            agents_alive.minimum_total_agents_alive
            and len(agent_ids) < agents_alive.minimum_total_agents_alive
        ):
            return True
        if agents_alive.agent_lists_alive:
            for agents_list_alive in agents_alive.agent_lists_alive:
                assert isinstance(
                    agents_list_alive.agents_list, (List, Set, Tuple)
                ), "Please specify a list of agent ids to watch"
                assert isinstance(
                    agents_list_alive.minimum_agents_alive_in_list, int
                ), "Please specify an int for minimum number of alive agents in the list"
                assert (
                    agents_list_alive.minimum_agents_alive_in_list >= 0
                ), "minimum_agents_alive_in_list should not be negative"
                agents_alive_check = [
                    1 if id in agent_ids else 0 for id in agents_list_alive.agents_list
                ]
                if (
                    agents_alive_check.count(1)
                    < agents_list_alive.minimum_agents_alive_in_list
                ):
                    return True

        return False

    @classmethod
    def _actors_alive_done_check(
        cls,
        vehicle_ids,
        sensor_state,
        actors_alive: Optional[ActorsAliveDoneCriteria],
    ):
        if actors_alive is None:
            return False

        sensor_state: SensorState = sensor_state

        pattern = re.compile(
            "|".join(rf"(?:{aoi})" for aoi in actors_alive.actors_of_interest)
        )
        ## TODO optimization to get vehicles that were added and removed last step
        ## TODO second optimization to check for already known vehicles
        for vehicle_id in vehicle_ids:
            # get vehicles by pattern
            if pattern.match(vehicle_id):
                sensor_state.seen_interest_actors = True
                return False
        if actors_alive.strict or sensor_state.seen_interest_actors:
            # if agent requires the actor to exist immediately
            # OR if previously seen relevant actors but no actors match anymore
            return True

        ## if never seen a relevant actor
        return False

    @classmethod
    def _is_done_with_events(
        cls,
        sim_frame: SimulationFrame,
        sim_local_constants,
        agent_id,
        vehicle_state: VehicleState,
        sensor_state,
        plan,
        vehicle_sensors,
    ):
        vehicle_sensors = sim_frame.vehicle_sensors[vehicle_state.actor_id]
        interface = sim_frame.agent_interfaces[agent_id]
        done_criteria = interface.done_criteria
        event_config = interface.event_configuration

        # TODO:  the following calls nearest_lanes (expensive) 6 times
        reached_goal = cls._agent_reached_goal(
            sensor_state, plan, vehicle_state, vehicle_sensors.get("trip_meter_sensor")
        )
        collided = sim_frame.vehicle_did_collide(vehicle_state.actor_id)
        is_off_road = cls._vehicle_is_off_road(
            sim_local_constants.road_map, vehicle_state
        )
        is_on_shoulder = cls._vehicle_is_on_shoulder(
            sim_local_constants.road_map, vehicle_state
        )
        is_not_moving = cls._vehicle_is_not_moving(
            sim_frame,
            event_config.not_moving_time,
            event_config.not_moving_distance,
            vehicle_sensors.get("driven_path_sensor"),
        )
        reached_max_episode_steps = sensor_state.reached_max_episode_steps
        is_off_route, is_wrong_way = cls._vehicle_is_off_route_and_wrong_way(
            sim_frame, sim_local_constants, vehicle_state, plan
        )
        agents_alive_done = cls._agents_alive_done_check(
            sim_frame.ego_ids, sim_frame.potential_agent_ids, done_criteria.agents_alive
        )
        actors_alive_done = cls._actors_alive_done_check(
            sim_frame.vehicle_ids, sensor_state, done_criteria.actors_alive
        )

        done = not sim_frame.resetting and (
            (is_off_road and done_criteria.off_road)
            or reached_goal
            or reached_max_episode_steps
            or (is_on_shoulder and done_criteria.on_shoulder)
            or (collided and done_criteria.collision)
            or (is_not_moving and done_criteria.not_moving)
            or (is_off_route and done_criteria.off_route)
            or (is_wrong_way and done_criteria.wrong_way)
            or agents_alive_done
            or actors_alive_done
        )

        events = Events(
            collisions=sim_frame.filtered_vehicle_collisions(vehicle_state.actor_id),
            off_road=is_off_road,
            reached_goal=reached_goal,
            reached_max_episode_steps=reached_max_episode_steps,
            off_route=is_off_route,
            on_shoulder=is_on_shoulder,
            wrong_way=is_wrong_way,
            not_moving=is_not_moving,
            agents_alive_done=agents_alive_done,
            actors_alive_done=actors_alive_done,
        )

        return done, events

    @classmethod
    def _agent_reached_goal(
        cls, sensor_state, plan, vehicle_state: VehicleState, trip_meter_sensor
    ):
        if not trip_meter_sensor:
            return False
        distance_travelled = trip_meter_sensor()
        mission = plan.mission
        return mission.is_complete(vehicle_state, distance_travelled)

    @classmethod
    def _vehicle_is_off_road(cls, road_map, vehicle_state: VehicleState):
        return not road_map.road_with_point(vehicle_state.pose.point)

    @classmethod
    def _vehicle_is_on_shoulder(cls, road_map, vehicle_state: VehicleState):
        # XXX: this isn't technically right as this would also return True
        #      for vehicles that are completely off road.
        for corner_coordinate in vehicle_state.bounding_box_points:
            if not road_map.road_with_point(Point(*corner_coordinate)):
                return True
        return False

    @classmethod
    def _vehicle_is_not_moving(
        cls, sim, last_n_seconds_considered, min_distance_moved, driven_path_sensor
    ):
        # Flag if the vehicle has been immobile for the past 'last_n_seconds_considered' seconds
        if sim.elapsed_sim_time < last_n_seconds_considered:
            return False

        distance = sys.maxsize
        if driven_path_sensor is not None:
            distance = driven_path_sensor.distance_travelled(
                sim.elapsed_sim_time, last_n_seconds=last_n_seconds_considered
            )

        # Due to controller instabilities there may be some movement even when a
        # vehicle is "stopped".
        return distance < min_distance_moved

    @classmethod
    def _vehicle_is_off_route_and_wrong_way(
        cls,
        sim_frame: SimulationFrame,
        sim_local_constants,
        vehicle_state: VehicleState,
        plan,
    ):
        """Determines if the agent is on route and on the correct side of the road.

        Args:
            sim: An instance of the simulator.
            agent_id: The id of the agent to check.

        Returns:
            A tuple (is_off_route, is_wrong_way)
            is_off_route:
                Actor's vehicle is not on its route or an oncoming traffic lane.
            is_wrong_way:
                Actor's vehicle is going against the lane travel direction.
        """

        route_roads = plan.route.roads

        vehicle_pos = vehicle_state.pose.point
        vehicle_minimum_radius_bounds = (
            np.linalg.norm(vehicle_state.dimensions.as_lwh[:2]) * 0.5
        )
        # Check that center of vehicle is still close to route
        radius = vehicle_minimum_radius_bounds + 5
        nearest_lane = sim_local_constants.road_map.nearest_lane(
            vehicle_pos, radius=radius
        )

        # No road nearby, so we're not on route!
        if not nearest_lane:
            return (True, False)

        # Check whether vehicle is in wrong-way
        is_wrong_way = cls._check_wrong_way_event(nearest_lane, vehicle_state)

        # Check whether vehicle has no-route or is on-route
        if (
            not route_roads  # Vehicle has no-route. E.g., endless mission with a random route
            or nearest_lane.road in route_roads  # Vehicle is on-route
            or nearest_lane.in_junction
        ):
            return (False, is_wrong_way)

        veh_offset = nearest_lane.offset_along_lane(vehicle_pos)

        # so we're obviously not on the route, but we might have just gone
        # over the center line into an oncoming lane...
        for on_lane in nearest_lane.oncoming_lanes_at_offset(veh_offset):
            if on_lane.road in route_roads:
                return (False, is_wrong_way)

        # Check for case if there was an early merge into another incoming lane. This means the
        # vehicle should still be following the lane direction to be valid as still on route.
        if not is_wrong_way:
            # See if the lane leads into the current route
            for lane in nearest_lane.outgoing_lanes:
                if lane.road in route_roads:
                    return (False, is_wrong_way)
                # If outgoing lane is in a junction see if the junction lane leads into current route.
                if lane.in_junction:
                    for out_lane in lane.outgoing_lanes:
                        if out_lane.road in route_roads:
                            return (False, is_wrong_way)

        # Vehicle is completely off-route
        return (True, is_wrong_way)

    @staticmethod
    def _vehicle_is_wrong_way(vehicle_state: VehicleState, closest_lane):
        target_pose = closest_lane.center_pose_at_point(vehicle_state.pose.point)
        # Check if the vehicle heading is oriented away from the lane heading.
        return (
            np.fabs(vehicle_state.pose.heading.relative_to(target_pose.heading))
            > 0.5 * np.pi
        )

    @classmethod
    def _check_wrong_way_event(cls, lane_to_check, vehicle_state):
        # When the vehicle is in an intersection, turn off the `wrong way` check to avoid
        # false positive `wrong way` events.
        if lane_to_check.in_junction:
            return False
        return cls._vehicle_is_wrong_way(vehicle_state, lane_to_check)


class WorkerKwargs:
    """Used to serialize arguments for a worker upfront."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = self._serialize(kwargs)

    def merged(self, o_worker_kwargs: "WorkerKwargs") -> "WorkerKwargs":
        """Merge two worker arguments and return a new copy."""
        new = type(self)()
        new.kwargs = {**self.kwargs, **o_worker_kwargs.kwargs}
        return new

    @staticmethod
    def _serialize(kwargs: Dict):
        return {
            k: serializer.dumps(a) if a is not None else a for k, a in kwargs.items()
        }

    def deserialize(self):
        """Deserialize all objects in the arguments and return a dictionary copy."""
        return {
            k: serializer.loads(a) if a is not None else a
            for k, a in self.kwargs.items()
        }


class ProcessWorker:
    """A utility class that defines a persistant worker which will continue to operate in the background."""

    class WorkerDone:
        """The done signal for a worker."""

        pass

    @dataclass
    class Request:
        """A request to made to the process worker"""

        id: Any
        data: WorkerKwargs

    def __init__(self, serialize_results=False) -> None:
        parent_connection, child_connection = mp.Pipe()
        self._parent_connection = parent_connection
        self._child_connection = child_connection
        self._serialize_results = serialize_results
        self._proc: Optional[mp.Process] = None

    @classmethod
    def _do_work(cls, state):
        raise NotImplementedError()

    @classmethod
    def _on_request(cls, state: Dict, request: Request) -> bool:
        """
        Args:
            state: The persistant state on the worker
            request: A request made to the worker.

        Returns:
            bool: If the worker method `_do_work` should be called.
        """
        raise NotImplementedError()

    @classmethod
    def _run(
        cls: "ProcessWorker",
        connection: mp.connection.Connection,
        serialize_results,
    ):
        state: Dict[Any, Any] = {}
        while True:
            run_work = False
            work = connection.recv()
            if isinstance(work, cls.WorkerDone):
                break
            if isinstance(work, cls.Request):
                run_work = cls._on_request(state, request=work)
            with timeit("do work", logger.info):
                if not run_work:
                    continue
                result = cls._do_work(state=state.copy())
                with timeit("reserialize", logger.info):
                    if serialize_results:
                        result = serializer.dumps(result)
                with timeit("put back to main thread", logger.info):
                    connection.send(result)

    def run(self):
        """Start the worker seeded with the given data."""
        kwargs = dict(serialize_results=self._serialize_results)
        self._proc = mp.Process(
            target=self._run,
            args=(self._child_connection,),
            kwargs=kwargs,
            daemon=True,
        )
        self._proc.start()
        return self._parent_connection

    def send(self, request: Request):
        """Sends a request to the worker."""
        assert isinstance(request, self.Request)
        self._parent_connection.send(request)

    def result(self, timeout=None):
        """The most recent result from the worker."""
        with timeit("main thread blocked", logger.info):
            conn = mp.connection.wait([self._parent_connection], timeout=timeout).pop()
            # pytype: disable=attribute-error
            result = conn.recv()
            # pytype: enable=attribute-error
        with timeit("deserialize for main thread", logger.info):
            if self._serialize_results:
                result = serializer.loads(result)
        return result

    def stop(self):
        """Sends a stop signal to the worker."""
        self._parent_connection.send(self.WorkerDone())

    @property
    def running(self) -> bool:
        """If this current worker is still running."""
        return self._proc is not None and self._proc.exitcode is None

    @property
    def connection(self):
        """The underlying connection to send data to the worker."""
        return self._parent_connection


class SensorsWorkerRequestId(IntEnum):
    """Options for update requests to a sensor worker."""

    SIMULATION_FRAME = 1
    SIMULATION_LOCAL_CONSTANTS = 2


class SensorsWorker(ProcessWorker):
    """A worker for sensors."""

    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def _do_work(cls, state):
        return cls.local(state=state)

    @classmethod
    def _on_request(cls, state: Dict, request: ProcessWorker.Request) -> bool:
        assert request.data is None or isinstance(request.data, WorkerKwargs)
        if request.id == SensorsWorkerRequestId.SIMULATION_FRAME:
            state.update(request.data.deserialize())
            return True
        if request.id == SensorsWorkerRequestId.SIMULATION_LOCAL_CONSTANTS:
            state.update(request.data.deserialize())

        return False

    @staticmethod
    def local(state: Dict):
        """The work method on the local thread."""
        sim_local_constants = state["sim_local_constants"]
        sim_frame = state["sim_frame"]
        agent_ids = state["agent_ids"]
        return Sensors.observe_serializable_sensor_batch(
            sim_frame, sim_local_constants, agent_ids
        )
