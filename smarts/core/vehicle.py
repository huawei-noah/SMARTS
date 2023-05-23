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
import importlib.resources as pkg_resources
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from smarts.core.agent_interface import AgentInterface
from smarts.core.plan import Mission, Plan

from . import models
from .actor import ActorRole
from .chassis import AckermannChassis, BoxChassis, Chassis
from .colors import SceneColors
from .coordinates import Dimensions, Heading, Pose
from .sensors import (
    AccelerometerSensor,
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
from .utils.custom_exceptions import RendererException
from .utils.math import rotate_cw_around_point
from .vehicle_state import VEHICLE_CONFIGS, VehicleConfig, VehicleState


class Vehicle:
    """Represents a single vehicle."""

    _HAS_DYNAMIC_ATTRIBUTES = True  # dynamic pytype attribute

    def __init__(
        self,
        id: str,
        chassis: Chassis,
        vehicle_config_type: str = "passenger",
        color: Optional[SceneColors] = None,
        action_space=None,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._id = id

        self._chassis: Chassis = chassis
        self._vehicle_config_type = vehicle_config_type
        self._action_space = action_space

        self._meta_create_sensor_functions()
        self._sensors = {}

        # Color override
        self._color: Optional[SceneColors] = color
        if self._color is None:
            config = VEHICLE_CONFIGS[vehicle_config_type]
            self._color = config.color

        self._initialized = True
        self._has_stepped = False

    def _assert_initialized(self):
        assert self._initialized, f"Vehicle({self.id}) is not initialized"

    def __eq__(self, __o: object) -> bool:
        if self is __o:
            return True
        if isinstance(__o, self.__class__) and self.state == __o.state:
            return True
        return False

    def __repr__(self):
        return f"""Vehicle({self.id},
  pose={self.pose},
  speed={self.speed},
  type={self.vehicle_type},
  w={self.width},
  l={self.length},
  h={self.height}
)"""

    @property
    def id(self):
        """The id of this vehicle."""
        return self._id

    @property
    def length(self) -> float:
        """The length of this vehicle."""
        self._assert_initialized()
        return self._chassis.dimensions.length

    @property
    def max_steering_wheel(self) -> Optional[float]:
        """The max steering value the chassis steering wheel can turn to.

        Some chassis types do not support this.
        """
        self._assert_initialized()
        return getattr(self._chassis, "max_steering_wheel", None)

    @property
    def width(self) -> float:
        """The width of this vehicle."""
        self._assert_initialized()
        return self._chassis.dimensions.width

    @property
    def height(self) -> float:
        """The height of this vehicle."""
        self._assert_initialized()
        return self._chassis.dimensions.height

    @property
    def speed(self) -> float:
        """The current speed of this vehicle."""
        self._assert_initialized()
        return self._chassis.speed

    @property
    def sensors(self) -> Dict[str, Sensor]:
        """The sensors attached to this vehicle."""
        self._assert_initialized()
        return self._sensors

    # # TODO: See issue #898 This is a currently a no-op
    # @speed.setter
    # def speed(self, speed):
    #     self._chassis.speed = speed

    @property
    def vehicle_color(self) -> Union[SceneColors, None]:
        """The color of this vehicle (generally used for rendering purposes.)"""
        self._assert_initialized()
        return self._color

    @property
    def state(self) -> VehicleState:
        """The current state of this vehicle."""
        self._assert_initialized()
        return VehicleState(
            actor_id=self.id,
            actor_type=self.vehicle_type,
            source="SMARTS",  # this is the "ground truth" state
            vehicle_config_type=self._vehicle_config_type,
            pose=self.pose,
            dimensions=self._chassis.dimensions,
            speed=self.speed,
            # pytype: disable=attribute-error
            steering=self._chassis.steering,
            # pytype: enable=attribute-error
            yaw_rate=self._chassis.yaw_rate,
            linear_velocity=self._chassis.velocity_vectors[0],
            angular_velocity=self._chassis.velocity_vectors[1],
        )

    @property
    def action_space(self):
        """The action space this vehicle uses."""
        self._assert_initialized()
        return self._action_space

    @property
    def pose(self) -> Pose:
        """The pose of this vehicle. Pose is defined as position and orientation."""
        self._assert_initialized()
        return self._chassis.pose

    @property
    def chassis(self) -> Chassis:
        """The underlying chassis of this vehicle."""
        self._assert_initialized()
        return self._chassis

    @property
    def heading(self) -> Heading:
        """The heading of this vehicle.

        Note: Heading rotates counterclockwise with north as 0.
        """
        self._assert_initialized()
        return self._chassis.pose.heading

    @property
    def position(self) -> np.ndarray:
        """The position of this vehicle."""
        self._assert_initialized()
        return self._chassis.pose.position

    @property
    def bounding_box(self) -> List[np.ndarray]:
        """The minimum fitting heading aligned bounding box. Four 2D points representing the minimum fitting box."""
        # XXX: this doesn't return a smarts.core.coordinates.BoundingBox!
        self._assert_initialized()
        # Assuming the position is the center,
        # calculate the corner coordinates of the bounding_box
        origin = self.position[:2]
        dimensions = np.array([self.width, self.length])
        corners = np.array([(-1, 1), (1, 1), (1, -1), (-1, -1)]) / 2
        heading = self.heading
        return [
            rotate_cw_around_point(
                point=origin + corner * dimensions,
                radians=Heading.flip_clockwise(heading),
                origin=origin,
            )
            for corner in corners
        ]

    @property
    def vehicle_type(self) -> str:
        """Get the vehicle type identifier."""
        return VEHICLE_CONFIGS[self._vehicle_config_type].vehicle_type

    @property
    def valid(self) -> bool:
        """Check if the vehicle still `exists` and is still operable."""
        return self._initialized

    @staticmethod
    def agent_vehicle_dims(mission: Mission) -> Dimensions:
        """Get the vehicle dimensions from the mission requirements.
        Args:
            A mission for the agent.
        Returns:
            The mission vehicle spec dimensions XOR the default "passenger" vehicle dimensions.
        """
        if mission.vehicle_spec:
            # mission.vehicle_spec.veh_config_type will always be "passenger" for now,
            # but we use that value here in case we ever expand our history functionality.
            vehicle_config_type = mission.vehicle_spec.veh_config_type
            return Dimensions.copy_with_defaults(
                mission.vehicle_spec.dimensions,
                VEHICLE_CONFIGS[vehicle_config_type].dimensions,
            )
        # non-history agents can currently only control passenger vehicles.
        vehicle_config_type = "passenger"
        return VEHICLE_CONFIGS[vehicle_config_type].dimensions

    @classmethod
    def build_agent_vehicle(
        cls,
        sim,
        vehicle_id: str,
        agent_interface: AgentInterface,
        plan: Plan,
        vehicle_filepath: str,
        tire_filepath: str,
        trainable: bool,
        surface_patches: List[Dict[str, Any]],
        initial_speed: Optional[float] = None,
    ) -> "Vehicle":
        """Create a new vehicle and set up sensors and planning information as required by the
        ego agent.
        """
        mission = plan.mission

        chassis_dims = cls.agent_vehicle_dims(mission)

        start = mission.start
        if start.from_front_bumper:
            start_pose = Pose.from_front_bumper(
                front_bumper_position=np.array(start.position[:2]),
                heading=start.heading,
                length=chassis_dims.length,
            )
        else:
            start_pose = Pose.from_center(start.position, start.heading)

        vehicle_color = SceneColors.Agent if trainable else SceneColors.SocialAgent

        if agent_interface.vehicle_type == "sedan":
            urdf_name = "vehicle"
        elif agent_interface.vehicle_type == "bus":
            urdf_name = "bus"
        else:
            raise Exception("Vehicle type does not exist!!!")

        if (vehicle_filepath is None) or not os.path.exists(vehicle_filepath):
            with pkg_resources.path(models, urdf_name + ".urdf") as path:
                vehicle_filepath = str(path.absolute())

        controller_parameters = sim.vehicle_index.controller_params_for_vehicle_type(
            agent_interface.vehicle_type
        )

        chassis = None
        if agent_interface and agent_interface.action in sim.dynamic_action_spaces:
            if mission.vehicle_spec:
                logger = logging.getLogger(cls.__name__)
                logger.warning(
                    "setting vehicle dimensions on a AckermannChassis not yet supported"
                )
            chassis = AckermannChassis(
                pose=start_pose,
                bullet_client=sim.bc,
                vehicle_filepath=vehicle_filepath,
                tire_parameters_filepath=tire_filepath,
                friction_map=surface_patches,
                controller_parameters=controller_parameters,
                initial_speed=initial_speed,
            )
        else:
            chassis = BoxChassis(
                pose=start_pose,
                speed=initial_speed,
                dimensions=chassis_dims,
                bullet_client=sim.bc,
            )

        vehicle = Vehicle(
            id=vehicle_id,
            chassis=chassis,
            color=vehicle_color,
        )

        return vehicle

    @staticmethod
    def build_social_vehicle(sim, vehicle_id, vehicle_state: VehicleState) -> "Vehicle":
        """Create a new unassociated vehicle."""
        dims = Dimensions.copy_with_defaults(
            vehicle_state.dimensions,
            VEHICLE_CONFIGS[vehicle_state.vehicle_config_type].dimensions,
        )
        chassis = BoxChassis(
            pose=vehicle_state.pose,
            speed=vehicle_state.speed,
            dimensions=dims,
            bullet_client=sim.bc,
        )
        return Vehicle(
            id=vehicle_id,
            chassis=chassis,
            vehicle_config_type=vehicle_state.vehicle_config_type,
        )

    @staticmethod
    def attach_sensors_to_vehicle(
        sensor_manager,
        sim,
        vehicle: "Vehicle",
        agent_interface,
    ):
        """Attach sensors as required to satisfy the agent interface's requirements"""
        # The distance travelled sensor is not optional b/c it is used for the score
        # and reward calculation
        vehicle_state = vehicle.state
        sensor = TripMeterSensor()
        vehicle.attach_trip_meter_sensor(sensor)

        # The distance travelled sensor is not optional b/c it is used for visualization
        # done criteria
        sensor = DrivenPathSensor()
        vehicle.attach_driven_path_sensor(sensor)

        if agent_interface.neighborhood_vehicle_states:
            sensor = NeighborhoodVehiclesSensor(
                radius=agent_interface.neighborhood_vehicle_states.radius,
            )
            vehicle.attach_neighborhood_vehicle_states_sensor(sensor)

        if agent_interface.accelerometer:
            sensor = AccelerometerSensor()
            vehicle.attach_accelerometer_sensor(sensor)

        if agent_interface.lane_positions:
            sensor = LanePositionSensor()
            vehicle.attach_lane_position_sensor(sensor)

        if agent_interface.waypoint_paths:
            sensor = WaypointsSensor(
                lookahead=agent_interface.waypoint_paths.lookahead,
            )
            vehicle.attach_waypoints_sensor(sensor)

        if agent_interface.road_waypoints:
            sensor = RoadWaypointsSensor(
                horizon=agent_interface.road_waypoints.horizon,
            )
            vehicle.attach_road_waypoints_sensor(sensor)

        if agent_interface.drivable_area_grid_map:
            if not sim.renderer:
                raise RendererException.required_to("add a drivable_area_grid_map")
            sensor = DrivableAreaGridMapSensor(
                vehicle_state=vehicle_state,
                width=agent_interface.drivable_area_grid_map.width,
                height=agent_interface.drivable_area_grid_map.height,
                resolution=agent_interface.drivable_area_grid_map.resolution,
                renderer=sim.renderer,
            )
            vehicle.attach_drivable_area_grid_map_sensor(sensor)
        if agent_interface.occupancy_grid_map:
            if not sim.renderer:
                raise RendererException.required_to("add an OGM")
            sensor = OGMSensor(
                vehicle_state=vehicle_state,
                width=agent_interface.occupancy_grid_map.width,
                height=agent_interface.occupancy_grid_map.height,
                resolution=agent_interface.occupancy_grid_map.resolution,
                renderer=sim.renderer,
            )
            vehicle.attach_ogm_sensor(sensor)
        if agent_interface.top_down_rgb:
            if not sim.renderer:
                raise RendererException.required_to("add an RGB camera")
            sensor = RGBSensor(
                vehicle_state=vehicle_state,
                width=agent_interface.top_down_rgb.width,
                height=agent_interface.top_down_rgb.height,
                resolution=agent_interface.top_down_rgb.resolution,
                renderer=sim.renderer,
            )
            vehicle.attach_rgb_sensor(sensor)
        if agent_interface.lidar_point_cloud:
            sensor = LidarSensor(
                vehicle_state=vehicle_state,
                sensor_params=agent_interface.lidar_point_cloud.sensor_params,
            )
            vehicle.attach_lidar_sensor(sensor)

        sensor = ViaSensor(
            # At lane change time of 6s and speed of 13.89m/s, acquistion range = 6s x 13.89m/s = 83.34m.
            lane_acquisition_range=80,
            speed_accuracy=1.5,
        )
        vehicle.attach_via_sensor(sensor)

        if agent_interface.signals:
            lookahead = agent_interface.signals.lookahead
            sensor = SignalsSensor(lookahead=lookahead)
            vehicle.attach_signals_sensor(sensor)

        for sensor_name, sensor in vehicle.sensors.items():
            if not sensor:
                continue
            sensor_manager.add_sensor_for_actor(vehicle.id, sensor_name, sensor)

    def step(self, current_simulation_time):
        """Update internal state."""
        self._has_stepped = True
        self._chassis.step(current_simulation_time)

    def control(self, *args, **kwargs):
        """Apply control values to this vehicle.

        Forwards control to the chassis.
        """
        self._chassis.control(*args, **kwargs)

    def update_state(self, state: VehicleState, dt: float):
        """Update the vehicle's state"""
        state.updated = True
        if state.role != ActorRole.External:
            assert isinstance(self._chassis, BoxChassis)
            self.control(pose=state.pose, speed=state.speed, dt=dt)
            return
        # External actors are "privileged", which means they work directly (bypass force application).
        # Conceptually, this is playing 'god' with physics and should only be used
        # to defer to a co-simulator's states.
        linear_velocity, angular_velocity = None, None
        if not np.allclose(
            self._chassis.velocity_vectors[0], state.linear_velocity
        ) or not np.allclose(self._chassis.velocity_vectors[1], state.angular_velocity):
            linear_velocity = state.linear_velocity
            angular_velocity = state.angular_velocity
        if not state.dimensions.equal_if_defined(self.length, self.width, self.height):
            self._log.warning(
                "Unable to change a vehicle's dimensions via external_state_update()."
            )
        # XXX:  any way to update acceleration in pybullet?
        self._chassis.state_override(dt, state.pose, linear_velocity, angular_velocity)

    def create_renderer_node(self, renderer):
        """Create the vehicle's rendering node in the renderer."""
        config = VEHICLE_CONFIGS[self._vehicle_config_type]
        return renderer.create_vehicle_node(
            config.glb_model, self._id, self.vehicle_color, self.pose
        )

    # @lru_cache(maxsize=1)
    def _warn_AckermannChassis_set_pose(self):
        if self._has_stepped and isinstance(self._chassis, AckermannChassis):
            logging.warning(
                f"Agent `{self._id}` has called set pose after step."
                "This may cause collision problems"
            )

    # TODO: Merge this w/ speed setter as a set GCD call?
    def set_pose(self, pose: Pose):
        """Use with caution. This will directly set the pose of the chassis.

        This may disrupt physics simulation of the chassis physics body for a few steps after use.
        """
        self._warn_AckermannChassis_set_pose()
        self._chassis.set_pose(pose)

    def swap_chassis(self, chassis: Chassis):
        """Swap the current chassis with the given chassis. Apply the GCD of the previous chassis
        to the new chassis ("greatest common denominator state" from front-end to back-end)
        """
        chassis.inherit_physical_values(self._chassis)
        self._chassis.teardown()
        self._chassis = chassis

    def teardown(self, renderer, exclude_chassis=False):
        """Clean up internal resources"""
        if not exclude_chassis:
            self._chassis.teardown()
        if renderer:
            renderer.remove_vehicle_node(self._id)
        self._initialized = False

    def _meta_create_sensor_functions(self):
        # Bit of metaprogramming to make sensor creation more DRY
        sensor_names = [
            "ogm_sensor",
            "rgb_sensor",
            "lidar_sensor",
            "driven_path_sensor",
            "trip_meter_sensor",
            "drivable_area_grid_map_sensor",
            "neighborhood_vehicle_states_sensor",
            "waypoints_sensor",
            "road_waypoints_sensor",
            "accelerometer_sensor",
            "lane_position_sensor",
            "via_sensor",
            "signals_sensor",
        ]
        for sensor_name in sensor_names:

            def attach_sensor(self, sensor, sensor_name=sensor_name):
                # replace previously-attached sensor with this one
                # (to allow updating its parameters).
                # Sensors might have been attached to a non-agent vehicle
                # (for example, for observation collection from history vehicles),
                # but if that vehicle gets hijacked, we want to use the sensors
                # specified by the hijacking agent's interface.
                detach = getattr(self, f"detach_{sensor_name}")
                if detach:
                    detach(sensor_name)
                    self._log.info(
                        f"replacing existing {sensor_name} on vehicle {self.id}"
                    )
                setattr(self, f"_{sensor_name}", sensor)
                self._sensors[sensor_name] = sensor

            def detach_sensor(self, sensor_name=sensor_name):
                sensor = getattr(self, f"_{sensor_name}", None)
                if sensor is not None:
                    setattr(self, f"_{sensor_name}", None)
                    del self._sensors[sensor_name]
                return sensor

            def subscribed_to(self, sensor_name=sensor_name):
                sensor = getattr(self, f"_{sensor_name}", None)
                return sensor is not None

            def sensor_property(self, sensor_name=sensor_name):
                sensor = getattr(self, f"_{sensor_name}", None)
                assert sensor is not None, f"{sensor_name} is not attached to {self.id}"
                return sensor

            setattr(Vehicle, f"_{sensor_name}", None)
            setattr(Vehicle, f"attach_{sensor_name}", attach_sensor)
            setattr(Vehicle, f"detach_{sensor_name}", detach_sensor)
            setattr(Vehicle, f"subscribed_to_{sensor_name}", property(subscribed_to))
            setattr(Vehicle, f"{sensor_name}", property(sensor_property))

        def detach_all_sensors_from_vehicle(vehicle):
            sensors = []
            for sensor_name in sensor_names:
                detach_sensor_func = getattr(vehicle, f"detach_{sensor_name}")
                sensors.append(detach_sensor_func())
            return sensors

        setattr(
            Vehicle,
            "detach_all_sensors_from_vehicle",
            staticmethod(detach_all_sensors_from_vehicle),
        )
