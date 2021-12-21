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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import importlib.resources as pkg_resources
import logging
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import yaml

from . import models
from .chassis import AckermannChassis, BoxChassis, Chassis
from .colors import SceneColors
from .coordinates import Dimensions, Heading, Pose
from .sensors import (
    AccelerometerSensor,
    DrivableAreaGridMapSensor,
    DrivenPathSensor,
    LidarSensor,
    NeighborhoodVehiclesSensor,
    OGMSensor,
    RGBSensor,
    RoadWaypointsSensor,
    TrafficLightSensor,
    TripMeterSensor,
    ViaSensor,
    WaypointsSensor,
)
from .utils.math import rotate_around_point


@dataclass
class VehicleState:
    vehicle_id: str
    pose: Pose
    dimensions: Dimensions
    vehicle_type: str = None
    vehicle_config_type: str = None  # key into VEHICLE_CONFIGS
    updated: bool = False
    speed: float = 0
    steering: float = None
    yaw_rate: float = None
    source: str = None  # the source of truth for this vehicle state
    linear_velocity: np.ndarray = None
    angular_velocity: np.ndarray = None
    linear_acceleration: np.ndarray = None
    angular_acceleration: np.ndarray = None
    _privileged: bool = False

    def set_privileged(self):
        """For deferring to external co-simulators only. Use with caution!"""
        self._privileged = True

    @property
    def privileged(self) -> bool:
        return self._privileged


@dataclass(frozen=True)
class VehicleConfig:
    vehicle_type: str
    color: tuple
    dimensions: Dimensions
    glb_model: str


# A mapping between SUMO's vehicle types and our internal vehicle config.
# TODO: Don't leak SUMO's types here.
# XXX: The GLB's dimensions must match the specified dimensions here.
# TODO: for traffic histories, vehicle (and road) dimensions are set by the dataset
VEHICLE_CONFIGS = {
    "passenger": VehicleConfig(
        vehicle_type="car",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=3.68, width=1.47, height=1.4),
        glb_model="simple_car.glb",
    ),
    "bus": VehicleConfig(
        vehicle_type="bus",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=7, width=2.25, height=3),
        glb_model="bus.glb",
    ),
    "coach": VehicleConfig(
        vehicle_type="coach",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=8, width=2.4, height=3.5),
        glb_model="coach.glb",
    ),
    "truck": VehicleConfig(
        vehicle_type="truck",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=5, width=1.91, height=1.89),
        glb_model="truck.glb",
    ),
    "trailer": VehicleConfig(
        vehicle_type="trailer",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=10, width=2.5, height=4),
        glb_model="trailer.glb",
    ),
    "pedestrian": VehicleConfig(
        vehicle_type="pedestrian",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=0.5, width=0.5, height=1.6),
        glb_model="pedestrian.glb",
    ),
    "motorcycle": VehicleConfig(
        vehicle_type="motorcycle",
        color=SceneColors.SocialVehicle.value,
        dimensions=Dimensions(length=2.5, width=1, height=1.4),
        glb_model="motorcycle.glb",
    ),
}

# TODO: Replace VehicleConfigs w/ the VehicleGeometry class
class VehicleGeometry:
    @classmethod
    def fromfile(cls, path, color):
        pass


class RendererException(Exception):
    """An exception raised if a renderer is required but not available."""

    @classmethod
    def required_to(cls, thing):
        return cls(
            f"""A renderer is required to {thing}. You may not have installed the [camera-obs] dependencies required to render the camera sensor observations. Install them first using the command `pip install -e .[camera-obs]` at the source directory."""
        )


class Vehicle:
    def __init__(
        self,
        id: str,
        chassis: Chassis,
        vehicle_config_type: str = "passenger",
        color=None,
        action_space=None,
    ):
        self._log = logging.getLogger(self.__class__.__name__)
        self._id = id

        self._chassis = chassis
        self._vehicle_config_type = vehicle_config_type
        self._action_space = action_space
        self._speed = None

        self._meta_create_sensor_functions()
        self._sensors = {}

        # Color override
        self._color = color
        if self._color is None:
            config = VEHICLE_CONFIGS[vehicle_config_type]
            self._color = config.color

        self._renderer = None

        self._initialized = True
        self._has_stepped = False

    def _assert_initialized(self):
        assert self._initialized, f"Vehicle({self.id}) is not initialized"

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
        return self._id

    @property
    def length(self):
        self._assert_initialized()
        return self._chassis.dimensions.length

    @property
    def max_steering_wheel(self):
        self._assert_initialized()
        return self._chassis.max_steering_wheel

    @property
    def width(self):
        self._assert_initialized()
        return self._chassis.dimensions.width

    @property
    def height(self):
        self._assert_initialized()
        return self._chassis.dimensions.height

    @property
    def speed(self):
        self._assert_initialized()
        if self._speed is not None:
            return self._speed
        else:
            return self._chassis.speed

    def set_speed(self, speed):
        self._speed = speed

    @property
    def sensors(self):
        self._assert_initialized()
        return self._sensors

    @property
    def renderer(self):
        return self._renderer

    # # TODO: See issue #898 This is a currently a no-op
    # @speed.setter
    # def speed(self, speed):
    #     self._chassis.speed = speed

    @property
    def vehicle_color(self):
        self._assert_initialized()
        return self._color

    @property
    def state(self):
        self._assert_initialized()
        return VehicleState(
            vehicle_id=self.id,
            vehicle_type=self.vehicle_type,
            vehicle_config_type=None,  # it's hard to invert
            pose=self.pose,
            dimensions=self._chassis.dimensions,
            speed=self.speed,
            steering=self._chassis.steering,
            yaw_rate=self._chassis.yaw_rate,
            source="SMARTS",
            linear_velocity=self._chassis.velocity_vectors[0],
            angular_velocity=self._chassis.velocity_vectors[1],
        )

    @property
    def action_space(self):
        self._assert_initialized()
        return self._action_space

    @property
    def pose(self) -> Pose:
        self._assert_initialized()
        return self._chassis.pose

    @property
    def chassis(self):
        self._assert_initialized()
        return self._chassis

    @property
    def heading(self) -> Heading:
        self._assert_initialized()
        return self._chassis.pose.heading

    @property
    def position(self):
        self._assert_initialized()
        pos, _ = self._chassis.pose.as_panda3d()
        return pos

    @property
    def bounding_box(self):
        self._assert_initialized()
        # Assuming the position is the centre,
        # calculate the corner coordinates of the bounding_box
        origin = self.position[:2]
        dimensions = np.array([self.width, self.length])
        corners = np.array([(-1, 1), (1, 1), (1, -1), (-1, -1)]) / 2
        heading = self.heading
        return [
            rotate_around_point(
                point=origin + corner * dimensions,
                radians=heading,
                origin=origin,
            )
            for corner in corners
        ]

    @property
    def vehicle_type(self):
        return VEHICLE_CONFIGS[self._vehicle_config_type].vehicle_type

    @staticmethod
    def agent_vehicle_dims(mission) -> Dimensions:
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
        vehicle_id,
        agent_interface,
        plan,
        vehicle_filepath,
        tire_filepath,
        trainable,
        surface_patches,
        initial_speed=None,
    ):
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

        vehicle_color = (
            SceneColors.Agent.value if trainable else SceneColors.SocialAgent.value
        )

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
        # change this to dynamic_action_spaces later when pr merged
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
    def build_social_vehicle(sim, vehicle_id, vehicle_state, vehicle_config_type):
        dims = Dimensions.copy_with_defaults(
            vehicle_state.dimensions, VEHICLE_CONFIGS[vehicle_config_type].dimensions
        )
        chassis = BoxChassis(
            pose=vehicle_state.pose,
            speed=vehicle_state.speed,
            dimensions=dims,
            bullet_client=sim.bc,
        )
        return Vehicle(
            id=vehicle_id, chassis=chassis, vehicle_config_type=vehicle_config_type
        )

    @staticmethod
    def attach_sensors_to_vehicle(sim, vehicle, agent_interface, plan):
        # The distance travelled sensor is not optional b/c it is used for the score
        # and reward calculation
        vehicle.attach_trip_meter_sensor(
            TripMeterSensor(
                vehicle=vehicle,
                sim=sim,
                plan=plan,
            )
        )

        # The distance travelled sensor is not optional b/c it is used for visualization
        # done criteria
        vehicle.attach_driven_path_sensor(DrivenPathSensor(vehicle=vehicle))

        if agent_interface.neighborhood_vehicles:
            vehicle.attach_neighborhood_vehicles_sensor(
                NeighborhoodVehiclesSensor(
                    vehicle=vehicle,
                    sim=sim,
                    radius=agent_interface.neighborhood_vehicles.radius,
                )
            )

        if agent_interface.accelerometer:
            vehicle.attach_accelerometer_sensor(AccelerometerSensor(vehicle=vehicle))

        if agent_interface.waypoints:
            vehicle.attach_waypoints_sensor(
                WaypointsSensor(
                    vehicle=vehicle,
                    plan=plan,
                    lookahead=agent_interface.waypoints.lookahead,
                )
            )

        if agent_interface.road_waypoints:
            vehicle.attach_road_waypoints_sensor(
                RoadWaypointsSensor(
                    vehicle=vehicle,
                    sim=sim,
                    plan=plan,
                    horizon=agent_interface.road_waypoints.horizon,
                )
            )

        if agent_interface.drivable_area_grid_map:
            if not sim.renderer:
                raise RendererException.required_to("add a drivable_area_grid_map")
            vehicle.attach_drivable_area_grid_map_sensor(
                DrivableAreaGridMapSensor(
                    vehicle=vehicle,
                    width=agent_interface.drivable_area_grid_map.width,
                    height=agent_interface.drivable_area_grid_map.height,
                    resolution=agent_interface.drivable_area_grid_map.resolution,
                    renderer=sim.renderer,
                )
            )
        if agent_interface.ogm:
            if not sim.renderer:
                raise RendererException.required_to("add an OGM")
            vehicle.attach_ogm_sensor(
                OGMSensor(
                    vehicle=vehicle,
                    width=agent_interface.ogm.width,
                    height=agent_interface.ogm.height,
                    resolution=agent_interface.ogm.resolution,
                    renderer=sim.renderer,
                )
            )
        if agent_interface.rgb:
            if not sim.renderer:
                raise RendererException.required_to("add an RGB camera")
            vehicle.attach_rgb_sensor(
                RGBSensor(
                    vehicle=vehicle,
                    width=agent_interface.rgb.width,
                    height=agent_interface.rgb.height,
                    resolution=agent_interface.rgb.resolution,
                    renderer=sim.renderer,
                )
            )
        if agent_interface.lidar:
            vehicle.attach_lidar_sensor(
                LidarSensor(
                    vehicle=vehicle,
                    bullet_client=sim.bc,
                    sensor_params=agent_interface.lidar.sensor_params,
                )
            )

        vehicle.attach_via_sensor(
            ViaSensor(
                vehicle=vehicle,
                plan=plan,
                lane_acquisition_range=40,
                speed_accuracy=1.5,
            )
        )

        if agent_interface.traffic_lights:
            assert (
                sim.scenario.traffic_history
            ), "Traffic history required for traffic lights sensor"
            vehicle.attach_traffic_lights_sensor(
                TrafficLightSensor(sim.scenario.traffic_history)
            )

    def step(self, current_simulation_time):
        self._has_stepped = True
        self._chassis.step(current_simulation_time)

    def control(self, *args, **kwargs):
        self._chassis.control(*args, **kwargs)

    def update_state(self, state: VehicleState, dt: float):
        state.updated = True
        if not state.privileged:
            assert isinstance(self._chassis, BoxChassis)
            self.control(pose=state.pose, speed=state.speed, dt=dt)
            return
        # "Privileged" means we can work directly (bypass force application).
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
        assert not self._renderer
        self._renderer = renderer
        config = VEHICLE_CONFIGS[self._vehicle_config_type]
        self._renderer.create_vehicle_node(
            config.glb_model, self._id, self.vehicle_color, self.pose
        )

    def sync_to_renderer(self):
        assert self._renderer
        self._renderer.update_vehicle_node(self._id, self.pose)

    @lru_cache(maxsize=1)
    def _warn_AckermannChassis_set_pose(self):
        if self._has_stepped and isinstance(self._chassis, AckermannChassis):
            logging.warning(
                f"Agent `{self._id}` has called set pose after step."
                "This may cause collision problems"
            )

    # TODO: Merge this w/ speed setter as a set GCD call?
    def set_pose(self, pose: Pose):
        self._warn_AckermannChassis_set_pose()
        self._chassis.set_pose(pose)

    def swap_chassis(self, chassis: Chassis):
        # Apply GCD ("greatest common denominator state" from front-end to back-end)
        chassis.inherit_physical_values(self._chassis)
        self._chassis.teardown()
        self._chassis = chassis

    def teardown(self, exclude_chassis=False):
        for sensor in [
            sensor
            for sensor in [
                self._drivable_area_grid_map_sensor,
                self._ogm_sensor,
                self._rgb_sensor,
                self._lidar_sensor,
                self._driven_path_sensor,
            ]
            if sensor is not None
        ]:
            sensor.teardown()

        if not exclude_chassis:
            self._chassis.teardown()
        if self._renderer:
            self._renderer.remove_vehicle_node(self._id)
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
            "neighborhood_vehicles_sensor",
            "waypoints_sensor",
            "road_waypoints_sensor",
            "accelerometer_sensor",
            "via_sensor",
            "traffic_lights_sensor",
        ]
        for sensor_name in sensor_names:

            def attach_sensor(self, sensor, sensor_name=sensor_name):
                assert (
                    getattr(self, f"_{sensor_name}", None) is None
                ), f"{sensor_name} already added to {self.id}"
                setattr(self, f"_{sensor_name}", sensor)
                self._sensors[sensor_name] = sensor

            def detach_sensor(self, sensor_name=sensor_name):
                sensor = getattr(self, f"_{sensor_name}", None)
                if sensor is not None:
                    sensor.teardown()
                    setattr(self, f"_{sensor_name}", None)
                    del self._sensors[sensor_name]

            def subscribed_to(self, sensor_name=sensor_name):
                sensor = getattr(self, f"_{sensor_name}", None)
                return sensor is not None

            def sensor_property(self, sensor_name=sensor_name):
                sensor = getattr(self, f"_{sensor_name}", None)
                assert sensor is not None, f"{sensor_name} is not attached"
                return sensor

            setattr(Vehicle, f"_{sensor_name}", None)
            setattr(Vehicle, f"attach_{sensor_name}", attach_sensor)
            setattr(Vehicle, f"detach_{sensor_name}", detach_sensor)
            setattr(Vehicle, f"subscribed_to_{sensor_name}", property(subscribed_to))
            setattr(Vehicle, f"{sensor_name}", property(sensor_property))

        def detach_all_sensors_from_vehicle(vehicle):
            for sensor_name in sensor_names:
                detach_sensor_func = getattr(vehicle, f"detach_{sensor_name}")
                detach_sensor_func()

        setattr(
            Vehicle,
            "detach_all_sensors_from_vehicle",
            staticmethod(detach_all_sensors_from_vehicle),
        )
