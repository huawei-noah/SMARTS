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
import math
import os
from typing import Optional, Sequence, Tuple

import numpy as np
import yaml
from cached_property import cached_property
from shapely.affinity import rotate as shapely_rotate
from shapely.geometry import Polygon
from shapely.geometry import box as shapely_box

from smarts.core import models
from smarts.core.coordinates import Dimensions, Heading, Pose
from smarts.core.tire_models import TireForces
from smarts.core.utils import pybullet
from smarts.core.utils.bullet import (
    BulletBoxShape,
    BulletPositionConstraint,
    ContactPoint,
    JointInfo,
    JointState,
)
from smarts.core.utils.math import (
    min_angles_difference_signed,
    radians_to_vec,
    vec_to_radians,
    yaw_from_quaternion,
)
from smarts.core.utils.pybullet import bullet_client as bc

with pkg_resources.path(models, "vehicle.urdf") as path:
    DEFAULT_VEHICLE_FILEPATH = str(path.absolute())
with pkg_resources.path(models, "controller_parameters.yaml") as controller_path:
    controller_filepath = str(controller_path.absolute())
with open(controller_filepath, "r") as controller_file:
    DEFAULT_CONTROLLER_PARAMETERS = yaml.safe_load(controller_file)["sedan"]


def _query_bullet_contact_points(bullet_client, bullet_id, link_index):
    contact_objects = set()

    # `getContactPoints` does not pick up collisions well so we cast a fast box check on the physics
    min_, max_ = bullet_client.getAABB(bullet_id, link_index)
    # note that getAABB returns a box around the link_index link only,
    # which means it's offset from the ground (min_ has a positive z)
    # if link_index=0 (the chassis link) is used.
    overlapping_objects = bullet_client.getOverlappingObjects(min_, max_)
    # the pairs returned by getOverlappingObjects() appear to be in the form (body_id, link_idx)
    if overlapping_objects is not None:
        contact_objects = set(oo for oo, _ in overlapping_objects if oo != bullet_id)

    contact_points = []
    for contact_object in contact_objects:
        # Give 0.05 meter leeway
        contact_points.extend(
            bullet_client.getClosestPoints(bullet_id, contact_object, distance=0.05)
        )

    return contact_points


class Chassis:
    """Represents a vehicle chassis."""

    def control(self, *args, **kwargs):
        """Apply control values to the chassis."""
        raise NotImplementedError

    def reapply_last_control(self):
        """Re-apply the last given control given to the chassis."""
        raise NotImplementedError

    def teardown(self):
        """Clean up resources."""
        raise NotImplementedError

    @property
    def dimensions(self) -> Dimensions:
        """The fitted front aligned dimensions of the chassis."""
        raise NotImplementedError

    @property
    def contact_points(self) -> Sequence:
        """The contact point of the chassis."""
        raise NotImplementedError

    @property
    def bullet_id(self) -> str:
        """The physics id of the chassis physics body."""
        raise NotImplementedError

    @property
    def velocity_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns linear velocity vector in m/s and angular velocity in rad/sec."""
        raise NotImplementedError

    @property
    def speed(self) -> float:
        """The speed of the chassis in the facing direction of the chassis."""
        raise NotImplementedError

    @speed.setter
    def speed(self, speed: float):
        """Apply GCD from front-end."""
        raise NotImplementedError

    @property
    def pose(self) -> Pose:
        """The pose of the chassis."""
        raise NotImplementedError

    @property
    def steering(self):
        """The steering value of the chassis in radians [-math.pi, math.pi]."""
        raise NotImplementedError

    @property
    def yaw_rate(self) -> float:
        """The turning rate of the chassis in radians."""
        raise NotImplementedError

    def inherit_physical_values(self, other: "Chassis"):
        """Apply GCD between the two chassis."""
        raise NotImplementedError

    @property
    def to_polygon(self) -> Polygon:
        """Convert the chassis to a 2D shape."""
        p = self.pose.position
        d = self.dimensions
        poly = shapely_box(
            p[0] - d.width * 0.5,
            p[1] - d.length * 0.5,
            p[0] + d.width * 0.5,
            p[1] + d.length * 0.5,
        )
        return shapely_rotate(poly, self.pose.heading, use_radians=True)

    def step(self, current_simulation_time):
        """Update the chassis state."""
        raise NotImplementedError

    def state_override(
        self,
        dt: float,
        force_pose: Pose,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
    ):
        """Use with care!  In essence, this is tinkering with the physics of the world,
        and may have unintended behavioral or performance consequences."""
        raise NotImplementedError


class BoxChassis(Chassis):
    """Control a vehicle by setting its absolute position and heading. The collision
    shape of the vehicle is a box of the provided dimensions.
    """

    def __init__(
        self,
        pose: Pose,
        speed: float,
        dimensions: Dimensions,
        bullet_client: bc.BulletClient,
    ):
        self._dimensions = dimensions
        self._bullet_body = BulletBoxShape(self._dimensions.as_lwh, bullet_client)
        self._bullet_constraint = BulletPositionConstraint(
            self._bullet_body, bullet_client
        )
        bullet_client.setCollisionFilterGroupMask(
            self._bullet_body._bullet_id, -1, 0x0, 0x0
        )
        self._pose = None
        self.control(pose, speed)
        self._client = bullet_client

    def control(self, pose: Pose, speed: float, dt: float = 0):
        if self._pose:
            self._last_heading = self._pose.heading
        self._last_dt = dt
        self._pose = pose
        self._speed = speed
        self._bullet_constraint.move_to(pose)

    def reapply_last_control(self):
        # no need to do anything here since we're not applying forces
        pass

    def state_override(
        self,
        dt: float,
        force_pose: Pose,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
    ):
        """Use with care!  In essence, this is tinkering with the physics of the world,
        and may have unintended behavioral or performance consequences."""
        if self._pose:
            self._last_heading = self._pose.heading
        self._last_dt = dt
        self._pose = force_pose
        if linear_velocity is not None or angular_velocity is not None:
            assert linear_velocity is not None
            assert angular_velocity is not None
            self._speed = np.linalg.norm(linear_velocity)
            self._client.resetBaseVelocity(
                self.bullet_id,
                linearVelocity=linear_velocity,
                angularVelocity=angular_velocity,
            )
        self._bullet_constraint.move_to(force_pose)

    @property
    def dimensions(self) -> Dimensions:
        return self._dimensions

    @property
    def contact_points(self) -> Sequence:
        contact_points = _query_bullet_contact_points(self._client, self.bullet_id, -1)
        return [
            ContactPoint(bullet_id=p[2], contact_point=p[5], contact_point_other=p[6])
            for p in contact_points
        ]

    @property
    def bullet_id(self) -> str:
        return self._bullet_body._bullet_id

    @property
    def speed(self) -> float:
        # in m/s
        return self._speed

    @property
    def velocity_vectors(self):
        # linear_velocity in m/s, angular_velocity in rad/s
        vh = radians_to_vec(self.pose.heading)
        if self._speed is not None:
            linear_velocity = np.array((vh[0], vh[1], 0.0)) * self._speed
        else:
            linear_velocity = None
        if self._last_dt and self._last_dt > 0:
            av = (vh - radians_to_vec(self._last_heading)) / self._last_dt
            angular_velocity = np.array((av[0], av[1], 0.0))
        else:
            angular_velocity = np.array((0.0, 0.0, 0.0))
        return (linear_velocity, angular_velocity)

    @speed.setter
    def speed(self, speed: Optional[float] = None):
        self._speed = speed

    @property
    def pose(self) -> Pose:
        return self._pose

    @property
    def steering(self):
        return None

    @property
    def yaw_rate(self) -> Optional[float]:
        # in rad/s
        if self._last_dt and self._last_dt > 0:
            delta = min_angles_difference_signed(self._pose.heading, self._last_heading)
            return delta / self._last_dt
        return None

    def inherit_physical_values(self, other: Chassis):
        self._pose = other.pose
        self.speed = other.speed
        # ignore physics

    def step(self, current_simulation_time):
        pass

    def teardown(self):
        self._bullet_constraint.teardown()
        self._bullet_body.teardown()


class AckermannChassis(Chassis):
    """Control a vehicle by applying forces on its joints. The joints and links are
    defined by a URDF file.
    """

    def __init__(
        self,
        pose: Pose,
        bullet_client: bc.BulletClient,
        vehicle_filepath=DEFAULT_VEHICLE_FILEPATH,
        tire_parameters_filepath=None,
        friction_map=None,
        controller_parameters=DEFAULT_CONTROLLER_PARAMETERS,
        initial_speed=None,
    ):
        assert isinstance(pose, Pose)
        self._log = logging.getLogger(self.__class__.__name__)

        # XXX: Parameterize these vehicle properties?
        self._client = bullet_client
        self._chassis_aero_force_gain = controller_parameters["chassis"][
            "chassis_aero_force_gain"
        ]
        self._max_brake_gain = controller_parameters["chassis"]["max_brake_gain"]
        # This value was found emperically. It causes the wheel steer joints to
        # reach their maximum. We use this value to map to the -1, 1 steering range.
        # If it were larger we'd cap out our turning radius before we hit -1, or 1.
        # If it were smaller we'd never reach the tightest turning radius we could.
        self._max_turn_radius = controller_parameters["chassis"]["max_turn_radius"]
        self._wheel_radius = controller_parameters["chassis"]["wheel_radius"]
        self._max_torque = controller_parameters["chassis"]["max_torque"]
        self._max_btorque = controller_parameters["chassis"]["max_btorque"]
        # 720 is the maximum driver steering wheel angle
        # which equals to two full rotation of steering wheel.
        # This corresponds to maximum 41.3 deg angle at tires.
        self._max_steering = controller_parameters["chassis"]["max_steering"]
        self._steering_gear_ratio = controller_parameters["chassis"][
            "steering_gear_ratio"
        ]
        self._tire_model = None
        self._lat_forces = np.zeros(4)
        self._lon_forces = np.zeros(4)
        self._plane_id = None
        self._friction_map = friction_map
        self._tire_parameters = None
        self._road_wheel_frictions = None

        self._bullet_id = self._client.loadURDF(
            vehicle_filepath,
            [0, 0, 0],
            [0, 0, 0, 1],
            useFixedBase=False,
            # We use cylinders for wheels, If we don't provide this flag, bullet
            # will subdivide the cylinder resulting in bouncy ride
            flags=pybullet.URDF_USE_IMPLICIT_CYLINDER
            | pybullet.URDF_ENABLE_CACHED_GRAPHICS_SHAPES,
        )
        for i in range(self._client.getNumBodies()):
            if self._client.getBodyInfo(i)[1].decode("ascii") == "plane":
                self._plane_id = self._client.getBodyUniqueId(i)
                self._road_wheel_frictions = {
                    "road_friction": self._client.getDynamicsInfo(
                        self._plane_id,
                        -1,
                    )[1],
                    "wheel_friction": self._client.getDynamicsInfo(
                        self._bullet_id,
                        2,
                    )[1],
                }
                break

        self._controller_parameters = controller_parameters["control"]

        if (tire_parameters_filepath is not None) and os.path.exists(
            tire_parameters_filepath
        ):
            self._client.changeDynamics(
                self._plane_id,
                -1,
                lateralFriction=1e-16,
            )
            with open(tire_parameters_filepath, "r") as tire_file:
                self._tire_parameters = yaml.safe_load(tire_file)
                self._tire_model = TireForces.build_tire_model(
                    [
                        self._tire_parameters["C_alpha_front"],
                        self._tire_parameters["C_alpha_rear"],
                        self._tire_parameters["C_x_front"],
                        self._tire_parameters["C_x_rear"],
                    ],
                    self._tire_parameters["tire_model"],
                    self._tire_parameters["road_friction"],
                )
            for j in [2, 4, 5, 6]:
                self._client.setCollisionFilterPair(
                    self._bullet_id, self._plane_id, j, -1, 0
                )

        for _id in range(len(self._load_joints(self._bullet_id)) + 1):
            self._client.changeDynamics(
                self._bullet_id,
                _id - 1,
                linearDamping=0,
                angularDamping=0.00001,
                maxJointVelocity=300,  # This is a limit after which the pybullet interfer with the dynamics to ensure the bound.
            )

        width, length, height = np.array(
            self._client.getCollisionShapeData(self._bullet_id, 0)[0][3]
        )
        self._dimensions = Dimensions(length=length, width=width, height=height)
        chassis_pos = self._client.getLinkState(self._bullet_id, 0)[4]
        center_offset = np.array(
            self._client.getVisualShapeData(self._bullet_id, 0)[0][5]
        )

        self._joints = self._load_joints(self._bullet_id)

        # 2,4,5,6 are the indices of wheels (FL,FR,RL,RR) and 1,3 are the
        # indices for front left and front right steer joints, 0 is
        # the index of the base link.
        link_states = self._client.getLinkStates(self._bullet_id, [0, 2, 4, 5, 6])
        link_positions = [np.array(state[0]) for state in link_states]
        base_pos = link_positions[0] + self._client.getLinkState(self._bullet_id, 0)[2]
        front_left_wheel_pos = link_positions[1]
        front_right_wheel_pos = link_positions[2]
        back_left_wheel_pos = link_positions[3]
        back_right_wheel_pos = link_positions[4]
        self._front_track_width = np.linalg.norm(
            front_left_wheel_pos - front_right_wheel_pos
        )
        self._rear_track_width = np.linalg.norm(
            back_left_wheel_pos - back_right_wheel_pos
        )
        self._front_distant_CG = np.linalg.norm(
            0.5 * (front_left_wheel_pos[0:2] + front_right_wheel_pos[0:2])
            - base_pos[0:2]
        )
        self._rear_distant_CG = np.linalg.norm(
            0.5 * (back_left_wheel_pos[0:2] + back_right_wheel_pos[0:2]) - base_pos[0:2]
        )

        # distance between axles
        front_axle_pos = (front_left_wheel_pos + front_right_wheel_pos) / 2
        back_axle_pos = (back_left_wheel_pos + back_right_wheel_pos) / 2
        self._wheel_base_length = np.linalg.norm(front_axle_pos - back_axle_pos)
        self._clear_joint_forces()

        self.set_pose(pose)

        if initial_speed is not None:
            self._initialize_speed(initial_speed)

    @cached_property
    def _cached_props(self):
        return {
            a
            for a, v in self.__class__.__dict__.items()
            if isinstance(v, cached_property)
        }

    def _clear_step_cache(self):
        cached_props = self._cached_props
        self.__dict__ = {
            a: v
            for a, v in self.__dict__.items()
            if a not in cached_props or a == "_cached_props"
        }

    @cached_property
    def pose(self) -> Pose:
        pos, orn = self._client.getBasePositionAndOrientation(self._bullet_id)
        heading = Heading(yaw_from_quaternion(orn))
        # NOTE: we're inefficiently creating a new Pose object on every call here,
        # but it's too risky to change this because our clients now rely on this behavior.
        return Pose.from_explicit_offset(
            [0, 0, 0],
            np.array(pos),
            heading,
            local_heading=Heading(0),
        )

    def set_pose(self, pose: Pose):
        """Use with caution since it disrupts the physics simulation. Sets the pose of the
        chassis.
        """
        position, orientation = pose.as_bullet()
        self._client.resetBasePositionAndOrientation(
            self._bullet_id, position, orientation
        )
        self._clear_step_cache()

    @cached_property
    def steering(self):
        """Current steering value in radians."""
        steering_radians = np.mean(
            [
                joint.position
                for joint in self._joint_states(
                    ["front_left_steer_joint", "front_right_steer_joint"]
                )
            ]
        )
        # Convert to clockwise rotation to be consistent with our action
        #    space where 1 is a right turn and -1 is a left turn. We'd expect
        #    an action with a positive steering activation to result in
        #    positive steering values read back from the vehicle.
        return -steering_radians

    @property
    def speed(self) -> float:
        """Returns speed in m/s."""
        velocity, _ = np.array(self._client.getBaseVelocity(self._bullet_id))
        return math.sqrt(velocity.dot(velocity))

    @cached_property
    def velocity_vectors(self):
        linear_velocity, angular_velocity = np.array(
            self._client.getBaseVelocity(self._bullet_id)
        )
        return (np.array(self.longitudinal_lateral_speed + (0,)), angular_velocity)

    @speed.setter
    def speed(self, speed: Optional[float] = None):
        # TODO: Temporary, figure out the required joint velocities to achieve the
        #       requested speed
        if not speed or self.speed < speed:
            self.control(brake=1)
        elif self.speed > speed:
            self.control(throttle=1)

    @cached_property
    def yaw_rate(self) -> float:
        """Returns 2-D rotational speed in rad/sec."""
        _, velocity_rotational = np.array(self._client.getBaseVelocity(self._bullet_id))
        return vec_to_radians(velocity_rotational[:2])

    @cached_property
    def longitudinal_lateral_speed(self):
        """Returns speed in m/s."""
        velocity, _ = np.array(self._client.getBaseVelocity(self._bullet_id))
        heading = self.pose.heading
        return (
            (velocity[1] * math.cos(heading) - velocity[0] * math.sin(heading)),
            (velocity[1] * math.sin(heading) + velocity[0] * math.cos(heading)),
        )

    @property
    def front_rear_stiffness(self):
        """The front and rear stiffness values of the tires on this chassis."""
        if self._tire_parameters is not None:
            return (
                self._tire_parameters["C_alpha_front"],
                self._tire_parameters["C_alpha_rear"],
            )
        else:
            raise ValueError("MPC requires providing tire stiffnesses in tire model")

    @property
    def approx_max_speed(self):
        """This is the scientifically discovered maximum speed of this vehicle model"""
        return 95

    @cached_property
    def contact_points(self):
        ## 0 is the chassis link index (which means ground won't be included)
        contact_points = _query_bullet_contact_points(self._client, self._bullet_id, 0)
        return [
            ContactPoint(bullet_id=p[2], contact_point=p[5], contact_point_other=p[6])
            for p in contact_points
        ]

    @cached_property
    def mass_and_inertia(self):
        """The mass and inertia values of this chassis."""
        return (
            self._client.getDynamicsInfo(self._bullet_id, 0)[0],
            self._client.getDynamicsInfo(self._bullet_id, 0)[2][2],
        )

    @property
    def controller_parameters(self):
        """The current controller parameters for this chassis."""
        return self._controller_parameters

    @property
    def dimensions(self):
        return self._dimensions

    @property
    def max_steering_wheel(self):
        """Maximum steering output for the current gear ratio."""
        return self._max_steering / self._steering_gear_ratio

    @property
    def wheel_radius(self):
        """The wheel radius of the wheels on the chassis."""
        return self._wheel_radius

    @property
    def front_rear_axle_CG_distance(self):
        """The axle offsets from the vehicle base."""
        return (self._front_distant_CG, self._rear_distant_CG)

    @property
    def front_track_width(self):
        """The track width between the front wheels."""
        return self._front_track_width

    @property
    def rear_track_width(self):
        """The track width between the back wheels."""
        return self._rear_track_width

    @property
    def max_torque(self):
        """The maximum throttle torque."""
        return self._max_torque

    @property
    def max_btorque(self):
        """The maximum break torque."""
        return self._max_btorque

    @property
    def steering_ratio(self):
        """The steering gear ratio"""
        return self._steering_gear_ratio

    @property
    def bullet_id(self) -> str:
        return self._bullet_id

    @property
    def bullet_client(self):
        """The bullet physics simulator."""
        return self._client

    def step(self, current_simulation_time):
        if self._friction_map != None:
            self._set_road_friction(current_simulation_time)
        self._clear_step_cache()

    def inherit_physical_values(self, other: BoxChassis):
        self.set_pose(other.pose)
        self._initialize_speed(other.speed)

    def _initialize_speed(self, speed: float):
        self.speed = speed
        velocity = radians_to_vec(self.pose.heading) * speed
        self._client.resetBaseVelocity(self._bullet_id, [*velocity, 0])

    def teardown(self):
        self._client.removeBody(self._bullet_id)
        self._bullet_id = None
        self._clear_step_cache()

    def control(self, throttle=0, brake=0, steering=0):
        """Apply throttle [0, 1], brake [0, 1], and steering [-1, 1] values for this
        timestep.
        """
        self._last_control = (throttle, brake, steering)

        if isinstance(throttle, np.ndarray):
            assert all(
                0 <= x <= 1 for x in throttle
            ), f"throttle ({throttle}) must be in [0, 1]"
            throttle_list = list(throttle * self._max_torque)
        else:
            assert 0 <= throttle <= 1, f"throttle ({throttle}) must be in [0, 1]"
            throttle_list = [throttle * self._max_torque] * 4
        assert 0 <= brake <= 1, f"brake ({brake}) must be in [0, 1]"
        assert -1 <= steering <= 1, f"steering ({steering}) must be in [-1, 1]"

        # If we apply brake at low speed using reverse torque
        # the vehicle starts to roll back. we need to apply a condition
        # on brake such that, the reverse torque is only applied after
        # a threshold is passed for vehicle velocity.
        # Thus, brake is applied if: vehicle speed > 1/36 (m/s)
        if brake > 0 and self.longitudinal_lateral_speed[0] < 1 / 36:
            brake = 0

        self._apply_steering(steering)

        # If the tire parameters yaml file exists, then the throttle and
        # brake forces are applied according to the requested tire model.
        # Otherwise, it uses bullet to calculate the reaction forces.
        if self._tire_model != None:
            self._lat_forces, self._lon_forces = self._tire_model.apply_tire_forces(
                self,
                self.bullet_client,
                [(1 / self._max_torque) * np.array(throttle_list), brake, steering],
            )
            self._clear_step_cache()
            return
        self._apply_throttle(throttle_list)
        self._apply_brake(brake)
        self._clear_step_cache()

    def reapply_last_control(self):
        assert self._last_control
        self.control(*self._last_control)

    def state_override(
        self,
        dt: float,
        force_pose: Pose,
        linear_velocity: Optional[np.ndarray] = None,
        angular_velocity: Optional[np.ndarray] = None,
    ):
        """Use with care!  In essence, this is tinkering with the physics of the world,
        and may have unintended behavioral or performance consequences."""
        self.set_pose(force_pose)
        if linear_velocity is not None or angular_velocity is not None:
            assert linear_velocity is not None
            assert angular_velocity is not None
            self._client.resetBaseVelocity(
                self._bullet_id,
                linearVelocity=linear_velocity,
                angularVelocity=angular_velocity,
            )
        self._clear_step_cache()

    def _apply_throttle(self, throttle_list):
        self._client.setJointMotorControlArray(
            self._bullet_id,
            [
                self._joints[name].index
                for name in [
                    "front_left_wheel_joint",
                    "front_right_wheel_joint",
                    "rear_left_wheel_joint",
                    "rear_right_wheel_joint",
                ]
            ],
            pybullet.TORQUE_CONTROL,
            forces=throttle_list,
        )

    def _apply_brake(self, brake):
        self._client.setJointMotorControlArray(
            self._bullet_id,
            [
                self._joints[name].index
                for name in [
                    "front_left_wheel_joint",
                    "front_right_wheel_joint",
                    "rear_left_wheel_joint",
                    "rear_right_wheel_joint",
                ]
            ],
            pybullet.TORQUE_CONTROL,
            forces=[-brake * self._max_btorque] * 4,
        )

    def _apply_steering(self, steering):
        # Apply steering (following Ackermann steering geometry)
        # See http://datagenetics.com/blog/december12016/index.html

        self._client.setJointMotorControlArray(
            self._bullet_id,
            [
                self._joints[name].index
                for name in [
                    "front_left_steer_joint",
                    "front_right_steer_joint",
                ]
            ],
            pybullet.POSITION_CONTROL,
            targetPositions=[
                -steering * self._max_steering * (1 / self._steering_gear_ratio),
                -steering * self._max_steering * (1 / self._steering_gear_ratio),
            ],
        )

    def _log_states(self):
        wheel_joint_states = self._joint_states(
            [
                "front_left_wheel_joint",
                "front_right_wheel_joint",
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
            ]
        )
        state_summary = "\t|\t".join(
            "{:.2f}".format(s.velocity) for s in wheel_joint_states
        )
        self._log.debug(
            f"wheel_states: {state_summary}\t vehicle speed: {self.speed:.2f}",
            end="\r",
        )

    def _load_joints(self, bullet_id):
        joints = {}
        for i in range(self._client.getNumJoints(self._bullet_id)):
            info = self._client.getJointInfo(self._bullet_id, i)
            name = info[1].decode()
            joints[name] = JointInfo(
                index=info[0],
                type_=info[2],
                lower_limit=info[8],
                upper_limit=info[9],
                max_force=info[10],
                max_velocity=info[11],
            )
        return joints

    def _joint_states(self, names):
        joint_indices = [self._joints[name].index for name in names]
        joint_states = self._client.getJointStates(self._bullet_id, joint_indices)
        joint_states = [JointState(position=s[0], velocity=s[1]) for s in joint_states]
        return joint_states

    def _clear_joint_forces(self):
        self._client.setJointMotorControlArray(
            self._bullet_id,
            [
                self._joints[name].index
                for name in [
                    "front_left_wheel_joint",
                    "front_right_wheel_joint",
                    "rear_left_wheel_joint",
                    "rear_right_wheel_joint",
                ]
            ],
            pybullet.TORQUE_CONTROL,
            forces=[0] * 4,
        )
        self._clear_step_cache()

    def _set_road_friction(self, current_simulation_time):
        """Sets the road friction coefficient if fricition map
        exists and the vehicle is located in the defined regions
        in scenario file.
        """
        pos = self.pose.point.as_shapely
        # A check to see if we are in a surface patch.
        for surface_patch in self._friction_map:
            if pos.within(surface_patch["zone"].to_geometry()) and (
                surface_patch["begin_time"]
                < current_simulation_time
                < surface_patch["end_time"]
            ):

                self._update_tire_parameters(surface_patch["friction coefficient"])
                return
        # If we are not in any surface patch then use the initial
        # tire parameters values.
        self._reset_tire_parameters()

    def _reset_tire_parameters(self):
        self._update_tire_parameters(None)

    def _update_tire_parameters(self, tire_model_parameters):
        """Updates the tire parameters if we are inside a surface patch
        defined in scenario file. If it is used with None argument,
        then it resets the tire paramters to their respective
        initial values.
        """
        if self._tire_model is not None:
            if tire_model_parameters is None:
                tire_model_parameters = self._tire_parameters["road_friction"]
            self._tire_model.road_friction = tire_model_parameters
            return
        if tire_model_parameters is None:
            tire_model_parameters = (
                self._road_wheel_frictions["wheel_friction"]
                * self._road_wheel_frictions["road_friction"]
            )
        wheel_indices = [
            self._joints[name].index
            for name in [
                "front_left_wheel_joint",
                "front_right_wheel_joint",
                "rear_left_wheel_joint",
                "rear_right_wheel_joint",
            ]
        ]
        for wheel_id in wheel_indices:
            self._client.changeDynamics(
                self._bullet_id,
                wheel_id,
                lateralFriction=tire_model_parameters
                / self._road_wheel_frictions["road_friction"],
            )
        self._clear_step_cache()
