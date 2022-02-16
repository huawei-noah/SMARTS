#!/usr/bin/env python3

# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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

import json
import logging
import math
import os
import sys
import time
from collections import deque
from threading import Lock
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import rospy
from smarts_ros.msg import (
    AgentReport,
    AgentSpec,
    AgentsStamped,
    EntitiesStamped,
    EntityState,
    SmartsReset,
)
from smarts_ros.srv import SmartsInfo, SmartsInfoRequest, SmartsInfoResponse

from envision.client import Client as Envision
from smarts.core.agent import Agent
from smarts.core.coordinates import Dimensions, Heading, Pose
from smarts.core.plan import (
    EndlessGoal,
    Mission,
    PositionalGoal,
    Start,
    VehicleSpec,
    default_entry_tactic,
)
from smarts.core.scenario import Scenario
from smarts.core.sensors import Observation
from smarts.core.smarts import SMARTS
from smarts.core.utils.math import (
    fast_quaternion_from_angle,
    vec_to_radians,
    yaw_from_quaternion,
)
from smarts.core.vehicle import VehicleState
from smarts.ros.logging import log_everything_to_ROS
from smarts.sstudio.types import MapSpec
from smarts.zoo import registry


class ROSDriver:
    """Wraps SMARTS as a ROS (v1) node.
    See the README.md in `examples/ros` for
    instructions on how to setup and run."""

    def __init__(self):
        self._state_lock = Lock()
        self._reset_lock = Lock()
        self._smarts = None
        self._reset()

    def _reset(self):
        if self._smarts:
            self._smarts.destroy()
        self._smarts = None
        self._target_freq = None
        self._state_topic = None
        self._state_publisher = None
        self._agents_publisher = None
        self._most_recent_state_sent = None
        self._warned_about_freq = False
        with self._state_lock:
            self._recent_state = deque(maxlen=3)
        with self._reset_lock:
            self._reset_msg = None
            self._scenario_path = None
            self._agents = {}
            self._agents_to_add = {}

    def setup_ros(
        self,
        node_name: str = "SMARTS",
        def_namespace: str = "SMARTS/",
        buffer_size: int = 3,
        target_freq: float = None,
        time_ratio: float = 1.0,
        pub_queue_size: int = 10,
    ):
        """Set up the SMARTS ros node."""
        assert not self._state_publisher

        # enforce only one SMARTS instance per ROS network...
        # NOTE: The node name specified here may be overridden by ROS
        # remapping arguments from the command line invocation.
        rospy.init_node(node_name, anonymous=False)

        # If the namespace is aready set in the environment, we use it,
        # otherwise we use our default.
        namespace = def_namespace if not os.environ.get("ROS_NAMESPACE") else ""

        self._service_name = f"{namespace}{node_name}_info"

        self._state_publisher = rospy.Publisher(
            f"{namespace}entities_out", EntitiesStamped, queue_size=pub_queue_size
        )
        self._agents_publisher = rospy.Publisher(
            f"{namespace}agents_out", AgentsStamped, queue_size=pub_queue_size
        )

        rospy.Subscriber(f"{namespace}reset", SmartsReset, self._smarts_reset_callback)

        self._state_topic = f"{namespace}entities_in"
        rospy.Subscriber(self._state_topic, EntitiesStamped, self._entities_callback)

        rospy.Subscriber(f"{namespace}agent_spec", AgentSpec, self._agent_spec_callback)

        buffer_size = rospy.get_param("~buffer_size", buffer_size)
        if buffer_size and buffer_size != self._recent_state.maxlen:
            assert buffer_size > 0
            self._recent_state = deque(maxlen=buffer_size)

        # If target_freq is not specified, SMARTS is allowed to
        # run as quickly as it can with no delay between steps.
        target_freq = rospy.get_param("~target_freq", target_freq)
        if target_freq:
            assert target_freq > 0.0
            self._target_freq = target_freq

        log_everything_to_ROS(level=logging.WARNING)

    def setup_smarts(
        self, headless: bool = True, seed: int = 42, time_ratio: float = 1.0
    ):
        """Do the setup of the underlying SMARTS instance."""
        assert not self._smarts
        if not self._state_publisher:
            raise RuntimeError("must call setup_ros() first.")

        self._zoo_module = rospy.get_param("~zoo_module", "zoo")

        headless = rospy.get_param("~headless", headless)
        seed = rospy.get_param("~seed", seed)
        time_ratio = rospy.get_param("~time_ratio", time_ratio)
        assert time_ratio > 0.0
        self._time_ratio = time_ratio

        self._smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            fixed_timestep_sec=None,
            envision=None if headless else Envision(),
            external_provider=True,
        )
        assert self._smarts.external_provider
        self._last_step_time = None
        with self._reset_lock:
            self._reset_msg = None
            self._scenario_path = None
            self._agents = {}
            self._agents_to_add = {}

    def _smarts_reset_callback(self, reset_msg: SmartsReset):
        with self._reset_lock:
            self._reset_msg = reset_msg
            self._agents = {}
            self._agents_to_add = {}
        for ros_agent_spec in reset_msg.initial_agents:
            self._agent_spec_callback(ros_agent_spec)

    def _get_smarts_info(self, req: SmartsInfoRequest) -> SmartsInfoResponse:
        resp = SmartsInfoResponse()
        resp.header.stamp = rospy.Time.now()
        if not self._smarts:
            rospy.logwarn("get_smarts_info() called before SMARTS set up.")
            return resp
        resp.version = self._smarts.version
        resp.step_count = self._smarts.step_count
        resp.elapsed_sim_time = self._smarts.elapsed_sim_time
        if self._smarts.scenario:
            resp.current_scenario_path = self._smarts.scenario.root_filepath
        return resp

    def _entities_callback(self, entities: EntitiesStamped):
        # note: push/pop is thread safe on a deque but
        # in our smoothing we are accessing all elements
        # so we still need to protect it.
        with self._state_lock:
            self._recent_state.append(entities)

    @staticmethod
    def _decode_entity_type(entity_type: int) -> str:
        if entity_type == EntityState.ENTITY_TYPE_CAR:
            return "passenger"
        if entity_type == EntityState.ENTITY_TYPE_TRUCK:
            return "truck"
        if entity_type == EntityState.ENTITY_TYPE_TRAILER:
            return "trailer"
        if entity_type == EntityState.ENTITY_TYPE_BUS:
            return "bus"
        if entity_type == EntityState.ENTITY_TYPE_COACH:
            return "coach"
        if entity_type == EntityState.ENTITY_TYPE_PEDESTRIAN:
            return "pedestrian"
        if entity_type == EntityState.ENTITY_TYPE_MOTORCYCLE:
            return "motorcycle"
        if entity_type == EntityState.ENTITY_TYPE_UNSPECIFIED:
            return "passenger"
        rospy.logwarn(
            f"unsupported entity_type {entity_type}. defaulting to passenger car."
        )
        return "passenger"

    @staticmethod
    def _encode_entity_type(entity_type: str) -> int:
        if entity_type in ["passenger", "car"]:
            return EntityState.ENTITY_TYPE_CAR
        if entity_type == "truck":
            return EntityState.ENTITY_TYPE_TRUCK
        if entity_type == "trailer":
            return EntityState.ENTITY_TYPE_TRAILER
        if entity_type == "bus":
            return EntityState.ENTITY_TYPE_BUS
        if entity_type == "coach":
            return EntityState.ENTITY_TYPE_COACH
        if entity_type == "pedestrian":
            return EntityState.ENTITY_TYPE_PEDESTRIAN
        if entity_type == "motorcycle":
            return EntityState.ENTITY_TYPE_MOTORCYCLE
        if entity_type is None:
            return EntityState.ENTITY_TYPE_UNSPECIFIED
        rospy.logwarn(f"unsupported entity_type {entity_type}. defaulting to 'car'.")
        return EntityState.ENTITY_TYPE_CAR

    @staticmethod
    def _decode_vehicle_type(vehicle_type: int) -> str:
        if vehicle_type == AgentSpec.VEHICLE_TYPE_CAR:
            return "passenger"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_TRUCK:
            return "truck"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_TRAILER:
            return "trailer"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_BUS:
            return "bus"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_COACH:
            return "coach"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_PEDESTRIAN:
            return "pedestrian"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_MOTORCYCLE:
            return "motorcycle"
        if vehicle_type == AgentSpec.VEHICLE_TYPE_UNSPECIFIED:
            return "passenger"
        rospy.logwarn(
            f"unsupported vehicle_type {vehicle_type}. defaulting to passenger car."
        )
        return "passenger"

    @staticmethod
    def _pose_from_ros(ros_pose) -> Pose:
        return Pose(
            position=(ros_pose.position.x, ros_pose.position.y, ros_pose.position.z),
            orientation=(
                ros_pose.orientation.x,
                ros_pose.orientation.y,
                ros_pose.orientation.z,
                ros_pose.orientation.w,
            ),
        )

    def _agent_spec_callback(self, ros_agent_spec: AgentSpec):
        assert (
            len(ros_agent_spec.tasks) == 1
        ), "more than 1 task per agent is not yet supported"
        task = ros_agent_spec.tasks[0]
        task_params = json.loads(task.params_json) if task.params_json else {}
        task_version = task.task_ver or "latest"
        agent_locator = f"{self._zoo_module}:{task.task_ref}-{task_version}"
        agent_spec = None
        try:
            agent_spec = registry.make(agent_locator, **task_params)
        except ImportError as ie:
            rospy.logerr(f"Unable to locate agent with locator={agent_locator}:  {ie}")
        if not agent_spec:
            rospy.logwarn(
                f"got unknown task_ref '{task.task_ref}' in AgentSpec message with params='{task.param_json}'.  ignoring."
            )
            return
        if (
            ros_agent_spec.end_pose.position.x != 0.0
            or ros_agent_spec.end_pose.position.y != 0.0
        ):
            goal = PositionalGoal(
                (
                    ros_agent_spec.end_pose.position.x,
                    ros_agent_spec.end_pose.position.y,
                ),
                ros_agent_spec.veh_length,
            )
        else:
            goal = EndlessGoal()
        mission = Mission(
            start=Start.from_pose(ROSDriver._pose_from_ros(ros_agent_spec.start_pose)),
            goal=goal,
            # TODO:  how to prevent them from spawning on top of another existing vehicle? (see how it's done in SUMO traffic)
            entry_tactic=default_entry_tactic(ros_agent_spec.start_speed),
            vehicle_spec=VehicleSpec(
                veh_id=f"veh_for_agent_{ros_agent_spec.agent_id}",
                veh_config_type=ROSDriver._decode_vehicle_type(ros_agent_spec.veh_type),
                dimensions=Dimensions(
                    ros_agent_spec.veh_length,
                    ros_agent_spec.veh_width,
                    ros_agent_spec.veh_height,
                ),
            ),
        )
        with self._reset_lock:
            if (
                ros_agent_spec.agent_id in self._agents
                or ros_agent_spec.agent_id in self._agents_to_add
            ):
                rospy.logwarn(
                    f"trying to add new agent with existing agent_id '{ros_agent_spec.agent_id}'.  ignoring."
                )
                return
            self._agents_to_add[ros_agent_spec.agent_id] = (agent_spec, mission)

    @staticmethod
    def _xyz_to_vect(xyz) -> np.ndarray:
        return np.array((xyz.x, xyz.y, xyz.z))

    @staticmethod
    def _xyzw_to_vect(xyzw) -> np.ndarray:
        return np.array((xyzw.x, xyzw.y, xyzw.z, xyzw.w))

    @staticmethod
    def _entity_to_vs(entity: EntityState) -> VehicleState:
        veh_id = entity.entity_id
        veh_type = ROSDriver._decode_entity_type(entity.entity_type)
        veh_dims = Dimensions(entity.length, entity.width, entity.height)
        vs = VehicleState(
            source="EXTERNAL",
            vehicle_id=veh_id,
            vehicle_config_type=veh_type,
            pose=Pose(
                ROSDriver._xyz_to_vect(entity.pose.position),
                ROSDriver._xyzw_to_vect(entity.pose.orientation),
            ),
            dimensions=veh_dims,
            linear_velocity=ROSDriver._xyz_to_vect(entity.velocity.linear),
            angular_velocity=ROSDriver._xyz_to_vect(entity.velocity.angular),
            linear_acceleration=ROSDriver._xyz_to_vect(entity.acceleration.linear),
            angular_acceleration=ROSDriver._xyz_to_vect(entity.acceleration.angular),
        )
        vs.set_privileged()
        vs.speed = np.linalg.norm(vs.linear_velocity)
        return vs

    class _VehicleStateVector:
        def __init__(self, vs: VehicleState, stamp: float):
            self.vs = vs
            self.vector = np.concatenate(
                (
                    np.array((stamp,)),
                    vs.pose.position,
                    np.array((vs.pose.heading,)),
                    np.array(vs.dimensions.as_lwh),
                    vs.linear_velocity,
                    vs.angular_velocity,
                    vs.linear_acceleration,
                    vs.angular_acceleration,
                )
            )

        @property
        def stamp(self):
            """The estimated timestamp of this vehicle state."""
            return self.vector[0]

        def average_with(self, other_vect: np.ndarray):
            """Update this vehicle state with the average between this state and the given new
            state.
            """
            self.vector += other_vect
            self.vector /= 2
            self.update_vehicle_state()

        def update_vehicle_state(self):
            """Update this vehicle state."""
            assert len(self.vector) == 20
            self.vs.pose = Pose.from_center(
                self.vector[1:4], Heading(self.vector[4] % (2 * math.pi))
            )
            self.vs.dimensions = Dimensions(*self.vector[5:8])
            self.linear_velocity = self.vector[8:11]
            self.angular_velocity = self.vector[11:14]
            self.linear_acceleration = self.vector[14:17]
            self.angular_acceleration = self.vector[17:]
            self.vs.speed = np.linalg.norm(self.linear_velocity)

    @staticmethod
    def _moving_average(
        vehicle_id: str, states: Sequence[EntitiesStamped]
    ) -> Tuple[VehicleState, float, float, float]:
        prev_states = []
        for s in range(len(states)):
            prev_state = states[-1 - s]
            for entity in prev_state.entities:
                if entity.entity_id == vehicle_id:
                    vs = ROSDriver._entity_to_vs(entity)
                    stamp = prev_state.header.stamp.to_sec()
                    prev_states.append(ROSDriver._VehicleStateVector(vs, stamp))
        assert prev_states
        if len(prev_states) == 1:
            return (prev_states[0].vs, prev_states[0].stamp, 0, 0)
        prev_states[0].average_with(prev_states[1].vector)
        if len(prev_states) == 2:
            return (prev_states[0].vs, prev_states[0].stamp, 0, 0)
        prev_states[1].average_with(prev_states[2].vector)
        dt = prev_states[0].stamp - prev_states[1].stamp
        lin_acc_slope = (
            prev_states[0].vs.linear_acceleration
            - prev_states[1].vs.linear_acceleration
        ) / dt
        ang_acc_slope = (
            prev_states[0].vs.angular_acceleration
            - prev_states[1].vs.angular_acceleration
        ) / dt
        return (prev_states[0].vs, prev_states[0].stamp, lin_acc_slope, ang_acc_slope)

    @staticmethod
    def _extrapolate_to_now(
        vs: VehicleState, staleness: float, lin_acc_slope: float, ang_acc_slope: float
    ):
        """Here we just linearly extrapolate the acceleration to "now" from the previous state for
        each vehicle and then use standard kinematics to project the velocity and position from that."""
        # The following ~10 lines are a hack b/c I'm too stupid to figure out
        # how to do calculus on quaternions...
        heading = vs.pose.heading
        heading_delta_vec = staleness * (
            vs.angular_velocity
            + 0.5 * vs.angular_acceleration * staleness
            + ang_acc_slope * staleness ** 2 / 6.0
        )
        heading += vec_to_radians(heading_delta_vec[:2]) + (0.5 * math.pi)
        heading %= 2 * math.pi
        vs.pose.orientation = fast_quaternion_from_angle(heading)
        # XXX: also need to remove the cached heading_ since we've changed orientation
        vs.pose.heading_ = None

        # I assume the following should be updated based on changing
        # heading from above, but I'll leave that for now...
        vs.pose.position += staleness * (
            vs.linear_velocity
            + 0.5 * vs.linear_acceleration * staleness
            + lin_acc_slope * staleness ** 2 / 6.0
        )

        vs.linear_velocity += staleness * (
            vs.linear_acceleration + 0.5 * lin_acc_slope * staleness
        )
        vs.speed = np.linalg.norm(vs.linear_velocity)
        vs.angular_velocity += staleness * (
            vs.angular_acceleration + 0.5 * ang_acc_slope * staleness
        )
        vs.linear_acceleration += staleness * lin_acc_slope
        vs.angular_acceleration += staleness * ang_acc_slope

    def _update_smarts_state(self) -> float:
        with self._state_lock:
            if (
                not self._recent_state
                or self._most_recent_state_sent == self._recent_state[-1]
            ):
                rospy.logdebug(
                    f"No messages received on topic {self._state_topic} yet to send to SMARTS."
                )
                states = None
            else:
                states = [s for s in self._recent_state]

        rosnow = rospy.get_rostime()
        if self._last_step_time:
            step_delta = (rosnow - self._last_step_time).to_sec()
        else:
            step_delta = None
        self._last_step_time = rosnow
        if not states:
            return step_delta

        # Note: when the source of these states is a co-simulator
        # running on another machine across the network, for accurate
        # extrapolation and staleness-related computations, it is
        # a good idea to either use an external time server or a
        # ROS /clock node (in which case the /use_sim_time parameter
        # shoule be set to True).
        entities = []
        most_recent_state = states[-1]
        for entity in most_recent_state.entities:
            vs, stamp, lin_acc_slope, ang_acc_slope = ROSDriver._moving_average(
                entity.entity_id, states
            )
            entity_staleness = rosnow.to_sec() - stamp
            if entity_staleness > 0:
                ROSDriver._extrapolate_to_now(
                    vs, entity_staleness, lin_acc_slope, ang_acc_slope
                )
            entities.append(vs)

        rospy.logdebug(
            f"sending state to SMARTS w/ step_delta={step_delta}, approximate staleness={(rosnow - most_recent_state.header.stamp).to_sec()}..."
        )
        self._smarts.external_provider.state_update(entities, step_delta)
        self._most_recent_state_sent = most_recent_state
        return step_delta

    @staticmethod
    def _vector_to_xyz(v, xyz):
        xyz.x, xyz.y, xyz.z = v[0], v[1], v[2]

    @staticmethod
    def _vector_to_xyzw(v, xyzw):
        xyzw.x, xyzw.y, xyzw.z, xyzw.w = v[0], v[1], v[2], v[3]

    def _publish_state(self):
        smarts_state = self._smarts.external_provider.all_vehicle_states
        entities = EntitiesStamped()
        entities.header.stamp = rospy.Time.now()
        for vehicle in smarts_state:
            entity = EntityState()
            entity.entity_id = vehicle.vehicle_id
            entity.entity_type = ROSDriver._encode_entity_type(vehicle.vehicle_type)
            entity.length = vehicle.dimensions.length
            entity.width = vehicle.dimensions.width
            entity.height = vehicle.dimensions.height
            ROSDriver._vector_to_xyz(vehicle.pose.position, entity.pose.position)
            ROSDriver._vector_to_xyzw(vehicle.pose.orientation, entity.pose.orientation)
            ROSDriver._vector_to_xyz(vehicle.linear_velocity, entity.velocity.linear)
            ROSDriver._vector_to_xyz(vehicle.angular_velocity, entity.velocity.angular)
            if vehicle.linear_acceleration:
                ROSDriver._vector_to_xyz(
                    vehicle.linear_acceleration, entity.acceleration.linear
                )
            if vehicle.angular_acceleration:
                ROSDriver._vector_to_xyz(
                    vehicle.angular_acceleration, entity.acceleration.angular
                )
            entities.entities.append(entity)
        self._state_publisher.publish(entities)

    def _publish_agents(
        self, observations: Dict[str, Observation], dones: Dict[str, bool]
    ):
        agents = AgentsStamped()
        agents.header.stamp = rospy.Time.now()
        for agent_id, agent_obs in observations.items():
            veh_state = agent_obs.ego_vehicle_state
            pose = Pose.from_center(veh_state.position, veh_state.heading)
            agent = AgentReport()
            agent.agent_id = agent_id
            ROSDriver._vector_to_xyz(pose.position, agent.pose.position)
            ROSDriver._vector_to_xyzw(pose.orientation, agent.pose.orientation)
            agent.speed = veh_state.speed
            agent.distance_travelled = agent_obs.distance_travelled
            agent.is_done = dones[agent_id]
            agent.reached_goal = agent_obs.events.reached_goal
            agent.did_collide = bool(agent_obs.events.collisions)
            agent.is_wrong_way = agent_obs.events.wrong_way
            agent.is_off_route = agent_obs.events.off_route
            agent.is_off_road = agent_obs.events.off_road
            agent.is_on_shoulder = agent_obs.events.on_shoulder
            agent.is_not_moving = agent_obs.events.not_moving
            agents.agents.append(agent)
        self._agents_publisher.publish(agents)

    def _do_agents(self, observations: Dict[str, Observation]) -> Dict[str, Any]:
        with self._reset_lock:
            actions = {
                agent_id: self._agents[agent_id].act(agent_obs)
                for agent_id, agent_obs in observations.items()
            }
            for agent_id, agent in self._agents_to_add.items():
                spec, mission = agent[0], agent[1]
                assert agent_id not in self._agents
                agent = spec.build_agent()
                # XXX: hack! in the future this injection must be removed or done another way...
                agent.sim = self._smarts
                self._agents[agent_id] = agent
                self._smarts.add_agent_with_mission(agent_id, spec.interface, mission)
            self._agents_to_add = {}
            return actions

    def _get_map_spec(self) -> MapSpec:
        """SMARTS ROS nodes can extend from this ROSDriver base class
        and implement this method to return an alternative MapSpec object
        designating the map builder to use in the Scenario (returning None
        indicates to use the default for the current Scenario).
        self._scenario_path can be used to construct a MapSpec object."""
        return None

    def _check_reset(self) -> Dict[str, Observation]:
        with self._reset_lock:
            if self._reset_msg:
                self._scenario_path = self._reset_msg.scenario
                rospy.loginfo(f"resetting SMARTS w/ scenario={self._scenario_path}")
                self._reset_msg = None
                self._last_step_time = None
                self._recent_state = deque(maxlen=3)
                self._most_recent_state_sent = None
                self._warned_about_freq = False
                map_spec = self._get_map_spec()
                return self._smarts.reset(
                    Scenario(self._scenario_path, map_spec=map_spec)
                )
        return None

    def run_forever(self):
        """Publish the SMARTS ros node and run indefinitely."""
        if not self._state_publisher:
            raise RuntimeError("must call setup_ros() first.")
        if not self._smarts:
            raise RuntimeError("must call setup_smarts() first.")

        rospy.Service(self._service_name, SmartsInfo, self._get_smarts_info)

        warned_scenario = False
        observations = {}
        step_delta = None
        if self._target_freq:
            rate = rospy.Rate(self._target_freq)
        rospy.loginfo(f"starting to spin")
        try:
            while not rospy.is_shutdown():

                obs = self._check_reset()
                if not self._scenario_path:
                    if not warned_scenario:
                        rospy.loginfo("waiting for scenario on reset channel...")
                        warned_scenario = True
                    elif self._last_step_time:
                        rospy.loginfo("no more scenarios.  exiting...")
                        break
                    continue
                if obs is not None:
                    observations = obs

                try:
                    actions = self._do_agents(observations)

                    step_delta = self._update_smarts_state()

                    observations, _, dones, _ = self._smarts.step(actions, step_delta)

                    self._publish_state()
                    self._publish_agents(observations, dones)
                except Exception as e:
                    if isinstance(e, rospy.ROSInterruptException):
                        raise e
                    batch_mode = rospy.get_param("~batch_mode", False)
                    if not batch_mode:
                        raise e
                    import traceback

                    rospy.logerr(f"SMARTS raised exception:  {e}")
                    rospy.logerr(traceback.format_exc())
                    rospy.logerr("Will wait for next reset...")
                    self._smarts = None
                    self.setup_smarts()

                if self._target_freq:
                    if rate.remaining().to_sec() <= 0.0:
                        msg = f"SMARTS unable to maintain requested target_freq of {self._target_freq} Hz."
                        if self._warned_about_freq:
                            rospy.loginfo(msg)
                        else:
                            rospy.logwarn(msg)
                            self._warned_about_freq = True

                    rate.sleep()

        except rospy.ROSInterruptException:
            rospy.loginfo("ROS interrupted.  exiting...")

        self._reset()  # cleans up the SMARTS instance...


if __name__ == "__main__":
    driver = ROSDriver()
    driver.setup_ros()
    driver.setup_smarts()
    driver.run_forever()
