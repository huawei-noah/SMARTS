#!/usr/bin/env python3

import rospy
import std_msgs
from smarts_ros.msg import EntitiesStamped, EntityState

import time
import numpy as np
from threading import Lock

from envision.client import Client as Envision
from smarts.core.coordinates import BoundingBox, Heading, Pose
from smarts.core.scenario import Scenario
from smarts.core.smarts import SMARTS
from smarts.core.vehicle import VehicleState


class ROSDriver:
    """Wraps SMARTS as a ROS (v1) node.
    See the README.md in `examples/ros` for
    instructions on how to setup and run."""

    def __init__(self):
        self._state_lock = Lock()
        self._smarts = None
        self._reset()

    def _reset(self):
        if self._smarts:
            self._smarts.destroy()
        self._smarts = None
        self._scenarios_iterator = None
        self._state_topic = None
        self._publisher = None
        with self._state_lock:
            self._reset_smarts = False
            self._latest_state = None

    def setup_ros(
        self,
        node_name: str = "SMARTS",
        namespace: str = "SMARTS",
        pub_queue_size: int = 10,
    ):
        assert not self._publisher
        # enforce only one SMARTS instance per ROS network...
        rospy.init_node(node_name, anonymous=False)

        out_topic = f"{namespace}/entities_out"
        self._publisher = rospy.Publisher(
            out_topic, EntitiesStamped, queue_size=pub_queue_size
        )

        self._state_topic = f"{namespace}/entities_in"
        rospy.Subscriber(self._state_topic, EntitiesStamped, self._entities_callback)
        rospy.Subscriber(f"{namespace}/reset", std_msgs.msg.Bool, self._reset_callback)

    def setup_smarts(self, scenarios: str, headless: bool, seed: int):
        assert not self._smarts
        self._smarts = SMARTS(
            agent_interfaces={},
            traffic_sim=None,
            fixed_time_step=None,
            envision=None if headless else Envision(),
            external_provider=True,
        )
        assert self._smarts.external_provider
        self._scenarios_iterator = Scenario.scenario_variations(scenarios, list([]))
        self._last_step_time = None
        self._reset_smarts = True

    def _reset_callback(self, param):
        with self._state_lock:
            self._reset_smarts = param.data

    def _entities_callback(self, entities: EntitiesStamped):
        # Here we just only keep the latest msg.
        # In the future, we may want to buffer them so that
        # when we update the SMARTS state we can do something clever
        # like smooth/interpolate-over/extrapolate-from them.
        with self._state_lock:
            self._latest_state = entities

    def _update_smarts_state(self, step_delta: float) -> bool:
        with self._state_lock:
            state_to_send = self._latest_state
            self._latest_state = None  # ensure we don't resend same one later
        if not state_to_send:
            rospy.logdebug(
                f"No messages received on topic {self._state_topic} yet to send to SMARTS."
            )
            return False
        entities = []
        for entity in state_to_send.entities:
            pos = entity.pose.position
            pos = np.array((pos.x, pos.y, pos.z))
            qt = entity.pose.orientation
            qt = np.array((qt.x, qt.y, qt.z, qt.w))
            vv = entity.velocity.linear
            linear_velocity = np.array((vv.x, vv.y, vv.z))
            av = entity.velocity.angular
            angular_velocity = np.array((av.x, av.y, av.z))
            lacc = entity.acceleration.linear
            linear_acc = np.array((lacc.x, lacc.y, lacc.z))
            aacc = entity.acceleration.angular
            angular_acc = np.array((aacc.x, aacc.y, aacc.z))
            vs = VehicleState(
                source="EXTERNAL",
                vehicle_id=entity.entity_id,
                vehicle_config_type="passenger",
                privileged=True,
                pose=Pose(pos, qt),
                dimensions=BoundingBox(entity.length, entity.width, entity.height),
                speed=np.linalg.norm(linear_velocity),
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
                linear_acceleration=linear_acc,
                angular_acceleration=angular_acc,
            )
            entities.append(vs)
        staleness = (rospy.get_rostime() - state_to_send.header.stamp).to_sec()
        rospy.logdebug(
            f"sending state to SMARTS w/ step_delta={step_delta}, staleness={staleness}..."
        )
        self._smarts.external_provider.state_update(entities, step_delta, staleness)
        return True

    @staticmethod
    def _vector_to_xyz(v, xyz):
        xyz.x, xyz.y, xyz.z = v[0], v[1], v[2]

    @staticmethod
    def _vector_to_xyzw(v, xyz):
        xyz.x, xyz.y, xyz.z, xyz.w = v[0], v[1], v[2], v[3]

    def _publish_state(self):
        smarts_state = self._smarts.external_provider.all_vehicle_states
        entities = EntitiesStamped()
        entities.header.stamp = rospy.Time.now()
        for vehicle in smarts_state:
            entity = EntityState()
            entity.entity_id = vehicle.vehicle_id
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
        self._publisher.publish(entities)

    def _check_reset(self):
        with self._state_lock:
            if self._reset_smarts:
                rospy.loginfo(f"resetting SMARTS")
                self._smarts.reset(next(self._scenarios_iterator))
                self._reset_smarts = False

    def run_forever(self):
        if not self._publisher:
            raise Exception("must call setup_ros() first.")
        if not self._smarts or not self._scenarios_iterator:
            raise Exception("must call setup_smarts() first.")
        step_delta = None
        while not rospy.is_shutdown():
            self._check_reset()
            if self._last_step_time:
                step_delta = rospy.get_time() - self._last_step_time
            self._last_step_time = rospy.get_time()
            self._update_smarts_state(step_delta)
            self._smarts.step({}, step_delta)
            self._publish_state()
        self._reset()


def _parse_args():
    from examples.argument_parser import default_argument_parser

    parser = default_argument_parser()
    parser.add_argument(
        "--node-name",
        help="The name to use for this ROS node.",
        type=str,
        default="SMARTS",
    )
    parser.add_argument(
        "--namespace",
        help="The ROS namespace to use for published/subscribed toipc names.",
        type=str,
        default="SMARTS",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    driver = ROSDriver()
    driver.setup_ros(args.node_name, args.namespace)
    driver.setup_smarts(args.scenarios, args.headless, args.seed)
    driver.run_forever()
