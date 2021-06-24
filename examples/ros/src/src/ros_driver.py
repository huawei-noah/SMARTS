#!/usr/bin/env python3

import rospy
import std_msgs
from smarts_ros.msg import EntitiesStamped, EntityState, EntityEvents

import time
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
        self._reset_smarts = False
        self._scenarios_iterator = None
        self._state_topic = None
        self._publisher = None
        with self._state_lock:
            self._latest_state = None

    def setup_ros(self, node_name="SMARTS", namespace="SMARTS", pub_queue_size=10):
        assert not self._publisher
        rospy.init_node(node_name, anonymous=False)  # enforce only one SMARTS instance per ROS network

        out_topic = f"{namespace}/entities_out"
        self._publisher = rospy.Publisher(out_topic, EntitiesStamped, queue_size=pub_queue_size)

        self._state_topic = f"{namespace}/entities_in"
        rospy.Subscriber(self._state_topic, EntitiesStamped, self._entities_callback)
        rospy.Subscriber(f"{namespace}/reset", std_msgs.msg.Bool, self._reset_callback)

    def setup_smarts(self, scenarios: str, headless: bool, seed: int):
        assert not self._smarts
        self._smarts = SMARTS(agent_interfaces={}, traffic_sim=None, envision=None if headless else Envision(), external_state_access=True)
        self._scenarios_iterator = Scenario.scenario_variations(scenarios, list([]))
        self._reset_smarts = True

    def _reset_callback(self, param):
        self._reset_smarts = param.data

    def _entities_callback(self, entities):
        # Here we just only keep the latest msg.
        # In the future, we may want to buffer them so that
        # when we update the SMARTS state we can do something clever
        # like smooth/interpolate-over/extrapolate-from them.
        with self._state_lock:
            self._latest_state = entities

    def _update_smarts_state(self):
        with self._state_lock:
            state_to_send = self._latest_state
        if not state_to_send:
            rospy.loginfo(f"No messages received on topic {self._state_topic} yet to send to SMARTS.")
            return False
        entities = []
        for entity in state_to_send.entities:
            pos = entity.pose.position
            pos = (pos.x, pos.y, pos.z)
            qt = entity.pose.orientation
            qt = (qt.x, qt.y, qt.z, qt.w)
            vv = entity.velocity.linear
            linear_velocity = np.array((vv.x, vv.y, vv.z))
            av = entity.velocity.angular
            angular_velocity = np.array((av.x, av.y, av.z))
            vs = VehicleState(
                source="EXTERNAL",
                vehicle_id=entity.entity_id,
                pose=Pose(pos, qt),
                dimensions=BoundingBox(entity.length, entity.width, entity.height),
                speed=np.linalg.norm(linear_velocity),
                linear_velocity=linear_velocity,
                angular_velocity=angular_velocity,
            )
            entities.append(vs)
        # TODO sim_time / real_time, etc.
        sim_time = time.time()
        self._smarts.external_state_update(sim_time, entities)
        return True

    @staticmethod
    def _vector_to_xyz(v, xyz):
        xyz.x, xyz.y, xyz.z = v[0], v[1], v[2]

    @staticmethod
    def _vector_to_xyzw(v, xyz):
        xyz.x, xyz.y, xyz.z, xyz.w = v[0], v[1], v[2], v[3]

    def _publish_state(self):
        smarts_state = self._smarts.external_state_query()
        entities = EntitiesStamped()
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
            # TODO:  done, events, rewards
            entities.entities.append(entity)
        self._publisher.publish(entities)

    def run_forever(self):
        if not self._publisher:
            raise Exception("must call setup_ros() first.")
        if not self._smarts or not self._scenarios_iterator:
            raise Exception("must call setup_smarts() first.")
        while not rospy.is_shutdown():
            if self._reset_smarts:
                rospy.loginfo(f"resetting SMARTS")
                self._smarts.reset(next(self._scenarios_iterator))
                self._reset_smarts = False
            self._update_smarts_state()
            self._smarts.step({})
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
