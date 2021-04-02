import math
import random
from pathlib import Path
from unittest import mock
import multiprocessing as mp

import numpy as np

from smarts.core.coordinates import Heading, Pose
from smarts.core.scenario import Scenario
from smarts.core.vehicle import VEHICLE_CONFIGS, Vehicle, VehicleState
from smarts.core.chassis import BoxChassis
from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc
from smarts.core.sumo_traffic_simulation import SumoTrafficSimulation


TIMESTEP_SEC = 1 / 240
INITIAL_KINEMATICS_VEHICLES = 50


def look_at(client, position=(0, 0, 0), top_down=True):
    if top_down:
        client.resetDebugVisualizerCamera(
            cameraTargetPosition=position,
            cameraDistance=100,
            cameraYaw=0,
            cameraPitch=-89.999,
        )
    else:
        client.resetDebugVisualizerCamera(
            cameraTargetPosition=position,
            cameraDistance=30,
            cameraYaw=15,
            cameraPitch=0,
        )


def social_spin_on_bumper_cw(step, front_bumper_position, length):
    return Pose.from_front_bumper(
        np.array(front_bumper_position[:2]), Heading.from_sumo(step), length
    )


def social_spin_on_centre_ccw(step, position):
    return Pose.from_center(np.array(position), Heading.from_panda3d(step))


def social_spin_on_axle_cw(step, base_position, offset_from_centre):
    return Pose.from_explicit_offset(
        offset_from_centre,
        np.array(base_position),
        Heading.from_sumo(step),
        local_heading=Heading(0),
    )


def run(
    client,
    traffic_sim: SumoTrafficSimulation,
    plane_body_id,
    n_steps=1e6,
):
    prev_friction_sum = None
    scenario = next(
        Scenario.variations_for_all_scenario_roots(
            ["scenarios/loop"], agents_to_be_briefed=["007"]
        )
    )
    previous_provider_state = traffic_sim.setup(scenario)
    traffic_sim.sync(previous_provider_state)
    previous_vehicle_ids = set()
    vehicles = dict()

    passenger_dimen = VEHICLE_CONFIGS["passenger"].dimensions

    for step in range(n_steps):
        if not client.isConnected():
            print("Client got disconnected")
            return

        injected_poses = [
            social_spin_on_bumper_cw(step * 0.1, [8, 6, 0], passenger_dimen.length),
            # social_spin_on_centre_ccw(step * 0.1, [8, 0, passenger_dimen[2] / 2]),
            # social_spin_on_axle_cw(
            #     step * 0.1, [0, 0, 0], [2 * passenger_dimen[0], 0, 0]
            # ),
            # Pose(
            #     [0, -6, passenger_dimen[2] * 0.5],
            #     fast_quaternion_from_angle(Heading(0)),
            # ),
        ]

        current_provider_state = traffic_sim.step(0.01)
        for pose, i in zip(injected_poses, range(len(injected_poses))):
            converted_to_provider = VehicleState(
                vehicle_id=f"EGO{i}",
                vehicle_type="passenger",
                pose=pose,
                dimensions=passenger_dimen,
                speed=0,
                source="TESTS",
            )
            current_provider_state.vehicles.append(converted_to_provider)
        traffic_sim.sync(current_provider_state)

        current_vehicle_ids = {v.vehicle_id for v in current_provider_state.vehicles}
        vehicle_ids_removed = previous_vehicle_ids - current_vehicle_ids
        vehicle_ids_added = current_vehicle_ids - previous_vehicle_ids

        for v_id in vehicle_ids_added:
            pose = Pose.from_center([0, 0, 0], Heading(0))
            vehicles[v] = Vehicle(
                id=v_id,
                pose=pose,
                chassis=BoxChassis(
                    pose=pose,
                    speed=0,
                    dimensions=vehicle_config.dimensions,
                    bullet_client=client,
                ),
            )

        # Hide any additional vehicles
        for v in vehicle_ids_removed:
            veh = vehicles.pop(v, None)
            veh.teardown()

        for pv in current_provider_state.vehicles:
            vehicles[pv.vehicle_id].control(pv.pose, pv.speed)

        client.stepSimulation()

        look_at(client, tuple([0, 0, 0]), top_down=False)

        previous_vehicle_ids = current_vehicle_ids

    traffic_sim.teardown()


if __name__ == "__main__":
    # https://turtlemonvh.github.io/python-multiprocessing-and-corefoundation-libraries.html
    # mp.set_start_method('spawn', force=True)

    client = bc.BulletClient(pybullet.GUI)
    # client = BulletClient(pybullet.GUI)
    # client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    client.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME, 1)

    while True:
        client.resetSimulation()
        client.setGravity(0, 0, -9.8)
        client.setPhysicsEngineParameter(
            fixedTimeStep=TIMESTEP_SEC,
            numSubSteps=int(TIMESTEP_SEC / (1 / 240)),
            # enableConeFriction=False,
            # erp=0.1,
            # contactERP=0.1,
            # frictionERP=0.1,
        )

        path = Path(__file__).parent / "../smarts/core/models/plane.urdf"
        path = str(path.absolute())
        plane_body_id = client.loadURDF(path, useFixedBase=True)

        vehicle_config = VEHICLE_CONFIGS["passenger"]

        traffic_sim = SumoTrafficSimulation(
            headless=False, time_resolution=0.1, debug=True
        )

        run(
            client,
            traffic_sim,
            plane_body_id,
            n_steps=int(1e6),
        )
