import math
from pathlib import Path
from unittest import mock
import matplotlib.pyplot as plt

from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc
from smarts.core.coordinates import Pose, Heading
from smarts.core.vehicle import Vehicle
from smarts.core.chassis import AckermannChassis
from smarts.core.controllers import (
    TrajectoryTrackingController,
    TrajectoryTrackingControllerState,
)


TIMESTEP_SEC = 1 / 240
(vel, tim, axx, des, head, xdes, ydes, xx, yy) = ([] for i in range(9))


def look_at(client, position=(0, 0, 0), top_down=True):
    if top_down:
        client.resetDebugVisualizerCamera(
            cameraTargetPosition=position,
            cameraDistance=10,
            cameraYaw=0,
            cameraPitch=-89.999,
        )
    else:
        client.resetDebugVisualizerCamera(
            cameraTargetPosition=position,
            cameraDistance=20,
            cameraYaw=15,
            cameraPitch=-50,
        )


def run(client, vehicle, plane_body_id, sliders, n_steps=1e6):
    prev_friction_sum = None

    controller_state = TrajectoryTrackingControllerState()
    for step_num in range(n_steps):
        if not client.isConnected():
            print("Client got disconnected")
            return

        # Simple circular trajectory with radius R and angular velocity Omega in rad/sec
        num_trajectory_points = 15
        R = 40
        omega_1 = 0.1
        omega_2 = 0.2
        if step_num > 1.4 * 3.14 / (TIMESTEP_SEC * omega_1):
            raise ValueError("Terminate and plot.")

        if step_num > 3.14 / (TIMESTEP_SEC * omega_1):
            Omega = omega_2
            alph = ((omega_1 - omega_2) / omega_2) * 3.14 / (TIMESTEP_SEC * omega_1)
        else:
            Omega = omega_1
            alph = 0
        desheadi = step_num * Omega * TIMESTEP_SEC
        trajectory = [
            [
                -(R - R * math.cos((step_num + i + alph) * Omega * TIMESTEP_SEC))
                for i in range(num_trajectory_points)
            ],
            [
                R * math.sin((step_num + i + alph) * Omega * TIMESTEP_SEC)
                for i in range(num_trajectory_points)
            ],
            [
                (step_num + i + alph) * Omega * TIMESTEP_SEC
                for i in range(num_trajectory_points)
            ],
            [R * Omega for i in range(num_trajectory_points)],
        ]

        TrajectoryTrackingController.perform_trajectory_tracking_PD(
            trajectory,
            vehicle,
            controller_state,
            dt_sec=TIMESTEP_SEC,
            heading_gain=0.05,
            lateral_gain=0.65,
            velocity_gain=1.8,
            traction_gain=2,
            derivative_activation=False,
            speed_reduction_activation=False,
        )

        client.stepSimulation()
        vehicle.sync_chassis()

        frictions_ = frictions(sliders)

        if prev_friction_sum is not None and not math.isclose(
            sum(frictions_.values()), prev_friction_sum
        ):
            print("Updating")
            return  # will reset and take us to the next episode

        prev_friction_sum = sum(frictions_.values())

        look_at(client, vehicle.position, top_down=True)
        print("Speed: {:.2f} m/s".format(vehicle.speed), end="\r")
        vel.append(vehicle.speed)
        head.append(vehicle.heading)
        des.append(desheadi)
        tim.append(step_num * TIMESTEP_SEC)
        xx.append(vehicle.position[0])
        yy.append(vehicle.position[1])
        xdes.append(trajectory[0][0])
        ydes.append(trajectory[1][0])


def frictions(sliders):
    return dict(
        throttle=client.addUserDebugParameter("Throttle", 0, 1, 0.0),
        brake=client.addUserDebugParameter("Brake", 0, 1, 0),
        steering=client.addUserDebugParameter("Steering", -10, 10, 0),
        lateralFriction=client.readUserDebugParameter(sliders["lateral_friction"]),
        spinningFriction=client.readUserDebugParameter(sliders["spinning_friction"]),
        rollingFriction=client.readUserDebugParameter(sliders["rolling_friction"]),
        contactStiffness=client.readUserDebugParameter(sliders["contact_stiffness"]),
        contactDamping=client.readUserDebugParameter(sliders["contact_damping"]),
        # anisotropicFriction=client.readUserDebugParameter(
        #     sliders["anisotropic_friction"]
        # ),
    )


if __name__ == "__main__":
    # https://turtlemonvh.github.io/python-multiprocessing-and-corefoundation-libraries.html
    # mp.set_start_method('spawn', force=True)

    client = bc.BulletClient(pybullet.GUI)
    # client = BulletClient(pybullet.GUI)
    # client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
    client.configureDebugVisualizer(pybullet.COV_ENABLE_WIREFRAME, 1)

    sliders = dict(
        lateral_friction=client.addUserDebugParameter("Lateral Friction", 0, 10, 3),
        spinning_friction=client.addUserDebugParameter("Spinning Friction", 0, 1, 0),
        rolling_friction=client.addUserDebugParameter("Rolling Friction", 0, 1, 0),
        contact_stiffness=client.addUserDebugParameter(
            "Contact Stiffness", 0, 1e6, 100000
        ),
        contact_damping=client.addUserDebugParameter("Contact Damping", 0, 1e6, 35000),
        # anisotropic_friction=client.addUserDebugParameter(
        #     "Anisotropic Friction", -1e3, 1e3, 0
        # ),
    )
    try:
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

            client.changeDynamics(plane_body_id, -1, **frictions(sliders))
            pose = pose = Pose.from_center((0, 0, 0), Heading(0))

            vehicle = Vehicle(
                id="vehicle",
                pose=pose,
                chassis=AckermannChassis(
                    pose=pose,
                    bullet_client=client,
                ),
            )

            run(
                client,
                vehicle,
                plane_body_id,
                sliders,
                n_steps=int(1e6),
            )
    except Exception as e:
        print(e)
        # Uncomment for calculating the acceleration
        # ax=[(x-vel[i-1])/TIMESTEP_SEC for i, x in enumerate(vel)][1:]
        plt.figure(1)
        plt.plot(tim, vel)
        plt.title("Velocity")
        plt.xlabel("Time (sec)")
        plt.ylabel("Velocity m/s")
        plt.figure(2)
        plt.title("Path")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.plot(xdes, ydes)
        plt.plot(xx, yy)
        plt.show()
