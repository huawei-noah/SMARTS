import math
from pathlib import Path
from unittest import mock

from smarts.core.controllers.actuator_dynamic_controller import (
    ActuatorDynamicController,
    ActuatorDynamicControllerState,
)

from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc
from smarts.core.coordinates import Pose, Heading
from smarts.core.vehicle import Vehicle
from smarts.core.chassis import AckermannChassis


TIMESTEP_SEC = 1 / 240


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

    controller_state = ActuatorDynamicControllerState()
    for _ in range(n_steps):
        if not client.isConnected():
            print("Client got disconnected")
            return

        action = [
            client.readUserDebugParameter(sliders["throttle"]),
            client.readUserDebugParameter(sliders["brake"]),
            client.readUserDebugParameter(sliders["steering"]),
        ]
        ActuatorDynamicController.perform_action(
            vehicle, action, controller_state, dt_sec=TIMESTEP_SEC
        )

        client.stepSimulation()
        vehicle.sync_to_renderer()

        frictions_ = frictions(sliders)

        if prev_friction_sum is not None and not math.isclose(
            sum(frictions_.values()), prev_friction_sum
        ):
            print("Updating")
            return  # will reset and take us to the next episode

        prev_friction_sum = sum(frictions_.values())

        look_at(client, vehicle.position, top_down=True)
        print(
            "Speed: {:.2f} m/s, Position: {}, Heading:{:.2f}, Sumo-Heading:{:.2f}".format(
                vehicle.speed,
                vehicle.position,
                vehicle.heading,
                vehicle.heading.as_sumo,
            ),
            end="\r",
        )


def frictions(sliders):
    return dict(
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
        throttle=client.addUserDebugParameter("Throttle", 0, 1, 0.0),
        brake=client.addUserDebugParameter("Brake", 0, 1, 0),
        steering=client.addUserDebugParameter("Steering", -10, 10, 0),
        lateral_friction=client.addUserDebugParameter("Lateral Friction", 0, 1, 0.6),
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
