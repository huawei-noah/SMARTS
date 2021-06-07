import math
from pathlib import Path
from unittest import mock
import numpy as np
import matplotlib.pyplot as plt

from smarts.core.controllers.actuator_dynamic_controller import (
    ActuatorDynamicController,
    ActuatorDynamicControllerState,
)
from smarts.core.coordinates import Pose, Heading
from smarts.core.vehicle import Vehicle
from smarts.core.chassis import AckermannChassis
from smarts.core.utils import pybullet
from smarts.core.utils.pybullet import bullet_client as bc


TIMESTEP_SEC = 1 / 240
# TIMESTEP_SEC=0.0005
xx = []
yy = []
time = []
speed = []
vy = []
ay = []
rvx = []
flN = []
frN = []
rlN = []
rrN = []
latforce = []
latmoment = []
rlslipangle = []
rlFy = []
yaw = []


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
    for yi in range(n_steps):

        if not client.isConnected():
            print("Client got disconnected")
            return

        action = [
            client.readUserDebugParameter(sliders["throttle"]),
            client.readUserDebugParameter(sliders["brake"]),
            client.readUserDebugParameter(sliders["steering"]),
        ]
        # if (yi * TIMESTEP_SEC) > 2:
        #     action[0] = 1
        # if (yi * TIMESTEP_SEC) > 5 and (yi * TIMESTEP_SEC) < 10:
        #     action[2] = 0.5
        # if (yi * TIMESTEP_SEC) > 10 and (yi * TIMESTEP_SEC) < 15:
        #     action[2] = -0.5
        # if (yi * TIMESTEP_SEC) > 15:
        #     raise Exception("time is more than 20")

        # ActuatorDynamicController.perform_action(
        #     vehicle, action, controller_state, dt_sec=TIMESTEP_SEC
        # )
        ActuatorDynamicController.perform_action(
            vehicle, action, controller_state, dt_sec=TIMESTEP_SEC
        )

        z_yaw = vehicle.chassis.velocity_vectors[1][2]
        xx.append(vehicle.position[0])
        yy.append(vehicle.position[1])
        rvx.append(z_yaw * vehicle.chassis.longitudinal_lateral_speed[0])
        vy.append(vehicle.chassis.longitudinal_lateral_speed[1])
        latforce.append(
            (1 / 2500)
            * (
                sum(vehicle.chassis._lat_forces[2:4])
                + math.cos(vehicle.chassis.steering)
                * sum(vehicle.chassis._lat_forces[0:2])
                - math.sin(vehicle.chassis.steering)
                * sum(vehicle.chassis._lon_forces[0:2])
            )
        )
        print("Lateral Forces:", vehicle.chassis._lat_forces)
        print("Steering:", vehicle.chassis.steering)
        print(
            "Distribution:",
            vehicle.chassis.front_rear_axle_CG_distance[1],
            sum(vehicle.chassis._lat_forces[2:4]),
        )

        kk = 1.5

        latmoment.append(
            (-1 / 3150)
            * (
                (3 - kk) * sum(vehicle.chassis._lat_forces[2:4])
                - kk
                * (
                    +math.cos(vehicle.chassis.steering)
                    * sum(vehicle.chassis._lat_forces[0:2])
                    - math.sin(vehicle.chassis.steering)
                    * sum(vehicle.chassis._lon_forces[0:2])
                )
            )
        )

        speed.append(vehicle.speed)
        yaw.append(vehicle.chassis.yaw_rate[2])
        time.append(yi * TIMESTEP_SEC)
        # print(client.getDynamicsInfo(vehicle._chassis._bullet_id,-1),"ppppppppppppppppp")

        print(client.getLinkState(vehicle._chassis._bullet_id, 0), "ppppppppppppppppp")
        # client.changeDynamics(vehicle._chassis._bullet_id,-1,lateralFriction=0)
        # client.changeDynamics(plane_body_id,0,lateralFriction=0)
        # client.changeDynamics(plane_body_id,-1,lateralFriction=0)

        # for iii in range(7):
        #     client.changeDynamics(vehicle._chassis._bullet_id,iii,lateralFriction=0)
        # print(client.getDynamicsInfo(vehicle._chassis._bullet_id,iii),"ppppppppppppppppp",iii)
        ss = list(
            client.getContactPoints(plane_body_id, vehicle._chassis._bullet_id, -1, 2)
        )
        ss1 = list(
            client.getContactPoints(plane_body_id, vehicle._chassis._bullet_id, -1, 4)
        )
        ss2 = list(
            client.getContactPoints(plane_body_id, vehicle._chassis._bullet_id, -1, 5)
        )
        ss3 = list(
            client.getContactPoints(plane_body_id, vehicle._chassis._bullet_id, -1, 6)
        )

        for i in ss:
            flN.append(i[9])
            break

        for i in ss1:
            frN.append(i[9])
            break

        for i in ss2:
            rlN.append(i[9])
            break

        for i in ss3:
            rrN.append(i[9])
            break

        client.stepSimulation()

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
        lateral_friction=client.addUserDebugParameter("Lateral Friction", 0, 1, 0.0),
        spinning_friction=client.addUserDebugParameter("Spinning Friction", 0, 1, 0),
        rolling_friction=client.addUserDebugParameter("Rolling Friction", 0, 1, 0),
        contact_stiffness=client.addUserDebugParameter(
            "Contact Stiffness", 0, 1e16, 100000
        ),
        contact_damping=client.addUserDebugParameter("Contact Damping", 0, 1e16, 3500),
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
                "hello",
                pose=pose,
                chassis=AckermannChassis(
                    pose=pose,
                    bullet_client=client,
                    tire_parameters_filepath="/home/kyber/MainProjectSMARTS/SMARTS/tools/tire_parameters.yaml",
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
        # plt.plot(xx,yy)
        # plt.plot(time, speed)
        # vy=savgol_filter(vy, 15, 7)
        ay = [0] + [(x - vy[i - 1]) / TIMESTEP_SEC for i, x in enumerate(vy)][1:]
        ayaw = [0] + [(x - yaw[i - 1]) / TIMESTEP_SEC for i, x in enumerate(yaw)][1:]
        ayfilt = []
        for i in range(len(ay)):
            if i > 20:
                ayfilt.append(sum(ay[i - 15 : i]) / 15)
            else:
                ayfilt.append(ay[i])

        plt.figure(1)
        # plt.plot(time,-np.array(savgol_filter(ay, 5, 2))+np.array(rvx))
        plt.plot(time, -np.array(ay) + np.array(rvx))
        plt.title("Lateral acceleration")
        plt.xlabel("Time (sec)")
        plt.ylabel("Ay(blue) m/(sec^2)")
        # plt.plot(time,-np.array(ayfilt)+np.array(rvx[0:len(ayfilt)]))
        # # plt.plot(time,savgol_filter(ay, 11, 3))
        plt.plot(time, latforce)
        # plt.plot(time,ay)
        # plt.figure(2)

        # plt.plot(time[:-1],flN,'red',time[:-1],frN,'green',time[:-1],rlN,'blue',time[:-1],rrN,'brown')
        # plt.title('Normal Forces')
        # plt.xlabel('Time(sec)')
        # plt.ylabel('Normal Forces N')
        # plt.scatter(rlslipangle,rlFy)
        # plt.xlabel('RL slip angle')
        # plt.ylabel('Rl Fy(lateral force) N')
        # plt.title('red:fl,green:fr')
        # plt.xlabel('Time (sec)')
        # plt.ylabel('Normal Forces')
        # plt.ylabel('Lateral Accelartion (m/sec^2)')
        plt.figure(2)
        plt.plot(time, ayaw)
        plt.plot(time, latmoment)
        plt.title("Time Step=1/240")
        plt.xlabel("Time (sec)")
        plt.ylabel(
            "Angular Acceleration(blue),Sum of moments/inertia(orange) Rad/(sec^2)"
        )

        plt.figure(3)
        plt.plot(time, yaw)

        plt.show()
