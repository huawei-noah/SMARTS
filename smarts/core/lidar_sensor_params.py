from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SensorParams:
    start_angle: float
    end_angle: float
    laser_angles: list
    angle_resolution: float
    max_distance: float
    noise_mu: float
    noise_sigma: float


VelodyneHDL32E = SensorParams(
    start_angle=0,
    end_angle=2 * np.pi,
    laser_angles=np.linspace(-np.radians(30.67), np.radians(10.67), 24),
    angle_resolution=0.1728,
    max_distance=100,
    noise_mu=0,
    noise_sigma=0.078,
)

BasicLidar = SensorParams(
    start_angle=0,
    end_angle=2 * np.pi,
    laser_angles=np.linspace(-np.radians(4), np.radians(10), 50),
    angle_resolution=1,
    max_distance=20,
    noise_mu=0,
    noise_sigma=0.078,
)
