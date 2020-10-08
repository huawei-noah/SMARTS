from unittest import mock

import pytest

from smarts.core.sensors import DrivenPathSensor


def test_driven_path_sensor():
    vehicle = mock.Mock()
    sim = mock.Mock()

    max_path_length = 5
    sensor = DrivenPathSensor(vehicle, max_path_length=max_path_length)

    positions = [(x, 0, 0) for x in range(0, 100, 10)]
    sim_times = list(range(0, 50, 5))
    for idx, (position, sim_time) in enumerate(zip(positions, sim_times)):
        sim.elapsed_sim_time = sim_time
        vehicle.position = position
        sensor.track_latest_driven_path(sim)

        if idx >= 3:
            assert sensor.distance_travelled(sim, last_n_steps=3) == 30
            assert sensor.distance_travelled(sim, last_n_seconds=10) == 20

        assert len(sensor()) <= max_path_length

    sensor.teardown()
