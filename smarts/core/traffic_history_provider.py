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
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import logging
from functools import lru_cache
from typing import Iterable, Optional, Set

from cached_property import cached_property
from shapely.geometry import Polygon

from .actor_role import ActorRole
from .controllers import ActionSpaceType
from .coordinates import Dimensions, Heading, Point, Pose
from .provider import ProviderState
from .traffic_provider import TrafficProvider
from .utils.math import rounder_for_dt
from .vehicle import VEHICLE_CONFIGS, VehicleState


class TrafficHistoryProvider(TrafficProvider):
    """A provider that replays traffic history for simulation."""

    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._histories = None
        self._scenario = None
        self._is_setup = False
        self._replaced_vehicle_ids = set()
        self._last_step_vehicles = set()
        self._this_step_dones = set()
        self._vehicle_id_prefix = "history-vehicle-"
        self._start_time_offset = 0

    @property
    def start_time(self):
        """The start time of the traffic playback"""
        return self._start_time_offset

    @start_time.setter
    def start_time(self, start_time: float):
        assert start_time >= 0, "start_time should be positive"
        self._start_time_offset = start_time

    @property
    def done_this_step(self):
        """The vehicles that are to be removed this step."""
        return self._this_step_dones

    def setup(self, scenario) -> ProviderState:
        """Initialize this provider with the given scenario."""
        self._scenario = scenario
        self._histories = scenario.traffic_history
        if self._histories:
            self._histories.connect_for_multiple_queries()
        self._is_setup = True
        return ProviderState(source=__file__)

    def set_replaced_ids(self, vehicle_ids: Iterable[str]):
        """Replace the given vehicles, excluding them from control by this provider."""
        self._replaced_vehicle_ids.update(self._get_base_id(v_id) for v_id in vehicle_ids)

    @lru_cache(maxsize=128)
    def _get_base_id(self, vehicle_id: str):
        if vehicle_id.startswith(self._vehicle_id_prefix):
            return vehicle_id[len("history-vehicle-"):]
        return vehicle_id

    def reset(self):
        pass

    def teardown(self):
        self._is_setup = False
        if self._histories:
            self._histories.disconnect()
            self._histories = None
        self._scenario = None
        self._replaced_vehicle_ids = set()

    def destroy(self):
        pass

    @property
    def action_spaces(self) -> Set[ActionSpaceType]:
        return set()

    def sync(self, provider_state):
        # Ignore other sim state
        pass

    def _dbid_to_vehicle_id(self, dbid) -> str:
        return self._vehicle_id_prefix + str(dbid)

    def step(
        self, provider_actions, dt: float, elapsed_sim_time: float
    ) -> ProviderState:
        if not self._histories:
            return ProviderState(source=__file__, vehicles=[])
        vehicles = []
        vehicle_ids = set()
        rounder = rounder_for_dt(dt)
        history_time = rounder(self._start_time_offset + elapsed_sim_time)
        prev_time = rounder(history_time - dt)
        rows = self._histories.vehicles_active_between(prev_time, history_time)
        for hr in rows:
            v_id = self._dbid_to_vehicle_id(hr.vehicle_id)
            if v_id in vehicle_ids or v_id in self._replaced_vehicle_ids:
                continue
            vehicle_ids.add(v_id)
            vehicle_config_type = self._histories.decode_vehicle_type(hr.vehicle_type)
            vehicles.append(
                VehicleState(
                    vehicle_id=v_id,
                    vehicle_config_type=vehicle_config_type,
                    pose=Pose.from_center(
                        (hr.position_x, hr.position_y, 0), Heading(hr.heading_rad)
                    ),
                    # Note: Neither NGSIM nor INTERACTION provide the vehicle height
                    dimensions=Dimensions.init_with_defaults(
                        hr.vehicle_length,
                        hr.vehicle_width,
                        hr.vehicle_height,
                        defaults=VEHICLE_CONFIGS[vehicle_config_type].dimensions,
                    ),
                    speed=hr.speed,
                    source=self.source_str,
                    role=ActorRole.Social,
                )
            )
        self._this_step_dones = self._last_step_vehicles - vehicle_ids
        self._last_step_vehicles = vehicle_ids
        return ProviderState(vehicles=vehicles)

    @property
    def _history_vehicle_ids(self) -> Set[str]:
        if not self._histories:
            return set()
        return {
            self._dbid_to_vehicle_id(hvid) for hvid in self._histories.all_vehicle_ids()
        }

    @property
    def _my_vehicles(self) -> Set[str]:
        return self._history_vehicle_ids - self._replaced_vehicle_ids

    def manages_vehicle(self, vehicle_id: str) -> bool:
        return vehicle_id in self._my_vehicles

    def stop_managing(self, vehicle_id: str):
        self._replaced_vehicle_ids.add(vehicle_id)

    def reserve_traffic_location_for_vehicle(
        self, vehicle_id: str, reserved_location: Polygon
    ):
        pass

    def vehicle_collided(self, vehicle_id: str):
        # Here we might remove the vehicle_id from history replay, i.e.:
        #     self.stop_managing(vehicle_id)
        # OR we might consider handing the vehicle off to another
        # provider to manage from here on out.
        # But this collision MIGHT have explicitly been part of the original
        # traffic history data, so we don't do either of those things just in case.
        pass

    def vehicle_dest_road(self, vehicle_id: str) -> Optional[str]:
        try:
            vid = int(vehicle_id.replace(self._vehicle_id_prefix, ""))
            pos_x, pos_y = self._histories.vehicle_final_position(vid)
            final_lane = self._scenario.road_map.nearest_lane(Point(pos_x, pos_y))
            return final_lane.road.road_id
        except:
            self._logger.warning(
                "called vehicle_dest_road() for non-history vehicle_id: {vehicle_id}"
            )
            return None

    def can_accept_vehicle(self, state: VehicleState) -> bool:
        # TAI consider:
        # return state.vehicle_id in self._replaced_vehicle_ids and state.pose "is close to" self._histories.vehicle_pose_at_time(state.vehicle_id, self._sim.elapsed_sim_time)
        return False
