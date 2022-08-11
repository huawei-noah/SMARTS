# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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


from typing import Optional, Sequence

from shapely.geometry import Polygon

from .provider import Provider
from .road_map import RoadMap


class TrafficProvider(Provider):
    """A TrafficProvider is a Provider that controls/owns a (sub)set of vehicles
    that all share the same action space."""

    def reserve_traffic_location_for_vehicle(
        self,
        vehicle_id: str,
        reserved_location: Polygon,
    ):
        """Reserve an area around a location where vehicles cannot spawn until a given vehicle
        is added.
        Args:
            vehicle_id: The vehicle to wait for.
            reserved_location: The space the vehicle takes up.
        """
        raise NotImplementedError

    def vehicle_collided(self, vehicle_id: str):
        """Called when a vehicle this provider manages is detected to have
        collided with any other vehicles in the scenario."""
        raise NotImplementedError

    def update_route_for_vehicle(self, vehicle_id: str, new_route: RoadMap.Route):
        """Set a new route for the given vehicle."""
        raise NotImplementedError

    def vehicle_dest_road(self, vehicle_id: str) -> Optional[str]:
        """Get the final road_id in the route of the given vehicle."""
        raise NotImplementedError

    def route_for_vehicle(self, vehicle_id: str) -> Optional[RoadMap.Route]:
        """Gets the current Route for the specified vehicle, if known."""
        return None

    def destroy(self):
        """Clean up any connections/resources."""
        raise NotImplementedError
