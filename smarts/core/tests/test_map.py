# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
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
import math
from os import path
from pathlib import Path

import pytest
from smarts.core.opendrive_road_network import OpenDriveRoadNetwork
from smarts.core.scenario import Scenario
from smarts.core.default_map_factory import create_road_map
from smarts.core.sumo_road_network import SumoRoadNetwork


@pytest.fixture
def sumo_scenario():
    return Scenario(scenario_root="scenarios/intersections/4lane")


@pytest.fixture
def opendrive_scenario():
    return Scenario(scenario_root="scenarios/opendrive")


def test_sumo_map(sumo_scenario):
    road_map = sumo_scenario.road_map
    assert isinstance(road_map, SumoRoadNetwork)

    point = (125.20, 139.0, 0)
    lane = road_map.nearest_lane(point)
    assert lane.lane_id == "edge-north-NS_0"
    assert lane.road.road_id == "edge-north-NS"
    assert lane.index == 0
    assert lane.road.contains_point(point)
    assert lane.is_drivable
    assert len(lane.shape()) >= 2

    right_lane, direction = lane.lane_to_right
    assert not right_lane

    left_lane, direction = lane.lane_to_left
    assert left_lane
    assert direction
    assert left_lane.lane_id == "edge-north-NS_1"
    assert left_lane.index == 1

    lefter_lane, direction = left_lane.lane_to_left
    assert not lefter_lane

    on_roads = lane.road.oncoming_roads_at_point(point)
    assert on_roads
    assert len(on_roads) == 1
    assert on_roads[0].road_id == "edge-north-SN"

    reflinept = lane.to_lane_coord(point)
    assert reflinept.s == 1.0
    assert reflinept.t == 0.0

    offset = reflinept.s
    assert lane.width_at_offset(offset) == 3.2
    assert lane.curvature_radius_at_offset(offset) == math.inf

    on_lanes = lane.oncoming_lanes_at_offset(offset)
    assert not on_lanes
    on_lanes = left_lane.oncoming_lanes_at_offset(offset)
    assert len(on_lanes) == 1
    assert on_lanes[0].lane_id == "edge-north-SN_1"

    in_lanes = lane.incoming_lanes
    assert not in_lanes

    out_lanes = lane.outgoing_lanes
    assert out_lanes
    assert len(out_lanes) == 2
    assert out_lanes[0].lane_id == ":junction-intersection_0_0"
    assert out_lanes[1].lane_id == ":junction-intersection_1_0"

    foes = out_lanes[0].foes
    assert foes
    assert len(foes) == 3
    foe_set = set(f.lane_id for f in foes)
    assert "edge-east-EW_0" in foe_set  # entering from east
    assert "edge-north-NS_0" in foe_set  # entering from north
    assert ":junction-intersection_5_0" in foe_set  # crossing from east-to-west

    r1 = road_map.road_by_id("edge-north-NS")
    assert r1
    assert r1.is_drivable
    assert len(r1.shape()) >= 2
    r2 = road_map.road_by_id("edge-east-WE")
    assert r2
    assert r2.is_drivable
    assert len(r2.shape()) >= 2

    routes = road_map.generate_routes(r1, r2)
    assert routes
    assert len(routes[0].roads) == 4

    route = routes[0]
    db = route.distance_between(point, (198, 65.20, 0))
    assert db == 134.01

    cands = route.project_along(point, 134.01)
    for r2lane in r2.lanes:
        assert (r2lane, 53.6) in cands

    cands = left_lane.project_along(offset, 134.01)
    assert len(cands) == 6
    for r2lane in r2.lanes:
        if r2lane.index == 1:
            assert any(
                r2lane == cand[0] and math.isclose(cand[1], 53.6) for cand in cands
            )


def test_od_map_junction():
    root = path.join(Path(__file__).parent.absolute(), "maps")
    road_map = OpenDriveRoadNetwork.from_file(
        path.join(root, "UC_Simple-X-Junction.xodr")
    )
    assert isinstance(road_map, OpenDriveRoadNetwork)

    # Expected properties for all roads and lanes
    for road_id, road in road_map._roads.items():
        assert type(road_id) == str
        assert road.is_junction is not None
        assert road.length is not None
        assert road.length >= 0
        assert road.parallel_roads == []
        for lane in road.lanes:
            assert lane.in_junction is not None
            assert lane.length is not None
            assert lane.length >= 0

    # Road tests
    r0 = road_map.road_by_id("0_0")
    assert r0
    assert not r0.is_junction
    assert r0.length == 103
    assert len(r0.lanes) == 8
    assert r0.lane_at_index(0) is None
    assert r0.lane_at_index(1).road.road_id == "0_0"
    r0_in_road_ids = set([r.road_id for r in r0.incoming_roads])
    r0_out_road_ids = set([r.road_id for r in r0.outgoing_roads])
    assert r0_in_road_ids == {"5_0", "7_0", "9_0"}
    assert r0_out_road_ids == {"3_0", "8_0", "15_0"}

    r13 = road_map.road_by_id("13_0")
    assert r13
    assert not r13.is_junction
    assert r13.length == 103
    assert len(r13.lanes) == 8
    assert r13.lane_at_index(0) is None
    assert r13.lane_at_index(1).road.road_id == "13_0"
    r13_in_road_ids = set([r.road_id for r in r13.incoming_roads])
    r13_out_road_ids = set([r.road_id for r in r13.outgoing_roads])
    assert r13_in_road_ids == {"10_0", "12_0", "15_0"}
    assert r13_out_road_ids == {"9_0", "11_0", "14_0"}

    # Lane tests
    l1 = road_map.lane_by_id("0_0_1")
    assert l1
    assert l1.road.road_id == "0_0"
    assert l1.index == 1
    assert len(l1.lanes_in_same_direction) == 3
    assert l1.length == 103
    assert l1.is_drivable

    right_lane, direction = l1.lane_to_right
    assert right_lane
    assert direction
    assert right_lane.lane_id == "0_0_2"
    assert right_lane.index == 2

    left_lane, direction = l1.lane_to_left
    assert left_lane
    assert not direction
    assert left_lane.lane_id == "0_0_-1"
    assert left_lane.index == -1

    further_right_lane, direction = right_lane.lane_to_right
    assert further_right_lane
    assert direction
    assert further_right_lane.lane_id == "0_0_3"
    assert further_right_lane.index == 3

    l1_in_lanes = l1.incoming_lanes
    assert not l1_in_lanes

    l1_out_lanes = l1.outgoing_lanes
    assert l1_out_lanes
    assert len(l1_out_lanes) == 3
    assert l1_out_lanes[0].lane_id == "3_0_-1"
    assert l1_out_lanes[1].lane_id == "8_0_-1"
    assert l1_out_lanes[2].lane_id == "15_0_-1"

    # point on lane
    point = (118.0, 170.0, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 33.0
    assert round(refline_pt.t, 2) == 2.0

    offset = refline_pt.s
    assert l1.width_at_offset(offset) == 3.75
    assert l1.curvature_radius_at_offset(offset) == math.inf
    assert l1.contains_point(point)
    assert l1.road.contains_point(point)

    # lane edges on point
    left_edge, right_edge = l1.edges_at_point(point)
    assert (round(left_edge.x, 2), round(left_edge.y, 2)) == (109.7, 170.0)
    assert (round(right_edge.x, 2), round(right_edge.y, 2)) == (120.0, 170.0)

    # road edges on point
    road_left_edge, road_right_edge = r0.edges_at_point(point)
    assert (round(road_left_edge.x, 2), round(road_left_edge.y, 2)) == (130.3, 170.0)
    assert (round(road_right_edge.x, 2), round(road_right_edge.y, 2)) == (116.25, 170.0)

    # point not on lane but on road
    point = (122.0, 170.0, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 33.0
    assert round(refline_pt.t, 2) == -2.0

    offset = refline_pt.s
    assert l1.width_at_offset(offset) == 3.75
    assert l1.curvature_radius_at_offset(offset) == math.inf
    assert not l1.contains_point(point)
    assert l1.road.contains_point(point)

    l2 = road_map.lane_by_id("0_0_-1")
    assert l2
    assert l2.road.road_id == "0_0"
    assert l2.index == -1
    assert l2.is_drivable

    left_lane, direction = l2.lane_to_left
    assert left_lane
    assert not direction
    assert left_lane.lane_id == "0_0_1"
    assert left_lane.index == 1

    l2_in_lanes = l2.incoming_lanes
    assert l2_in_lanes
    assert len(l2_in_lanes) == 3
    assert l2_in_lanes[0].lane_id == "5_0_-1"
    assert l2_in_lanes[1].lane_id == "7_0_-1"
    assert l2_in_lanes[2].lane_id == "9_0_-1"

    l2_out_lanes = l2.outgoing_lanes
    assert not l2_out_lanes

    l3 = road_map.lane_by_id("9_0_-1")
    assert l3
    assert l3.road.road_id == "9_0"
    assert l3.index == -1
    assert l3.is_drivable

    foes = l3.foes
    assert foes
    assert len(foes) == 2
    foe_set = set(f.lane_id for f in foes)
    assert "7_0_-1" in foe_set
    assert "5_0_-1" in foe_set


def test_od_map_figure_eight():
    root = path.join(Path(__file__).parent.absolute(), "maps")
    road_map = OpenDriveRoadNetwork.from_file(path.join(root, "Figure-Eight.xodr"))
    assert isinstance(road_map, OpenDriveRoadNetwork)

    # Expected properties for all roads and lanes
    for road_id, road in road_map._roads.items():
        assert type(road_id) == str
        assert road.is_junction is not None
        assert road.length is not None
        assert road.length >= 0
        assert road.parallel_roads == []
        for lane in road.lanes:
            assert lane.in_junction is not None
            assert lane.length is not None
            assert lane.length >= 0

    # Road tests
    r0 = road_map.road_by_id("508_0")
    assert r0
    assert not r0.is_junction
    assert len(r0.lanes) == 8
    r0_in_road_ids = set([r.road_id for r in r0.incoming_roads])
    r0_out_road_ids = set([r.road_id for r in r0.outgoing_roads])
    assert r0_in_road_ids == {"516_0"}
    assert r0_out_road_ids == {"501_0"}

    # Lane tests
    l1 = road_map.lane_by_id("508_0_-1")
    assert l1
    assert l1.road.road_id == "508_0"
    assert l1.index == -1
    assert l1.is_drivable

    assert len(l1.lanes_in_same_direction) == 3
    assert round(l1.length, 2) == 541.50

    l1_out_lanes = l1.outgoing_lanes
    assert l1_out_lanes
    assert len(l1_out_lanes) == 1
    assert l1_out_lanes[0].lane_id == "501_0_1"

    l1_in_lanes = l1.incoming_lanes
    assert l1_in_lanes
    assert len(l1_in_lanes) == 1
    assert l1_in_lanes[0].lane_id == "516_0_-1"

    l2 = road_map.lane_by_id("501_0_1")

    l2_out_lanes = l2.outgoing_lanes
    assert l2_out_lanes
    assert len(l2_out_lanes) == 3
    assert l2_out_lanes[0].lane_id == "503_0_-1"
    assert l2_out_lanes[1].lane_id == "504_0_-1"
    assert l2_out_lanes[2].lane_id == "513_0_-1"

    l2_in_lanes = l2.incoming_lanes
    assert l2_in_lanes
    assert len(l2_in_lanes) == 1
    assert l2_in_lanes[0].lane_id == "508_0_-1"

    # point on straight part of the lane
    point = (13.0, -17.0, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 7.21
    assert round(refline_pt.t, 2) == -2.83

    offset = refline_pt.s
    assert l1.width_at_offset(offset) == 3.75
    assert l1.curvature_radius_at_offset(offset) == math.inf
    assert l1.contains_point(point)
    assert l1.road.contains_point(point)

    # point on curved part of the lane
    point = (163.56, 75.84, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 358.08
    assert round(refline_pt.t, 2) == -1.75

    offset = refline_pt.s
    assert l1.width_at_offset(offset) == 3.75
    assert round(l1.curvature_radius_at_offset(offset), 2) == 100.0
    assert l1.contains_point(point)
    assert l1.road.contains_point(point)

    # edges on curved part
    left_edge, right_edge = l1.edges_at_point(point)
    assert (round(left_edge.x, 2), round(left_edge.y, 2)) == (162.63, 74.36)
    assert (round(right_edge.x, 2), round(right_edge.y, 2)) == (164.63, 77.53)

    # point not on lane but on road
    point = (163.48, 71.80, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 355.95
    assert round(refline_pt.t, 2) == 1.68

    assert not l1.contains_point(point)
    assert l1.road.contains_point(point)

    # point not on lane, not on road
    l3 = road_map.lane_by_id("508_0_-4")
    assert not l3.is_drivable
    point = (12.0, -28.0, 0)
    refline_pt = l3.to_lane_coord(point)
    offset = refline_pt.s
    assert round(l3.width_at_offset(offset), 2) == 4.7
    assert round(refline_pt.s, 2) == 14.28
    assert round(refline_pt.t, 2) == -5.71

    assert not l3.contains_point(point)
    assert not l3.road.contains_point(point)


def test_od_map_lane_offset():
    root = path.join(Path(__file__).parent.absolute(), "maps")
    road_map = OpenDriveRoadNetwork.from_file(
        path.join(root, "Ex_Simple-LaneOffset.xodr")
    )
    assert isinstance(road_map, OpenDriveRoadNetwork)

    # Expected properties for all roads and lanes
    for road_id, road in road_map._roads.items():
        assert type(road_id) == str
        assert road.is_junction is not None
        assert road.length is not None
        assert road.length >= 0
        assert road.parallel_roads == []
        for lane in road.lanes:
            assert lane.in_junction is not None
            assert lane.length is not None
            assert lane.length >= 0

    # Road tests
    r0 = road_map.road_by_id("1_1")
    assert r0
    assert len(r0.lanes) == 6
    assert not r0.is_junction

    r0_in_road_ids = set([r.road_id for r in r0.incoming_roads])
    r0_out_road_ids = set([r.road_id for r in r0.outgoing_roads])

    assert r0_in_road_ids == {"1_0"}
    assert r0_out_road_ids == {"1_2"}

    r1 = road_map.road_by_id("1_0")
    assert r1
    assert len(r1.lanes) == 5
    assert not r1.is_junction

    r1_in_road_ids = set([r.road_id for r in r1.incoming_roads])
    r1_out_road_ids = set([r.road_id for r in r1.outgoing_roads])
    assert len(r1_in_road_ids) == 0
    assert r1_out_road_ids == {"1_1"}

    r2 = road_map.road_by_id("1_2")
    assert r2
    assert len(r2.lanes) == 5
    assert not r2.is_junction

    r2_in_road_ids = set([r.road_id for r in r2.incoming_roads])
    r2_out_road_ids = set([r.road_id for r in r2.outgoing_roads])
    assert r2_in_road_ids == {"1_1"}
    assert len(r2_out_road_ids) == 0

    # Lane tests
    l0 = road_map.lane_by_id("1_1_1")
    assert l0
    assert l0.road.road_id == "1_1"
    assert l0.index == 1
    assert l0.is_drivable

    assert set([lane.lane_id for lane in l0.incoming_lanes]) == {"1_0_1"}
    assert set([lane.lane_id for lane in l0.outgoing_lanes]) == set()

    right_lane, direction = l0.lane_to_right
    assert right_lane
    assert direction
    assert right_lane.lane_id == "1_1_2"
    assert right_lane.index == 2

    left_lane, direction = l0.lane_to_left
    assert left_lane
    assert not direction
    assert left_lane.lane_id == "1_1_-1"
    assert left_lane.index == -1

    further_right_lane, direction = right_lane.lane_to_right
    assert further_right_lane
    assert direction
    assert further_right_lane.lane_id == "1_1_3"
    assert further_right_lane.index == 3

    l1 = road_map.lane_by_id("1_1_-2")
    assert l1
    assert l1.road.road_id == "1_1"
    assert l1.index == -2
    assert l1.is_drivable

    assert set([lane.lane_id for lane in l1.incoming_lanes]) == set()
    assert set([lane.lane_id for lane in l1.outgoing_lanes]) == {"1_2_-2"}

    right_lane, direction = l1.lane_to_right
    assert right_lane
    assert direction
    assert right_lane.lane_id == "1_1_-3"
    assert right_lane.index == -3

    left_lane, direction = l1.lane_to_left
    assert left_lane
    assert direction
    assert left_lane.lane_id == "1_1_-1"
    assert left_lane.index == -1

    further_left_lane, direction = left_lane.lane_to_left
    assert further_left_lane
    assert not direction
    assert further_left_lane.lane_id == "1_1_1"
    assert further_left_lane.index == 1
