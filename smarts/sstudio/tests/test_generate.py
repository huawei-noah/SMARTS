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
import os
import tempfile
from typing import Sequence
from xml.etree.ElementTree import ElementTree

import pytest

from smarts.core.scenario import Scenario
from smarts.sstudio import gen_map, gen_missions, gen_traffic
from smarts.sstudio.types import (
    Distribution,
    Flow,
    JunctionModel,
    LaneChangingModel,
    MapSpec,
    Mission,
    Route,
    Traffic,
    TrafficActor,
)


@pytest.fixture
def traffic() -> Traffic:
    car1 = TrafficActor(
        name="car",
        speed=Distribution(sigma=0.2, mean=1.0),
    )
    car2 = TrafficActor(
        name="car",
        speed=Distribution(sigma=0.2, mean=0.8),
        lane_changing_model=LaneChangingModel(impatience=1, cooperative=0.25),
        junction_model=JunctionModel(drive_after_yellow_time=1.0, impatience=0.5),
    )

    return Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=(f"edge-{r[0]}", 0, 30), end=(f"edge-{r[1]}", 0, -30)
                ),
                rate=1.0,
                actors={
                    car1: 0.5,
                    car2: 0.5,
                },
            )
            for r in [("west-WE", "east-WE"), ("east-EW", "west-EW")]
        ]
    )


@pytest.fixture
def missions() -> Sequence[Mission]:
    return [
        Mission(Route(begin=("edge-west-WE", 0, 0), end=("edge-south-NS", 0, 0))),
        Mission(Route(begin=("edge-south-SN", 0, 30), end=("edge-west-EW", 0, 0))),
    ]


def test_generate_traffic(traffic: Traffic):
    with tempfile.TemporaryDirectory() as temp_dir:
        gen_traffic(
            "scenarios/intersections/4lane_t",
            traffic,
            output_dir=temp_dir,
            name="generated",
        )

        with open("smarts/sstudio/tests/baseline.rou.xml") as f:
            items = [x.items() for x in ElementTree(file=f).iter()]

        with open(os.path.join(temp_dir, "traffic", "generated.rou.xml")) as f:
            generated_items = [x.items() for x in ElementTree(file=f).iter()]

        print(sorted(items))
        print(sorted(generated_items))
        assert sorted(items) == sorted(generated_items)


def _gen_map_from_spec(scenario_root: str, map_spec: MapSpec):
    with tempfile.TemporaryDirectory() as temp_dir:
        gen_map(scenario_root, map_spec, output_dir=temp_dir)
        found_map_spec = Scenario.discover_map(temp_dir)
        assert found_map_spec
        road_map = found_map_spec.builder_fn(found_map_spec)
        assert road_map


def test_generate_map():
    scenario_root = "scenarios/intersections/4lane_t"
    map_file = "map.net.xml"

    map_path = os.path.join(scenario_root, map_file)
    map_spec = MapSpec(map_path)
    _gen_map_from_spec(scenario_root, map_spec)

    lw = 5.2
    lps = 3.14

    def fake_map_builder(map_spec: MapSpec) -> bool:
        assert map_spec.source == map_path
        assert map_spec.default_lane_width == lw
        assert map_spec.lanepoint_spacing == lps
        return True

    map_spec = MapSpec(map_path, lps, lw, fake_map_builder)
    _gen_map_from_spec(scenario_root, map_spec)
