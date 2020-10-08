import os
import pytest
import tempfile
from xml.etree.ElementTree import ElementTree
from typing import Sequence

from smarts.sstudio import gen_traffic, gen_missions
from smarts.sstudio.types import (
    Traffic,
    Flow,
    Route,
    TrafficActor,
    Distribution,
    LaneChangingModel,
    JunctionModel,
    Mission,
)


@pytest.fixture
def traffic() -> Traffic:
    car1 = TrafficActor(name="car", speed=Distribution(sigma=0.2, mean=1.0),)
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
                actors={car1: 0.5, car2: 0.5,},
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
